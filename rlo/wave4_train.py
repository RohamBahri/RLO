"""Wave 4 research: searched-policy learning, GVF-CAPI, V-trace and curriculum.

Never uses solver/expert training targets. Source is a detached research branch.
Resume is exact at completed collection boundaries (including replay and RNG).
All travel, including automatically forced moves, is accounted physically.
"""
import argparse
import copy
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .campaign_train import evaluate_greedy
from .data import StatefulTrainingBatches
from .env import CVRPEnv
from .wave2_replay import Snapshot, observations
from .wave3_core import environments, trajectories, policy_objective
from .wave3_model import ResearchPolicy
from .wave4_mcts import Tree, SearchStats, search_many, supported_behavior
from .wave4_vtrace import loss as vtrace_loss


WARM = {'az', 'az_rollout', 'vtrace_mcts', 'gvf_mean', 'gvf_blend', 'self_imitation'}
SCRATCH = {'offpolicy', 'recent_vtrace', 'dualstream', 'az_scratch', 'curriculum'}
ALL = WARM | SCRATCH | {'calibrate'}


def policy_tensors(model, snaps):
    dev = next(model.parameters()).device
    node, glob, dist, masks = [torch.as_tensor(x, device=dev)
                                for x in observations(snaps)]
    return model(node, glob, dist, masks), (node, glob, dist, masks)


class ActionGVF(nn.Module):
    """State-action cost-to-go prediction; targets are the policy's own rollouts."""
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*width+2, width), nn.ReLU(),
                                 nn.Linear(width, 1))

    def forward(self, policy, tensors):
        node, glob, dist, _ = tensors
        static = policy.encode(node, dist)
        h = static + policy.dynamic(node[..., 4:6])
        here = h[torch.arange(len(h), device=h.device), node[..., 5].argmax(-1)]
        feat = torch.cat([h, here[:, None, :].expand_as(h),
                          glob[:, None, :].expand(len(h), h.shape[1], 2)], -1)
        return self.net(feat).squeeze(-1)


def _count_physical(env, before):
    return len(env.tour) - before


def _batch_policy_episodes(model, envs, rng, *, greedy=False):
    """Vectorized own-policy experience; cache one encoding per unique instance."""
    if not envs:raise ValueError('empty rollout batch')
    dev=next(model.parameters()).device
    records=[[] for _ in envs]
    physical=0
    unique,ids,lookup=[],[],{}
    for i,e in enumerate(envs):
        key=id(e.instance)
        if key not in lookup:
            lookup[key]=len(unique);unique.append(i)
        ids.append(lookup[key])
    with torch.no_grad():
        obs=[envs[i].observe() for i in unique]
        nodes=torch.as_tensor(np.stack([o[0] for o in obs]),device=dev)
        distances=torch.as_tensor(np.stack([o[2] for o in obs]),device=dev)
        static=model.encode(nodes,distances)[torch.as_tensor(ids,device=dev)]
        while True:
            active=[i for i,e in enumerate(envs) if not e.done]
            if not active:break
            snaps=[Snapshot.take(envs[i]) for i in active]
            current=[envs[i].observe() for i in active]
            node=torch.as_tensor(np.stack([o[0] for o in current]),device=dev)
            glob=torch.as_tensor(np.stack([o[1] for o in current]),device=dev)
            mask=torch.as_tensor(np.stack([o[3] for o in current]),device=dev)
            logits,_=model.decode(static[active],node,glob,mask)
            probs=logits.softmax(-1).cpu().numpy()
            for j,i in enumerate(active):
                row=probs[j]
                action=int(row.argmax()) if greedy else int(rng.choice(len(row),p=row))
                before=len(envs[i].tour)
                reward,done=envs[i].step(action)
                physical+=len(envs[i].tour)-before
                records[i].append(dict(before=snaps[j],action=action,
                    behavior_logp=float(np.log(max(row[action],1.e-40))),
                    reward=float(reward),done=done))
    return records,physical


def _policy_episode(model, env, rng, *, greedy=False):
    episodes,physical=_batch_policy_episodes(model,[env],rng,greedy=greedy)
    return episodes[0],physical


def _searched_episodes(model, envs, rng, *, simulations=32, leaf_type='value',
                       mixture=.05, c_puct=1.25):
    """MCTS behavior on fresh instances, with frozen policy for a whole collection."""
    records = [[] for _ in envs]
    encoding_cache={}
    trees = [None if e.done else Tree(e, rng, c_puct=c_puct, root_noise=True,
                                     encoder_cache=encoding_cache)
             for e in envs]
    physical, stats = 0, SearchStats()
    while True:
        active = [i for i,e in enumerate(envs) if not e.done]
        if not active:
            break
        chosen_trees = [trees[i] for i in active]
        targets, add = search_many(model, chosen_trees, simulations, leaf_type=leaf_type)
        stats.add(add)
        snaps = [Snapshot.take(envs[i]) for i in active]
        with torch.no_grad():
            (logits, _), _ = policy_tensors(model, snaps)
            priors = logits.softmax(-1).cpu().numpy()
        for j, i in enumerate(active):
            # Fixed stochastic mixture ensures coverage; no solver instructions.
            behavior = supported_behavior(targets[j], priors[j], mixture)
            action = int(rng.choice(len(behavior), p=behavior))
            before = len(envs[i].tour)
            reward, done = envs[i].step(action)
            physical += len(envs[i].tour) - before
            records[i].append(dict(before=snaps[j], action=action, reward=float(reward),
                                   behavior_logp=float(np.log(behavior[action])),
                                   visit_target=targets[j], done=done))
            trees[i].advance(action, envs[i])
            if not done:
                trees[i].root_noise = True
    return records, physical, stats


def _actor_critic_loss(model, episodes, *, is_vtrace=True):
    if not episodes:
        raise ValueError("empty actor-critic batch")
    stats, per_episode = [], []
    dev = next(model.parameters()).device
    for episode in episodes:
        if not episode or not episode[-1]['done']:
            raise ValueError("replay expects complete terminal episodes")
        (logits, values), _ = policy_tensors(model, [s['before'] for s in episode])
        dist = Categorical(logits=logits)
        acts = torch.tensor([s['action'] for s in episode], device=dev)
        target_logp = dist.log_prob(acts)
        rewards = torch.tensor([s['reward'] for s in episode], device=dev)
        old = torch.tensor([s['behavior_logp'] for s in episode], device=dev)
        loss, components = vtrace_loss(rewards, values, old, target_logp,
                                       entropy=dist.entropy(), entropy_weight=0.01)
        per_episode.append(loss)
        stats.append(components)
    return torch.stack(per_episode).mean(), dict(
        actor=float(np.mean([x['actor'] for x in stats])),
        critic=float(np.mean([x['critic'] for x in stats])))


def _imitation_loss(model, episodes, *, value_weight=.5):
    snaps, targets, returns = [], [], []
    for ep in episodes:
        for step in ep:
            snaps.append(step['before'])
            targets.append(step['visit_target'])
    # returns from reversed episodes must be aligned with their corresponding steps
    returns = []
    for ep in episodes:
        future = np.cumsum([s['reward'] for s in ep][::-1])[::-1]
        returns.extend(future.tolist())
    (logits, vals), _ = policy_tensors(model, snaps)
    target = torch.tensor(np.stack(targets), dtype=logits.dtype, device=logits.device)
    ce = -(target * logits.log_softmax(-1)).sum(-1).mean()
    mc = torch.tensor(returns, dtype=vals.dtype, device=vals.device)
    critic = .5 * F.smooth_l1_loss(vals, mc)
    return ce + value_weight * critic, dict(actor=float(ce.detach()),
                                           critic=float(critic.detach()))


def _sample_q(model, root, actions, rng, rollouts):
    """Independent counterfactual branches, batched across root actions."""
    envs=[];labels=[];moves=0
    for action in actions:
        for _ in range(rollouts):
            env=root.restore()
            before=len(env.tour)
            env.step(int(action))
            moves+=len(env.tour)-before
            envs.append(env);labels.append(int(action))
    _,suffix=_batch_policy_episodes(model,envs,rng)
    moves+=suffix
    out={int(a):[] for a in actions}
    for action,env in zip(labels,envs):out[action].append(env.cost)
    return {a:np.asarray(v,dtype=np.float64) for a,v in out.items()},moves,len(envs)


def _gvf_update(model, gvf, optimizer, episodes, rng, blend=False, *, rollouts=4, topk=4,
                eta=.2):
    dev = next(model.parameters()).device
    roots = [s['before'] for ep in episodes for s in ep]
    if not roots:
        raise AssertionError('no GVF roots')
    roots = [roots[int(i)] for i in rng.choice(len(roots), size=min(4,len(roots)),
                                               replace=False)]
    targets, labels, support = [], [], []
    synthetic, samples, all_legal_recall = 0, 0, []
    for snap in roots:
        legal = np.flatnonzero(snap.restore().legal())
        (logits, _), tensors = policy_tensors(model, [snap])
        with torch.no_grad():
            estimate = gvf(model, tensors)[0].cpu().numpy()
        k = min(topk, len(legal))
        selected = sorted(legal, key=lambda a: (estimate[a], a))[:k]
        # One out-of-shortlist exploration action, if available, for GVF coverage.
        omitted = [int(x) for x in legal if x not in selected]
        if omitted:
            selected.append(int(rng.choice(omitted)))
        q, moves, nroll = _sample_q(model, snap, selected, rng, rollouts)
        synthetic += moves
        samples += nroll
        costs = np.array([q[a].mean() for a in selected])
        if blend:
            tails = np.array([np.sort(q[a])[:max(1,int(np.ceil(.1*len(q[a]))))].mean()
                              for a in selected])
            costs = .5 * costs + .5 * tails
        # KL projection restricted to sampled candidates, with action-prior support.
        prior = logits.detach().log_softmax(-1)[0]
        t = torch.zeros_like(prior)
        inds = torch.tensor(selected, dtype=torch.long, device=dev)
        raw = prior[inds] - torch.as_tensor(costs-costs.min(),device=dev,dtype=prior.dtype)/eta
        t[inds] = raw.softmax(-1)
        targets.append(t)
        labs = torch.zeros_like(prior)
        labs[inds] = torch.as_tensor([q[a].mean() for a in selected],
                                    dtype=prior.dtype, device=dev)
        labels.append(labs)
        support.append([int(x) for x in selected])
        all_legal_recall.append(len(selected)/len(legal))
    ((logits, _), tensors) = policy_tensors(model, roots)
    gv = gvf(model, tensors)
    qt = torch.stack(targets)
    actor = -(qt * logits.log_softmax(-1)).sum(-1).mean()
    losses=[]
    for i, selected in enumerate(support):
        ids = torch.tensor(selected, device=dev)
        losses.append(F.smooth_l1_loss(gv[i,ids], labels[i][ids]))
    aux = torch.stack(losses).mean()
    obj = actor + .5*aux
    optimizer.zero_grad(set_to_none=True)
    obj.backward()
    clip = torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(gvf.parameters()), 1.)
    if not torch.isfinite(obj) or not torch.isfinite(clip):
        raise FloatingPointError('nonfinite GVF-CAPI loss')
    optimizer.step()
    return dict(loss=float(obj.detach()), actor=float(actor.detach()),
                critic=float(aux.detach()), synthetic=synthetic,
                rollout_samples=samples, candidates=float(np.mean(all_legal_recall)),updates=1)


def _self_imitation(model, optimizer, cases, rng, *, k=16):
    dev=next(model.parameters()).device
    all_envs=[CVRPEnv(inst) for inst in cases for _ in range(k)]
    episodes,moves=_batch_policy_episodes(model,all_envs,rng)
    fresh=sum(len(e.tour)-1 for e in all_envs[::k])
    simulated=moves-fresh
    chosen,weights=[],[]
    for i,inst in enumerate(cases):
        costs=[all_envs[i*k+j].cost for j in range(k)]
        best=int(np.argmin(costs))
        advantage=(float(np.mean(costs))-costs[best])/np.sqrt(len(inst['demands'])-1)
        chosen.append(episodes[i*k+best]);weights.append(max(0.,advantage))
    if not any(w>0 for w in weights):
        return dict(loss=0.,fresh=fresh,synthetic=simulated,updates=0)
    flat=[s for ep in chosen for s in ep]
    (logits,_),_=policy_tensors(model,[s['before'] for s in flat])
    acts=torch.tensor([s['action'] for s in flat],device=dev)
    weights=torch.tensor([w for ep,w in zip(chosen,weights) for _ in ep],
                         dtype=logits.dtype,device=dev)
    obj=-(logits.log_softmax(-1).gather(1,acts[:,None])[:,0]*weights).sum()/weights.sum().clamp_min(1.)
    optimizer.zero_grad(set_to_none=True);obj.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(obj) or not torch.isfinite(grad):
        raise FloatingPointError('nonfinite self imitation')
    optimizer.step()
    return dict(loss=float(obj.detach()),fresh=fresh,synthetic=simulated,updates=1)


def _calibration_update(model, opt, episodes):
    snaps, ret = [], []
    for ep in episodes:
        running=0.
        values=[]
        for step in reversed(ep):
            running += step['reward']
            values.append(running)
        snaps.extend(s['before'] for s in ep)
        ret.extend(reversed(values))
    (logits,pred), _ = policy_tensors(model,snaps)
    target = torch.tensor(ret, dtype=pred.dtype,device=pred.device)
    loss = F.smooth_l1_loss(pred,target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.value.parameters(),1.)
    if not torch.isfinite(loss) or not torch.isfinite(grad):
        raise FloatingPointError('nonfinite critic calibration')
    opt.step()
    return float(loss.detach())


class EpisodeReplay:
    def __init__(self, limit=100000):
        self.limit = limit
        self.by_size={}
        self.size_counts={}

    def append(self, episode):
        n = len(episode[0]['before'].visited)-1
        queue=self.by_size.setdefault(n,deque())
        queue.append(episode)
        self.size_counts[n]=self.size_counts.get(n,0)+len(episode)
        while self.size_counts[n] > self.limit and len(queue)>1:
            self.size_counts[n] -= len(queue.popleft())

    def sample(self, rng, n, count):
        queue=self.by_size[n]
        return [queue[int(rng.integers(len(queue)))] for _ in range(count)]


def sha256_file(filename):
    h=hashlib.sha256()
    with open(filename,'rb') as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def parser(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=sorted(ALL))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--initial-checkpoint',default=None)
    p.add_argument('--expected-initial-moves',type=int,default=25000000)
    p.add_argument('--sizes',type=int,nargs='+',default=[10,20,30])
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=None)
    p.add_argument('--max-moves',type=int,default=None)
    p.add_argument('--validate-every',type=int,default=2500000)
    p.add_argument('--checkpoint-every',type=int,default=1000000)
    p.add_argument('--validation-count',type=int,default=256)
    p.add_argument('--simulations',type=int,default=32)
    p.add_argument('--rollouts',type=int,default=4)
    p.add_argument('--k',type=int,default=16)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-moves',type=int)
    a=p.parse_args(argv)
    if a.batch<1 or a.validate_every<1 or a.checkpoint_every<1 or a.simulations<1 or a.k<2 or a.rollouts<1 or a.validation_count<1:
        p.error('positive batch, budget, rollout and validation settings required')
    if a.lr is None:a.lr=0.0001 if a.method in WARM or a.method=='calibrate' else .0003
    if a.max_moves is None:a.max_moves=1000000 if a.method=='calibrate' else (25000000 if a.method in WARM else 50000000)
    if a.method in WARM | {'calibrate'} and not a.initial_checkpoint and not a.resume:
        p.error('warm starts and critic calibration require --initial-checkpoint')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA requested but unavailable')
    return a


def _initial_model(a):
    model=ResearchPolicy(width=128,layers=2,heads=4,logit_clip=10.).to(a.device)
    if a.initial_checkpoint:
        original=torch.load(a.initial_checkpoint,map_location='cpu',weights_only=False)
        if original.get('wave4_version')==1:
            if original.get('method')!='calibrate' or original.get('seed')!=a.seed:
                raise ValueError('warm source must be same-seed Wave4 calibrated model')
            source=original['model']
        else:
            provenance=original.get('campaign_config',{})
            if original.get('campaign_version')!=1 or provenance.get('method')!='pomo' or provenance.get('seed')!=a.seed:
                raise ValueError('calibration source must be matching Wave1 POMO checkpoint')
            moves=int(original.get('transitions_total',-1))
            if not a.expected_initial_moves <= moves < a.expected_initial_moves+100000:
                raise ValueError('wrong POMO milestone')
            source=original['model']
        model.load_state_dict(source,strict=True)
    return model


def main(argv=None):
    a=parser(argv)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    stream=StatefulTrainingBatches(a.seed,batch=a.batch,sizes=a.sizes,
                                   capacity_factors=(.75,1.25),demand_maxes=(10,))
    model=_initial_model(a)
    gvf=ActionGVF(128).to(a.device) if a.method.startswith('gvf_') else None
    if a.method=='calibrate':
        for name,param in model.named_parameters():
            param.requires_grad=name.startswith('value.')
    params=list(filter(lambda p:p.requires_grad,model.parameters()))
    if gvf is not None:params+=list(gvf.parameters())
    opt=torch.optim.Adam(params,lr=a.lr)
    out=Path(a.out).expanduser().resolve()
    config=vars(a).copy();config.pop('resume');config.pop('stop_after_moves')
    config['out']=str(out)
    if a.initial_checkpoint:config['initial_digest']=sha256_file(a.initial_checkpoint)
    counters=dict(iteration=0,real_moves=0,synthetic_moves=0,optimizer_updates=0,
                  replay_samples=0,network_states=0,model_calls=0,search_traversals=0,
                  episodes=0,train_seconds=0.,elapsed_seconds=0.,
                  last_validate=0,last_checkpoint=0)
    replay=EpisodeReplay()
    if a.resume:
        path=out/'last.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('wave4_version')!=1 or saved['config']!=config:
            raise ValueError('resume with different configuration or version')
        model.load_state_dict(saved['model'])
        if gvf is not None:gvf.load_state_dict(saved['gvf'])
        opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(a.device)
        counters=saved['counters'];replay=saved['replay']
        stream.load_state_dict(saved['stream_rng'])
        rng.bit_generator.state=saved['numpy_rng']
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if a.device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        hist=out/'history.jsonl'
        rows=[json.loads(x) for x in hist.read_text().splitlines()]
        rows=[x for x in rows if x['iteration']<=counters['iteration']]
        if not rows or rows[-1]['iteration']!=counters['iteration']:
            raise ValueError('checkpoint-history mismatch')
        temp=out/'history.jsonl.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        os.replace(temp,hist)
    else:
        if out.exists() and set(x.name for x in out.iterdir())-{'commit.txt'}:
            raise ValueError('refusing populated output directory')
        out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    stop_requested=False
    def on_signal(sig,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGUSR1,on_signal)
    signal.signal(signal.SIGTERM,on_signal)
    started=time.perf_counter()

    def budget():
        if a.method in WARM|{'dualstream','az_scratch'}:
            return counters['real_moves']+counters['synthetic_moves']
        return counters['real_moves']

    def checkpoint():
        counters['last_checkpoint']=budget()
        elapsed=counters['elapsed_seconds']+time.perf_counter()-started
        state=dict(wave4_version=1,method=a.method,seed=a.seed,config=config,
                   model=model.state_dict(),gvf=None if gvf is None else gvf.state_dict(),
                   optimizer=opt.state_dict(),counters=copy.deepcopy({**counters,'elapsed_seconds':elapsed}),
                   replay=replay,stream_rng=stream.state_dict(),
                   numpy_rng=copy.deepcopy(rng.bit_generator.state),python_rng=random.getstate(),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else None)
        tmp=out/'last.pt.tmp';torch.save(state,tmp);os.replace(tmp,out/'last.pt')
        # A safe, inference-only export. Value is trained ONLY in relevant methods.
        public=dict(wave4_version=1,architecture=model.configuration(),model=model.state_dict(),
                    method=a.method,seed=a.seed)
        tmp=out/'policy.pt.tmp';torch.save(public,tmp);os.replace(tmp,out/'policy.pt')
        for mark in ([10_000_000,25_000_000,40_000_000,50_000_000] if a.method not in WARM
                     else [5_000_000,10_000_000,20_000_000,25_000_000]):
            target=out/f'milestone_{mark}.pt'
            if budget()>=mark and not target.exists():os.link(out/'last.pt',target)

    def checked_step(objective):
        opt.zero_grad(set_to_none=True)
        objective.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(objective) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite Wave4 update')
        opt.step()
        return float(objective.detach())

    if budget()>=a.max_moves:
        print('ALREADY_COMPLETE',budget());return
    with (out/'history.jsonl').open('a') as log:
        while budget()<a.max_moves:
            t0=time.perf_counter()
            if a.method=='curriculum':
                stream.sizes=([10] if budget()<10_000_000 else
                              [10,20] if budget()<25_000_000 else [10,20,30])
            cases=next(stream)
            n=len(cases[0]['demands'])-1
            record=dict(n=n,loss=None,optimizer_added=0,fresh_added=0,simulated_added=0)
            if a.method=='calibrate':
                episodes,moves=_batch_policy_episodes(model,[CVRPEnv(c) for c in cases],rng)
                record['fresh_added']=moves
                record['loss']=_calibration_update(model,opt,episodes)
                record['optimizer_added']=1
            elif a.method in {'az','az_rollout','vtrace_mcts','az_scratch'}:
                envs=[CVRPEnv(x) for x in cases]
                ep,real,search=_searched_episodes(model,envs,rng,
                    simulations=a.simulations,
                    leaf_type='rollout' if a.method=='az_rollout' else 'value')
                record.update(fresh_added=real,simulated_added=search.edges+search.rollout_moves,
                    network_states=search.network_states,model_calls=search.model_calls,
                    search_traversals=search.traversals)
                for _ in range(4):
                    if a.method=='vtrace_mcts':obj,details=_actor_critic_loss(model,ep)
                    else:obj,details=_imitation_loss(model,ep)
                    record['loss']=checked_step(obj)
                    record['optimizer_added']+=1
                record.update(details)
            elif a.method.startswith('gvf_'):
                envs=[CVRPEnv(x) for x in cases]
                ep,done,real=trajectories(model,envs,grad=False)
                record['fresh_added']=real
                result=_gvf_update(model,gvf,opt,ep,rng,a.method=='gvf_blend',
                                   rollouts=a.rollouts)
                record.update(loss=result['loss'],actor=result['actor'],critic=result['critic'],
                              sampled_candidates=result['candidates'],
                              synthetic_added=result['synthetic'])
                record['simulated_added']=result['synthetic'];record['optimizer_added']=1
            elif a.method=='self_imitation':
                result=_self_imitation(model,opt,cases,rng,k=a.k)
                record.update(loss=result['loss'],fresh_added=result['fresh'],
                              simulated_added=result['synthetic'],
                              optimizer_added=result['updates'])
            elif a.method in {'offpolicy','recent_vtrace'}:
                ep,moves=_batch_policy_episodes(model,[CVRPEnv(c) for c in cases],rng)
                record['fresh_added']=moves
                for rows in ep:replay.append(rows)
                # Match replay gradient exposure for both arms. S1c samples
                # only current trajectories, with replacement; S1 uses history.
                sample_count=2*len(ep)
                source=(replay.sample(rng,n,sample_count) if a.method=='offpolicy'
                        else [ep[int(rng.integers(len(ep)))] for _ in range(sample_count)])
                for group in (source[i:i+len(ep)] for i in range(0,len(source),len(ep))):
                    obj,details=_actor_critic_loss(model,group)
                    record['loss']=checked_step(obj)
                    record['optimizer_added']+=1
                    record['replay_added']=record.get('replay_added',0)+sum(len(e) for e in group)
                record.update(details)
            elif a.method in {'dualstream','curriculum'}:
                envs=environments(cases,starts=n,rng=rng)
                forced=sum(len(e.tour)-1 for e in envs)
                traced,done,moves=trajectories(model,envs,grad=True)
                record['fresh_added']=moves+forced
                obj,_=policy_objective(model,traced,done,'pomo',starts=n)
                if a.method=='dualstream':
                    # POMO alone does not train the value head; MCTS requires
                    # returns learned from own-policy fresh trajectories.
                    value_predictions=[];mc_returns=[]
                    for seq in traced:
                        suffix=0.
                        future=[]
                        for item in reversed(seq):
                            suffix+=item['reward'];future.append(suffix)
                        value_predictions.extend(item['value'] for item in seq)
                        mc_returns.extend(reversed(future))
                    if value_predictions:
                        predictions=torch.stack(value_predictions)
                        truth=predictions.new_tensor(mc_returns)
                        obj=obj+.5*F.smooth_l1_loss(predictions,truth)

                    roots=[s['before'] for ep in traced for s in ep]
                    take=max(1,round(.25*len(roots)))
                    indices=rng.choice(len(roots),size=take,replace=False)
                    selected=[roots[int(i)] for i in indices]
                    encoder_cache={}
                    trees=[Tree(r.restore(),rng,root_noise=True,encoder_cache=encoder_cache)
                           for r in selected]
                    target,search=search_many(model,trees,a.simulations)
                    (logits,_),_=policy_tensors(model,selected)
                    qt=torch.tensor(np.stack(target),device=logits.device,dtype=logits.dtype)
                    searchloss=-(qt*logits.log_softmax(-1)).sum(-1).mean()
                    obj=obj+searchloss
                    record['simulated_added']=search.edges+search.rollout_moves
                    record.update(network_states=search.network_states,
                                  model_calls=search.model_calls,
                                  search_traversals=search.traversals)
                record['loss']=checked_step(obj)
                record['optimizer_added']=1
            else:raise ValueError('unsupported method')
            counters['real_moves']+=record['fresh_added']
            counters['synthetic_moves']+=record['simulated_added']
            counters['optimizer_updates']+=record['optimizer_added']
            counters['replay_samples']+=record.get('replay_added',0)
            counters['network_states']+=record.get('network_states',0)
            counters['model_calls']+=record.get('model_calls',0)
            counters['search_traversals']+=record.get('search_traversals',0)
            counters['episodes']+=len(cases)
            counters['iteration']+=1
            counters['train_seconds']+=time.perf_counter()-t0
            record.update(iteration=counters['iteration'],real_moves=counters['real_moves'],
                          simulated_moves=counters['synthetic_moves'],
                          total_moves=counters['real_moves']+counters['synthetic_moves'],
                          budget_moves=budget(),optimizer_updates=counters['optimizer_updates'],
                          replay_samples=counters['replay_samples'],
                          train_seconds=counters['train_seconds'],
                          elapsed_seconds=counters['elapsed_seconds']+time.perf_counter()-started,
                          gpu_peak_memory=int(torch.cuda.max_memory_allocated()) if a.device=='cuda' else 0)
            stopping=budget()>=a.max_moves or stop_requested or (a.stop_after_moves is not None and
                                                                 budget()>=a.stop_after_moves)
            if stopping or budget()-counters['last_validate']>=a.validate_every:
                record['validation']=evaluate_greedy(model,[10,20,30],a.validation_count)
                counters['last_validate']=budget()
            log.write(json.dumps(record)+'\n');log.flush()
            if stopping or 'validation' in record or budget()-counters['last_checkpoint']>=a.checkpoint_every:
                checkpoint()
            if stopping or 'validation' in record:
                print(json.dumps({k:record[k] for k in ('iteration','budget_moves','real_moves',
                                'simulated_moves','optimizer_updates','validation') if k in record}),flush=True)
            if stopping:
                print('FINISHED_OR_PAUSED',a.method,flush=True)
                break


if __name__=='__main__':main()
