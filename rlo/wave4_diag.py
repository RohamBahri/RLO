"""Exact small-instance stochastic-policy Q^pi diagnostics, never training labels.

Computes expected cost under the *fixed network policy*, NOT the optimal Q*.
For n=10 dynamic programming memoizes all reachable nonterminal states.
This is expensive and should be run on CPU with controlled instance count.
"""
import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from .env import CVRPEnv
from .data import evaluation_instances,fingerprint
from .wave2_replay import Snapshot
from .wave4_eval import load_policy
from .wave4_inference import probabilities
from .wave4_train import _policy_episode


class ExactPolicyEvaluator:
    def __init__(self,model,instance):
        if len(instance['demands'])-1>10:
            raise ValueError('exact Qpi diagnostics limited to n <= 10')
        self.model,self.instance=model,instance
        self.cache={}

    def _env(self,key):
        visited,current,load=key
        e=CVRPEnv(self.instance)
        e.visited=np.asarray(visited,dtype=bool)
        e.current=current;e.load=load
        e.done=bool(e.visited.all() and e.current==0)
        e.infeasible=False
        return e

    @staticmethod
    def key(e):
        return (tuple(bool(z) for z in e.visited),int(e.current),int(e.load))

    def _step(self,key,action):
        e=self._env(key)
        r,done=e.step(action)
        return float(r),None if done else self.key(e)

    def value(self,key):
        if key in self.cache:return self.cache[key]
        e=self._env(key)
        if e.done:return 0.
        acts=np.flatnonzero(e.legal())
        if len(acts)==1:
            reward,nxt=self._step(key,int(acts[0]))
            v=reward+(0. if nxt is None else self.value(nxt))
        else:
            p=probabilities(self.model,[Snapshot.take(e)])[0]
            v=sum(p[a]*(self.action_value(key,int(a))) for a in acts)
        self.cache[key]=float(v)
        return float(v)

    def action_value(self,key,action):
        reward,next_key=self._step(key,action)
        return reward+(0. if next_key is None else self.value(next_key))

    def action_values(self,snap):
        e=snap.restore()
        key=self.key(e)
        return {int(a):self.action_value(key,int(a)) for a in np.flatnonzero(e.legal())}


def sampled_q(model,snap,M,rng):
    """Own-rollout Q estimates; common sampled-policy distribution for all actions."""
    result={};moves=0
    for a in np.flatnonzero(snap.restore().legal()):
        returns=[]
        for _ in range(M):
            e=snap.restore()
            before=len(e.tour);e.step(int(a))
            _,used=_policy_episode(model,e,rng)
            moves+=len(e.tour)-before   # first action plus policy continuation
            returns.append(-e.cost)
        result[int(a)]=np.asarray(returns)
    return result,moves


def kl_target(prior,q,eta=.2):
    actions=sorted(q)
    vals=np.asarray([q[a] for a in actions])
    # Q is negative cost; higher is better.
    raw=np.log(np.maximum(prior[actions],1.e-40))+(vals-max(vals))/eta
    weights=np.exp(raw-max(raw));weights/=sum(weights)
    return actions,weights


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--roots',type=int,default=16)
    p.add_argument('--rollouts',type=int,nargs='+',default=[2,4,8,32])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.count<1 or a.roots<1 or min(a.rollouts)<1:p.error('invalid sample sizes')
    model=load_policy(a.checkpoint,'cpu')
    rng=np.random.default_rng(a.seed)
    rows=[]
    cases=evaluation_instances(10,'validation',a.count,1.)
    for inst in cases:
        agent=CVRPEnv(inst)
        episodes,_=_policy_episode(model,agent,rng)
        candidates=[s['before'] for s in episodes]
        if len(candidates)>a.roots:
            candidates=[candidates[int(i)] for i in rng.choice(len(candidates),a.roots,replace=False)]
        oracle=ExactPolicyEvaluator(model,inst)
        for root in candidates:
            qexact=oracle.action_values(root)
            prior=probabilities(model,[root])[0]
            exact_actions,exact_target=kl_target(prior,qexact)
            old_cost=-sum(prior[a]*qexact[a] for a in exact_actions)
            oracle_cost=-sum(exact_target[i]*qexact[a] for i,a in enumerate(exact_actions))
            for m in a.rollouts:
                samples,moves=sampled_q(model,root,m,rng)
                mean={a:float(v.mean()) for a,v in samples.items()}
                actions,target=kl_target(prior,mean)
                mse=float(np.mean([(mean[a]-qexact[a])**2 for a in actions]))
                target_cost=-sum(target[i]*qexact[a] for i,a in enumerate(actions))
                divergence=float(np.sum(exact_target*np.log(
                    np.maximum(exact_target,1.e-40)/np.maximum(target,1.e-40))))
                rows.append(dict(instance=fingerprint(inst),state=str(oracle.key(root.restore())),
                                 M=m,rmse=math.sqrt(mse),
                                 local_initial_cost=old_cost,local_exact_target_cost=oracle_cost,
                                 local_sample_target_cost=float(target_cost),
                                 exact_to_sample_target_kl=divergence,
                                 simulated_moves=moves,memoized_states=len(oracle.cache)))
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(rows,indent=2)+'\n')
    for m in a.rollouts:
        subset=[r for r in rows if r['M']==m]
        print(json.dumps(dict(M=m,roots=len(subset),
             mean_rmse=float(np.mean([r['rmse'] for r in subset])),
             mean_local_target_cost=float(np.mean([r['local_sample_target_cost'] for r in subset])))))


if __name__=='__main__':main()
