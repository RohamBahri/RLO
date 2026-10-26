"""Fair inference methods on fixed frozen masked-CVRP policies.

All produced tours are independently validated. No solver output trains policy.
Residual dynamic programming is exact with respect to fixed route prefix;
it is deliberately labelled a hybrid inference method.
"""
from functools import lru_cache
import math
import time
import numpy as np
import torch

from .env import CVRPEnv
from .problem import validate
from .wave2_replay import Snapshot
from .wave4_mcts import Tree, search_many


def probabilities(model, snaps):
    from .wave4_train import policy_tensors
    with torch.no_grad():
        (logits,_),_=policy_tensors(model,snaps)
        return logits.softmax(-1).cpu().numpy()


def _advance(e,action):
    before=len(e.tour)
    e.step(int(action))
    return len(e.tour)-before


def complete(model,instance,*,rng=None,first=None,greedy=True):
    e=CVRPEnv(instance, first)
    moves=len(e.tour)-1
    rng=np.random.default_rng(0) if rng is None else rng
    calls=0
    while not e.done:
        p=probabilities(model,[Snapshot.take(e)])[0]
        calls+=1
        action=int(np.argmax(p)) if greedy else int(rng.choice(len(p),p=p))
        moves+=_advance(e,action)
    e.finish()
    return (e.cost,list(e.tour)),dict(simulated_moves=moves,network_states=calls)


def multistart(model,instance):
    n=len(instance['demands'])-1
    solutions=[];moves=0;calls=0
    for first in range(1,n+1):
        solution,extra=complete(model,instance,first=first)
        solutions.append(solution)
        moves+=extra['simulated_moves'];calls+=extra['network_states']
    return min(solutions,key=lambda r:r[0]),dict(simulated_moves=moves,network_states=calls)


def best_of_k(model,instance,k,rng):
    routes=[];moves=0;calls=0
    for j in range(k):
        route,extra=complete(model,instance,rng=rng,greedy=(k==1))
        routes.append(route)
        moves+=extra['simulated_moves'];calls+=extra['network_states']
    return min(routes,key=lambda r:r[0]),dict(simulated_moves=moves,network_states=calls)


def mcts(model,instance,*,sims,rng,leaf='value'):
    e=CVRPEnv(instance)
    tree=Tree(e,rng,c_puct=1.25,root_noise=False)
    simulated,calls,traversed=0,0,0
    while not e.done:
        targets,stats=search_many(model,[tree],sims,leaf_type=leaf)
        simulated+=stats.edges+stats.rollout_moves
        calls+=stats.network_states
        traversed+=stats.traversals
        action=int(np.argmax(targets[0]))
        e.step(action)
        tree.advance(action,e)
    e.finish()
    return (e.cost,list(e.tour)),dict(simulated_moves=simulated,
           network_states=calls,tree_traversals=traversed)


def _clone(e):
    restored=Snapshot.take(e).restore()
    # Prefix only needed by independent validator after the final solution.
    restored.tour=list(e.tour)
    restored.cost=float(e.cost)
    restored.decisions=e.decisions
    return restored


def beam_search(model,instance,width,*,stochastic=False,rng=None):
    if width<1:raise ValueError('beam width must be positive')
    rng=np.random.default_rng(0) if rng is None else rng
    candidates=[CVRPEnv(instance)]
    used,calls=0,0
    while any(not x.done for x in candidates):
        expanded=[]
        for e in candidates:
            if e.done:
                expanded.append((e.cost,e));continue
            prior=probabilities(model,[Snapshot.take(e)])[0];calls+=1
            actions=np.flatnonzero(e.legal())
            for action in actions:
                child=_clone(e)
                used+=_advance(child,int(action))
                # Greedy complete from branch state; gives a comparable
                # full-tour score rather than a biased prefix-only score.
                if child.done:
                    estimate=child.cost
                else:
                    tail=_clone(child)
                    while not tail.done:
                        p=probabilities(model,[Snapshot.take(tail)])[0];calls+=1
                        used+=_advance(tail,int(np.argmax(p)))
                    estimate=tail.cost
                # Explicit heuristic *perturbed beam*, NOT exact stochastic
                # beam search and not an unbiased without-replacement sample.
                if stochastic:
                    estimate+=float(rng.gumbel()) * 0.1
                expanded.append((estimate,child))
        expanded.sort(key=lambda x:(x[0],x[1].cost,len(x[1].tour)))
        candidates=[x[1] for x in expanded[:width]]
    winner=min(candidates,key=lambda e:e.cost)
    winner.finish()
    return (winner.cost,list(winner.tour)),dict(simulated_moves=used,network_states=calls)


def exact_residual(e,max_remaining=10):
    """Certified optimal completion of the current partial tour, not the global CVRP.

    State=(unserved subset, current location, current used capacity). A return
    to depot is allowed only when currently on a route; no fleet cap applies.
    """
    remaining=[int(x) for x in np.flatnonzero(~e.visited) if x!=0]
    if len(remaining)>max_remaining:
        raise ValueError('residual too large for certified dynamic programming')
    Q=int(e.instance['capacity'])
    q=np.asarray(e.instance['demands'],dtype=int)
    D=e.dist
    ids={c:i for i,c in enumerate(remaining)}
    full=(1<<len(remaining))-1
    decisions={}

    @lru_cache(None)
    def dp(bits,loc,load):
        if not bits:
            return 0. if loc==0 else float(D[loc,0])
        best=float('inf');best_action=None
        if loc!=0:
            v=float(D[loc,0])+dp(bits,0,0)
            if v<best:best,best_action=v,0
        for node in remaining:
            bit=1<<ids[node]
            if bits&bit and load+q[node]<=Q:
                v=float(D[loc,node])+dp(bits^bit,node,load+q[node])
                if v<best-1.e-12:
                    best,best_action=v,node
        if best_action is None:raise AssertionError('unlimited-fleet residual infeasible')
        decisions[(bits,loc,load)]=best_action
        return best

    target=dp(full,int(e.current),int(e.load))
    bits,loc,load=full,int(e.current),int(e.load)
    actions=[]
    while bits:
        a=decisions[(bits,loc,load)]
        actions.append(a)
        if a==0:loc,load=0,0
        else:bits^=1<<ids[a];loc=a;load+=q[a]
    if loc!=0:actions.append(0)
    return target,actions,dp.cache_info().currsize


def handoff(model,instance,remaining=10):
    e=CVRPEnv(instance)
    used,calls=0,0
    while not e.done and int(np.sum(~e.visited[1:]))>remaining:
        p=probabilities(model,[Snapshot.take(e)])[0];calls+=1
        used+=_advance(e,int(np.argmax(p)))
    if not e.done:
        additional,plan,states=exact_residual(e,remaining)
        old=e.cost
        for a in plan:
            if e.done:break
            # CVRPEnv applies forced moves between policy decisions. Skip any
            # actions already executed by that deterministic closure.
            if a==0 and e.current==0:
                continue
            if a>0 and e.visited[a]:
                continue
            if not e.legal()[a]:
                raise AssertionError('residual planner action illegal')
            # CVRPEnv.step() may perform deterministic forced transitions.
            # If it does, consume them from the DP action sequence.
            before=len(e.tour)
            e.step(a)
            used+=len(e.tour)-before
        if abs(e.cost-old-additional)>1.e-7:
            raise AssertionError('residual DP and environment costs disagree')
    e.finish()
    return (e.cost,list(e.tour)),dict(simulated_moves=used,network_states=calls)


def checked(instance,solution):
    cost,tour=solution
    actual=validate(instance,tour)
    if not math.isclose(cost,actual,abs_tol=1.e-7,rel_tol=1.e-7):
        raise AssertionError('independent tour cost disagreement')
    return float(actual)
