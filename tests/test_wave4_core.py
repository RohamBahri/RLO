import numpy as np
import pytest
import torch
from rlo.wave4_vtrace import targets,loss
from rlo.wave4_mcts import _backup,_select,Node,supported_behavior,SearchStats


def test_onpolicy_vtrace_returns_for_gamma_one():
    rewards=torch.tensor([-2.,-3.,-4.])
    values=torch.tensor([-5.,-5.,-1.])
    log=torch.tensor([-0.4,-1.1,-0.8])
    vt,adv=targets(rewards,values,log,log)
    assert torch.allclose(vt,torch.tensor([-9.,-7.,-4.]))
    assert torch.allclose(adv,vt-values)


def test_offpolicy_terminal_and_gradient():
    r=torch.tensor([-3.,-4.]);v=torch.tensor([-.1,-.2],requires_grad=True)
    behavior=torch.tensor([-0.3,-0.3])
    policy=torch.tensor([-0.4,-0.5],requires_grad=True)
    obj,comp=loss(r,v,behavior,policy)
    obj.backward()
    assert torch.isfinite(obj) and torch.isfinite(policy.grad).all()
    assert torch.isfinite(v.grad).all()


def test_vtrace_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        targets(torch.tensor([1.]),torch.tensor([0.,1.]),torch.tensor([0.]),torch.tensor([0.]))


def test_search_backup_edge_reward_and_counts():
    root=Node(prior={0:.4,1:.6},visits={0:0,1:0},reward_sum={0:0.,1:0.})
    child=Node(prior={2:1.},visits={2:0},reward_sum={2:0.})
    _backup([(root,1,-2.),(child,2,-3.)],-4.)
    assert root.visits[1]==1 and root.reward_sum[1]==-9
    assert child.visits[2]==1 and child.reward_sum[2]==-7
    assert _select(root,1.25) in root.prior


def test_behavior_full_support():
    visits=np.array([0.,1.,0.,0.]);prior=np.array([.1,.2,.7,0.])
    mu=supported_behavior(visits,prior)
    assert np.isclose(sum(mu),1) and all(mu[prior>0]>0) and mu[3]==0
    with pytest.raises(ValueError):supported_behavior(visits,prior,0)


def test_mcts_batched_and_reused_tree():
    from rlo.wave4_mcts import search_many,Tree
    try:
        from tiny_test_env import TinyEnv
        env=TinyEnv({})
    except ImportError:
        from rlo.env import CVRPEnv
        env=CVRPEnv(dict(coords=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]],dtype=np.float32),
                         demands=np.array([0.,1.,1.,1.],dtype=np.float32),capacity=1))
    class Mock(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,node,glob,dist,mask):
            logits=torch.zeros(mask.shape,device=node.device)
            return logits.masked_fill(~mask,-1e9),node.new_zeros((len(node),))
    rng=np.random.default_rng(7);policy=Mock()
    tree=Tree(env,rng)
    target,stats=search_many(policy,[tree],simulations=20)
    assert abs(target[0].sum()-1)<1.e-12
    assert np.all(target[0][~env.legal()]==0)
    action=int(np.argmax(target[0]));env.step(action);tree.advance(action,env)
    if not env.done:
        target2,stats2=search_many(policy,[tree],simulations=20)
        assert abs(target2[0].sum()-1)<1.e-12
        assert np.all(target2[0][~env.legal()]==0)


def test_exact_residual_cvrp_state_and_prefix():
    from rlo.wave4_inference import exact_residual
    try:
        from tiny_test_env import TinyEnv
        env=TinyEnv({'capacity':1,'demands':np.array([0,1,1])})
    except ImportError:
        from rlo.env import CVRPEnv
        env=CVRPEnv(dict(coords=np.array([[0.,0.],[1.,0.],[2.,0.]],dtype=np.float32),
                         demands=np.array([0.,1.,1.],dtype=np.float32),capacity=1))
    dist,actions,nstates=exact_residual(env,2)
    assert np.isclose(dist,6.)
    assert actions==[1,0,2,0] or actions==[2,0,1,0]
    assert nstates>0
