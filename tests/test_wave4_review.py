"""Regression tests for October 26 independent research-code audit."""
import numpy as np
import pytest
import torch


def test_only_new_tree_edges_charged_to_simulator():
    from rlo.wave4_mcts import Tree,search_many
    try:
        from tiny_test_env import TinyEnv
        env = TinyEnv({})
    except ImportError:
        from rlo.env import CVRPEnv
        env = CVRPEnv(dict(coords=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]], dtype=np.float32),
                           demands=np.array([0,1,1,1], dtype=np.float32),capacity=1))
    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__();self.param=torch.nn.Parameter(torch.tensor(0.))
        def encode(self,nodes,dist):return nodes
        def decode(self,static,nodes,glob,mask):
            logits = torch.zeros(mask.shape, device=nodes.device).masked_fill(~mask, -1e9)
            return logits, nodes.new_zeros(len(nodes))
    tree=Tree(env,np.random.default_rng(3));model=Flat()
    _,stats=search_many(model,[tree],simulations=64)
    assert stats.traversals > 0
    # New edges are instantiated at most once each, even with 64 PUCT walks.
    assert stats.edges == sum(moves for node in _all_nodes(tree.root)
                              for _,moves in node.edges.values())
    assert stats.edges < stats.traversals+stats.edges
    assert stats.expansions > 0


def _all_nodes(root):
    yield root
    for child in root.children.values():yield from _all_nodes(child)


def test_shortlist_reweight_does_not_prune_other_legal_actions():
    pytest.importorskip('rlo.wave3_model')
    from rlo.wave4_train import _conditional_kl_target
    prior=torch.tensor([.1,.2,.3,.4],dtype=torch.float32).log()
    improved=_conditional_kl_target(prior,[1,3],[5.,1.],.2)
    assert torch.allclose(improved[[0,2]],prior.exp()[[0,2]])
    assert torch.allclose(improved.sum(),torch.tensor(1.))
    assert improved[3]>prior.exp()[3]
    with pytest.raises(ValueError):_conditional_kl_target(prior,[1,1],[2.,3.],.2)


def test_dualstream_and_replay_budget_axes():
    pytest.importorskip('rlo.wave3_model')
    from rlo.wave4_train import _budget_for
    counts={'real_moves':100,'synthetic_moves':900}
    assert _budget_for('dualstream',counts)==100
    assert _budget_for('az_scratch',counts)==1000
    assert _budget_for('gvf_mean',counts)==1000
    assert _budget_for('offpolicy',counts)==100


def test_default_entropy_zero():
    pytest.importorskip('rlo.wave3_model')
    from rlo.wave4_train import parser
    args=parser(['--method','offpolicy','--out','/tmp/only-parser','--device','cpu'])
    assert args.entropy==0.
