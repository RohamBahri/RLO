import copy
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rlo.env import CVRPEnv
from rlo.data import instances
from rlo.model import Policy
from rlo.learning import rollout, update
from rlo.mbpi import state_from_experience, estimate_q, improve
from rlo.problem import validate


def small_model():
    torch.manual_seed(11)
    return Policy(width=16, layers=1)


def test_policy_is_permutation_equivariant():
    model = small_model().eval()
    env = CVRPEnv(instances(6, "test", 1)[0])
    nodes, glob, dist, mask = env.observe()
    perm = np.array([0, 3, 1, 6, 2, 5, 4])
    with torch.no_grad():
        out, val = model(torch.tensor(nodes[None]), torch.tensor(glob[None]),
                         torch.tensor(dist[None]), torch.tensor(mask[None]))
        out_p, val_p = model(torch.tensor(nodes[perm][None]), torch.tensor(glob[None]),
                             torch.tensor(dist[np.ix_(perm, perm)][None]),
                             torch.tensor(mask[perm][None]))
    np.testing.assert_allclose(out_p[0].numpy(), out[0][perm].numpy(), atol=1e-5)
    assert val_p.item() == pytest.approx(val.item(), abs=1e-6)


@pytest.mark.parametrize("method", ["reinforce", "rollout", "a2c", "pomo"])
def test_all_four_rl_updates_complete(method):
    model = small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cases = instances(8, "validation", 4)
    before = model.embed.weight.detach().clone()
    metrics = update(model, optimizer, cases, method, starts=2, rng=np.random.default_rng(0),
                     baseline_policy=copy.deepcopy(model) if method == "rollout" else None)
    assert np.isfinite(metrics["loss"])
    assert metrics["feasibility_rate"] == 1
    assert metrics["transitions"] >= 8
    if method == "rollout":
        assert metrics["baseline_episodes"] == 4
        assert metrics["baseline_transitions"] >= 8
        assert metrics["episodes"] == 8
    else:
        assert metrics["baseline_transitions"] == 0
    assert not torch.equal(before, model.embed.weight.detach())
    with torch.no_grad():
        routes, trace = rollout(model, cases, greedy=True)
    assert all(validate(inst, env.finish()) == pytest.approx(env.cost)
               for inst, env in zip(cases, routes))


def test_mbpi_training_uses_policy_rollouts_and_exact_is_only_metric():
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(12)
    env = state_from_experience(model, instances(4, "test", 1)[0], rng)
    assert env is not None
    q, steps = estimate_q(model, env, samples=2)
    assert np.isfinite(q[env.legal()]).all() and steps > 0
    m = improve(model, opt, [env], samples=2, oracle_metrics=True)
    assert m["rollout_transitions"] > 0
    assert m["collection_transitions"] > 0
    assert m["transitions_total"] == m["rollout_transitions"] + m["collection_transitions"]
    assert m["mean_optimal_action_regret"] >= -1e-8
