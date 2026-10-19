import numpy as np
import pytest
from rlo.data import instances
from rlo.env import CVRPEnv, BatchCVRP
from rlo.oracle import ExactOracle
from rlo.exact import solve
from rlo.problem import generate, validate


def test_legacy_is_unchanged_and_tight_capacity_is_feasible():
    rng = np.random.default_rng(42)
    from rlo.data import fingerprint
    assert [fingerprint(generate(20, rng)) for _ in range(4)] == [
        fingerprint(x) for x in instances(20, "legacy", 4)]
    for n in (2, 10, 20):
        inst = generate(n, rng, capacity_factor=0.5)
        assert inst["capacity"] >= max(inst["demands"])


def test_actions_mask_and_auto_moves_match_independent_validator():
    rng = np.random.default_rng(123)
    for n in (1, 3, 10, 20):
        for _ in range(5):
            inst = generate(n, rng, capacity_factor=float(rng.uniform(0.5, 1.3)))
            env = CVRPEnv(inst)
            rewards = []
            while not env.done:
                options = env.legal()
                assert options.sum() >= 2  # forced moves are already executed
                assert options[0] == (env.current != 0)
                for j in range(1, n + 1):
                    expected = not env.visited[j] and env.load + inst["demands"][j] <= inst["capacity"]
                    assert bool(options[j]) == expected
                r, done = env.step(int(rng.choice(np.flatnonzero(options))))
                rewards.append(r)
            assert not env.infeasible
            assert done if n != 1 else env.done
            assert sum(rewards) == pytest.approx(-env.cost, abs=1e-8) if n > 1 else env.done
            assert env.cost == pytest.approx(validate(inst, env.finish()), abs=1e-8)


def test_singleton_actions_and_forced_start():
    instance = instances(4, "validation", 1)[0]
    env = CVRPEnv(instance, first_action=3)
    assert env.tour[1] == 3
    assert env.decisions == 0  # forced start does not enter policy gradient
    with pytest.raises(ValueError, match="illegal"):
        env.move(3)
    with pytest.raises(ValueError, match="step requires") if env.legal().sum() < 2 else pytest.raises(ValueError, match="illegal"):
        env.step(3)
    while not env.done:
        env.step(int(np.flatnonzero(env.legal())[0]))
    assert env.cost == pytest.approx(validate(instance, env.finish()))
    single = CVRPEnv(instances(1, "test", 1)[0])
    assert single.done and single.decisions == 0
    single.finish()


def test_batch_only_one_size():
    batch = BatchCVRP(instances(5, "validation", 3))
    nodes, glob, dist, mask = batch.observe()
    assert nodes.shape == (3, 6, 6)
    assert glob.shape == (3, 2) and dist.shape == (3, 6, 6)
    assert mask.shape == (3, 6)
    with pytest.raises(ValueError):
        BatchCVRP(instances(4, "validation", 1) + instances(5, "validation", 1))


def test_residual_cost_to_go_matches_exact_and_bellman():
    for n in (3, 6, 10):
        inst = instances(n, "test", 1)[0]
        oracle = ExactOracle(inst)
        env = CVRPEnv(inst)
        optimal = validate(inst, solve(inst))
        assert oracle.cost_to_go(env.state) == pytest.approx(optimal, abs=1e-8)
        for _ in range(min(4, n)):
            if env.done:
                break
            values = oracle.action_values(env.state)
            assert np.isfinite(values[env.legal()]).all()
            assert oracle.cost_to_go(env.state) == pytest.approx(min(values), abs=1e-8)
            env.step(int(np.flatnonzero(env.legal())[0]))
            # Prefix cost plus a truly optimal remaining route is an upper bound
            # on the globally optimal tour, because the chosen prefix may be bad.
            assert env.cost + oracle.cost_to_go(env.state) + 1e-8 >= optimal


def test_terminal_residual_has_no_cost():
    inst = instances(4, "test", 1)[0]
    env = CVRPEnv(inst)
    while not env.done:
        env.step(int(np.flatnonzero(env.legal())[0]))
    assert ExactOracle(inst).cost_to_go(env.state) == 0
