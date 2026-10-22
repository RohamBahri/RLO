"""Correctness and tiny end-to-end execution gates for Wave 2 replay/Dyna."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rlo.env import CVRPEnv
from rlo.data import instances
from rlo.model import Policy
from rlo.wave2_replay import (Snapshot, Replay, collect_real, simulate,
                              dyna_from_real, observations, double_dqn_targets,
                              replay_q_update, n_step_real_targets)


def test_decision_reward_contains_all_forced_moves():
    ins = {"coords": np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]], dtype=np.float32),
           "demands": np.array([0.,1.,1.,1.], dtype=np.float32),
           "capacity": 1}
    env = CVRPEnv(ins)
    snapshot = Snapshot.take(env)
    expected = simulate(snapshot, 1)
    assert expected.action == 1
    assert expected.physical_moves == 2  # customer, forced return to depot
    assert expected.reward == pytest.approx(-2.0)
    assert expected.after is not None
    after = expected.after.restore()
    assert after.current == 0
    np.testing.assert_array_equal(after.visited, [True, True, False, False])
    assert after.load == 0
    reward, terminal = env.step(1)
    assert not terminal and reward == pytest.approx(expected.reward)
    # Simulate rest of the route with every transition having >=2 legal actions.
    while not env.done:
        act = int(np.flatnonzero(env.legal())[0])
        env.step(act)
    env.finish()


def test_double_dqn_uses_online_masked_argmax_and_target_value():
    online = torch.tensor([[1., 99., 3.], [5., 4., 8.]])
    target = torch.tensor([[10., 300., 30.], [50., 40., 80.]])
    legal = torch.tensor([[True, False, True], [True, True, False]])
    reward = torch.tensor([-2., -1.])
    terminal = torch.tensor([False, True])
    y = double_dqn_targets(online, target, legal, reward, terminal)
    torch.testing.assert_close(y, torch.tensor([28., -1.]))


def test_counterfactuals_are_from_real_decision_states():
    rng = np.random.default_rng(7)
    cases = instances(5, "validation", 3)
    model = Policy(width=16, layers=1, logit_clip=None)
    buffer = Replay(capacity_per_size=100)
    real, envs = collect_real(model, cases, buffer, rng, epsilon=.4)
    assert all(e.done and not e.infeasible for e in envs)
    assert sum(t.physical_moves for t in real) == sum(len(e.tour)-1 for e in envs)
    assert buffer.real_count() == len(real)
    synth = dyna_from_real(buffer, rng, 10)
    assert len(synth) == 10
    assert all(t.synthetic and t.physical_moves >= 1 for t in synth)
    assert buffer.total_count() == len(real) + 10
    assert all(t.after is None or t.after.restore().legal().sum() >= 2 for t in synth)
    samples = buffer.sample(rng, 8)
    assert len(samples) == 8
    assert len({t.n for t in samples}) == 1
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    target = Policy(width=16, layers=1, logit_clip=None)
    target.load_state_dict(model.state_dict())
    before = model.key.weight.detach().clone()
    rec = replay_q_update(model, target, opt, samples, "cpu")
    assert math.isfinite(rec["loss"]) and math.isfinite(rec["gradient"])
    assert not torch.equal(model.key.weight, before)


def test_mixed_size_replay_rejects_mixed_minibatch():
    a = Snapshot.take(CVRPEnv(instances(5, "validation", 1)[0]))
    b = Snapshot.take(CVRPEnv(instances(6, "validation", 1)[0]))
    with pytest.raises(ValueError, match="mixed-size"):
        observations([a, b])


def run_train(method, dest, *extra):
    cmd = [sys.executable, "-m", "rlo.wave2_train", "--method", method,
           "--sizes", "4", "6", "--width", "16", "--layers", "1",
           "--batch", "2", "--replay-batch", "4",
           "--replay-capacity-per-size", "100",
           "--replay-warmup-decisions", "1", "--updates-per-collection", "2",
           "--target-sync-updates", "3",
           "--max-real-transitions", "180",
           "--validate-every-transitions", "60",
           "--checkpoint-every-transitions", "25",
           "--milestones", "40", "80", "160",
           "--validation-count", "2", "--seed", "10",
           "--device", "cpu", "--out", str(dest), *extra]
    return subprocess.run(cmd, capture_output=True, text=True,
                          env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))


@pytest.mark.parametrize("method", ["ddqn", "dyna"])
@pytest.mark.parametrize("nstep", [1, 3])
def test_full_train_and_exact_resume_cpu(method, nstep, tmp_path):
    continuous = tmp_path / "continuous"
    split = tmp_path / "split"
    initial = run_train(method, continuous, "--n-step", str(nstep))
    assert initial.returncode == 0, initial.stdout + "\n" + initial.stderr
    interrupted = run_train(method, split, "--n-step", str(nstep), "--stop-after-transitions", "70")
    assert interrupted.returncode == 0, interrupted.stdout + "\n" + interrupted.stderr
    resumed = run_train(method, split, "--n-step", str(nstep), "--resume", str(split / "resume.pt"))
    assert resumed.returncode == 0, resumed.stdout + "\n" + resumed.stderr
    a = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
    b = torch.load(split / "last.pt", map_location="cpu", weights_only=True)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for folder in (continuous, split):
        assert (folder / "resume.pt").is_file()
        assert (folder / "milestone_40.pt").is_file()
        assert (folder / "milestone_80.pt").is_file()
        assert (folder / "milestone_160.pt").is_file()
        hist = [json.loads(x) for x in (folder / "history.jsonl").read_text().splitlines()]
        assert hist[-1]["transitions_total"] >= 180
        assert hist[-1]["real_transitions"] > 0
        assert hist[-1]["optimizer_updates"] > 0
        assert hist[-1]["replay_samples"] > 0
        assert hist[-1]["validation"]["4"]["feasible"] == 1.0
        if method == "dyna":
            assert hist[-1]["dyna_transitions"] > 0
        else:
            assert hist[-1]["dyna_transitions"] == 0
    assert run_train(method, continuous, "--n-step", str(nstep)).returncode != 0


def test_replay_train_rejects_invalid_configuration(tmp_path):
    run = run_train("ddqn", tmp_path / "invalid", "--dyna-per-real-decision", "1")
    assert run.returncode != 0
    assert "DDQN control cannot use Dyna" in run.stderr


@pytest.mark.parametrize("method", ["ddqn", "dyna"])
@pytest.mark.parametrize("nstep", [1, 3])
def test_full_wave2_preflight_on_cpu(method, nstep):
    args = [sys.executable, "-m", "rlo.wave2_preflight",
            "--method", method, "--n-step", str(nstep), "--device", "cpu", "--n", "6",
            "--batch", "3", "--replay-batch", "4",
            "--width", "16", "--layers", "1"]
    run = subprocess.run(args, capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    assert f"WAVE2_PREFLIGHT_OK method={method}" in run.stdout

def test_real_nstep_path_reward_and_no_double_counting():
    rng = np.random.default_rng(23)
    model = Policy(width=16, layers=1, logit_clip=None)
    group = instances(6, "validation", 2)
    buffer = Replay(100)
    fresh, envs = collect_real(model, group, buffer, rng, epsilon=.6, n_step=3)
    assert len(fresh) == buffer.real_count()
    assert sum(t.physical_moves for t in fresh) == sum(len(e.tour)-1 for e in envs)
    grouped = {}
    for t in fresh:
        grouped.setdefault(id(t.before.instance), []).append(t)
    recorded = {id(t.before.instance): t for seq in buffer.real.values() for t in seq}
    assert recorded
    for episode in grouped.values():
        stored = [t for t in buffer.real[6] if id(t.before.instance)==id(episode[0].before.instance)]
        assert len(stored)==len(episode)
        for i, t in enumerate(stored):
            seq = episode[i:i+3]
            assert t.n_steps==len(seq)
            assert t.reward==pytest.approx(sum(x.reward for x in seq))
            assert t.after is seq[-1].after
            assert t.physical_moves == seq[0].physical_moves

def test_optimised_replay_observation_matches_environment_exactly():
    env = CVRPEnv(instances(8, "validation", 1)[0])
    for _ in range(4):
        snap = Snapshot.take(env)
        from_env = env.observe()
        from_replay = observations([snap])
        for x, y in zip(from_env, from_replay):
            np.testing.assert_array_equal(x, y[0])
        act = int(np.flatnonzero(env.legal())[0])
        env.step(act)
        if env.done: break

def test_matched_epsilon_and_replay_allocation():
    from rlo.wave2_train import epsilon_at, allocated_updates
    for method in ("ddqn", "dyna"):
        assert epsilon_at(0, 50_000_000)==pytest.approx(1)
        assert epsilon_at(5_000_000, 50_000_000)==pytest.approx(.55)
        assert epsilon_at(10_000_000, 50_000_000)==pytest.approx(.1)
        assert epsilon_at(25_000_000, 50_000_000)==pytest.approx(.1)
    baseline, rest = allocated_updates(8, 500, 0)
    model_based, extra = allocated_updates(8, 500, 500)
    assert baseline==8 and model_based==16 and rest==extra==0
    assert baseline*64/500 == model_based*64/1000
    # Nonintegral Dyna ratios carry forward without dropping samples.
    a, remainder = allocated_updates(8, 500, 73)
    b, remainder = allocated_updates(8, 500, 73, remainder)
    assert (a+b)==int(2*8*573/500)
