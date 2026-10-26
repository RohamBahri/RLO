"""Production-method correctness and resumability checks for Packages A/B."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from rlo.data import instances
from rlo.wave2_replay import Snapshot
from rlo.wave3_model import ResearchPolicy, from_saved
from rlo.wave3_core import (environments, trajectories, policy_objective,
                            future_targets, ppo_update, critic_only_update)


@pytest.mark.parametrize("variant", [
    dict(encoder_type="attention"),
    dict(encoder_type="raw"),
    dict(encoder_type="mlp"),
    dict(encoder_type="codebook"),
    dict(encoder_type="attention", dynamic_reencode=True),
    dict(encoder_type="attention", hierarchy=True),
    dict(encoder_type="attention", glimpse=True),
    dict(encoder_type="attention", auxiliary=True),
    dict(encoder_type="attention", width=32, heads=8, layers=4),
    dict(encoder_type="attention", distance_bias=False),
])
def test_architecture_valid_mask_and_gradient(variant):
    settings = dict(width=16, layers=1, logit_clip=10.)
    settings.update(variant)
    model = ResearchPolicy(**settings)
    cases = instances(5, "train", 3)
    traces, envs, steps = trajectories(model, environments(cases), grad=True)
    assert steps > 0 and all(e.done and not e.infeasible for e in envs)
    assert all(len(t) >= 1 for t in traces)
    cost, parts = policy_objective(
        model, traces, envs, "a2c",
        aux_weight=.1 if model.auxiliary_enabled else 0.)
    assert torch.isfinite(cost)
    cost.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    if model.hierarchy:
        x = cases[0]
        e = environments([x])[0]
        obs = [torch.as_tensor(y[None]) for y in e.observe()]
        logits, _ = model(*obs)
        mass = logits.softmax(-1)
        assert torch.allclose(mass.sum(-1), torch.ones(1), atol=1e-5)
        assert torch.all(mass[~obs[-1]] < 1e-6)


def test_auxiliary_future_targets_are_realized_quantities():
    model = ResearchPolicy(width=16, layers=1, auxiliary=True)
    seqs, envs, _ = trajectories(model, environments(instances(6, "train", 2)))
    for seq, env in zip(seqs, envs):
        labels = future_targets(seq, n=6)
        assert labels[0][0] > 0
        assert abs(labels[0][0] * (6 ** .5) + sum(x["reward"] for x in seq)) < 1e-5
        assert labels[0][1] >= 0
        assert all(len(x) == 3 for x in labels)


def test_ppo_and_dyna_update_finite():
    model = ResearchPolicy(width=16, layers=1, auxiliary=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    seqs, envs, _ = trajectories(model, environments(instances(6, "train", 3)),
                                 grad=False)
    before = next(model.parameters()).detach().clone()
    rec = ppo_update(model, opt, seqs, minibatch=8, epochs=2, aux_weight=.1,
                     temperature=1.0)
    assert rec["optimizer_updates"] >= 2
    assert not torch.equal(before, next(model.parameters()))
    roots = [s["before"] for seq in seqs for s in seq]
    synthetic, final, moved = trajectories(model, [root.restore() for root in roots[:3]],
                                            grad=True)
    assert moved > 0 and len(synthetic) == 3
    critic = torch.optim.Adam(model.value.parameters(), lr=1e-4)
    actor_before = model.key.weight.detach().clone()
    vrec = critic_only_update(model, critic, synthetic)
    assert vrec["optimizer_updates"] == 1
    torch.testing.assert_close(model.key.weight, actor_before, rtol=0, atol=0)


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "rlo.wave3_train", *args],
        capture_output=True, text=True, timeout=180,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))


def _args(method, path, *extra):
    return ["--method", method, "--sizes", "4", "6", "--eval-sizes", "4", "6", "--width", "16",
            "--layers", "1", "--batch", "2", "--device", "cpu",
            "--max-moves", "140", "--validation-count", "2",
            "--validate-every-moves", "90", "--checkpoint-every-moves", "20",
            "--milestones", "50", "100", "--ppo-epochs", "2",
            "--ppo-minibatch", "8", "--dyna-roots", "2",
            "--mbpi-roots", "1", "--mbpi-rollouts", "1",
            "--out", str(path), "--seed", "7", *extra]


@pytest.mark.parametrize("method,extra", [
    ("ppo", []),
    ("dyna_critic", []),
    ("dyna_actor", []),
    ("pomo", ["--starts", "2"]),
    ("pomo", ["--encoder", "raw", "--starts", "2"]),
    ("pomo", ["--encoder", "codebook", "--starts", "2"]),
    ("pomo", ["--auxiliary", "--starts", "2"]),
    ("pomo", ["--hierarchy", "--starts", "2"]),
    ("pomo", ["--dynamic-reencode", "--starts", "2"]),
])
def test_package_ab_end_to_end_cpu(method, extra, tmp_path):
    run = _run(_args(method, tmp_path / "full", *extra))
    assert run.returncode == 0, run.stderr + run.stdout
    state = torch.load(tmp_path / "full" / "policy.pt", weights_only=True)
    assert state["wave3_version"] == 1
    policy = from_saved(tmp_path / "full" / "policy.pt")
    assert isinstance(policy, ResearchPolicy)
    history = [json.loads(s) for s in
               (tmp_path / "full" / "history.jsonl").read_text().splitlines()]
    assert history[-1]["real_transitions"] >= 140
    assert history[-1]["optimizer_updates"] >= 1
    assert history[-1]["validation"]["4"]["feasible"] == 1.
    if method.startswith("dyna"):
        assert history[-1]["synthetic_transitions"] > 0


@pytest.mark.parametrize("method,options", [
    ("ppo", ["--encoder", "mlp"]),
    ("dyna_critic", []),
    ("dyna_actor", []),
    ("pomo", ["--starts", "2", "--auxiliary"]),
])
def test_resume_exact_policy_and_counters(method, options, tmp_path):
    full = tmp_path / "full"
    split = tmp_path / "split"
    x = _run(_args(method, full, *options))
    assert x.returncode == 0, x.stderr
    x = _run(_args(method, split, *options, "--stop-after-moves", "70"))
    assert x.returncode == 0, x.stderr
    x = _run(_args(method, split, *options, "--resume", str(split / "last.pt")))
    assert x.returncode == 0, x.stderr
    a = torch.load(full / "policy.pt", weights_only=True)["model"]
    b = torch.load(split / "policy.pt", weights_only=True)["model"]
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)


@pytest.mark.parametrize("roots,rollouts", [(4, 8), (16, 4)])
def test_mbpi_from_seed_matched_pomo_anchor(roots, rollouts, tmp_path):
    # Produce a real original-format POMO anchor; synthetic MBPI states
    # must not be trained using an A2C checkpoint mislabeled as POMO.
    anchor = tmp_path / "anchor"
    command = [sys.executable, "-m", "rlo.campaign_train", "--method", "pomo",
               "--sizes", "4", "--updates", "1", "--batch", "2", "--seed", "7",
               "--width", "16", "--layers", "1", "--validation-count", "2",
               "--device", "cpu", "--out", str(anchor)]
    r = subprocess.run(command, capture_output=True, text=True, timeout=90,
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    output = tmp_path / "mbpi"
    args = _args("mbpi", output, "--initial-checkpoint", str(anchor / "last.pt"),
                 "--mbpi-roots", str(roots), "--mbpi-rollouts", str(rollouts))
    r = _run(args)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(s) for s in (output / "history.jsonl").read_text().splitlines()]
    assert rows[-1]["synthetic_transitions"] > 0
    assert rows[-1]["real_transitions"] > 0
    assert rows[-1]["transitions_total"] >= 140
    assert rows[-1]["mbpi_rollouts"] == rows[-1]["mbpi_actions"] * rollouts
    assert rows[-1]["mbpi_roots"] == roots
    assert rows[-1]["mbpi_unique_roots"] == min(roots, rows[-1]["mbpi_root_candidates"])
    assert rows[-1]["mbpi_root_candidates"] >= 2

    # Wrong seed and wrong requested milestone must be rejected BEFORE training.
    wrong_seed = _run(_args("mbpi", tmp_path / "wrong_seed",
                           "--initial-checkpoint", str(anchor / "last.pt"),
                           "--seed", "8"))
    assert wrong_seed.returncode != 0
    assert "seed-matched" in wrong_seed.stderr

    wrong_milestone = _run(_args(
        "mbpi", tmp_path / "wrong_milestone",
        "--initial-checkpoint", str(anchor / "last.pt"),
        "--expected-initial-moves", "25000000"))
    assert wrong_milestone.returncode != 0
    assert "milestone" in wrong_milestone.stderr


def test_pomo_matches_original_wave1_update():
    """Representation ablations must keep the actual Wave-1 POMO objective."""
    from rlo.model import Policy
    from rlo.learning import update as old_update
    torch.manual_seed(2026)
    old = Policy(width=16, layers=1, glimpse=False)
    new = ResearchPolicy(width=16, layers=1, glimpse=False)
    new.load_state_dict(old.state_dict(), strict=True)
    cases = instances(6, "train", 3)
    a = torch.optim.Adam(old.parameters(), lr=3e-4)
    b = torch.optim.Adam(new.parameters(), lr=3e-4)
    rng_old = np.random.default_rng(17)
    rng_new = np.random.default_rng(17)
    torch.manual_seed(200)
    original = old_update(old, a, cases, "pomo", starts=6, rng=rng_old)
    torch.manual_seed(200)
    starting = environments(cases, 6, rng_new)
    forced_prefix = sum(len(e.tour) - 1 for e in starting)
    seq, envs, moves = trajectories(new, starting, grad=True)
    cost, _ = policy_objective(new, seq, envs, "pomo", starts=6, entropy_weight=0.)
    b.zero_grad()
    cost.backward()
    b.step()
    assert abs(float(cost.detach()) - original["loss"]) < 1e-5
    assert moves + forced_prefix == original["transitions"]
    for key, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, new.state_dict()[key], atol=1e-5, rtol=1e-5)


def test_shared_validation_grid_and_sealed_test(tmp_path):
    run = _run(_args("ppo", tmp_path / "training"))
    assert run.returncode == 0, run.stderr
    weights = tmp_path / "training" / "policy.pt"
    out = tmp_path / "grid.json"
    cmd = [sys.executable, "-m", "rlo.wave3_eval", "--checkpoint", str(weights),
           "--sizes", "4", "6", "--count", "2", "--k", "1", "2",
           "--device", "cpu", "--out", str(out)]
    check = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                           env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    assert check.returncode == 0, check.stdout + check.stderr
    rows = json.loads(out.read_text())
    assert len(rows) == 4
    assert all(r["feasibility_rate"] == 1. for r in rows)
    assert all(len(r["instances"]) == 2 for r in rows)
    blocked = subprocess.run(cmd + ["--split", "test"], capture_output=True, text=True,
                             env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert blocked.returncode != 0
    assert "sealed" in blocked.stderr
