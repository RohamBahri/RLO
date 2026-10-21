"""Correctness gates for long-run campaign checkpoint and data-stream resume."""
import os
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rlo.data import StatefulTrainingBatches, training_batches


def test_campaign_stream_reproduces_original_generator():
    expected = training_batches(seed=9, batch=3, sizes=(4, 6), capacity_factors=(.75, 1.25))
    actual = StatefulTrainingBatches(seed=9, batch=3, sizes=(4, 6), capacity_factors=(.75, 1.25))
    for _ in range(3):
        a, b = next(expected), next(actual)
        for x, y in zip(a, b):
            assert np.array_equal(x["coords"], y["coords"])
            assert np.array_equal(x["demands"], y["demands"])
            assert x["capacity"] == y["capacity"]
    state = actual.state_dict()
    after = next(actual)
    restarted = StatefulTrainingBatches(seed=20, batch=3, sizes=(4, 6), capacity_factors=(.75, 1.25))
    restarted.load_state_dict(state)
    replay = next(restarted)
    for x, y in zip(after, replay):
        assert np.array_equal(x["coords"], y["coords"])
        assert x["capacity"] == y["capacity"]


def check_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            check_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            check_same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("method", ["reinforce", "rollout", "a2c", "pomo"])
def test_campaign_resume_identical_to_uninterrupted_cpu(method, tmp_path):
    budget = 700 if method == "rollout" else 140
    common = [
        sys.executable, "-m", "rlo.campaign_train",
        "--method", method, "--sizes", "4", "--width", "16", "--layers", "1",
        "--batch", "2", "--lr", "0.0003", "--seed", "9", "--device", "cpu",
        "--max-transitions", str(budget), "--validation-count", "2",
        "--validate-every", "1000000", "--validate-every-transitions", "90",
        "--checkpoint-every-transitions", "25",
        "--milestones", "40", "80", "120",
        "--baseline-check-every", "2", "--baseline-count", "30",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    continuous = tmp_path / f"{method}_continuous"
    split = tmp_path / f"{method}_split"
    def execute(*flags):
        subprocess.run(common + list(flags), check=True, capture_output=True, text=True, env=env)
    execute("--out", str(continuous))
    execute("--out", str(split), "--stop-after-transitions", "70")
    execute("--out", str(split), "--resume", str(split / "last.pt"))
    expected_history = [json.loads(line) for line in (continuous / "history.jsonl").read_text().splitlines()]
    actual_history = [json.loads(line) for line in (split / "history.jsonl").read_text().splitlines()]
    assert len(expected_history) == len(actual_history), (method, [(r["iteration"], r["transitions_total"], r.get("baseline_check_transitions")) for r in expected_history], [(r["iteration"], r["transitions_total"], r.get("baseline_check_transitions")) for r in actual_history])
    for x, y in zip(expected_history, actual_history):
        for key in ("iteration", "transitions_total", "baseline_accepted", "mean_cost", "loss"):
            assert x.get(key) == y.get(key), (method, key, x.get("iteration"), x.get(key), y.get(key))
    lhs = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
    rhs = torch.load(split / "last.pt", map_location="cpu", weights_only=True)
    for field in ("model", "optimizer", "rollout_baseline", "training_stream_rng",
                  "pomo_rng", "ema_by_size", "python_rng", "torch_cpu_rng",
                  "transitions_total", "episodes_total", "iteration"):
        check_same(lhs[field], rhs[field])
    assert lhs["transitions_total"] >= budget
    for path in (continuous, split):
        assert (path / "milestone_40.pt").exists()
        assert (path / "milestone_80.pt").exists()
        assert (path / "milestone_120.pt").exists()


def test_incompatible_resume_rejected(tmp_path):
    target = tmp_path / "run"
    base = [sys.executable, "-m", "rlo.campaign_train",
            "--method", "reinforce", "--sizes", "4", "--width", "16",
            "--layers", "1", "--batch", "2", "--device", "cpu",
            "--max-transitions", "30", "--out", str(target), "--validation-count", "2"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(base, check=True, capture_output=True, env=env)
    rejected = subprocess.run(base + ["--lr", "0.005", "--resume", str(target / "last.pt")],
                              capture_output=True, text=True, env=env)
    assert rejected.returncode != 0
    assert "CLI settings differ" in rejected.stderr
