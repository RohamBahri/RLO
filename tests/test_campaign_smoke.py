"""Run real end-to-end gradient steps at the campaign CLI and verify outputs.

CPU test intentionally uses one complete episode (not one action, because
actions alone do not produce full episodic training targets). No oracle labels.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from rlo.study import model_from_checkpoint
from rlo.data import instances
from rlo.learning import rollout
from rlo.problem import validate


@pytest.mark.parametrize("method", ["reinforce", "rollout", "a2c", "pomo"])
def test_campaign_startup_full_episode_and_checkpoint(method, tmp_path):
    dest = tmp_path / method
    dest.mkdir()
    (dest / "commit.txt").write_text("smoke-test-pinned-commit\n")
    cmd = [
        sys.executable, "-m", "rlo.campaign_train",
        "--method", method, "--sizes", "4", "--capacity-factors", "0.75", "1.25",
        "--demand-maxes", "10",
        "--width", "16", "--layers", "2", "--device", "cpu",
        "--batch", "2", "--lr", "0.0003", "--seed", "3",
        "--max-transitions", "1", "--validation-count", "2",
        "--validate-every", "100000000", "--validate-every-transitions", "2500000",
        "--checkpoint-every-transitions", "1",
        "--milestones", "1", "--selection-metric", "greedy",
        "--out", str(dest),
    ]
    result = subprocess.run(
        cmd, text=True, capture_output=True,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    history = [json.loads(line) for line in (dest / "history.jsonl").read_text().splitlines()]
    assert len(history) == 1
    row = history[0]
    assert row["transitions_total"] >= 1 and row["iteration"] == 1
    assert row["feasibility_rate"] == 1.0
    assert math.isfinite(row["loss"]) and math.isfinite(row["gradient"])
    assert row["validation"]["4"]["feasible"] == 1.0
    saved = torch.load(dest / "last.pt", map_location="cpu", weights_only=True)
    assert "optimizer" in saved and "torch_cpu_rng" in saved
    assert (dest / "milestone_1.pt").is_file()
    restored = model_from_checkpoint(dest / "last.pt", device="cpu")
    with torch.no_grad():
        cases = instances(4, split="validation", count=2)
        envs, _ = rollout(restored, cases, greedy=True)
    for case, env in zip(cases, envs):
        assert env.done and not env.infeasible
        assert math.isclose(validate(case, env.finish()), env.cost, abs_tol=1e-8)
    assert (dest / "commit.txt").read_text() == "smoke-test-pinned-commit\n"


@pytest.mark.parametrize("method,batch", [("reinforce", 32), ("rollout", 32), ("a2c", 32), ("pomo", 16)])
def test_campaign_production_like_startup(method, batch, tmp_path):
    """One full-shape update with real CLI settings and cross-size validation."""
    dest = tmp_path / f"{method}_production"
    dest.mkdir()
    (dest / "commit.txt").write_text("smoke-test-pinned-commit\n")
    cmd = [
        sys.executable, "-m", "rlo.campaign_train",
        "--method", method, "--sizes", "10", "20", "30",
        "--capacity-factors", "0.75", "1.25", "--demand-maxes", "10",
        "--width", "128", "--layers", "2", "--device", "cpu",
        "--batch", str(batch), "--lr", "0.0003", "--seed", "0",
        "--max-transitions", "50000000",
        "--stop-after-transitions", "1",
        "--validate-every", "100000000",
        "--validate-every-transitions", "2500000",
        "--checkpoint-every-transitions", "1000000",
        "--milestones", "10000000", "25000000", "40000000", "50000000",
        "--validation-count", "2", "--selection-metric", "greedy",
        "--out", str(dest),
    ]
    result = subprocess.run(
        cmd, text=True, capture_output=True,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    history = [json.loads(line) for line in (dest / "history.jsonl").read_text().splitlines()]
    assert len(history) == 1
    row = history[0]
    assert row["transitions_total"] >= 1 and row["iteration"] == 1
    for n in ("10", "20", "30"):
        assert row["validation"][n]["feasible"] == 1.0
        if method == "pomo":
            assert row["validation"][n]["multistart_cost"] <= row["validation"][n]["cost"] + 1e-8
    assert (dest / "last.pt").is_file()
    assert (dest / "config.json").is_file()


@pytest.mark.parametrize("method,batch", [("reinforce", 2), ("rollout", 2), ("a2c", 2), ("pomo", 2)])
def test_preflight_real_cpu_step(method, batch):
    result = subprocess.run([
        sys.executable, "-m", "rlo.campaign_preflight",
        "--method", method, "--batch", str(batch),
        "--width", "16", "--layers", "1", "--n", "5", "--device", "cpu"
    ], text=True, capture_output=True,
       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "PREFLIGHT_OK" in result.stdout
