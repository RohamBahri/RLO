"""End-to-end standard-library campaign audit tests, without loading test data."""
import csv
import json
import subprocess
import sys

from scripts.campaign.audit_onpolicy50 import ARMS, MARKS, SOURCE


def make_results(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    for method, (batch, lr) in ARMS.items():
        for seed in range(3):
            d = root / f"{method}_b{batch}_lr{lr}_seed{seed}"
            d.mkdir()
            cfg = {
                "method": method, "batch": batch, "lr": lr, "seed": seed,
                "width": 128, "layers": 2, "sizes": [10, 20, 30],
                "capacity_factors": [.75, 1.25], "demand_maxes": [10],
                "max_transitions": 50_000_000, "validation_count": 256,
                "selection_metric": "greedy",
                "architecture": {"glimpse": False, "distance_bias": True},
            }
            (d / "config.json").write_text(json.dumps(cfg))
            (d / "commit.txt").write_text(SOURCE + "\n")
            for mark in MARKS:
                (d / f"milestone_{mark}.pt").write_bytes(b"0" * 2048)
            (d / "last.pt").write_bytes(b"0" * 2048)
            history = []
            for i, mark in enumerate(MARKS):
                bias = {"reinforce": .10, "rollout": .07, "a2c": .20, "pomo": 0.0}[method]
                obs = mark + 10000
                history.append({
                    "transitions_total": obs, "iteration": i + 1,
                    "elapsed_seconds": 500 * (i + 1), "train_seconds": 250 * (i + 1),
                    "validation": {
                        str(n): {"cost": (1.7 - i * .01 + bias + seed * .001) * n**.5,
                                 "feasible": 1.0} for n in (10, 20, 30)
                    },
                })
            (d / "history.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in history))
    return root


def invoke(root, out):
    return subprocess.run(
        [sys.executable, "-m", "scripts.campaign.audit_onpolicy50", str(root),
         "--out", str(out)],
        capture_output=True, text=True)


def test_campaign_audit_reports_complete_runs_without_test(tmp_path):
    root = make_results(tmp_path)
    out = tmp_path / "analysis"
    result = invoke(root, out)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "12/12" in (out / "report.md").read_text()
    with (out / "runs.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    with (out / "milestones.csv").open() as f:
        points = list(csv.DictReader(f))
    assert len(points) == 48
    assert all(int(p["observed_at"]) > int(p["milestone"]) for p in points)
    with (out / "method_summary.csv").open() as f:
        summary = {r["method"]: r for r in csv.DictReader(f)}
    assert len(summary) == 4
    assert float(summary["pomo"]["score"]) < float(summary["rollout"]["score"])
    assert float(summary["pomo"]["gain_40_to_50_pct"]) > 0.0


def test_campaign_audit_detects_missing_checkpoint(tmp_path):
    root = make_results(tmp_path)
    missing = root / "pomo_b16_lr0.0003_seed2" / "milestone_25000000.pt"
    missing.unlink()
    out = tmp_path / "analysis"
    result = invoke(root, out)
    assert result.returncode == 2
    assert "11/12" in (out / "report.md").read_text()
    assert "milestone checkpoint missing: 25,000,000" in (out / "report.md").read_text()
