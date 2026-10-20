"""Small deterministic tests for Group A summary and selection rules."""
import json
from pathlib import Path
from scripts.group_a_report import analyze, read_runs


def test_best_lr_uses_seed_mean_not_best_individual_seed(tmp_path):
    for lr, scores in [(1e-4, [1.0, 1.2, 1.3]), (3e-4, [1.12, 1.13, 1.14])]:
        for seed, score in enumerate(scores):
            folder = tmp_path / f"reinforce_lr{lr:g}_seed{seed}"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps(
                dict(method="reinforce", lr=lr, seed=seed, sizes=[10])))
            (folder / "history.jsonl").write_text(json.dumps(
                dict(iteration=7, transitions_total=10_000_001,
                     train_seconds=10, validation={"10": {"cost": score}})) + "\n")
    runs = read_runs(tmp_path, 10_000_000)
    summary, winners = analyze(runs)
    assert len(summary) == 2
    assert winners["reinforce"]["lr"] == 3e-4
    assert all(s["eligible"] for s in summary)


def test_incomplete_runs_are_excluded(tmp_path):
    folder = tmp_path / "partial"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        dict(method="a2c", lr=1e-4, seed=0, sizes=[10])))
    (folder / "history.jsonl").write_text(json.dumps(
        dict(iteration=1, transitions_total=1_000,
             train_seconds=1, validation={"10": {"cost": 5.}})) + "\n")
    assert read_runs(tmp_path, 10_000_000) == []
