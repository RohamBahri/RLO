#!/usr/bin/env python3
"""Read-only Group A convergence audit. Standard library only.

Run from any checkout; never reads held-out test data or changes training runs.
Usage:
  python3 scripts/oneoff/phase_a_convergence.py \
    "$SCRATCH/RLO_runs/group_a_gpu" \
    --out "$SCRATCH/RLO_analysis/phase_a_convergence"
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

SIZES = (10, 20, 30)
METHODS = ("reinforce", "rollout", "a2c", "pomo")
LRS = (0.0001, 0.0003, 0.0005)
SEEDS = (0, 1, 2)
MARKS = (5_000_000, 8_000_000, 10_000_000)


def score(validation):
    return mean(float(validation[str(n)]["cost"]) / math.sqrt(n) for n in SIZES)


def se(numbers):
    return stdev(numbers) / math.sqrt(len(numbers)) if len(numbers) > 1 else float("nan")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def format_number(x, digits=5):
    return "NA" if x is None or not math.isfinite(float(x)) else f"{x:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-milestone-offset", type=int, default=300_000)
    args = p.parse_args()
    root, dest = args.root.resolve(), args.out.resolve()
    if not root.is_dir():
        p.error(f"Missing Group A directory: {root}")
    if dest == root or root in dest.parents:
        p.error("Output cannot be inside completed training directory")
    if dest.exists() and any(dest.iterdir()):
        p.error(f"Output already exists and is not empty: {dest}")

    runs, errors, warnings, commits = {}, [], [], set()
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "history.jsonl").exists():
            continue
        try:
            config = json.loads((folder / "config.json").read_text())
            method, lr, seed = config["method"], float(config["lr"]), int(config["seed"])
            key = method, lr, seed
            if key in runs:
                errors.append(f"Duplicate run: {key}")
                continue
            valid_config = (
                method in METHODS and lr in LRS and seed in SEEDS
                and tuple(config["sizes"]) == SIZES
                and config["width"] == 128 and config["layers"] == 2
                and not config.get("glimpse", False)
                and config["batch"] == 32 and config["selection_metric"] == "greedy"
                and tuple(config["capacity_factors"]) == (0.75, 1.25)
                and tuple(config["demand_maxes"]) == (10,)
                and config.get("max_transitions") == 10_000_000
            )
            if not valid_config:
                errors.append(f"Unexpected configuration in {folder.name}: {key}")
                continue
            history = []
            with (folder / "history.jsonl").open() as stream:
                for line_num, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON in {folder.name}:{line_num}")
                        continue
                    if history and record["transitions_total"] <= history[-1]["transitions_total"]:
                        errors.append(f"Non-increasing transitions in {folder.name}:{line_num}")
                    history.append(record)
            if not history:
                errors.append(f"Empty history: {folder.name}")
                continue
            measured = [r for r in history if "validation" in r]
            if not measured:
                errors.append(f"No validation data: {folder.name}")
                continue
            final = history[-1]
            if "validation" not in final or final["transitions_total"] < 10_000_000:
                errors.append(f"Incomplete run: {folder.name}")
            if not (folder / "last.pt").is_file():
                errors.append(f"No last.pt: {folder.name}")
            if (folder / "commit.txt").is_file():
                commit = (folder / "commit.txt").read_text().strip()
                commits.add(commit)
            else:
                commit = "unknown"
                warnings.append(f"No commit.txt: {folder.name}")
            near = {}
            for mark in MARKS:
                record = min(measured, key=lambda x: abs(x["transitions_total"] - mark))
                offset = abs(record["transitions_total"] - mark)
                if offset > args.max_milestone_offset:
                    errors.append(f"No observation near {mark:,} in {folder.name} (offset {offset:,})")
                near[mark] = record
            runs[key] = dict(folder=folder.name, final=final, near=near, commit=commit)
        except Exception as exc:
            errors.append(f"Cannot read {folder.name}: {type(exc).__name__}: {exc}")

    expected = {(m, lr, seed) for m in METHODS for lr in LRS for seed in SEEDS}
    for key in sorted(expected - runs.keys()):
        errors.append(f"Missing configuration: {key}")
    if len(commits) > 1:
        warnings.append(f"Multiple source commits: {sorted(commits)}")
    if errors:
        print("AUDIT FAILED. No training runs were modified.")
        for msg in errors:
            print("ERROR:", msg)
        for msg in warnings:
            print("WARNING:", msg)
        raise SystemExit(1)

    lr_rows, selected = [], {}
    for method in METHODS:
        for lr in LRS:
            values = [score(runs[method, lr, seed]["final"]["validation"]) for seed in SEEDS]
            lr_rows.append(dict(method=method, lr=lr, seeds=3,
                                score_mean=mean(values), score_se=se(values)))
        choices = [r for r in lr_rows if r["method"] == method]
        selected[method] = min(choices, key=lambda r: r["score_mean"])["lr"]

    all_run_rows, trajectories, summaries = [], [], []
    for method in METHODS:
        for lr in LRS:
            for seed in SEEDS:
                run = runs[method, lr, seed]
                near = run["near"]
                scores = {mark: score(near[mark]["validation"]) for mark in MARKS}
                improvement = 100 * (scores[8_000_000] - scores[10_000_000]) / scores[8_000_000]
                all_run_rows.append(dict(
                    method=method, lr=lr, seed=seed, selected=lr == selected[method],
                    folder=run["folder"], commit=run["commit"],
                    final_score=score(run["final"]["validation"]),
                    final_transitions=run["final"]["transitions_total"],
                    final_updates=run["final"]["iteration"],
                    train_seconds=run["final"]["train_seconds"],
                    elapsed_seconds=run["final"]["elapsed_seconds"],
                    improvement_8_to_10_pct=improvement))
                for mark in MARKS:
                    rec = near[mark]
                    trajectories.append(dict(
                        method=method, lr=lr, seed=seed, selected=lr == selected[method],
                        milestone=mark, actual_transitions=rec["transitions_total"],
                        score=scores[mark], updates=rec["iteration"],
                        train_seconds=rec["train_seconds"],
                        **{f"cost_n{n}": rec["validation"][str(n)]["cost"] for n in SIZES}))

        chosen = [r for r in all_run_rows if r["method"] == method and r["selected"]]
        trajectories_chosen = [r for r in trajectories if r["method"] == method and r["selected"]]
        cost_at = lambda mark: [r["score"] for r in trajectories_chosen if r["milestone"] == mark]
        changes = [r["improvement_8_to_10_pct"] for r in chosen]
        summaries.append(dict(
            method=method, lr=selected[method], seeds=3,
            score_at_5m=mean(cost_at(5_000_000)),
            score_at_8m=mean(cost_at(8_000_000)),
            score_at_10m=mean(cost_at(10_000_000)),
            se_at_10m=se(cost_at(10_000_000)),
            mean_improvement_8_to_10_pct=mean(changes),
            seeds_improved_8_to_10=sum(x > 0 for x in changes),
            mean_updates=mean(r["final_updates"] for r in chosen),
            mean_train_hours=mean(r["train_seconds"] for r in chosen) / 3600))

    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "learning_rates.csv", lr_rows)
    write_csv(dest / "all_runs.csv", all_run_rows)
    write_csv(dest / "all_trajectories.csv", trajectories)
    write_csv(dest / "method_summary.csv", summaries)
    (dest / "selected_lrs.json").write_text(json.dumps(selected, indent=2) + "\n")
    lines = [
        "# Group A convergence diagnosis", "",
        f"Verified {len(runs)} / 36 configurations.",
        "All 36 runs are included in the CSV files; this table shows only the",
        "learning rate selected for each method by the preregistered final",
        "three-seed greedy validation score, averaged over n=10,20,30.",
        "No test cases or references were read. Milestones are nearby observed",
        "validation records, not interpolations. Exact transitions are in the CSV.", "",
        "| Method | LR | 5M | 8M | 10M | SE 10M | 8→10 change | Improved seeds | Mean updates | Training hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda x: x["score_at_10m"]):
        vals = [
            s["method"], str(s["lr"]),
            format_number(s["score_at_5m"]), format_number(s["score_at_8m"]),
            format_number(s["score_at_10m"]), format_number(s["se_at_10m"]),
            format_number(s["mean_improvement_8_to_10_pct"], 3) + "%",
            f'{s["seeds_improved_8_to_10"]}/3',
            format_number(s["mean_updates"], 0),
            format_number(s["mean_train_hours"], 2),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "Positive 8→10 change indicates decreasing validation cost.",
              "Three seeds and one late window do not prove convergence.", ""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {x}" for x in warnings] + [""]
    else:
        lines += ["All run configurations and checkpoints passed the completeness checks.", ""]
    (dest / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nWrote analysis to", dest)


if __name__ == "__main__":
    main()
