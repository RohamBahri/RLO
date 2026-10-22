#!/usr/bin/env python3
"""Read-only 50M CVRP campaign audit. Standard library; no test set or checkpoints loaded."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ARMS = {"reinforce": (32, .0001), "rollout": (32, .0001),
        "a2c": (32, .0003), "pomo": (16, .0003)}
SIZES = (10, 20, 30)
MARKS = (10_000_000, 25_000_000, 40_000_000, 50_000_000)
SOURCE = "e0d188e3027e3b0c83e245eb912c58bcc121f682"


def read_history(path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def metric(row):
    return mean(float(row["validation"][str(n)]["cost"]) / math.sqrt(n) for n in SIZES)


def fmt(x, digits=5):
    return "NA" if x is None else f"{x:.{digits}f}"


def save_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group-a", type=Path, default=None)
    a = ap.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    if not root.is_dir():
        ap.error(f"Missing root {root}")
    if out == root or root in out.parents:
        ap.error("Output cannot be inside training runs")
    if out.exists() and any(out.iterdir()):
        ap.error("Output already exists and is nonempty")
    checks, runs, milestones, errors = [], [], [], []
    expected = set()
    for method, (batch, lr) in ARMS.items():
        for seed in range(3):
            name = f"{method}_b{batch}_lr{lr}_seed{seed}"
            expected.add(name)
            path = root / name
            problem = []
            try:
                if not path.is_dir():
                    raise FileNotFoundError("run directory missing")
                cfg = json.loads((path / "config.json").read_text())
                required = {"method": method, "batch": batch, "lr": lr, "seed": seed,
                            "width": 128, "layers": 2, "sizes": list(SIZES),
                            "capacity_factors": [0.75, 1.25], "demand_maxes": [10],
                            "max_transitions": 50_000_000, "validation_count": 256,
                            "selection_metric": "greedy"}
                for key, val in required.items():
                    if cfg.get(key) != val:
                        problem.append(f"{key} unexpected: {cfg.get(key)!r}")
                if cfg.get("architecture", {}).get("glimpse", True):
                    problem.append("architecture glimpse unexpectedly enabled")
                if cfg.get("architecture", {}).get("distance_bias") is not True:
                    problem.append("distance bias unexpectedly disabled")
                if (path / "commit.txt").read_text().strip() != SOURCE:
                    problem.append("unexpected source commit")
                history = read_history(path / "history.jsonl")
                if not history:
                    raise ValueError("history empty")
                total = [int(r["transitions_total"]) for r in history]
                updates = [int(r["iteration"]) for r in history]
                if any(y <= x for x, y in zip(total, total[1:])):
                    problem.append("non-increasing transitions")
                if any(y <= x for x, y in zip(updates, updates[1:])):
                    problem.append("non-increasing updates")
                last = history[-1]
                if total[-1] < 50_000_000:
                    problem.append(f"incomplete run: {total[-1]:,} transitions")
                if "validation" not in last:
                    problem.append("no final validation")
                for mark in MARKS:
                    chk = path / f"milestone_{mark}.pt"
                    if not chk.is_file() or chk.stat().st_size < 1024:
                        problem.append(f"milestone checkpoint missing: {mark:,}")
                chk = path / "last.pt"
                if not chk.is_file() or chk.stat().st_size < 1024:
                    problem.append("final checkpoint missing/empty")
                observed = [r for r in history if "validation" in r]
                for r in observed:
                    if not math.isfinite(metric(r)) or metric(r) <= 0:
                        problem.append("nonfinite validation score")
                        break
                    if any(r["validation"][str(n)]["feasible"] != 1.0 for n in SIZES):
                        problem.append("infeasible validation solutions")
                        break
                item = {"method": method, "seed": seed, "batch": batch, "lr": lr,
                        "transitions": total[-1], "updates": updates[-1],
                        "elapsed_hours": float(last["elapsed_seconds"]) / 3600,
                        "train_hours": float(last["train_seconds"]) / 3600,
                        "score": metric(last) if "validation" in last else None}
                if "validation" in last:
                    item.update({f"cost_n{n}": last["validation"][str(n)]["cost"] for n in SIZES})
                if a.group_a:
                    oldpath = a.group_a / f"{method}_w128_plain_lr{lr}_seed{seed}/history.jsonl"
                    if oldpath.exists():
                        older = read_history(oldpath)
                        if older and "validation" in older[-1]:
                            item["group_a_10m_score"] = metric(older[-1])
                runs.append(item)
                for mark in MARKS:
                    # Report the first validation >= milestone, never call it exact.
                    r = next((v for v in observed if v["transitions_total"] >= mark), None)
                    if r is None:
                        problem.append(f"missing validation after {mark:,}")
                        continue
                    milestones.append({
                        "method": method, "seed": seed, "milestone": mark,
                        "observed_at": r["transitions_total"],
                        "score": metric(r),
                        "elapsed_hours": r["elapsed_seconds"] / 3600,
                        **{f"cost_n{n}": r["validation"][str(n)]["cost"] for n in SIZES}
                    })
            except (OSError, ValueError, TypeError, KeyError) as e:
                problem.append(f"read failure: {type(e).__name__}: {e}")
            checks.append({"run": name, "status": "FAIL" if problem else "PASS",
                           "issues": "; ".join(problem)})
            errors.extend(f"{name}: {p}" for p in problem)
    for d in root.iterdir():
        if d.is_dir() and d.name not in expected:
            errors.append(f"unexpected additional directory: {d.name}")

    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "checks.csv", checks, ["run", "status", "issues"])
    save_csv(out / "runs.csv", runs,
             ["method", "seed", "batch", "lr", "transitions", "updates",
              "elapsed_hours", "train_hours", "score",
              "cost_n10", "cost_n20", "cost_n30", "group_a_10m_score"])
    save_csv(out / "milestones.csv", milestones,
             ["method", "seed", "milestone", "observed_at", "score", "elapsed_hours",
              "cost_n10", "cost_n20", "cost_n30"])

    lines = [
        "# 50M on-policy CVRP campaign audit", "",
        f"Completed, validated run folders: {sum(r['status']=='PASS' for r in checks)}/12.",
        "Primary metric: mean single-start greedy validation cost/sqrt(n), n=10,20,30.",
        "Milestone scores are the *first evaluation at or after* each threshold.",
        "All observations are validation-only; no test cases or solver references read.",
        "", "| Method | Seeds | Final score | SE | Mean elapsed hours | 40M→50M gain | Improved seeds |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    grouped = {}
    for method in ARMS:
        arm = [r for r in runs if r["method"] == method and r["score"] is not None]
        if not arm:
            continue
        values = [r["score"] for r in arm]
        score = mean(values)
        se = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        changes = []
        for r in arm:
            curve = {p["milestone"]: p for p in milestones if
                     p["method"] == method and p["seed"] == r["seed"]}
            if 40_000_000 in curve and 50_000_000 in curve:
                start, end = curve[40_000_000]["score"], curve[50_000_000]["score"]
                changes.append(100 * (start - end) / start)
        gain = mean(changes) if changes else None
        row = {"method": method, "seeds": len(arm), "score": score, "se": se,
               "mean_elapsed_hours": mean(r["elapsed_hours"] for r in arm),
               "gain_40_to_50_pct": gain, "improving_seeds": sum(x > 0 for x in changes),
               "late_seed_count": len(changes)}
        grouped[method] = row
    for row in sorted(grouped.values(), key=lambda x: x["score"]):
        lines.append(f"| {row['method']} | {row['seeds']} | {fmt(row['score'])} | "
                     f"{fmt(row['se'])} | {fmt(row['mean_elapsed_hours'],2)} | "
                     f"{fmt(row['gain_40_to_50_pct'],3)}% | "
                     f"{row['improving_seeds']}/{row['late_seed_count']} |")
    save_csv(out / "method_summary.csv", list(grouped.values()),
             ["method", "seeds", "score", "se", "mean_elapsed_hours",
              "gain_40_to_50_pct", "improving_seeds", "late_seed_count"])
    lines += ["", "## Interpretation", "",
              "POMO uses batch 16, unlike Group A batch 32. All 50M runs start fresh.",
              "POMO multistart and other methods' single-start results must not be mixed.",
              "SE describes variation among only three training seeds.",
              "Checkpoint files were checked for existence/size, not deserialized.",
              "Elapsed hours exclude separate Slurm startup/preflight and scheduling delay.",
              "Late positive gains indicate continued improvement, not proven nonconvergence.",
              ""]
    if errors:
        lines += ["## Issues needing attention", ""] + [f"- {x}" for x in errors]
    else:
        lines += ["## Next step", "",
                  "Freeze the method-level validation comparison; perform common validation",
                  "generalization and matched inference-budget tests from retained checkpoints.",
                  "Meanwhile prepare off-policy replay, Dyna, and batched MBPI correctness gates."]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    print(report)
    print("Outputs in", out)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
