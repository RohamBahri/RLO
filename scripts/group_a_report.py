"""Summarize complete Group A runs without consulting test data.

Uses the predeclared final validation metric: average of greedy cost/sqrt(n)
over sizes 10,20,30. Reports each seed, grouped means and standard errors.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def read_runs(root, budget):
    runs = []
    for path in sorted(Path(root).glob("*/history.jsonl")):
        cfg = json.loads((path.parent / "config.json").read_text())
        rows = []
        with path.open() as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # active job may have an incomplete final line
                    pass
        if not rows:
            continue
        last = rows[-1]
        if last["transitions_total"] < budget or "validation" not in last:
            continue
        sizes = tuple(int(x) for x in cfg["sizes"])
        scores = {n: last["validation"][str(n)]["cost"] for n in sizes}
        record = dict(method=cfg["method"], lr=cfg["lr"], seed=cfg["seed"],
                      transitions=last["transitions_total"],
                      updates=last["iteration"], train_seconds=last["train_seconds"],
                      score=mean(scores[n] / math.sqrt(n) for n in sizes),
                      **{f"cost_n{n}": scores[n] for n in sizes})
        runs.append(record)
    return runs


def analyze(runs, min_seeds=3):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["method"], run["lr"])].append(run)
    summaries = []
    for (method, lr), rows in sorted(groups.items()):
        values = [r["score"] for r in rows]
        n = len(values)
        summaries.append(dict(method=method, lr=lr, seeds=n, score_mean=mean(values),
                              score_sd=stdev(values) if n > 1 else None,
                              score_se=stdev(values) / math.sqrt(n) if n > 1 else None,
                              mean_updates=mean(r["updates"] for r in rows),
                              mean_train_seconds=mean(r["train_seconds"] for r in rows),
                              eligible=n >= min_seeds))
    winners = {}
    for row in summaries:
        if row["eligible"] and (row["method"] not in winners or
                                row["score_mean"] < winners[row["method"]]["score_mean"]):
            winners[row["method"]] = row
    return summaries, winners


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Group A experiment root")
    p.add_argument("--budget", type=int, default=10_000_000)
    p.add_argument("--min-seeds", type=int, default=3)
    p.add_argument("--out", default="results/group_a_report")
    args = p.parse_args()
    runs = read_runs(args.folder, args.budget)
    summary, winners = analyze(runs, args.min_seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", runs)
    write_csv(out / "learning_rates.csv", summary)
    (out / "selected.json").write_text(json.dumps(winners, indent=2) + "\n")
    print(f"Complete runs: {len(runs)}, eligible methods: {len(winners)}")
    for key, row in sorted(winners.items()):
        print(f"{key:10s} lr={row['lr']:g} normalized greedy={row['score_mean']:.5f}"
              f" SE={row['score_se']:.5f}")


if __name__ == "__main__":
    main()
