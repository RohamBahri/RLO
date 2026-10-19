"""Plot validation learning curves across seeds, against transitions and time.

pip install -e '.[plots]'
python -m scripts.curves results/group_a
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def collect(folder):
    """Group runs by method/LR. Score = mean per-size validation cost / sqrt(n)."""
    groups = defaultdict(list)
    for path in Path(folder).rglob("history.jsonl"):
        config = json.loads((path.parent / "config.json").read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if "validation" in r]
        if not rows:
            continue
        sizes = sorted(rows[0]["validation"], key=int)
        curve = [{**{key: r[key] for key in ("transitions_total", "elapsed_seconds")},
                  "score": float(np.mean([r["validation"][n]["cost"] / np.sqrt(int(n))
                                          for n in sizes]))} for r in rows]
        groups[(config["method"], config["lr"])].append(curve)
    return groups


def plot(groups, x, output):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for (method, lr), curves in sorted(groups.items()):
        if not curves:
            continue
        start = max(c[0][x] for c in curves)
        end = min(c[-1][x] for c in curves)
        if end < start:
            continue
        grid = np.linspace(start, end, 80) if end > start else np.array([start])
        values = np.array([np.interp(grid, [r[x] for r in c], [r["score"] for r in c]) for c in curves])
        average = values.mean(0)
        ax.plot(grid, average, label=f"{method} lr={lr:g} (seeds={len(curves)})")
        if len(curves) > 1:
            deviation = values.std(0, ddof=1)
            ax.fill_between(grid, average - deviation, average + deviation, alpha=.15)
    ax.set(xlabel=x, ylabel="Mean validation cost / sqrt(n)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="directory containing experiment histories")
    p.add_argument("--out", default="results/curves")
    args = p.parse_args()
    groups = collect(args.folder)
    if not groups:
        p.error("no validation histories found")
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for axis in ("transitions_total", "elapsed_seconds"):
        plot(groups, axis, dest / f"{axis}.png")
    print(f"Saved two learning-curve plots to {dest}")


if __name__ == "__main__":
    main()
