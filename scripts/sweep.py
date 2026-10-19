"""Small, explicit sweep runner. Selection uses validation data only.

Example:
  python -m scripts.sweep --methods reinforce rollout a2c pomo \\
      --lrs 0.0001 0.0003 --seeds 0 1 2 --updates 1000
"""

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["reinforce", "rollout", "a2c", "pomo"])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.0001, 0.0003])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--updates", type=int, default=100000)
    p.add_argument("--max-transitions", type=int, default=10000000,
                   help="screening default: 10M environment transitions per run")
    p.add_argument("--validation-count", type=int, default=256)
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--widths", type=int, nargs="+", default=[64])
    p.add_argument("--architectures", nargs="+", choices=["plain", "glimpse"], default=["plain"])
    p.add_argument("--validation-multistart", action="store_true")
    p.add_argument("--selection-metric", choices=["greedy", "multistart"], default="greedy")
    p.add_argument("--out", default="results/group_a")
    args = p.parse_args()
    if args.selection_metric == "multistart" and not args.validation_multistart:
        p.error("--selection-metric multistart requires --validation-multistart")
    for method in args.methods:
        if method not in {"reinforce", "rollout", "a2c", "pomo"}:
            p.error(f"invalid method: {method}")
        for width in args.widths:
            for architecture in args.architectures:
                for lr in args.lrs:
                    for seed in args.seeds:
                        target = f"{args.out}/{method}_w{width}_{architecture}_lr{lr:g}_seed{seed}"
                        cmd = [sys.executable, "-m", "rlo.train", "--method", method,
                               "--lr", str(lr), "--seed", str(seed), "--width", str(width),
                               "--updates", str(args.updates), "--max-transitions", str(args.max_transitions),
                               "--selection-metric", args.selection_metric,
                               "--validation-count", str(args.validation_count), "--batch", str(args.batch),
                               "--sizes", *map(str, args.sizes), "--out", target]
                        if architecture == "glimpse":
                            cmd.append("--glimpse")
                        if args.validation_multistart:
                            cmd.append("--validation-multistart")
                        print(" ".join(cmd), flush=True)
                        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
