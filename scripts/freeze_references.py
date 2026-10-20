"""Freeze exact small or optional PyVRP route references; always validate all tours.

Run: python -m scripts.freeze_references --n 20 --split validation --method pyvrp
"""

import argparse
import importlib.metadata
from pathlib import Path
from rlo.data import evaluation_instances
from rlo.exact import solve as exact
from rlo.problem import validate
from rlo.reference import load, pyvrp_solver, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--split", choices=["legacy", "validation", "test"], required=True)
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--capacity-factor", type=float, default=1.0)
    p.add_argument("--method", choices=["exact", "pyvrp"], default="exact")
    p.add_argument("--seconds", type=float, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output")
    p.add_argument("--update", action="store_true", help="keep shortest verified route per case")
    args = p.parse_args()

    cases = evaluation_instances(args.n, args.split, args.count, args.capacity_factor)
    path = Path(args.output or f"benchmarks/{args.split}_n{args.n}_f{args.capacity_factor:g}.json")
    if path.exists() and not args.update:
        p.error(f"{path} exists; use --update to improve verified routes")
    if args.method == "exact":
        routes, label = [exact(inst) for inst in cases], "exact subset DP"
    else:
        routes = []
        for i, inst in enumerate(cases):
            options = [pyvrp_solver(inst, args.seconds, seed) for seed in args.seeds]
            routes.append(min(options, key=lambda t: validate(inst, t)))
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(cases)}", flush=True)
        version = importlib.metadata.version("pyvrp")
        label = f"PyVRP {version}; {args.seconds}s per seed; seeds={args.seeds} (heuristic)"
    if path.exists():
        import json
        old = json.loads(path.read_text())
        load(path, cases)
        routes = [min((a, b), key=lambda t: validate(inst, t))
                  for inst, a, b in zip(cases, old["routes"], routes)]
        label = old["method"] + " + " + label
    save(path, cases, routes, label, args.n, args.split, capacity_factor=args.capacity_factor, overwrite=args.update)
    print(f"Saved {len(routes)} verified tours to {path}")


if __name__ == "__main__":
    main()
