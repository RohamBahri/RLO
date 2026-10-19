"""Small CLI for quick reference and solution-quality sanity checks."""

import argparse
import json
from .data import instances
from .evaluate import evaluate
from .reference import load, ortools
from .solvers import nearest_neighbor, clarke_wright
from .exact import solve as exact


def main():
    p = argparse.ArgumentParser(description="Evaluate CVRP solutions on fixed instances")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--split", choices=["legacy", "validation", "test"], default="validation")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--solver", choices=["nearest", "savings", "exact", "ortools", "mvp5"], default="savings")
    p.add_argument("--checkpoint", help="archived MVP5 .pt checkpoint (required for --solver mvp5)")
    p.add_argument("--reference", default="exact", help="exact, ortools, none, or a frozen JSON path")
    p.add_argument("--seconds", type=int, default=10, help="OR-Tools per-instance limit")
    p.add_argument("--output", help="optional JSON report path")
    args = p.parse_args()
    data = instances(args.n, args.split, args.count)
    methods = {"nearest": nearest_neighbor, "savings": clarke_wright, "exact": exact,
               "ortools": lambda inst: ortools(inst, args.seconds)}
    if args.solver == "mvp5":
        if not args.checkpoint:
            p.error("--solver mvp5 requires --checkpoint")
        from .legacy import MVP5Policy
        methods["mvp5"] = MVP5Policy(args.checkpoint)
    if args.reference == "none":
        refs, method = None, None
    elif args.reference == "exact":
        from .problem import validate
        refs = [validate(inst, exact(inst)) for inst in data]
        method = "exact subset DP"
    elif args.reference == "ortools":
        from .problem import validate
        refs = [validate(inst, ortools(inst, args.seconds)) for inst in data]
        method = f"OR-Tools feasible heuristic, {args.seconds}s"
    else:
        refs, method = load(args.reference, data)
    report = evaluate(data, methods[args.solver], refs)
    report.update({"n": args.n, "split": args.split, "solver": args.solver, "reference_method": method})
    print(f"{args.solver}: {report['feasibility_rate']:.0%} feasible, "
          f"mean gap {report['mean_gap_pct']}%, {report['total_seconds']:.2f}s")
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
