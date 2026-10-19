"""Run quality-vs-compute evaluations on untrained and held-out instance sets."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import SEEDS, instances
from .learning import device_auto, rollout
from .model import Policy
from .problem import generate, validate
from .reference import load
from .solvers import clarke_wright, nearest_neighbor


def model_from_checkpoint(path, device="auto"):
    device = device_auto() if device == "auto" else device
    state = torch.load(path, map_location=device, weights_only=True)
    model = Policy(**state["architecture"]).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def best_of_k(model, cases, k=1, seed=0, temperature=1):
    """Sample k solutions per instance; return the shortest verified route and cost."""
    torch.manual_seed(seed)
    best = [None] * len(cases)
    with torch.no_grad():
        for rep in range(k):
            envs, _ = rollout(model, cases, greedy=(k == 1), temperature=temperature)
            for i, env in enumerate(envs):
                try:
                    cost = validate(cases[i], env.finish())
                except ValueError:
                    continue
                if best[i] is None or cost < best[i][0]:
                    best[i] = (cost, env.tour)
    return best


def main():
    p = argparse.ArgumentParser(description="Held-out generalization and inference-budget evaluation")
    p.add_argument("--checkpoint", help="new RL checkpoint; omit for a classical heuristic")
    p.add_argument("--baseline", choices=["savings", "nearest"], default="savings")
    p.add_argument("--legacy", action="store_true", help="interpret --checkpoint as old MVP5 PyTorch model")
    p.add_argument("--split", choices=["validation", "test", "legacy"], default="validation")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--factors", type=float, nargs="+", default=[1.0])
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--k", type=int, nargs="+", default=[1, 16, 128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference", help="route-based frozen reference file (one size and factor only)")
    p.add_argument("--out", help="optional JSON summary")
    args = p.parse_args()
    if args.reference and (len(args.sizes) != 1 or args.factors != [1.0]):
        p.error("--reference requires one size and factor 1.0")
    if args.legacy and not args.checkpoint:
        p.error("--legacy requires --checkpoint")
    if args.legacy:
        from .legacy import MVP5Policy
        model = MVP5Policy(args.checkpoint)
    else:
        model = model_from_checkpoint(args.checkpoint) if args.checkpoint else None
    all_rows = []
    for n in args.sizes:
        for factor in args.factors:
            if factor == 1.0:
                cases = instances(n, args.split, args.count)
            else:
                rng = np.random.default_rng(SEEDS[args.split] + n * 100003 + round(factor * 1000))
                cases = [generate(n, rng, capacity_factor=factor) for _ in range(args.count)]
            refs = load(args.reference, cases)[0] if args.reference else None
            for k in args.k if model else [1]:
                start = time.perf_counter()
                if model:
                    if args.legacy:
                        solutions = []
                        for j, inst in enumerate(cases):
                            tours = [model(inst)] if k == 1 else [
                                model.sample(inst, seed=args.seed + j * 100000 + rep)
                                for rep in range(k)]
                            feasible = []
                            for tour in tours:
                                try:
                                    feasible.append((validate(inst, tour), tour))
                                except ValueError:
                                    pass
                            solutions.append(min(feasible, default=None, key=lambda item: item[0]))
                    else:
                        solutions = best_of_k(model, cases, k, args.seed)
                else:
                    solver = clarke_wright if args.baseline == "savings" else nearest_neighbor
                    solutions = []
                    for inst in cases:
                        tour = solver(inst)
                        solutions.append((validate(inst, tour), tour))
                lengths = [s[0] for s in solutions if s is not None]
                paired_gaps = [100 * (s[0] / ref - 1) for s, ref in zip(solutions, refs)
                               if s is not None] if refs is not None else []
                row = dict(n=n, capacity_factor=factor, k=k, count=len(cases),
                           feasible=len(lengths), feasibility_rate=len(lengths) / len(cases),
                           mean_feasible_cost=float(np.mean(lengths)) if lengths else None,
                           mean_feasible_gap_pct=float(np.mean(paired_gaps)) if paired_gaps else None,
                           elapsed_seconds=time.perf_counter() - start)
                all_rows.append(row)
                print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(all_rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
