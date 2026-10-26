"""Common generalization / inference-budget evaluation for Wave 3 policies.

Reads only exported, safe-to-load policy.pt model weights. The sealed test split
requires an explicit --unlock-test flag after all model choices are frozen.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .data import evaluation_instances, fingerprint
from .learning import device_auto
from .study import best_of_k
from .wave3_model import from_saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--unlock-test", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 12, 20, 25, 30, 40, 50])
    p.add_argument("--factors", nargs="+", type=float, default=[1.0])
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--k", nargs="+", type=int, default=[1, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.split == "test" and not a.unlock_test:
        p.error("test split is sealed until method/configuration selection is frozen")
    if (a.count < 1 or min(a.sizes) < 2 or min(a.k) < 1 or
        min(a.factors) <= 0 or len(set(a.k)) != len(a.k)):
        p.error("invalid size, factor, sample count or K")
    device = device_auto() if a.device == "auto" else a.device
    model = from_saved(a.checkpoint, device)
    rows = []
    for n in a.sizes:
        for factor in a.factors:
            cases = evaluation_instances(n, a.split, a.count, factor)
            for k in a.k:
                solutions, moves = best_of_k(model, cases, k=k, seed=a.seed,
                                             return_transitions=True)
                costs = [x[0] for x in solutions if x is not None]
                row = dict(n=n, factor=factor, split=a.split, k=k,
                           count=len(cases), feasible=len(costs),
                           feasibility_rate=len(costs) / len(cases),
                           mean_cost=float(np.mean(costs)) if costs else None,
                           normalized_cost=float(np.mean(costs) / math.sqrt(n)) if costs else None,
                           inference_moves=moves,
                           instances=[dict(id=fingerprint(inst),
                                           cost=None if sol is None else float(sol[0]))
                                      for inst, sol in zip(cases, solutions)])
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "instances"}), flush=True)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
