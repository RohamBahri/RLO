"""One update of the actual RL algorithm, for cluster launch validation.

This is a software gate, not an experimental result. It never reads test data
or writes checkpoint/history files. Executed within the allocated Slurm GPU job
so it adds no separate queue wait.
"""
import argparse
import math
import numpy as np
import torch

from .data import training_batches
from .learning import update
from .model import Policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("reinforce", "rollout", "a2c", "pomo"))
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.batch < 1 or args.n < 2 or args.width < 4:
        p.error("positive batch, n>=2, width>=4 required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for A100 preflight")
    torch.manual_seed(127)
    rng = np.random.default_rng(127)
    cases = next(training_batches(127, batch=args.batch, sizes=(args.n,),
                                  capacity_factors=(0.75, 1.25), demand_maxes=(10,)))
    model = Policy(width=args.width, layers=args.layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    from copy import deepcopy
    frozen = deepcopy(model).eval() if args.method == "rollout" else None
    result = update(model, optimizer, cases, args.method,
                    baseline_policy=frozen, rng=rng)
    assert result["feasibility_rate"] == 1.0, result
    assert result["transitions"] > 0, result
    assert math.isfinite(result["loss"]), result
    assert math.isfinite(result["gradient"]), result
    assert math.isfinite(result["mean_cost"]), result
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise AssertionError("parameter became non-finite after first update")
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"PREFLIGHT_OK method={args.method} batch={args.batch} n={args.n} "
          f"loss={result['loss']:.5f} transitions={result['transitions']} "
          f"device={args.device}", flush=True)


if __name__ == "__main__":
    main()
