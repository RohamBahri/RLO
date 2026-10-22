"""One real forward/backward update of Wave 2 on one full-size batch.

This checks launchability within the SAME Slurm allocation, not a separate
queued GPU smoke job, and never writes to a training directory.
"""
import argparse
import math

import numpy as np
import torch

from .data import training_batches
from .model import Policy
from .wave2_replay import Replay, collect_real, dyna_from_real, replay_q_update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("ddqn", "dyna"))
    p.add_argument("--n-step", type=int, choices=(1, 3), default=1)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--replay-batch", type=int, default=64)
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA A100 requested but no GPU visible")
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    instances = next(training_batches(123, batch=args.batch, sizes=(args.n,),
                                      capacity_factors=(0.75, 1.25),
                                      demand_maxes=(10,)))
    model = Policy(width=args.width, layers=args.layers, logit_clip=None).to(args.device)
    target = Policy(width=args.width, layers=args.layers, logit_clip=None).to(args.device)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    replay = Replay(capacity_per_size=max(128, args.batch))
    actual, envs = collect_real(model, instances, replay, rng, epsilon=0.4, n_step=args.n_step)
    if args.method == "dyna":
        synthetics = dyna_from_real(replay, rng, len(actual), n_step=args.n_step, model=model, epsilon=.4)
        assert synthetics and all(t.synthetic for t in synthetics)
    samples = replay.sample(rng, args.replay_batch)
    result = replay_q_update(model, target, optimizer, samples, args.device)
    if not all(math.isfinite(result[k]) for k in ("loss", "gradient", "mean_q", "mean_target")):
        raise AssertionError(result)
    if args.device == "cuda":
        torch.cuda.synchronize()
    assert replay.real_count() == len(actual)
    assert all(env.done and not env.infeasible for env in envs)
    print(f"WAVE2_PREFLIGHT_OK method={args.method} n={args.n} "
          f"batch={args.batch} nstep={args.n_step} transitions={sum(t.physical_moves for t in actual)} "
          f"loss={result['loss']:.5f} device={args.device}", flush=True)


if __name__ == "__main__":
    main()
