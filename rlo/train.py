"""Small, reproducible on-policy training runner. Never consumes reference tours."""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import instances
from .learning import device_auto, rollout, update
from .model import Policy
from .problem import generate


def evaluate_greedy(model, sizes, count):
    model.eval()
    values = {}
    with torch.no_grad():
        for n in sizes:
            cases = instances(n, "validation", count)
            envs, _ = rollout(model, cases, greedy=True)
            values[str(n)] = dict(cost=float(np.mean([e.cost for e in envs])),
                                  feasible=float(np.mean([e.done and not e.infeasible for e in envs])))
    return values


def main():
    p = argparse.ArgumentParser(description="Experience-only on-policy CVRP training")
    p.add_argument("--method", choices=["reinforce", "rollout", "a2c", "pomo"], default="a2c")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--capacity-factors", type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument("--demand-maxes", type=int, nargs="+", default=[10])
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--without-distance-bias", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anneal-to", type=float, default=None)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--entropy", type=float, default=0.0)
    p.add_argument("--validate-every", type=int, default=20)
    p.add_argument("--validation-count", type=int, default=32)
    p.add_argument("--out", default="results/experiment")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()
    if min(args.sizes) < 2 or min(args.demand_maxes) < 1 or args.updates < 1 or args.batch < 1:
        p.error("sizes >=2, demand maxima and budgets must be positive")
    if args.capacity_factors[0] <= 0 or args.capacity_factors[0] > args.capacity_factors[1]:
        p.error("capacity factor interval must be positive and ordered")
    device = device_auto() if args.device == "auto" else args.device
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Policy(width=args.width, layers=args.layers,
                   distance_bias=not args.without_distance_bias).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(vars(args) | {"device_used": device}, indent=2))
    smoothed_reward = None
    elapsed = time.perf_counter()
    train_seconds = 0.0
    episodes = transitions = 0
    with (directory / "history.jsonl").open("w") as log:
        for iteration in range(1, args.updates + 1):
            n = int(rng.choice(args.sizes))
            cases = [generate(n, rng, capacity_factor=float(rng.uniform(*args.capacity_factors)),
                              demand_max=int(rng.choice(args.demand_maxes))) for _ in range(args.batch)]
            t = (iteration - 1) / max(1, args.updates - 1)
            temperature = args.temperature if args.anneal_to is None else (
                args.temperature * (1 - t) + args.anneal_to * t)
            step_start = time.perf_counter()
            record = update(model, optimizer, cases, args.method, temperature=temperature,
                            lam=args.lam, baseline=smoothed_reward, starts=args.starts,
                            rng=rng, entropy_weight=args.entropy)
            train_seconds += time.perf_counter() - step_start
            if args.method == "reinforce":
                reward = -record["mean_cost"]
                smoothed_reward = reward if smoothed_reward is None else .9 * smoothed_reward + .1 * reward
            transitions += record["transitions"]
            episodes += record["episodes"]
            record.update(iteration=iteration, n=n, episodes_total=episodes,
                          transitions_total=transitions, train_seconds=train_seconds,
                          elapsed_seconds=time.perf_counter() - elapsed)
            if iteration % args.validate_every == 0 or iteration == args.updates:
                record["validation"] = evaluate_greedy(model, args.sizes, args.validation_count)
                torch.save({"model": model.state_dict(), "architecture":
                            {"width": args.width, "layers": args.layers,
                             "distance_bias": not args.without_distance_bias},
                            "iteration": iteration}, directory / "last.pt")
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"{iteration}/{args.updates} n={n} cost={record['mean_cost']:.3f}", flush=True)


if __name__ == "__main__":
    main()
