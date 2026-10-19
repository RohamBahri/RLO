"""Reproducible, experience-only Group A training. No reference solutions are used."""

import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import instances, training_batches
from .learning import device_auto, rollout, update
from .model import Policy


def evaluate_greedy(model, sizes, count):
    model.eval()
    result = {}
    with torch.no_grad():
        for n in sizes:
            cases = instances(n, "validation", count)
            envs, _ = rollout(model, cases, greedy=True)
            result[str(n)] = dict(cost=float(np.mean([e.cost for e in envs])),
                                  feasible=float(np.mean([e.done and not e.infeasible for e in envs])))
    return result


def maybe_replace_baseline(model, baseline, cases_by_size):
    """Kool-style conservative baseline update on held-out validation instances.

    Use a paired one-sided ~2.5% test (normal approximation, n >= 30).
    Positive values mean the current policy improves the frozen baseline.
    """
    differences, transitions = [], 0
    with torch.no_grad():
        for cases in cases_by_size.values():
            old, _ = rollout(baseline, cases, greedy=True)
            new, _ = rollout(model, cases, greedy=True)
            differences.extend((a.cost - b.cost) / a.cost for a, b in zip(old, new))
            transitions += sum(len(e.tour) - 1 for e in old + new)
    differences = np.array(differences)
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    accept = mean > max(0, 1.96 * se)
    if accept:
        baseline.load_state_dict(model.state_dict())
    return accept, mean, transitions


def main():
    p = argparse.ArgumentParser(description="Experience-only masked CVRP training")
    p.add_argument("--method", choices=["reinforce", "rollout", "a2c", "pomo"], default="a2c")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--capacity-factors", type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument("--demand-maxes", type=int, nargs="+", default=[10])
    p.add_argument("--updates", type=int, default=1000, help="maximum optimizer updates")
    p.add_argument("--max-transitions", type=int, default=None, help="stop after this training compute budget")
    p.add_argument("--batch", type=int, default=32, help="base instances; POMO samples n starts per instance")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--without-distance-bias", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anneal-to", type=float, default=None)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--starts", type=int, default=None, help="POMO starts; default all n (fewer is POMO-k)")
    p.add_argument("--entropy", type=float, default=0.0)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--validate-every-transitions", type=int, default=100000)
    p.add_argument("--validation-count", type=int, default=256)
    p.add_argument("--baseline-check-every", type=int, default=100)
    p.add_argument("--baseline-count", type=int, default=256,
                   help="independent frozen training pool for rollout baseline promotion")
    p.add_argument("--out", default="results/experiment")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()
    if min(args.sizes) < 2 or min(args.demand_maxes) < 1 or args.updates < 1 or args.batch < 1:
        p.error("sizes >=2, demand maxima and budgets must be positive")
    if args.capacity_factors[0] <= 0 or args.capacity_factors[0] > args.capacity_factors[1]:
        p.error("capacity factor interval must be positive and ordered")
    if args.max_transitions is not None and args.max_transitions < 1:
        p.error("--max-transitions must be positive")
    if args.validation_count < 2 or args.baseline_check_every < 1 or args.validate_every < 1 or args.validate_every_transitions < 1:
        p.error("evaluation counts and intervals must be positive")
    if args.method == "rollout" and args.baseline_count < 30:
        p.error("frozen rollout baseline requires at least 30 comparison instances")
    device = device_auto() if args.device == "auto" else args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    batches = training_batches(args.seed, batch=args.batch, sizes=args.sizes,
                               capacity_factors=args.capacity_factors, demand_maxes=args.demand_maxes)
    architecture = dict(width=args.width, layers=args.layers,
                        distance_bias=not args.without_distance_bias, logit_clip=10.0)
    model = Policy(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline_policy = copy.deepcopy(model).eval() if args.method == "rollout" else None
    directory = Path(args.out)
    if (directory / "history.jsonl").exists():
        p.error(f"{directory} already contains an experiment; choose a new --out")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(vars(args) | {"device_used": device,
                        "architecture": architecture}, indent=2) + "\n")
    ema_by_size = {}
    start = time.perf_counter()
    train_seconds = 0.0
    episodes = transitions = last_validation_transitions = 0
    # Baseline promotion influences learning, so NEVER use validation or test.
    baseline_cases = {n: next(training_batches(args.seed + 104729 + n,
                       batch=args.baseline_count, sizes=(n,),
                       capacity_factors=args.capacity_factors,
                       demand_maxes=args.demand_maxes)) for n in args.sizes} if baseline_policy else {}
    with (directory / "history.jsonl").open("w") as log:
        for iteration in range(1, args.updates + 1):
            cases = next(batches)
            n = len(cases[0]["demands"]) - 1
            t = (min(1.0, transitions / args.max_transitions) if args.max_transitions
                 else (iteration - 1) / max(1, args.updates - 1))
            temperature = args.temperature if args.anneal_to is None else (
                args.temperature * (1 - t) + args.anneal_to * t)
            step_start = time.perf_counter()
            record = update(model, optimizer, cases, args.method, temperature=temperature,
                            lam=args.lam, baseline=ema_by_size.get(n), baseline_policy=baseline_policy,
                            starts=args.starts, rng=rng, entropy_weight=args.entropy)
            train_seconds += time.perf_counter() - step_start
            if args.method == "reinforce":
                reward = -record["mean_cost"]
                old = ema_by_size.get(n)
                ema_by_size[n] = reward if old is None else .9 * old + .1 * reward
            episodes += record["episodes"]
            transitions += record["transitions"]
            if baseline_policy is not None and iteration % args.baseline_check_every == 0:
                check_start = time.perf_counter()
                accepted, improvement, checked = maybe_replace_baseline(model, baseline_policy, baseline_cases)
                train_seconds += time.perf_counter() - check_start
                transitions += checked
                record.update(baseline_accepted=accepted, baseline_relative_improvement=improvement,
                              baseline_check_transitions=checked)
            record.update(iteration=iteration, n=n, episodes_total=episodes,
                          transitions_total=transitions, train_seconds=train_seconds,
                          elapsed_seconds=time.perf_counter() - start)
            stopping = iteration == args.updates or (
                args.max_transitions is not None and transitions >= args.max_transitions)
            if (iteration % args.validate_every == 0 or stopping or
                    transitions - last_validation_transitions >= args.validate_every_transitions):
                last_validation_transitions = transitions
                record["validation"] = evaluate_greedy(model, args.sizes, args.validation_count)
                torch.save({"model": model.state_dict(), "architecture": architecture,
                            "architecture_version": 2,
                            "iteration": iteration,
                            "rollout_baseline": baseline_policy.state_dict() if baseline_policy else None},
                           directory / "last.pt")
            log.write(json.dumps(record) + "\n")
            log.flush()
            if iteration % args.validate_every == 0 or stopping:
                print(f"{iteration}/{args.updates} n={n} cost={record['mean_cost']:.3f} "
                      f"transitions={transitions}", flush=True)
            if stopping:
                break


if __name__ == "__main__":
    main()
