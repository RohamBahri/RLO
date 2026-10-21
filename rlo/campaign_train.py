"""Long-run experience-only training with atomic, exactly resumable state."""

import argparse
import os
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import instances, training_batches, StatefulTrainingBatches
from .learning import device_auto, rollout, update
from .model import Policy


def evaluate_greedy(model, sizes, count, multistart=False):
    model.eval()
    result = {}
    with torch.no_grad():
        for n in sizes:
            cases = instances(n, "validation", count)
            envs, _ = rollout(model, cases, greedy=True)
            result[str(n)] = dict(cost=float(np.mean([e.cost for e in envs])),
                                  feasible=float(np.mean([e.done and not e.infeasible for e in envs])))
            if multistart:
                best = []
                for offset in range(0, count, 16):
                    group = cases[offset:offset + 16]
                    # Same instances, every customer as a forced first action.
                    repeated = [inst for inst in group for _ in range(n)]
                    starts = list(range(1, n + 1)) * len(group)
                    routes, _ = rollout(model, repeated, greedy=True, first_actions=starts)
                    best.extend(min(routes[j].cost for j in range(i * n, (i + 1) * n))
                                for i in range(len(group)))
                result[str(n)]["multistart_cost"] = float(np.mean(best))
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
    p.add_argument("--updates", type=int, default=None, help="maximum updates (default: 1000 unless --max-transitions is supplied)")
    p.add_argument("--max-transitions", type=int, default=None, help="stop after this training compute budget")
    p.add_argument("--batch", type=int, default=32, help="base instances; POMO samples n starts per instance")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--glimpse", action="store_true", help="optional masked multi-head decoder glimpse")
    p.add_argument("--without-distance-bias", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anneal-to", type=float, default=None)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--starts", type=int, default=None, help="POMO starts; default all n (fewer is POMO-k)")
    p.add_argument("--entropy", type=float, default=0.0)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--validate-every-transitions", type=int, default=100000)
    p.add_argument("--validation-count", type=int, default=256)
    p.add_argument("--validation-multistart", action="store_true", help="log n-start greedy cost for every method")
    p.add_argument("--selection-metric", choices=["greedy", "multistart"], default="greedy",
                   help="predeclared validation metric; use multistart at a shared n-start budget")
    p.add_argument("--baseline-check-every", type=int, default=100)
    p.add_argument("--baseline-count", type=int, default=256,
                   help="independent frozen training pool for rollout baseline promotion")
    p.add_argument("--resume", help="resume from this campaign last.pt, restoring RNG, optimizer and stream")
    p.add_argument("--checkpoint-every-transitions", type=int, default=1000000)
    p.add_argument("--milestones", type=int, nargs="*", default=[10000000, 25000000, 40000000, 50000000])
    p.add_argument("--stop-after-transitions", type=int, help="controlled interruption for resume testing")
    p.add_argument("--out", default="results/experiment")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()
    max_updates = args.updates if args.updates is not None else (10**9 if args.max_transitions else 1000)
    if min(args.sizes) < 2 or min(args.demand_maxes) < 1 or max_updates < 1 or args.batch < 1:
        p.error("sizes >=2, demand maxima and budgets must be positive")
    if args.capacity_factors[0] <= 0 or args.capacity_factors[0] > args.capacity_factors[1]:
        p.error("capacity factor interval must be positive and ordered")
    if args.max_transitions is not None and args.max_transitions < 1:
        p.error("--max-transitions must be positive")
    if args.validation_count < 2 or args.baseline_check_every < 1 or args.validate_every < 1 or args.validate_every_transitions < 1:
        p.error("evaluation counts and intervals must be positive")
    if args.selection_metric == "multistart" and not args.validation_multistart:
        p.error("--selection-metric multistart requires --validation-multistart for all methods")
    if args.method == "rollout" and args.baseline_count < 30:
        p.error("frozen rollout baseline requires at least 30 comparison instances")
    device = device_auto() if args.device == "auto" else args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    batches = StatefulTrainingBatches(args.seed, batch=args.batch, sizes=args.sizes,
                               capacity_factors=args.capacity_factors, demand_maxes=args.demand_maxes)
    architecture = dict(width=args.width, layers=args.layers,
                        distance_bias=not args.without_distance_bias, logit_clip=10.0,
                        glimpse=args.glimpse)
    model = Policy(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline_policy = copy.deepcopy(model).eval() if args.method == "rollout" else None
    directory = Path(args.out)
    if args.checkpoint_every_transitions < 1 or any(x < 1 for x in args.milestones):
        p.error("checkpoint interval and milestones must be positive")
    start_iteration = 0
    ema_by_size = {}
    train_seconds = 0.0
    episodes = transitions = last_validation_transitions = last_checkpoint_transitions = 0
    elapsed_offset = 0.0
    recorded_config = vars(args) | {"device_used": device, "architecture": architecture}
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.is_file():
            p.error(f"missing resume checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("campaign_version") != 1:
            p.error("checkpoint predates exact-resume campaign format")
        original = checkpoint["campaign_config"]
        exempt = {"resume", "stop_after_transitions"}
        if any(original.get(k) != v for k, v in vars(args).items() if k not in exempt):
            p.error("CLI settings differ from saved campaign configuration")
        if directory.resolve() != checkpoint_path.parent.resolve():
            p.error("resume checkpoint must belong to --out directory")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if baseline_policy is not None:
            baseline_policy.load_state_dict(checkpoint["rollout_baseline"])
        batches.load_state_dict(checkpoint["training_stream_rng"])
        rng.bit_generator.state = checkpoint["pomo_rng"]
        ema_by_size = dict(checkpoint["ema_by_size"])
        start_iteration = int(checkpoint["iteration"])
        episodes = int(checkpoint["episodes_total"])
        transitions = int(checkpoint["transitions_total"])
        last_validation_transitions = int(checkpoint["last_validation_transitions"])
        last_checkpoint_transitions = transitions
        train_seconds = float(checkpoint["train_seconds"])
        elapsed_offset = float(checkpoint["elapsed_seconds"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_cpu_rng"].cpu())
        if torch.cuda.is_available() and checkpoint["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng"])
        history_path = directory / "history.jsonl"
        raw_records = [json.loads(line) for line in history_path.read_text().splitlines()]
        retained = [line for line in raw_records if int(line["iteration"]) <= start_iteration]
        if not retained or int(retained[-1]["iteration"]) != start_iteration:
            p.error("history does not contain the checkpoint iteration")
        if len(retained) != len(raw_records):
            temp = history_path.with_suffix(".jsonl.tmp")
            temp.write_text("".join(json.dumps(line) + "\n" for line in retained))
            os.replace(temp, history_path)
    else:
        if directory.exists() and any(directory.iterdir()):
            p.error(f"{directory} is not empty; choose a new --out")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(recorded_config, indent=2) + "\n")
    # A completed checkpoint is a no-op. Never perform one extra optimizer
    # update when a previously interrupted job already crossed its budget.
    if args.resume and (start_iteration >= max_updates or
                        (args.max_transitions is not None and transitions >= args.max_transitions)):
        print(f"Already complete: iteration={start_iteration}, transitions={transitions}", flush=True)
        return
    start = time.perf_counter()

    def save_campaign_checkpoint(iteration, elapsed):
        nonlocal last_checkpoint_transitions
        state = {
            "campaign_version": 1, "campaign_config": recorded_config if not args.resume else checkpoint["campaign_config"],
            "model": model.state_dict(), "architecture": architecture,
            "architecture_version": 3 if args.glimpse else 2,
            "optimizer": optimizer.state_dict(),
            "rollout_baseline": baseline_policy.state_dict() if baseline_policy else None,
            "ema_by_size": ema_by_size,
            "training_stream_rng": batches.state_dict(),
            "pomo_rng": rng.bit_generator.state,
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "iteration": iteration, "episodes_total": episodes,
            "transitions_total": transitions, "last_validation_transitions": last_validation_transitions,
            "train_seconds": train_seconds, "elapsed_seconds": elapsed,
        }
        tmp = directory / f".checkpoint.{os.getpid()}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, directory / "last.pt")
        last_checkpoint_transitions = transitions
        for mark in args.milestones:
            target = directory / f"milestone_{mark}.pt"
            if transitions >= mark and not target.exists():
                # Atomic hard-link of the immutable, already-finished checkpoint.
                os.link(directory / "last.pt", target)

    # Baseline promotion influences learning, so NEVER use validation or test.
    baseline_cases = {n: next(training_batches(args.seed + 104729 + n,
                       batch=args.baseline_count, sizes=(n,),
                       capacity_factors=args.capacity_factors,
                       demand_maxes=args.demand_maxes)) for n in args.sizes} if baseline_policy else {}
    with (directory / "history.jsonl").open("a") as log:
        for iteration in range(start_iteration + 1, max_updates + 1):
            cases = next(batches)
            n = len(cases[0]["demands"]) - 1
            t = (min(1.0, transitions / args.max_transitions) if args.max_transitions
                 else (iteration - 1) / max(1, max_updates - 1))
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
                          elapsed_seconds=elapsed_offset + time.perf_counter() - start)
            stopping = iteration == max_updates or (
                args.max_transitions is not None and transitions >= args.max_transitions)
            interrupted = (args.stop_after_transitions is not None and
                           transitions >= args.stop_after_transitions)
            if (iteration % args.validate_every == 0 or stopping or interrupted or
                    transitions - last_validation_transitions >= args.validate_every_transitions):
                last_validation_transitions = transitions
                record["validation"] = evaluate_greedy(
                    model, args.sizes, args.validation_count,
                    multistart=(args.validation_multistart or args.method == "pomo"))
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (stopping or interrupted or "validation" in record or
                    transitions - last_checkpoint_transitions >= args.checkpoint_every_transitions):
                save_campaign_checkpoint(iteration, record["elapsed_seconds"])
            if iteration % args.validate_every == 0 or stopping or interrupted:
                print(f"{iteration}/{max_updates} n={n} cost={record['mean_cost']:.3f} "
                      f"transitions={transitions}", flush=True)
            if stopping or interrupted:
                break


if __name__ == "__main__":
    main()
