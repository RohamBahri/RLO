"""Reproducible experience-only masked Double-DQN and known-model Dyna training.

This is a new learning-family implementation, not a resumed on-policy run.
All training transitions come from the agent or its known CVRP simulator.
No exact/heuristic reference, held-out training label, or sealed test split.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import StatefulTrainingBatches
from .model import Policy
from .wave2_replay import Replay, collect_real, dyna_from_real, replay_q_update
from .train import evaluate_greedy


def epsilon_at(real_moves, real_budget, start=1.0, end=0.1, anneal_fraction=0.2):
    """A common fresh-experience schedule shared by DDQN and Dyna."""
    if real_budget <= 0 or not 0 < anneal_fraction <= 1:
        raise ValueError("invalid exploration schedule")
    progress = min(1.0, real_moves / (real_budget * anneal_fraction))
    return start + (end - start) * progress


def allocated_updates(base_updates, real_decisions, simulated_decisions, fractional=0.0):
    """Replay updates proportional to newly available real + model tuples.

    With one synthetic root per real decision, Dyna receives approximately
    twice the updates. Fractional remainder survives across batches/resume.
    """
    if base_updates < 1 or real_decisions < 1 or simulated_decisions < 0:
        raise ValueError("invalid replay update allocation")
    desired = fractional + base_updates * (real_decisions + simulated_decisions) / real_decisions
    whole = int(desired + 1e-10)
    return whole, desired - whole


def save_atomic(obj, path):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("ddqn", "dyna"), required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--capacity-factors", nargs=2, type=float, default=[0.75, 1.25])
    p.add_argument("--demand-maxes", nargs="+", type=int, default=[10])
    p.add_argument("--batch", type=int, default=32, help="fresh CVRP instances per collection batch")
    p.add_argument("--replay-batch", type=int, default=64)
    p.add_argument("--replay-capacity-per-size", type=int, default=100000)
    p.add_argument("--replay-warmup-decisions", type=int, default=256)
    p.add_argument("--updates-per-collection", type=int, default=8)
    p.add_argument("--target-sync-updates", type=int, default=1000)
    p.add_argument("--dyna-per-real-decision", type=float, default=None,
                   help="simulator counterfactual decisions per real decision; default 0 DDQN, 1 Dyna")
    p.add_argument("--synthetic-replay-fraction", type=float, default=0.5)
    p.add_argument("--n-step", type=int, choices=(1, 3), default=1)
    p.add_argument("--epsilon-anneal-fraction", type=float, default=0.2,
                   help="fraction of FRESH move budget until epsilon_end")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--max-real-transitions", type=int, default=50_000_000,
                   help="FRESH physical moves per run; Dyna extra model moves are additional")
    p.add_argument("--validate-every-transitions", type=int, default=2_500_000)
    p.add_argument("--checkpoint-every-transitions", type=int, default=1_000_000)
    p.add_argument("--milestones", nargs="*", type=int,
                   default=[10_000_000, 25_000_000, 40_000_000, 50_000_000])
    p.add_argument("--validation-count", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--stop-after-transitions", type=int, default=None,
                   help="testing/preemption drill; omitted from reproducible run configuration")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    if args.dyna_per_real_decision is None:
        args.dyna_per_real_decision = 1.0 if args.method == "dyna" else 0.0
    if args.method == "ddqn" and args.dyna_per_real_decision:
        p.error("DDQN control cannot use Dyna transitions")
    if not args.sizes or min(args.sizes) < 2 or not args.demand_maxes or min(args.demand_maxes) < 1:
        p.error("invalid instance distribution")
    if args.capacity_factors[0] <= 0 or args.capacity_factors[1] < args.capacity_factors[0]:
        p.error("invalid capacity factor range")
    if (min(args.batch, args.replay_batch, args.replay_capacity_per_size,
            args.replay_warmup_decisions, args.updates_per_collection,
            args.target_sync_updates, args.max_real_transitions,
            args.validate_every_transitions, args.checkpoint_every_transitions,
            args.validation_count) < 1
            or any(m < 1 for m in args.milestones)):
        p.error("budgets/counts must be positive")
    if not 0 <= args.synthetic_replay_fraction <= 1 or args.dyna_per_real_decision < 0:
        p.error("invalid Dyna fractions")
    if not (0 <= args.epsilon_end <= args.epsilon_start <= 1):
        p.error("epsilon schedule must be in [0,1], from high to low")
    if not 0 < args.epsilon_anneal_fraction <= 1:
        p.error("epsilon annealing fraction must be in (0,1]")
    if args.lr <= 0 or args.width % 4:
        p.error("positive LR and width divisible by 4 required")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = np.random.default_rng(args.seed)
    simulator_rng = np.random.default_rng(args.seed + 1_000_003)
    stream = StatefulTrainingBatches(args.seed, batch=args.batch,
                                    sizes=args.sizes,
                                    capacity_factors=args.capacity_factors,
                                    demand_maxes=args.demand_maxes)
    arch = dict(width=args.width, layers=args.layers, logit_clip=None,
                distance_bias=True, glimpse=False)
    model = Policy(**arch).to(args.device)
    target = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = Replay(args.replay_capacity_per_size)
    out = args.out.resolve()
    config = vars(args).copy()
    config["out"] = str(out)
    config["resume"] = None
    config["stop_after_transitions"] = None
    counters = dict(real_moves=0, dyna_moves=0, real_decisions=0,
                    dyna_decisions=0, replay_samples=0, optimizer_updates=0,
                    episodes=0, iteration=0, last_validation_moves=0,
                    last_checkpoint_moves=0, update_credit=0.0, train_seconds=0.0, elapsed_seconds=0.0)

    if args.resume:
        checkpoint = args.resume.resolve()
        if checkpoint != out / "resume.pt":
            p.error("--resume must be the output directory's resume.pt")
        # Only load this unsafe-pickle-capable object from trusted self-produced runs.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("wave2_format") != 1 or state["config"] != config:
            p.error("resume configuration mismatch")
        model.load_state_dict(state["model"])
        target.load_state_dict(state["target"])
        opt.load_state_dict(state["optimizer"])
        # Optimizer CPU->CUDA state move for resumed GPU jobs.
        for values in opt.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(args.device)
        replay = state["replay"]
        stream.load_state_dict(state["stream"])
        generator.bit_generator.state = state["numpy_rng"]
        simulator_rng.bit_generator.state = state["simulator_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if args.device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        counters = state["counters"]
        lines = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
        lines = [line for line in lines if line["iteration"] <= counters["iteration"]]
        if not lines or lines[-1]["iteration"] != counters["iteration"]:
            p.error("history does not contain resume checkpoint iteration")
        path = out / "history.jsonl"
        (out / "history.tmp").write_text("".join(json.dumps(line) + "\n" for line in lines))
        os.replace(out / "history.tmp", path)
    else:
        existing = {x.name for x in out.iterdir()} if out.exists() else set()
        if existing - {"commit.txt"}:
            p.error(f"experiment output already populated: {sorted(existing)}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def sim_count():
        return counters["real_moves"] + counters["dyna_moves"]

    if counters["real_moves"] >= args.max_real_transitions:
        print(f"Already complete: {counters[\'real_moves\']:,} fresh physical moves")
        return

    begin = time.perf_counter()

    def checkpoint(moves):
        counters["last_checkpoint_moves"] = moves
        elapsed_before = counters["elapsed_seconds"]
        counters["elapsed_seconds"] = elapsed_before + time.perf_counter() - begin
        state = dict(wave2_format=1, config=config, model=model.state_dict(),
                     target=target.state_dict(), optimizer=opt.state_dict(),
                     replay=replay, stream=stream.state_dict(),
                     numpy_rng=generator.bit_generator.state,
                     simulator_rng=simulator_rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
                     python_rng=random.getstate(), counters=copy.deepcopy(counters))
        save_atomic(state, out / "resume.pt")
        counters["elapsed_seconds"] = elapsed_before
        eval_model = dict(model=model.state_dict(), architecture=arch, architecture_version=2,
                          iteration=counters["iteration"],
                          simulator_transitions=sim_count(),
                          real_transitions=moves, method=args.method, n_step=args.n_step)
        save_atomic(eval_model, out / "last.pt")
        for mark in args.milestones:
            final = out / f"milestone_{mark}.pt"
            if moves >= mark and not final.exists():
                # Immutable checkpoint; weights saved immediately after the mark was reached.
                os.link(out / "last.pt", final)

    with (out / "history.jsonl").open("a") as fh:
        while counters["real_moves"] < args.max_real_transitions:
            tic = time.perf_counter()
            epsilon = epsilon_at(
                counters["real_moves"], args.max_real_transitions,
                args.epsilon_start, args.epsilon_end,
                args.epsilon_anneal_fraction)
            cases = next(stream)
            new, envs = collect_real(model, cases, replay, generator, epsilon,
                                    n_step=args.n_step)
            real_moves = sum(x.physical_moves for x in new)
            counters["real_moves"] += real_moves
            counters["real_decisions"] += len(new)
            counters["episodes"] += len(envs)
            if args.method == "dyna":
                want = args.dyna_per_real_decision * len(new)
                # Stochastic rounding: unbiased number of simulator decisions.
                count = int(want)
                if simulator_rng.random() < want - count:
                    count += 1
                generated = dyna_from_real(replay, simulator_rng, count,
                                           n_step=args.n_step, model=model, epsilon=epsilon)
                counters["dyna_moves"] += sum(x.physical_moves for x in generated)
                counters["dyna_decisions"] += len(generated)

            # Equal replay samples per newly stored item across the two methods.
            # Dyna is intentionally allowed more optimizer updates with its
            # additional simulated transitions; report them as compute overhead.
            new_synthetic = len(generated) if args.method == "dyna" else 0
            steps, counters["update_credit"] = allocated_updates(
                args.updates_per_collection, len(new), new_synthetic,
                counters["update_credit"])
            loss_rec = None
            if replay.real_count() >= args.replay_warmup_decisions:
                for _ in range(steps):
                    samples = replay.sample(generator, args.replay_batch,
                                            synthetic_fraction=args.synthetic_replay_fraction)
                    model.train()
                    loss_rec = replay_q_update(model, target, opt, samples, args.device)
                    counters["optimizer_updates"] += 1
                    counters["replay_samples"] += len(samples)
                    if counters["optimizer_updates"] % args.target_sync_updates == 0:
                        target.load_state_dict(model.state_dict())
            counters["train_seconds"] += time.perf_counter() - tic
            counters["iteration"] += 1
            moves = counters["real_moves"]
            stop = moves >= args.max_real_transitions
            interrupted = args.stop_after_transitions is not None and moves >= args.stop_after_transitions
            record = dict(iteration=counters["iteration"],
                          transitions_total=moves, real_transitions=counters["real_moves"],
                          total_simulated_transitions=sim_count(), n_step=args.n_step,
                          dyna_transitions=counters["dyna_moves"],
                          real_decisions=counters["real_decisions"],
                          dyna_decisions=counters["dyna_decisions"],
                          replay_samples=counters["replay_samples"],
                          optimizer_updates=counters["optimizer_updates"],
                          replay_items=replay.total_count(), epsilon=epsilon,
                          current_epsilon=epsilon_at(
                              moves, args.max_real_transitions,
                              args.epsilon_start, args.epsilon_end,
                              args.epsilon_anneal_fraction),
                          samples_per_collected_item=(counters["replay_samples"] /
                              max(1, counters["real_decisions"] + counters["dyna_decisions"])),
                          loss=None if loss_rec is None else loss_rec["loss"],
                          train_seconds=counters["train_seconds"],
                          elapsed_seconds=counters["elapsed_seconds"] + time.perf_counter() - begin)
            if (stop or interrupted or
                moves - counters["last_validation_moves"] >= args.validate_every_transitions):
                validation = evaluate_greedy(model, args.sizes, args.validation_count)
                record["validation"] = validation
                counters["last_validation_moves"] = moves
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if (stop or interrupted or "validation" in record or
                moves - counters["last_checkpoint_moves"] >= args.checkpoint_every_transitions):
                checkpoint(moves)
            if stop or interrupted:
                print(f"STOP method={args.method} moves={moves} updates={counters['optimizer_updates']}",
                      flush=True)
                break


if __name__ == "__main__":
    main()
