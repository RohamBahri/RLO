"""Wave 3 research campaigns: PPO/GAE, synthetic actor/critic, rollout-MBPI,
and controlled representation/auxiliary/hierarchy/POMO-distribution variants.

No expert actions, exact-oracle labels, or final-test instances enter training.
Physical moves, model-generated moves, replay updates and wall time are separate.
"""
import argparse
import copy
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .campaign_train import evaluate_greedy
from .data import StatefulTrainingBatches
from .env import CVRPEnv
from .wave2_replay import observations
from .wave3_model import ResearchPolicy
from .wave3_core import (environments, trajectories, policy_objective,
                         ppo_update, critic_only_update)


def save_atomic(state, path):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def mbpi_step(model, optimizer, traces, rng, *, roots_per_collection=4,
              rollouts_per_action=8, eta=.2, projection_epochs=1):
    """All-legal-action KL-MBPI using ONLY the current policy's sampled returns.

    Roots are real visited states; subsequent policy trajectories are simulated
    with the known CVRP transition function. Full physical move costs are
    counted, including each counterfactual root action.
    """
    candidates = [s["before"] for seq in traces for s in seq]
    if not candidates:
        raise ValueError("no real decision roots for MBPI")
    roots = [candidates[int(rng.integers(len(candidates)))]
             for _ in range(roots_per_collection)]
    dev = next(model.parameters()).device
    q_rows = []
    synthetic_moves, rollouts = 0, 0
    for root in roots:
        legal = np.flatnonzero(root.restore().legal())
        q = np.full(len(root.visited), np.inf, dtype=np.float64)
        for action in legal:
            costs = []
            for j in range(0, rollouts_per_action, 64):
                k = min(64, rollouts_per_action - j)
                envs = [root.restore() for _ in range(k)]
                for e in envs:
                    e.step(int(action))
                init = [len(e.tour) for e in envs]
                _, envs, moves = trajectories(model, envs, grad=False, record=False)
                # Every explicit root action is additional simulated work.
                synthetic_moves += moves + sum(length - 1 for length in init)
                rollouts += k
                costs.extend(float(e.cost) for e in envs)
            q[action] = np.mean(costs)
        q_rows.append(q)
    obs = observations(roots)
    nd, glob, dist, masks = [torch.as_tensor(x, device=dev) for x in obs]
    with torch.no_grad():
        old_logits, _ = model(nd, glob, dist, masks)
        priors = torch.log_softmax(old_logits, -1)
        target = torch.zeros_like(priors)
        for i, q in enumerate(q_rows):
            legal = masks[i].cpu().numpy()
            costs = torch.tensor(q[legal], device=dev, dtype=priors.dtype)
            target[i, masks[i]] = torch.softmax(
                priors[i, masks[i]] - (costs - costs.min()) / eta, 0)
    losses = []
    for _ in range(projection_epochs):
        logits, _ = model(nd, glob, dist, masks)
        loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(g):
            raise FloatingPointError("nonfinite MBPI projection")
        optimizer.step()
        losses.append(float(loss.detach()))
    return dict(loss=float(np.mean(losses)), optimizer_updates=projection_epochs,
                mbpi_roots=len(roots), mbpi_rollouts=rollouts,
                synthetic_moves=synthetic_moves)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("ppo", "dyna_critic", "dyna_actor",
                                      "pomo", "reinforce", "a2c", "mbpi"), required=True)
    p.add_argument("--encoder", choices=("attention", "raw", "mlp", "codebook"), default="attention")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--codebook-size", type=int, default=32)
    p.add_argument("--dynamic-reencode", action="store_true")
    p.add_argument("--glimpse", action="store_true")
    p.add_argument("--no-distance-bias", action="store_true")
    p.add_argument("--hierarchy", action="store_true")
    p.add_argument("--auxiliary", action="store_true")
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--capacity-factors", nargs=2, type=float, default=[.75, 1.25])
    p.add_argument("--demand-maxes", nargs="+", type=int, default=[10])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--starts", type=int, default=0, help="POMO starts; 0=all customers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anneal-to", type=float, default=None)
    p.add_argument("--lam", type=float, default=.95)
    p.add_argument("--entropy", type=float, default=.01)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-minibatch", type=int, default=64)
    p.add_argument("--ppo-clip", type=float, default=.2)
    p.add_argument("--dyna-roots", type=int, default=16)
    p.add_argument("--mbpi-roots", type=int, default=4)
    p.add_argument("--mbpi-rollouts", type=int, default=8)
    p.add_argument("--mbpi-eta", type=float, default=.2)
    p.add_argument("--mbpi-epochs", type=int, default=1)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--max-moves", type=int, default=50_000_000,
                   help="fresh CVRP physical moves, except MBPI: fresh+simulated budget")
    p.add_argument("--validate-every-moves", type=int, default=2_500_000)
    p.add_argument("--checkpoint-every-moves", type=int, default=1_000_000)
    p.add_argument("--milestones", nargs="*", type=int,
                   default=[10_000_000, 25_000_000, 40_000_000, 50_000_000])
    p.add_argument("--validation-count", type=int, default=256)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--stop-after-moves", type=int, default=None,
                   help="stop and checkpoint early for a reproducible interruption test")
    a = p.parse_args(argv)
    if (a.batch < 1 or min(a.sizes) < 2 or min(a.demand_maxes) < 1 or
        not 0 < a.capacity_factors[0] <= a.capacity_factors[1] or
        a.width < 8 or a.width % a.heads or a.max_moves < 1 or
        a.checkpoint_every_moves < 1 or a.validate_every_moves < 1 or
        a.validation_count < 2 or a.temperature <= 0 or a.lam < 0 or a.lam > 1 or
        a.ppo_epochs < 1 or a.ppo_minibatch < 1 or a.ppo_clip <= 0 or
        a.dyna_roots < 1 or a.mbpi_roots < 1 or a.mbpi_rollouts < 1 or
        a.mbpi_eta <= 0 or a.mbpi_epochs < 1 or (a.auxiliary and a.aux_weight <= 0)):
        p.error("invalid nonpositive budget, feature, exploration, or network parameters")
    if a.anneal_to is not None and a.anneal_to <= 0:
        p.error("--anneal-to must be positive")
    if a.glimpse and a.encoder != "attention":
        p.error("glimpse is only supported by the attention encoder")
    if a.glimpse and a.heads != 4:
        p.error("glimpse requires four heads")
    if a.method == "mbpi" and not a.initial_checkpoint and not a.resume:
        p.error("MBPI must start at a frozen 25M Wave-1 checkpoint")
    if a.method == "pomo" and a.starts == 1:
        p.error("POMO uses at least two forced starts")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but no GPU was visible")
    return a


def main(argv=None):
    a = parse_args(argv)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed + 7919)
    stream = StatefulTrainingBatches(a.seed, batch=a.batch, sizes=a.sizes,
                                      capacity_factors=a.capacity_factors,
                                      demand_maxes=a.demand_maxes)
    config_arch = dict(width=a.width, layers=a.layers, heads=a.heads,
                       distance_bias=not a.no_distance_bias, logit_clip=10.,
                       glimpse=a.glimpse, encoder_type=a.encoder,
                       dynamic_reencode=a.dynamic_reencode, hierarchy=a.hierarchy,
                       codebook_size=a.codebook_size, auxiliary=a.auxiliary)
    model = ResearchPolicy(**config_arch).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    value_opt = (torch.optim.Adam(model.value.parameters(), lr=a.lr)
                 if a.method == "dyna_critic" else None)
    directory = Path(a.out).expanduser().resolve()
    config = vars(a).copy()
    for key in ("resume", "stop_after_moves"):
        config[key] = None
    config["out"] = str(directory)
    counters = dict(iteration=0, real_moves=0, synthetic_moves=0, optimizer_updates=0,
                    replay_decisions=0, episodes=0, train_seconds=0., elapsed_seconds=0.,
                    last_validation_moves=0, last_checkpoint_moves=0)
    if a.resume:
        path = Path(a.resume).resolve()
        if path != directory / "last.pt":
            raise ValueError("resume must be the run's own last.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("wave3_version") != 1 or saved["config"] != config:
            raise ValueError("resume source/configuration mismatch")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        if value_opt:
            value_opt.load_state_dict(saved["value_optimizer"])
        for optimizer in [opt, value_opt]:
            if optimizer:
                for group in optimizer.state.values():
                    for name, item in group.items():
                        if isinstance(item, torch.Tensor):
                            group[name] = item.to(a.device)
        stream.load_state_dict(saved["stream_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if a.device == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        counters = saved["counters"]
        history = directory / "history.jsonl"
        rows = [json.loads(s) for s in history.read_text().splitlines()]
        rows = [x for x in rows if x["iteration"] <= counters["iteration"]]
        if not rows or rows[-1]["iteration"] != counters["iteration"]:
            raise ValueError("checkpoint/history mismatch")
        temp = directory / "history.jsonl.tmp"
        temp.write_text("".join(json.dumps(x) + "\n" for x in rows))
        os.replace(temp, history)
    else:
        existing = {f.name for f in directory.iterdir()} if directory.exists() else set()
        if existing - {"commit.txt"}:
            raise ValueError("refusing populated experiment directory: " + str(directory))
        directory.mkdir(parents=True, exist_ok=True)
        if a.initial_checkpoint:
            # This is a trusted local campaign-generated checkpoint, which
            # contains numpy RNG data in addition to model tensors.
            initial = torch.load(a.initial_checkpoint, map_location="cpu", weights_only=False)
            if (initial.get("architecture_version") not in (2, 3) or
                initial["architecture"] != dict(width=a.width, layers=a.layers,
                    distance_bias=not a.no_distance_bias, logit_clip=10., glimpse=a.glimpse)):
                raise ValueError("MBPI source checkpoint architecture mismatch")
            model.load_state_dict(initial["model"], strict=True)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if a.method == "mbpi" and (a.auxiliary or a.hierarchy or a.dynamic_reencode
                              or a.encoder != "attention" or a.heads != 4):
        raise ValueError("MBPI must match the pretrained flat attention policy")

    def budget():
        return counters["real_moves"] + (counters["synthetic_moves"] if a.method == "mbpi" else 0)

    if budget() >= a.max_moves:
        print("ALREADY_COMPLETE", budget(), flush=True)
        return
    stop_requested = False
    def handle_signal(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    started = time.perf_counter()

    def checkpoint():
        elapsed = counters["elapsed_seconds"] + time.perf_counter() - started
        state = dict(wave3_version=1, architecture=config_arch, config=config,
                     model=model.state_dict(), optimizer=opt.state_dict(),
                     value_optimizer=value_opt.state_dict() if value_opt else None,
                     stream_rng=stream.state_dict(), numpy_rng=copy.deepcopy(rng.bit_generator.state),
                     python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if a.device == "cuda" else None,
                     counters=copy.deepcopy({**counters, "elapsed_seconds":elapsed}))
        save_atomic(state, directory / "last.pt")
        save_atomic(dict(wave3_version=1, architecture=config_arch, model=model.state_dict()),
                    directory / "policy.pt")
        counters["last_checkpoint_moves"] = budget()
        for mark in a.milestones:
            target = directory / f"milestone_{mark}.pt"
            if budget() >= mark and not target.exists():
                os.link(directory / "last.pt", target)

    with (directory / "history.jsonl").open("a") as log:
        while budget() < a.max_moves:
            t0 = time.perf_counter()
            cases = next(stream)
            n = len(cases[0]["demands"]) - 1
            starts = (n if a.starts == 0 else a.starts) if a.method == "pomo" else 1
            initial_envs = environments(cases, starts, rng)
            extra_initial = sum(len(e.tour) - 1 for e in initial_envs)
            frac = min(1., budget() / a.max_moves)
            temp = (a.temperature if a.anneal_to is None
                    else a.temperature * (1-frac) + a.anneal_to * frac)
            traced, final, fresh = trajectories(
                model, initial_envs, grad=a.method not in ("ppo", "mbpi"),
                temperature=temp)
            fresh += extra_initial
            record = dict(n=n, mean_cost=float(np.mean([e.cost for e in final])),
                          fresh_moves=fresh, synthetic_added=0,
                          sampled_episodes=len(final), temperature=temp,
                          loss=None, optimizer_added=0)
            counters["real_moves"] += fresh
            counters["episodes"] += len(final)
            if a.method == "ppo":
                result = ppo_update(model, opt, traced, epochs=a.ppo_epochs,
                                    minibatch=a.ppo_minibatch, clip=a.ppo_clip,
                                    lam=a.lam, entropy_weight=a.entropy,
                                    aux_weight=a.aux_weight if a.auxiliary else 0.)
                record.update(result)
                record["optimizer_added"] = result["optimizer_updates"]
                counters["replay_decisions"] += result["replay_decisions"]
            elif a.method == "mbpi":
                result = mbpi_step(model, opt, traced, rng, roots_per_collection=a.mbpi_roots,
                                   rollouts_per_action=a.mbpi_rollouts,
                                   eta=a.mbpi_eta, projection_epochs=a.mbpi_epochs)
                counters["synthetic_moves"] += result["synthetic_moves"]
                record.update(result, synthetic_added=result["synthetic_moves"],
                              optimizer_added=result["optimizer_updates"])
            else:
                objective = "pomo" if a.method == "pomo" else (
                    "reinforce" if a.method == "reinforce" else "a2c")
                loss, components = policy_objective(
                    model, traced, final, objective, lam=a.lam,
                    aux_weight=a.aux_weight if a.auxiliary else 0.,
                    entropy_weight=a.entropy, starts=starts)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(loss) or not torch.isfinite(grad):
                    raise FloatingPointError("nonfinite policy optimization")
                opt.step()
                record.update(loss=float(loss.detach()), **components)
                record["optimizer_added"] = 1
                if a.method in ("dyna_critic", "dyna_actor"):
                    visited = [s["before"] for seq in traced for s in seq]
                    synth = [visited[int(rng.integers(len(visited)))].restore()
                             for _ in range(a.dyna_roots)]
                    synthetic, completed, used = trajectories(model, synth, grad=True,
                                                             temperature=temp)
                    counters["synthetic_moves"] += used
                    record["synthetic_added"] = used
                    if a.method == "dyna_critic":
                        add = critic_only_update(model, value_opt, synthetic)
                    else:
                        syn_loss, _ = policy_objective(model, synthetic, completed, "a2c",
                                                       lam=a.lam, entropy_weight=a.entropy,
                                                       aux_weight=a.aux_weight if a.auxiliary else 0.)
                        opt.zero_grad(set_to_none=True)
                        syn_loss.backward()
                        gr = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        if not torch.isfinite(syn_loss) or not torch.isfinite(gr):
                            raise FloatingPointError("nonfinite synthetic actor-critic update")
                        opt.step()
                        add = dict(loss=float(syn_loss.detach()), optimizer_updates=1)
                    record["synthetic_loss"] = add["loss"]
                    record["optimizer_added"] += add["optimizer_updates"]
            counters["optimizer_updates"] += record["optimizer_added"]
            counters["iteration"] += 1
            counters["train_seconds"] += time.perf_counter() - t0
            record.update(iteration=counters["iteration"],
                          real_transitions=counters["real_moves"],
                          synthetic_transitions=counters["synthetic_moves"],
                          transitions_total=budget(),
                          optimizer_updates=counters["optimizer_updates"],
                          replay_decisions=counters["replay_decisions"],
                          episodes=counters["episodes"],
                          train_seconds=counters["train_seconds"],
                          elapsed_seconds=counters["elapsed_seconds"] + time.perf_counter() - started)
            stopping = budget() >= a.max_moves or stop_requested or (
                a.stop_after_moves is not None and budget() >= a.stop_after_moves)
            if stopping or budget() - counters["last_validation_moves"] >= a.validate_every_moves:
                record["validation"] = evaluate_greedy(model, a.sizes, a.validation_count)
                counters["last_validation_moves"] = budget()
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (stopping or "validation" in record or
                budget() - counters["last_checkpoint_moves"] >= a.checkpoint_every_moves):
                checkpoint()
            if stopping or "validation" in record:
                print(json.dumps({k: record[k] for k in
                      ("iteration", "real_transitions", "synthetic_transitions",
                       "optimizer_updates", "loss", "validation") if k in record}), flush=True)
            if stopping:
                if stop_requested and budget() < a.max_moves:
                    print("PAUSED_AT_SIGNAL checkpoint_saved", flush=True)
                break


if __name__ == "__main__":
    main()
