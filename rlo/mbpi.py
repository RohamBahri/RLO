"""Experience-only KL policy improvement from simulated rollout returns.

No expert actions, exact-oracle labels, or heuristic demonstrations enter training.
The exact oracle may be used separately to diagnose action regret at n <= 10.
"""

import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import instances
from .env import CVRPEnv
from .learning import device_auto
from .model import Policy
from .oracle import ExactOracle
from .problem import generate
from .study import model_from_checkpoint


def policy_logits(model, env):
    obs = env.observe()
    device = next(model.parameters()).device
    args = [torch.as_tensor(x, device=device).unsqueeze(0) for x in obs]
    return model(*args)[0][0]


def sample_action(model, env):
    logits = policy_logits(model, env)
    return int(torch.distributions.Categorical(logits=logits).sample())


def state_from_experience(model, instance, rng):
    """Select one decision state on an on-policy trajectory (not a target tour)."""
    env = CVRPEnv(instance)
    states = []
    with torch.no_grad():
        while not env.done:
            states.append(copy.deepcopy(env))
            env.step(sample_action(model, env))
    if not states:
        return None
    selected = states[int(rng.integers(len(states)))]
    selected.collection_transitions = len(env.tour) - 1
    return selected


def estimate_q(model, env, samples):
    """Expected future travel cost after each legal action under current policy."""
    q = np.full(len(env.visited), np.inf)
    steps = 0
    with torch.no_grad():
        for action in np.flatnonzero(env.legal()):
            costs = []
            for _ in range(samples):
                sim = copy.deepcopy(env)
                start = sim.cost
                sim.step(int(action))
                while not sim.done:
                    sim.step(sample_action(model, sim))
                costs.append(sim.cost - start)
                steps += len(sim.tour) - len(env.tour)
            q[action] = float(np.mean(costs))
    return q, steps


def improve(model, optimizer, states, *, samples=8, eta=0.2, oracle_metrics=False):
    """One KL-regularized improvement update, computed from experience only."""
    if eta <= 0 or samples < 1:
        raise ValueError("eta and samples must be positive")
    losses, errors = [], []
    total_steps = 0
    collection_steps = sum(getattr(env, "collection_transitions", 0) for env in states if env is not None)
    search_rollouts = 0
    for env in states:
        if env is None:
            continue
        q, steps = estimate_q(model, env, samples)
        total_steps += steps
        search_rollouts += samples * int(env.legal().sum())
        logits = policy_logits(model, env)
        legal = torch.as_tensor(env.legal(), device=logits.device)
        q_tensor = torch.as_tensor(q[env.legal()], device=logits.device, dtype=logits.dtype)
        old_logp = logits.detach().log_softmax(dim=-1)[legal]
        improved = torch.softmax(old_logp - (q_tensor - q_tensor.min()) / eta, dim=-1)
        losses.append(-(improved.detach() * logits.log_softmax(-1)[legal]).sum())
        if oracle_metrics:
            exact = ExactOracle(env.instance).action_values(env.state)
            choices = np.flatnonzero(env.legal())
            errors.append(float(exact[choices[np.argmin(q[choices])]] - min(exact[choices])))
    if not losses:
        raise ValueError("no decision states to improve")
    objective = torch.stack(losses).mean()
    optimizer.zero_grad()
    objective.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return dict(loss=float(objective.detach()), rollout_transitions=total_steps,
                collection_transitions=collection_steps,
                transitions_total=total_steps + collection_steps,
                search_rollouts=search_rollouts,
                states=len(losses), mean_optimal_action_regret=(float(np.mean(errors)) if errors else None))


def main():
    p = argparse.ArgumentParser(description="KL-MBPI from the current policy's own rollouts")
    p.add_argument("--checkpoint", help="start from an existing policy checkpoint")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--updates", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diagnostic", action="store_true", help="report exact action regret for n<=10 (evaluation only)")
    p.add_argument("--out", default="results/mbpi")
    args = p.parse_args()
    if args.diagnostic and args.n > 10:
        p.error("exact diagnostics support n <= 10")
    if args.n < 2 or args.updates < 1 or args.batch < 1:
        p.error("n>=2, updates and batch must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = model_from_checkpoint(args.checkpoint) if args.checkpoint else (
        Policy(width=args.width, layers=args.layers).to(device_auto()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    start = time.perf_counter()
    cumulative = 0
    with (out / "history.jsonl").open("w") as log:
        for i in range(args.updates):
            cases = [generate(args.n, rng) for _ in range(args.batch)]
            states = [state_from_experience(model, inst, rng) for inst in cases]
            record = improve(model, optimizer, states, samples=args.samples, eta=args.eta,
                             oracle_metrics=args.diagnostic)
            cumulative += record["transitions_total"]
            record.update(iteration=i + 1, transitions_cumulative=cumulative,
                          elapsed_seconds=time.perf_counter() - start)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    torch.save({"model": model.state_dict(), "architecture_version": 3 if model.glimpse else 2, "architecture":
                {"width": model.width, "layers": len(model.encoder),
                 "distance_bias": model.distance_bias,
                 "logit_clip": model.logit_clip, "glimpse": model.glimpse}}, out / "last.pt")


if __name__ == "__main__":
    main()
