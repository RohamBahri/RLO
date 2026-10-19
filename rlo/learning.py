"""One cached-encoder rollout engine and four on-policy RL objectives."""

import numpy as np
import torch
from .env import BatchCVRP


def device_auto():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def rollout(model, instances, *, greedy=False, temperature=1.0, first_actions=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = BatchCVRP(instances, first_actions=first_actions)
    traces = [dict(logp=[], value=[], reward=[], entropy=[]) for _ in instances]
    device = next(model.parameters()).device
    # Static embeddings are shared by every decision in an episode. Autograd
    # correctly accumulates gradients from all decoder steps into this graph.
    # POMO repeats each instance for its n forced starts. Encode each unique
    # instance only once, then share its differentiable embeddings across starts.
    unique, replica, lookup = [], [], {}
    for i, instance in enumerate(instances):
        if id(instance) not in lookup:
            lookup[id(instance)] = len(unique)
            unique.append(i)
        replica.append(lookup[id(instance)])
    all_nodes, _, all_dist, _ = batch.observe(unique)
    encoded = model.encode(torch.as_tensor(all_nodes, device=device),
                           torch.as_tensor(all_dist, device=device))[replica]
    while indices := batch.active():
        nodes, glob, _, mask = batch.observe(indices)
        logits, values = model.decode(encoded[indices], torch.as_tensor(nodes, device=device),
                                      torch.as_tensor(glob, device=device),
                                      torch.as_tensor(mask, device=device))
        distribution = torch.distributions.Categorical(logits=logits / temperature)
        actions = logits.argmax(-1) if greedy else distribution.sample()
        logp, entropy = distribution.log_prob(actions), distribution.entropy()
        result = batch.step(indices, actions.detach().cpu().tolist())
        for k, i in enumerate(indices):
            traces[i]["logp"].append(logp[k])
            traces[i]["value"].append(values[k])
            traces[i]["entropy"].append(entropy[k])
            traces[i]["reward"].append(result[k][0])
    return batch.envs, traces


def loss(method, traces, envs, *, baseline=None, lam=0.9, value_weight=0.5, starts=1):
    device = next((x.device for tr in traces for x in tr["logp"]), torch.device("cpu"))
    total = torch.zeros((), device=device)
    costs = np.array([e.cost for e in envs])
    if method == "reinforce":
        targets = -costs
        baseline = float(targets.mean()) if baseline is None else baseline
    elif method == "rollout":
        if baseline is None or len(baseline) != len(traces):
            raise ValueError("frozen rollout baseline requires one greedy value per instance")
        targets = -costs
    elif method == "pomo":
        if len(envs) % starts or starts < 2:
            raise ValueError("POMO requires >=2 starts per instance")
        targets = -costs
        baseline = np.repeat(targets.reshape(-1, starts).mean(axis=1), starts)
    for i, tr in enumerate(traces):
        if not tr["logp"]:
            continue
        logp = torch.stack(tr["logp"])
        if method == "a2c":
            values = torch.stack(tr["value"])
            returns = []
            g = torch.zeros((), device=device)
            for t in reversed(range(len(logp))):
                next_v = values[t + 1].detach() if t + 1 < len(logp) else 0
                g = tr["reward"][t] + (1 - lam) * next_v + lam * g
                returns.append(g)
            target = torch.stack(returns[::-1]).detach()
            total = total - (logp * (target - values.detach())).sum()
            total = total + value_weight * (values - target).square().sum()
        else:
            base = baseline[i] if method in ("rollout", "pomo") else baseline
            total = total - logp.sum() * (targets[i] - base)
    return total / len(traces)


def update(model, optimizer, instances, method, *, temperature=1.0, lam=0.9,
           baseline=None, baseline_policy=None, starts=None, rng=None, entropy_weight=0.0):
    model.train()
    first = None
    if method == "pomo":
        rng = np.random.default_rng() if rng is None else rng
        n = len(instances[0]["demands"]) - 1
        starts = n if starts is None else min(starts, n)
        if starts < 2:
            raise ValueError("POMO requires at least two customers/starts")
        first = [int(a) for inst in instances for a in
                 rng.choice(np.arange(1, n + 1), size=starts, replace=False)]
        instances = [inst for inst in instances for _ in range(starts)]
    baseline_envs = []
    if method == "rollout":
        if baseline_policy is None:
            raise ValueError("rollout requires an independently frozen baseline policy")
        baseline_policy.eval()
        with torch.no_grad():
            baseline_envs, _ = rollout(baseline_policy, instances, greedy=True)
        baseline = [-e.cost for e in baseline_envs]
    envs, traces = rollout(model, instances, first_actions=first, temperature=temperature)
    objective = loss(method, traces, envs, baseline=baseline, lam=lam, starts=starts or 1)
    if entropy_weight:
        entropy = [x for tr in traces for x in tr["entropy"]]
        if entropy:
            objective = objective - entropy_weight * torch.stack(entropy).mean()
    optimizer.zero_grad()
    objective.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    baseline_steps = sum(len(e.tour) - 1 for e in baseline_envs)
    return dict(loss=float(objective.detach()), gradient=float(grad_norm),
                mean_cost=float(np.mean([e.cost for e in envs])),
                feasibility_rate=float(np.mean([e.done and not e.infeasible for e in envs])),
                transitions=sum(len(e.tour) - 1 for e in envs) + baseline_steps,
                baseline_transitions=baseline_steps,
                decisions=sum(e.decisions for e in envs),
                episodes=len(envs) + len(baseline_envs),
                baseline_episodes=len(baseline_envs),
                starts=starts if method == "pomo" else 1)
