"""Experience-only trajectories, auxiliary targets, and policy/value updates.

All action sampling uses the exact masked CVRPEnv; forced actions are charged
to physical transitions and their rewards are retained at the decision step.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import CVRPEnv
from .wave2_replay import Snapshot, observations


def environments(cases, starts=1, rng=None):
    if starts == 1:
        return [CVRPEnv(x) for x in cases]
    n = len(cases[0]["demands"]) - 1
    if not 2 <= starts <= n:
        raise ValueError("POMO starts must lie in [2,n]")
    rng = np.random.default_rng() if rng is None else rng
    return [CVRPEnv(x, int(a)) for x in cases
            for a in rng.choice(np.arange(1, n + 1), starts, replace=False)]


def trajectories(model, envs, *, grad=False, temperature=1., greedy=False,
                 record=True):
    """Roll out equal-sized CVRP states; supports resumed states and forced starts.

    Returns trajectories, final environments, physical moves since the supplied
    environments' initial state. Stored step paths permit objective-related
    future-depot auxiliary targets without solver labels.
    """
    if not envs or temperature <= 0 or len({len(e.visited) for e in envs}) != 1:
        raise ValueError("nonempty homogeneous environments and positive temperature required")
    dev = next(model.parameters()).device
    initial = [len(e.tour) for e in envs]
    traces = [[] for _ in envs]
    # Static encodings are shared by all states drawn from the same instance,
    # including the n forced-start POMO copies.
    unique, reverse, lookup = [], [], {}
    for i, e in enumerate(envs):
        key = id(e.instance)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(i)
        reverse.append(lookup[key])
    model.train(grad)
    with torch.enable_grad() if grad else torch.no_grad():
        static_obs = [envs[j].observe() for j in unique]
        node = torch.as_tensor(np.stack([v[0] for v in static_obs]), device=dev)
        dist = torch.as_tensor(np.stack([v[2] for v in static_obs]), device=dev)
        encoded = model.encode(node, dist)[torch.as_tensor(reverse, device=dev)]
        while True:
            active = [i for i, e in enumerate(envs) if not e.done]
            if not active:
                break
            obs = [envs[i].observe() for i in active]
            nodes = torch.as_tensor(np.stack([x[0] for x in obs]), device=dev)
            glob = torch.as_tensor(np.stack([x[1] for x in obs]), device=dev)
            mask = torch.as_tensor(np.stack([x[3] for x in obs]), device=dev)
            static = encoded[active]
            scores, values = model.decode(static, nodes, glob, mask)
            distribution = Categorical(logits=scores / temperature)
            actions = scores.argmax(-1) if greedy else distribution.sample()
            logp = distribution.log_prob(actions)
            entropy = distribution.entropy()
            aux = model.predict_aux(static, nodes, glob) if (record and model.auxiliary_enabled) else None
            for j, idx in enumerate(active):
                e = envs[idx]
                snap = Snapshot.take(e) if record else None
                before_len = len(e.tour)
                before_position = int(e.current)
                reward, _ = e.step(int(actions[j].item()))
                extension = tuple(int(x) for x in e.tour[before_len:])
                if record:
                    traces[idx].append(dict(
                        before=snap, action=int(actions[j].item()), reward=float(reward),
                        logp=logp[j], value=values[j], entropy=entropy[j],
                        aux=None if aux is None else aux[j],
                        path=extension, first_position=before_position))
        for e in envs:
            if not e.done or e.infeasible:
                raise AssertionError("uncompleted/infeasible policy rollout")
            # Complete ordinary episodes (including POMO forced starts) must
            # agree with independent tour-cost verification.
            if e.tour[0] == 0:
                e.finish()
    moves = sum(len(e.tour) - initial[i] for i, e in enumerate(envs))
    return traces, envs, moves


def future_targets(trace, *, n):
    """Per-decision total future distance, future depot returns, and distance
    until next depot. Derived entirely from the sampled environment trajectory.
    """
    out = []
    travel = 0.
    depot_returns = 0
    segments = []
    for t in trace:
        pos = t["first_position"]
        travel_in_step = []
        for nxt in t["path"]:
            coords = t["before"].instance["coords"]
            step_cost = float(np.linalg.norm(coords[pos] - coords[nxt]))
            travel_in_step.append((step_cost, nxt == 0))
            pos = nxt
        segments.append(travel_in_step)
    for i, t in enumerate(trace):
        remaining = [part for segment in segments[i:] for part in segment]
        travel = sum(x[0] for x in remaining)
        depot_returns = sum(int(x[1]) for x in remaining)
        to_close = 0.
        for cost, is_depot in remaining:
            to_close += cost
            if is_depot:
                break
        scale = math.sqrt(n)
        out.append([travel / scale, depot_returns / n, to_close / scale])
    return out


def advantages(trace, lam=.95):
    """Undiscounted GAE at decision steps; forced moves are inside r_t."""
    a = 0.
    result = []
    for j in reversed(range(len(trace))):
        cur = trace[j]
        v = float(cur["value"].detach())
        next_v = float(trace[j + 1]["value"].detach()) if j + 1 < len(trace) else 0.
        delta = cur["reward"] + next_v - v
        a = delta + lam * a
        result.append(a)
    return list(reversed(result))


def _normalize(x):
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6) if x.numel() > 1 else x


def policy_objective(model, traces, envs, method, lam=.95, aux_weight=0.,
                     entropy_weight=0., starts=1, normalize_adv=True):
    """REINFORCE/POMO or GAE actor-critic objective from true trajectories."""
    dev = next(model.parameters()).device
    logps = [torch.stack([s["logp"] for s in seq]) for seq in traces if seq]
    if not logps:
        raise AssertionError("no policy decisions")
    entropy = torch.stack([s["entropy"] for seq in traces for s in seq]).mean()
    if method in ("reinforce", "pomo"):
        costs = np.array([e.cost for e in envs], dtype=np.float64)
        if method == "pomo":
            if len(envs) % starts:
                raise ValueError("POMO groups do not match starts")
            baseline = np.repeat(costs.reshape(-1, starts).mean(-1), starts)
        else:
            baseline = np.full(len(envs), costs.mean())
        actor = torch.stack([-(lp.sum() * torch.tensor(
            -(costs[i] - baseline[i]), device=dev, dtype=lp.dtype))
                             for i, lp in enumerate([torch.stack([s["logp"] for s in seq])
                                                       if seq else None for seq in traces])
                             if lp is not None]).mean()
        critic = torch.zeros((), device=dev)
    else:
        flat = [(s, a) for seq in traces for s, a in zip(seq, advantages(seq, lam))]
        ads = torch.tensor([a for _, a in flat], dtype=torch.float32, device=dev)
        ads = _normalize(ads) if normalize_adv else ads
        logp = torch.stack([s["logp"] for s, _ in flat])
        value = torch.stack([s["value"] for s, _ in flat])
        targets = torch.tensor([s["reward"] for s, _ in flat], dtype=torch.float32, device=dev)
        # For GAE, the critic is trained to its fixed lambda-return.
        old_v = value.detach()
        gains = torch.tensor([a for _, a in flat], dtype=torch.float32, device=dev)
        actor = -(logp * ads.detach()).mean()
        critic = .5 * F.mse_loss(value, (old_v + gains).detach())
    aux_loss = torch.zeros((), device=dev)
    if model.auxiliary_enabled:
        if aux_weight <= 0:
            raise ValueError("auxiliary model requires positive aux_weight")
        pred = torch.stack([s["aux"] for seq in traces for s in seq])
        targets = torch.tensor([row for seq in traces for row in
                                future_targets(seq, n=len(seq[0]["before"].visited)-1)],
                               device=dev, dtype=pred.dtype)
        aux_loss = F.smooth_l1_loss(pred, targets)
    return actor + critic + aux_weight * aux_loss - entropy_weight * entropy, dict(
        actor=float(actor.detach()), critic=float(critic.detach()),
        auxiliary=float(aux_loss.detach()), entropy=float(entropy.detach()))


def ppo_update(model, optimizer, traces, *, epochs=4, minibatch=64, temperature=1.,
               clip=.2, lam=.95, value_weight=.5, entropy_weight=.01, aux_weight=0.):
    """Clipped PPO and GAE. Old action probabilities/advantages remain frozen."""
    rows = [(s, float(a)) for seq in traces for s, a in zip(seq, advantages(seq, lam))]
    if len(rows) < 2:
        raise ValueError("PPO requires at least two decisions")
    device = next(model.parameters()).device
    old_logp = torch.stack([s["logp"].detach() for s, _ in rows])
    old_value = torch.stack([s["value"].detach() for s, _ in rows])
    advantage = torch.tensor([a for _, a in rows], device=device)
    returns = old_value + advantage
    advantage = _normalize(advantage)
    extra = None
    if model.auxiliary_enabled:
        # Auxiliary truth may be computed from the same actual trajectory.
        lookup = {id(s): lab for seq in traces for s, lab in
                  zip(seq, future_targets(seq, n=len(seq[0]["before"].visited)-1))}
        extra = torch.tensor([lookup[id(s)] for s, _ in rows], device=device)
    stats, updates = [], 0
    for _ in range(epochs):
        permutation = torch.randperm(len(rows), device=device)
        for batch in permutation.split(minibatch):
            indices = batch.cpu().tolist()
            snapshots = [rows[i][0]["before"] for i in indices]
            sample = observations(snapshots)
            node, glob, dist, mask = [torch.as_tensor(x, device=device) for x in sample]
            enc = model.encode(node, dist)
            logits, value = model.decode(enc, node, glob, mask)
            distribution = Categorical(logits=logits / temperature)
            act = torch.tensor([rows[i][0]["action"] for i in indices], device=device)
            lp = distribution.log_prob(act)
            ratio = (lp - old_logp[batch]).exp()
            obj = torch.minimum(ratio * advantage[batch],
                                ratio.clamp(1 - clip, 1 + clip) * advantage[batch])
            v_loss = .5 * F.mse_loss(value, returns[batch])
            aux = torch.zeros((), device=device)
            if extra is not None:
                aux = F.smooth_l1_loss(model.predict_aux(enc, node, glob), extra[batch])
            loss = -obj.mean() + value_weight * v_loss \
                   - entropy_weight * distribution.entropy().mean() + aux_weight * aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise FloatingPointError("nonfinite PPO update")
            optimizer.step()
            updates += 1
            stats.append(float(loss.detach()))
    return dict(loss=float(np.mean(stats)), optimizer_updates=updates,
                replay_decisions=len(rows) * epochs)


def critic_only_update(model, optimizer, traces):
    """Synthetic Monte-Carlo critic fit: only the value head's weights change."""
    # Use explicit reverse sum; no solver-generated target.
    rows = []
    for seq in traces:
        total = 0.
        for s in reversed(seq):
            total += s["reward"]
            rows.append((s, total))
    if not rows:
        return dict(loss=0., optimizer_updates=0)
    predictions = torch.stack([s["value"] for s, _ in rows])
    targets = torch.as_tensor([t for _, t in rows], device=predictions.device, dtype=predictions.dtype)
    loss = .5 * F.mse_loss(predictions, targets)
    model.zero_grad(set_to_none=True)
    # Restrict the optimizer to model.value, not shared embeddings or actor.
    gradients = torch.autograd.grad(loss, tuple(model.value.parameters()))
    for p, g in zip(model.value.parameters(), gradients):
        p.grad = g
    grad = torch.nn.utils.clip_grad_norm_(model.value.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(grad):
        raise FloatingPointError("nonfinite synthetic critic step")
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return dict(loss=float(loss.detach()), optimizer_updates=1)
