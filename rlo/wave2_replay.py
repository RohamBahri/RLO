"""Shared, experience-only replay and counterfactual CVRP transition engine.

A decision is an action selected where >=2 actions are legal. CVRPEnv.step()
automatically follows forced moves. One replay tuple therefore represents one
decision and its entire *undiscounted* accumulated travel reward, while
physical_moves separately counts all underlying depot/customer moves.

Dyna uses the known, deterministic CVRP transition law, not an expert tour,
a fitted dynamics network, or a different reward/environment.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch

from .env import BatchCVRP, CVRPEnv


@dataclass
class Snapshot:
    """Enough Markov state to reproduce an action transition exactly."""
    instance: dict
    visited: np.ndarray
    current: int
    load: float

    @classmethod
    def take(cls, env):
        if env.done or int(env.legal().sum()) < 2:
            raise ValueError("only policy-decision states can be snapshotted")
        return cls(env.instance, env.visited.copy(), int(env.current), float(env.load))

    def restore(self):
        env = CVRPEnv(self.instance)
        env.visited = self.visited.copy()
        env.current = int(self.current)
        env.load = float(self.load)
        env.cost = 0.0
        env.tour = [int(self.current)]
        env.done = False
        env.infeasible = False
        env.decisions = 0
        if int(env.legal().sum()) < 2:
            raise ValueError("restored state does not have at least two legal actions")
        return env


@dataclass
class Transition:
    before: Snapshot
    action: int
    reward: float
    after: Snapshot | None
    physical_moves: int
    synthetic: bool = False

    @property
    def n(self):
        return len(self.before.instance["demands"]) - 1


class Replay:
    """Separate real/synthetic circular queues, grouped by problem size."""

    def __init__(self, capacity_per_size=4000):
        if capacity_per_size < 1:
            raise ValueError("capacity_per_size must be positive")
        self.capacity_per_size = int(capacity_per_size)
        self.real = defaultdict(lambda: deque(maxlen=self.capacity_per_size))
        self.synthetic = defaultdict(lambda: deque(maxlen=self.capacity_per_size))

    def add(self, transition):
        if transition.physical_moves < 1 or transition.reward > 1e-10:
            raise ValueError("invalid CVRP travel reward or transition length")
        store = self.synthetic if transition.synthetic else self.real
        store[transition.n].append(transition)

    def real_count(self):
        return sum(map(len, self.real.values()))

    def total_count(self):
        return self.real_count() + sum(map(len, self.synthetic.values()))

    def real_sample(self, rng):
        sizes = [n for n, seq in self.real.items() if seq]
        if not sizes:
            raise ValueError("cannot sample from empty real replay")
        n = int(rng.choice(sizes))
        seq = self.real[n]
        return seq[int(rng.integers(len(seq)))]

    def sample(self, rng, batch_size, synthetic_fraction=0.5):
        if batch_size < 1:
            raise ValueError("sample batch must be positive")
        sizes = sorted(n for n in set(self.real) | set(self.synthetic)
                       if self.real[n] or self.synthetic[n])
        if not sizes:
            raise ValueError("replay is empty")
        n = int(rng.choice(sizes))
        items = []
        for _ in range(batch_size):
            use_synthetic = bool(self.synthetic[n]) and (
                not self.real[n] or rng.random() < synthetic_fraction)
            source = self.synthetic[n] if use_synthetic else self.real[n]
            items.append(source[int(rng.integers(len(source)))])
        return items


def simulate(snapshot, action, synthetic=False):
    """Independent one-step counterfactual with forced-action costs included."""
    env = snapshot.restore()
    if not bool(env.legal()[int(action)]):
        raise ValueError("counterfactual action is illegal")
    moves_before = len(env.tour)
    reward, done = env.step(int(action))
    moves = len(env.tour) - moves_before
    return Transition(snapshot, int(action), float(reward),
                      None if done else Snapshot.take(env),
                      moves, synthetic=synthetic)


def dyna_from_real(replay, rng, count):
    """Explore alternative legal actions from *real* previously visited states.

    Excludes the behavior action when possible; does not recursively sample
    synthetic states. Reports simulator moves separately from fresh experience.
    """
    transitions = []
    if count < 0:
        raise ValueError("count must be nonnegative")
    for _ in range(count):
        observed = replay.real_sample(rng)
        env = observed.before.restore()
        alternatives = np.flatnonzero(env.legal())
        alternatives = alternatives[alternatives != observed.action]
        if len(alternatives) == 0:
            raise AssertionError("every policy state must have >=2 options")
        action = int(rng.choice(alternatives))
        simulated = simulate(observed.before, action, synthetic=True)
        replay.add(simulated)
        transitions.append(simulated)
    return transitions


def collect_real(model, instances, replay, rng, epsilon=0.1):
    """Batched epsilon-greedy self-experience with one static encoding per case."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    batch = BatchCVRP(instances)
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    collected = []
    with torch.no_grad():
        # All instances have the same n; cache node embeddings for full episodes.
        nodes, _, distances, _ = batch.observe()
        static = model.encode(torch.as_tensor(nodes, device=device),
                              torch.as_tensor(distances, device=device))
        while indices := batch.active():
            nodes, glob, _, mask = batch.observe(indices)
            q, _ = model.decode(
                static[indices],
                torch.as_tensor(nodes, device=device),
                torch.as_tensor(glob, device=device),
                torch.as_tensor(mask, device=device))
            greedy = q.argmax(dim=-1).cpu().numpy()
            for j, idx in enumerate(indices):
                env = batch.envs[idx]
                state = Snapshot.take(env)
                options = np.flatnonzero(mask[j])
                action = int(rng.choice(options)) if rng.random() < epsilon else int(greedy[j])
                before_moves = len(env.tour)
                reward, done = env.step(action)
                t = Transition(state, action, float(reward),
                               None if done else Snapshot.take(env),
                               len(env.tour) - before_moves, synthetic=False)
                replay.add(t)
                collected.append(t)
    model.train(previous_mode)
    for env in batch.envs:
        if not env.done or env.infeasible:
            raise AssertionError("infeasible/unfinished experience")
        env.finish()  # independently checks cumulative travel cost and constraints
    return collected, batch.envs


def observations(snapshots):
    """Build a homogeneous mini-batch of observations from replay snapshots."""
    if not snapshots:
        raise ValueError("empty observation batch")
    if len({len(s.instance["demands"]) for s in snapshots}) != 1:
        raise ValueError("mixed-size replay mini-batch")
    rows = [s.restore().observe() for s in snapshots]
    return tuple(np.stack([r[j] for r in rows]) for j in range(4))


def double_dqn_targets(online_next, target_next, next_masks, rewards, dones):
    """Undiscounted Double-DQN, masked online argmax and target evaluation.

    No value bootstrap at terminal states. DQN is defined at *policy decisions*,
    whose rewards include intervening automatic CVRP moves.
    """
    if online_next.shape != target_next.shape or online_next.shape != next_masks.shape:
        raise ValueError("Q/legality shape mismatch")
    actions = online_next.masked_fill(~next_masks, -torch.inf).argmax(-1)
    boot = target_next.gather(1, actions[:, None]).squeeze(1)
    boot = torch.where(dones, torch.zeros_like(boot), boot)
    return rewards + boot


def replay_q_update(model, target, optimizer, transitions, device):
    """One batched Bellman update. No solver labels or heuristic targets."""
    import torch.nn.functional as F
    if not transitions:
        raise ValueError("empty Bellman mini-batch")
    before = observations([t.before for t in transitions])
    data = [torch.as_tensor(x, device=device) for x in before]
    actions = torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=device)
    rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
    present = [i for i, t in enumerate(transitions) if t.after is not None]
    q, _ = model(*data)
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    expected = rewards.clone()
    if present:
        next_data = [torch.as_tensor(x, device=device)
                     for x in observations([transitions[i].after for i in present])]
        with torch.no_grad():
            online_next, _ = model(*next_data)
            target_next, _ = target(*next_data)
            next_masks = next_data[-1]
            gains = double_dqn_targets(
                online_next, target_next, next_masks,
                rewards[present],
                torch.zeros(len(present), dtype=torch.bool, device=device))
            expected[present] = gains
    loss = F.smooth_l1_loss(chosen, expected.detach())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if not torch.isfinite(loss) or not torch.isfinite(grad):
        raise FloatingPointError("nonfinite Bellman optimization")
    return {"loss": float(loss.detach()), "gradient": float(grad),
            "mean_q": float(chosen.detach().mean()),
            "mean_target": float(expected.detach().mean())}
