"""Single-player PUCT for the verified masked CVRP. Only self-generated targets.

Values are undiscounted *negative remaining distance*. Each decision may hide
several forced physical moves; these are included in edge reward and accounting.
The engine expands leaves in one batched GPU forward pass per simulation index.
It checkpoints only at collection boundaries, so trees never have to be pickled.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import torch
from .wave2_replay import Snapshot


@dataclass
class SearchStats:
    edges: int = 0               # physical moves simulated, including forced moves
    traversals: int = 0          # previously expanded tree edges traversed
    expansions: int = 0
    model_calls: int = 0         # batches passed to network
    network_states: int = 0      # rows evaluated by network
    rollout_moves: int = 0

    def add(self, other):
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass
class Node:
    snap: object = None
    terminal: bool = False
    expanded: bool = False
    prior: dict = field(default_factory=dict)
    visits: dict = field(default_factory=dict)
    reward_sum: dict = field(default_factory=dict)
    children: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)   # (reward, moves)
    n: int = 0


def _model_batch(model, snapshots, cache=None):
    """Batched masked priors and remaining-reward values, with static caching.

    The cache is valid only while the network weights are frozen. Training
    drivers discard it before every optimizer update; a different policy must
    never reuse these tensors.
    """
    device = next(model.parameters()).device
    obs = [s.restore().observe() for s in snapshots]
    nodes, glob, dist, mask = [torch.as_tensor(np.stack([o[j] for o in obs]), device=device)
                               for j in range(4)]
    with torch.no_grad():
        if cache is None:
            logits, val = model(nodes, glob, dist, mask)
        else:
            indices = [i for i,s in enumerate(snapshots) if id(s.instance) not in cache]
            # Deduplicate repeated snapshots of the same instance within this batch.
            missing = []
            seen = set()
            for i in indices:
                key = id(snapshots[i].instance)
                if key not in seen:missing.append(i);seen.add(key)
            if missing:
                encoded = model.encode(nodes[missing], dist[missing])
                for j, i in enumerate(missing):
                    cache[id(snapshots[i].instance)] = encoded[j].detach()
            static = torch.stack([cache[id(s.instance)] for s in snapshots])
            logits, val = model.decode(static, nodes, glob, mask)
        probs = logits.softmax(-1).cpu().numpy()
        values = val.cpu().numpy()
    return probs, values


def _select(node, c_puct):
    visited = [node.reward_sum[a] / node.visits[a]
               for a in node.prior if node.visits[a] > 0]
    low, high = (min(visited), max(visited)) if visited else (0., 0.)
    scale = max(high - low, 1.e-8)
    parent = math.sqrt(max(1, node.n))
    candidates = []
    for a, p in node.prior.items():
        count = node.visits[a]
        q = ((node.reward_sum[a] / count - low) / scale) if count else 0.5
        score = q + c_puct * p * parent / (1 + count)
        candidates.append((score, p, -a, a))
    return max(candidates)[-1]


def _expand_child(node, action, stats):
    if action in node.children:
        r, _ = node.edges[action]
        return node.children[action], r
    env = node.snap.restore()
    old_length = len(env.tour)
    reward, done = env.step(action)
    moves = len(env.tour) - old_length
    assert moves >= 1
    stats.edges += moves
    child = Node(terminal=done, snap=None if done else Snapshot.take(env))
    node.children[action] = child
    node.edges[action] = (float(reward), moves)
    return child, float(reward)


def _rollout(model, snap, rng, stats, cache):
    """One exact-simulator completion sampled under the current *frozen* actor."""
    env = snap.restore()
    total = 0.
    while not env.done:
        probs, _ = _model_batch(model, [Snapshot.take(env)], cache)
        stats.model_calls += 1
        stats.network_states += 1
        mask = env.legal()
        row = probs[0] * mask
        row = row / row.sum()
        action = int(rng.choice(len(row), p=row))
        initial = len(env.tour)
        r, _ = env.step(action)
        moves = len(env.tour) - initial
        stats.rollout_moves += moves
        total += float(r)
    return total


class Tree:
    def __init__(self, env, rng, *, c_puct=1.25, root_noise=False,
                 noise_alpha=.3, noise_weight=.25, encoder_cache=None):
        self.root = Node(snap=Snapshot.take(env))
        self.rng = rng
        self.c_puct = float(c_puct)
        self.root_noise = bool(root_noise)
        self.alpha = float(noise_alpha)
        self.weight = float(noise_weight)
        self.stats = SearchStats()
        self.encoder_cache = encoder_cache if encoder_cache is not None else {}

    def advance(self, action, env):
        """Reuse subtree for the chosen real action without treating it as simulation."""
        child = self.root.children.get(int(action))
        if child is not None and not env.done:
            if child.snap is None or child.snap.current != env.current or (
                    child.snap.load != env.load or not np.array_equal(child.snap.visited, env.visited)):
                raise AssertionError("tree and CVRP environment disagree")
            self.root = child
        elif not env.done:
            self.root = Node(snap=Snapshot.take(env))
        else:
            self.root = Node(terminal=True)
        self.root_noise = False    # training driver re-enables it per new decision


def _expand_batch(model, work, *, leaf_type, stats):
    if not work:
        return
    snaps = [node.snap for _, node, _ in work]
    probs, values = _model_batch(model, snaps, work[0][0].encoder_cache)
    stats.model_calls += 1
    stats.network_states += len(snaps)
    for (tree, node, path), row, v in zip(work, probs, values):
        legal = node.snap.restore().legal()
        node.prior = {int(a): float(row[a]) for a in np.flatnonzero(legal)}
        s = sum(node.prior.values())
        if not (s > 0 and np.isfinite(s)):
            raise FloatingPointError("invalid legal policy prior")
        node.prior = {a: p / s for a, p in node.prior.items()}
        if tree.root is node and tree.root_noise:
            noise = tree.rng.dirichlet(np.full(len(node.prior), tree.alpha))
            for (a, p), q in zip(list(node.prior.items()), noise):
                node.prior[a] = (1 - tree.weight) * p + tree.weight * q
            tree.root_noise = False
        node.visits = {a: 0 for a in node.prior}
        node.reward_sum = {a: 0. for a in node.prior}
        node.expanded = True
        stats.expansions += 1
        if leaf_type == "value":
            leaf = float(v)
        elif leaf_type == "rollout":
            leaf = _rollout(model, node.snap, tree.rng, stats, tree.encoder_cache)
        else:
            raise ValueError("leaf_type must be 'value' or 'rollout'")
        _backup(path, leaf)


def _backup(path, leaf):
    ret = float(leaf)
    for node, action, reward in reversed(path):
        ret += reward
        node.n += 1
        node.visits[action] += 1
        node.reward_sum[action] += ret


def search_many(model, trees, simulations=32, *, leaf_type="value"):
    """Search B trees. Each iteration batches all new leaf network evaluations.

    Visit-count targets sum to `simulations`. The first root expansion does not
    count as a simulation. Terminal nodes back up zero future return.
    """
    if simulations < 1 or not trees or leaf_type not in ("value", "rollout"):
        raise ValueError("invalid PUCT inputs")
    stats = SearchStats()
    initial = [(t, t.root, []) for t in trees if not t.root.expanded and not t.root.terminal]
    _expand_batch(model, initial, leaf_type=leaf_type, stats=stats)
    # A reused subtree can have preexisting visits. Targets describe only the
    # incremental search effort at this decision; old counts are priors only.
    before_counts = [{a: t.root.visits.get(a, 0) for a in t.root.prior}
                     for t in trees]
    for tree in trees:
        node = tree.root
        if tree.root_noise and node.expanded and not node.terminal:
            noise = tree.rng.dirichlet(np.full(len(node.prior), tree.alpha))
            for (a, p), q in zip(list(node.prior.items()), noise):
                node.prior[a] = (1-tree.weight)*p + tree.weight*q
            tree.root_noise = False
    for _ in range(simulations):
        leaves = []
        for tree in trees:
            node = tree.root
            path = []
            while node.expanded and not node.terminal:
                action = _select(node, tree.c_puct)
                exists = action in node.children
                child, r = _expand_child(node, action, stats)
                if exists:
                    # Replaying a known tree edge costs no new neural inference,
                    # but must still be charged as a simulated physical move.
                    stats.edges += node.edges[action][1]
                    stats.traversals += node.edges[action][1]
                path.append((node, action, r))
                node = child
            if node.terminal:
                _backup(path, 0.)
            else:
                leaves.append((tree, node, path))
        _expand_batch(model, leaves, leaf_type=leaf_type, stats=stats)
    targets = []
    for tree, previous in zip(trees, before_counts):
        root = tree.root
        counts = np.zeros(len(root.snap.visited), dtype=np.float64)
        for action, count in root.visits.items():
            counts[action] = count - previous.get(action, 0)
        if int(counts.sum()) != simulations:
            raise AssertionError("PUCT root visit accounting is incorrect")
        targets.append(counts / simulations)
    return targets, stats


def supported_behavior(visits, prior, epsilon=.05):
    """Search behavior has full support on legal target-policy actions."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    x, y = np.asarray(visits, float), np.asarray(prior, float)
    out = (1 - epsilon) * x + epsilon * y
    if not np.isclose(out.sum(), 1, atol=1e-6):
        raise AssertionError("search distribution not normalized")
    if np.any((y > 0) & (out <= 0)):
        raise AssertionError("behavior misses legal target-policy support")
    return out
