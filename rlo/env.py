"""CVRP environment: one mathematical transition model for evaluation and RL.

Every step made by a policy has >=2 legal choices. Moves with only one
option are automatically applied, but their travel cost is still counted.
"""

from dataclasses import dataclass
import numpy as np
from .constraints import Capacity
from .problem import distances, validate


@dataclass(frozen=True)
class State:
    visited: tuple  # depot is always marked visited
    current: int
    load: int


class CVRPEnv:
    def __init__(self, instance, first_action=None):
        self.instance = instance
        self.dist = distances(instance)
        self.capacity = Capacity(instance)
        self.visited = np.zeros(len(instance["demands"]), dtype=bool)
        self.visited[0] = True
        self.current = 0
        self.load = 0
        self.cost = 0.0
        self.tour = [0]
        self.done = False
        self.infeasible = False
        self.decisions = 0
        if first_action is not None:
            self.move(first_action)
        self._forced()

    @property
    def state(self):
        return State(tuple(bool(x) for x in self.visited), self.current, int(self.load))

    def legal(self):
        mask = self.capacity.legal(self.visited, self.load)
        mask[0] = self.current != 0  # no empty depot routes
        return mask

    def move(self, action):
        action = int(action)
        mask = self.legal()
        if self.done or action < 0 or action >= len(mask) or not mask[action]:
            raise ValueError(f"illegal CVRP action: {action}")
        old_cost = self.cost
        self.cost += float(self.dist[self.current, action])
        self.load = self.capacity.update(self.load, action)
        self.current = action
        self.tour.append(action)
        if action != 0:
            self.visited[action] = True
        self.done = bool(self.visited.all() and action == 0)
        return old_cost - self.cost  # reward: negative distance

    def _forced(self):
        while not self.done:
            options = np.flatnonzero(self.legal())
            if len(options) != 1:
                if len(options) == 0:
                    self.infeasible = True
                    self.done = True
                return
            self.move(int(options[0]))

    def step(self, action):
        if self.done or self.legal().sum() < 2:
            raise ValueError("step requires a nonterminal state with >=2 choices")
        before = self.cost
        self.move(action)
        self.decisions += 1
        self._forced()
        return -(self.cost - before), self.done

    def observe(self):
        nodes = len(self.visited)
        obs = np.zeros((nodes, 6), dtype=np.float32)
        obs[:, :2] = self.instance["coords"]
        obs[:, 2] = self.instance["demands"] / self.capacity.limit
        obs[0, 3] = 1
        obs[:, 4] = self.visited
        obs[self.current, 5] = 1
        global_features = np.array([*self.capacity.features(self.load), 1.0 / (nodes - 1)], dtype=np.float32)
        return obs, global_features, self.dist.astype(np.float32), self.legal()

    def finish(self):
        if not self.done or self.infeasible:
            raise ValueError("episode not complete and feasible")
        if abs(validate(self.instance, self.tour) - self.cost) > 1e-8:
            raise AssertionError("environment and validator disagree")
        return self.tour


class BatchCVRP:
    """One size per batch; completed members simply become inactive."""

    def __init__(self, instances, first_actions=None):
        if not instances or len({len(x["demands"]) for x in instances}) != 1:
            raise ValueError("batch requires nonempty instances of equal size")
        if first_actions is not None and len(first_actions) != len(instances):
            raise ValueError("first-action count mismatch")
        self.envs = [CVRPEnv(inst, None if first_actions is None else first_actions[i])
                     for i, inst in enumerate(instances)]

    def active(self):
        return [i for i, e in enumerate(self.envs) if not e.done]

    def observe(self, indices=None):
        ids = self.active() if indices is None else indices
        values = [self.envs[i].observe() for i in ids]
        return tuple(np.stack([v[j] for v in values]) for j in range(4))

    def step(self, indices, actions):
        return [self.envs[i].step(a) for i, a in zip(indices, actions)]
