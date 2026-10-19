"""Exact CVRP-10 residual cost-to-go from (visited, current, load).

No discounting; includes any automatically forced movement. Intended as
an independent diagnostic, never as a source of policy training targets.
"""

from functools import lru_cache
import numpy as np
from .problem import distances


class ExactOracle:
    def __init__(self, instance):
        self.d = distances(instance)
        self.q = tuple(map(int, instance["demands"]))
        self.Q = int(instance["capacity"])
        self.n = len(self.q) - 1
        if self.n > 10:
            raise ValueError("residual oracle is intended for n <= 10")
        self.full = (1 << self.n) - 1

    @lru_cache(maxsize=None)
    def value(self, remaining, current=0, load=0):
        if remaining == 0:
            return float(self.d[current, 0])
        best = float("inf")
        for j in range(1, self.n + 1):
            bit = 1 << (j - 1)
            if remaining & bit and load + self.q[j] <= self.Q:
                best = min(best, self.d[current, j] +
                           self.value(remaining ^ bit, j, load + self.q[j]))
        if current != 0:
            best = min(best, self.d[current, 0] + self.value(remaining, 0, 0))
        return best

    def remaining(self, state):
        return sum(1 << (j - 1) for j in range(1, self.n + 1)
                   if not state.visited[j])

    def cost_to_go(self, state):
        return self.value(self.remaining(state), state.current, state.load)

    def action_values(self, state):
        """Exact Q*(s,a) for legal next actions, inf for illegal ones."""
        r, c, load = self.remaining(state), state.current, state.load
        values = np.full(self.n + 1, np.inf)
        if c != 0:
            values[0] = self.d[c, 0] + self.value(r, 0, 0)
        for j in range(1, self.n + 1):
            bit = 1 << (j - 1)
            if (r & bit) and load + self.q[j] <= self.Q:
                values[j] = self.d[c, j] + self.value(r ^ bit, j, load + self.q[j])
        return values
