"""Exact subset dynamic programming for small unlimited-fleet CVRP (n <= 12)."""

import numpy as np
from .problem import distances, validate


def solve(instance):
    """Return an optimal tour under true Euclidean distances, not OR-Tools costs."""
    n = len(instance["demands"]) - 1
    if not 1 <= n <= 12:
        raise ValueError("exact subset DP supports 1 to 12 customers")
    size = 1 << n
    d = distances(instance)
    dp = np.full((size, n), np.inf)
    parent = np.full((size, n), -1, dtype=int)
    route_cost = np.full(size, np.inf)
    route_end = np.full(size, -1, dtype=int)
    load = np.zeros(size)

    # Held-Karp: optimal depot-to-last path through each customer subset.
    for mask in range(1, size):
        bit = mask & -mask
        load[mask] = load[mask ^ bit] + instance["demands"][bit.bit_length()]
        if load[mask] > instance["capacity"]:
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev = mask ^ (1 << j)
            if prev == 0:
                dp[mask, j] = d[0, j + 1]
            else:
                costs = dp[prev] + d[1:, j + 1]
                parent[mask, j] = int(np.argmin(costs))
                dp[mask, j] = costs[parent[mask, j]]
        ends = dp[mask] + d[1:, 0]
        route_end[mask] = int(np.argmin(ends))
        route_cost[mask] = ends[route_end[mask]]

    # Partition all customers into feasible routes, breaking route-order symmetry.
    best = np.full(size, np.inf)
    chosen = np.zeros(size, dtype=int)
    best[0] = 0
    for mask in range(1, size):
        pivot = mask & -mask
        subset = mask
        while subset:
            if (subset & pivot) and route_cost[subset] + best[mask ^ subset] < best[mask]:
                best[mask] = route_cost[subset] + best[mask ^ subset]
                chosen[mask] = subset
            subset = (subset - 1) & mask

    if not np.isfinite(best[-1]):
        raise ValueError("instance has no feasible solution")
    tour, remaining = [0], size - 1
    while remaining:
        subset = int(chosen[remaining])
        mask, j, route = subset, int(route_end[subset]), []
        while mask:
            route.append(j + 1)
            p = int(parent[mask, j])
            mask ^= 1 << j
            j = p
        tour.extend(reversed(route))
        tour.append(0)
        remaining ^= subset
    assert abs(validate(instance, tour) - best[-1]) < 1e-8
    return tour
