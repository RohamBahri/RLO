"""CVRP instances and independent solution validation. Depot is node 0."""

import numpy as np


def generate(n, rng, *, capacity_factor=1.0, demand_max=10):
    """Generate an unlabeled CVRP instance; defaults reproduce the 2026 benchmark.

    Capacity and demand ranges may vary during experience-based training.
    This changes problem parameters, not the type of constraints.
    """
    if n < 1 or demand_max < 1 or capacity_factor <= 0:
        raise ValueError("n, demand_max and capacity_factor must be positive")
    coords = rng.uniform(0, 1, (n + 1, 2)).astype(np.float32)
    demands = np.zeros(n + 1, dtype=np.float32)
    demands[1:] = rng.integers(1, demand_max + 1, n)
    typical_load = (demand_max + 1) / 2 * n / 3.5
    # Preserve the original generator exactly when called with default parameters.
    floor = 10 if demand_max == 10 and capacity_factor == 1.0 else int(demands.max())
    capacity = max(floor, round(typical_load * capacity_factor))
    return {"coords": coords, "demands": demands, "capacity": capacity}


def distances(instance):
    p = instance["coords"].astype(np.float64)
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)


def length(instance, tour):
    """Actual Euclidean length, with no integer-distance approximation."""
    nodes = np.asarray(tour, dtype=int)
    return float(distances(instance)[nodes[:-1], nodes[1:]].sum())


def validate(instance, tour):
    """Raise ValueError for an incomplete or capacity-infeasible route."""
    n = len(instance["demands"]) - 1
    tour = list(tour)
    if len(tour) < 3 or tour[0] != 0 or tour[-1] != 0:
        raise ValueError("tour must start and end at depot 0")
    if any(not isinstance(v, (int, np.integer)) or v < 0 or v > n for v in tour):
        raise ValueError("invalid node index")
    customers = [v for v in tour if v != 0]
    if sorted(customers) != list(range(1, n + 1)):
        raise ValueError("each customer must be visited exactly once")
    if any(a == b == 0 for a, b in zip(tour, tour[1:])):
        raise ValueError("empty depot-to-depot route")
    load = 0.0
    for node in tour[1:]:
        load = 0.0 if node == 0 else load + float(instance["demands"][node])
        if load > instance["capacity"] + 1e-6:
            raise ValueError("vehicle capacity exceeded")
    return length(instance, tour)


def features(instance, visited, current, load):
    """Seven features in the original RLO order (including global remaining capacity)."""
    n = len(instance["demands"])
    f = np.zeros((n, 7), dtype=np.float32)
    f[:, :2] = instance["coords"]
    f[:, 2] = instance["demands"] / instance["capacity"]
    f[0, 3] = 1
    f[:, 4] = visited
    f[current, 5] = 1
    f[:, 6] = (instance["capacity"] - load) / instance["capacity"]
    return f
