"""Small, inspectable constructive baselines. All return depot-separated tours."""

import numpy as np
from .problem import distances


def nearest_neighbor(instance):
    d, demands, cap = distances(instance), instance["demands"], instance["capacity"]
    left = set(range(1, len(d)))
    tour, current, load = [0], 0, 0
    while left:
        fits = [i for i in left if load + demands[i] <= cap]
        if not fits:
            tour.append(0)
            current, load = 0, 0
            continue
        i = min(fits, key=lambda j: d[current, j])
        tour.append(i)
        current, load = i, load + demands[i]
        left.remove(i)
    return tour + [0]


def clarke_wright(instance):
    """Parallel savings algorithm; same idea as the archived MVP 0 baseline."""
    d, demands, cap = distances(instance), instance["demands"], instance["capacity"]
    routes = [[i] for i in range(1, len(d))]
    savings = sorted(((d[0, i] + d[0, j] - d[i, j], i, j)
                      for i in range(1, len(d)) for j in range(i + 1, len(d))), reverse=True)
    for saving, i, j in savings:
        if saving <= 0:
            break
        a = next((r for r in routes if i in (r[0], r[-1])), None)
        b = next((r for r in routes if j in (r[0], r[-1]) and r is not a), None)
        if a is None or b is None or sum(demands[a + b]) > cap:
            continue
        merged = (a[::-1] if a[0] == i else a) + (b[::-1] if b[-1] == j else b)
        routes.remove(a)
        routes.remove(b)
        routes.append(merged)
    tour = [0]
    for route in routes:
        tour.extend(route + [0])
    return tour
