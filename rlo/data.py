"""Reproducible instance streams; the legacy split is the original 2026 test set."""

import hashlib
import numpy as np
from .problem import generate

SEEDS = {"legacy": 42, "validation": 202601, "test": 202602, "train": 202603}


def instances(n, split="validation", count=100):
    if split not in SEEDS:
        raise ValueError(f"unknown split: {split}")
    rng = np.random.default_rng(SEEDS[split])
    return [generate(n, rng) for _ in range(count)]


def fingerprint(instance):
    """Stable ID for preventing accidental reference/instance mismatches."""
    c = np.asarray(instance["coords"], dtype="<f4")
    d = np.asarray(instance["demands"], dtype="<f4")
    q = np.asarray([instance["capacity"]], dtype="<i8")
    return hashlib.sha256(c.tobytes() + d.tobytes() + q.tobytes()).hexdigest()


def training_stream(seed=0, *, sizes=(10, 15, 20, 30),
                    capacity_factors=(0.75, 1.25), demand_maxes=(8, 10, 12)):
    """Endless, reproducible *unlabeled* experiences across sizes and settings.

    Every iteration generates a new problem, never a solution or teacher label.
    Fixed legacy/validation/test streams are separate from this training stream.
    """
    if not sizes or not demand_maxes or not (0 < capacity_factors[0] <= capacity_factors[1]):
        raise ValueError("invalid training instance ranges")
    rng = np.random.default_rng(seed)
    while True:
        yield generate(int(rng.choice(sizes)), rng,
                       capacity_factor=float(rng.uniform(*capacity_factors)),
                       demand_max=int(rng.choice(demand_maxes)))


def training_batches(seed=0, *, batch=32, sizes=(10, 20, 30),
                     capacity_factors=(0.75, 1.25), demand_maxes=(10,)):
    """Shared generator: fresh unlabeled instances, one customer count per batch."""
    if batch < 1 or not sizes or not demand_maxes or capacity_factors[0] <= 0 or capacity_factors[0] > capacity_factors[1]:
        raise ValueError("invalid batch or generation settings")
    rng = np.random.default_rng(seed)
    while True:
        n = int(rng.choice(sizes))
        yield [generate(n, rng, capacity_factor=float(rng.uniform(*capacity_factors)),
                        demand_max=int(rng.choice(demand_maxes))) for _ in range(batch)]
