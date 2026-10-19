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
