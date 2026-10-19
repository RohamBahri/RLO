import numpy as np
import pytest
from rlo.problem import generate, validate
from rlo.data import instances, fingerprint
from rlo.solvers import nearest_neighbor, clarke_wright


def test_original_generator_and_legacy_split():
    old = np.random.default_rng(42)
    assert all(fingerprint(a) == fingerprint(generate(20, old))
               for a in instances(20, "legacy", 5))
    assert fingerprint(instances(20, "validation", 1)[0]) != fingerprint(instances(20, "test", 1)[0])


@pytest.mark.parametrize("solver", [nearest_neighbor, clarke_wright])
def test_baseline_routes_are_feasible(solver):
    for inst in instances(20, "validation", 10):
        assert validate(inst, solver(inst)) > 0


@pytest.mark.parametrize("tour", [[0, 1, 0], [1, 2, 0], [0, 1, 1, 2, 0], [0, 1, 0, 0, 2, 0]])
def test_rejects_invalid_tours(tour):
    inst = {"coords": np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32),
            "demands": np.array([0, 4, 4], dtype=np.float32), "capacity": 5}
    with pytest.raises(ValueError):
        validate(inst, tour)


def test_independent_capacity_check():
    inst = {"coords": np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32),
            "demands": np.array([0, 4, 4], dtype=np.float32), "capacity": 5}
    assert validate(inst, [0, 1, 0, 2, 0]) == 6
    with pytest.raises(ValueError, match="capacity"):
        validate(inst, [0, 1, 2, 0])
