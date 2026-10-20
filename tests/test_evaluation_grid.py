"""Regression tests for fixed evaluation grids and independently checked references."""

import json
import numpy as np
import pytest

from rlo.data import evaluation_instances, fingerprint, instances
from rlo.problem import validate
from rlo.reference import load, save
from rlo.solvers import nearest_neighbor


def test_original_validation_and_test_cases_remain_unchanged():
    for split in ("validation", "test"):
        a = instances(20, split, 6)
        b = evaluation_instances(20, split, 6, 1.0)
        assert [fingerprint(x) for x in a] == [fingerprint(x) for x in b]


def test_capacity_shifts_keep_coordinates_and_demands_paired():
    base = evaluation_instances(20, "validation", 5)
    small = evaluation_instances(20, "validation", 5, 0.5)
    large = evaluation_instances(20, "validation", 5, 1.5)
    for x, lo, hi in zip(base, small, large):
        assert np.array_equal(x["coords"], lo["coords"])
        assert np.array_equal(x["coords"], hi["coords"])
        assert np.array_equal(x["demands"], lo["demands"])
        assert np.array_equal(x["demands"], hi["demands"])
        assert int(lo["capacity"]) < int(x["capacity"]) < int(hi["capacity"])
        assert int(lo["capacity"]) >= max(x["demands"])
    assert [fingerprint(x) for x in small] == [
        fingerprint(x) for x in evaluation_instances(20, "validation", 5, 0.5)
    ]


def test_invalid_factor_rejected():
    for factor in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="capacity_factor"):
            evaluation_instances(10, "test", 3, factor)


def test_shifted_reference_roundtrip_and_mismatch_guard(tmp_path):
    cases = evaluation_instances(10, "validation", 3, 0.5)
    routes = [nearest_neighbor(case) for case in cases]
    path = tmp_path / "shift.json"
    save(path, cases, routes, "test feasible", 10, "validation",
         capacity_factor=0.5)
    actual, method = load(path, cases)
    assert method == "test feasible"
    np.testing.assert_allclose(actual, [validate(x, r) for x, r in zip(cases, routes)])
    assert json.loads(path.read_text())["capacity_factor"] == 0.5
    with pytest.raises(ValueError, match="match"):
        load(path, evaluation_instances(10, "validation", 3, 1.5))
    with pytest.raises(ValueError, match="match"):
        save(tmp_path / "wrong.json", cases, routes, "bad", 10, "validation")


def test_pyvrp_reference_distance_scaling_and_client_indexing(monkeypatch):
    """A fake 0.14 model checks the adapter without requiring optional PyVRP."""
    import sys
    import types
    from rlo.reference import pyvrp_solver

    captured = []
    class Act:
        def __init__(self, idx):
            self.idx = idx
        def is_client(self):
            return True
    class Model:
        def __init__(self):
            self.locations = []
            self.clients = []
        def add_location(self, x, y):
            o = object()
            self.locations.append(o)
            return o
        def add_depot(self, location):
            return None
        def add_vehicle_type(self, **kwargs):
            return None
        def add_client(self, location, delivery):
            self.clients.append(delivery)
        def add_edge(self, frm, to, distance):
            captured.append(distance)
        def solve(self, **kwargs):
            routes = [[Act(i)] for i in range(len(self.clients))]
            return types.SimpleNamespace(
                is_feasible=lambda: True,
                best=types.SimpleNamespace(routes=lambda: routes),
            )
    pkg = types.ModuleType("pyvrp")
    pkg.Model = Model
    stop = types.ModuleType("pyvrp.stop")
    stop.MaxRuntime = lambda x: x
    monkeypatch.setitem(sys.modules, "pyvrp", pkg)
    monkeypatch.setitem(sys.modules, "pyvrp.stop", stop)
    inst = evaluation_instances(6, "test", 1)[0]
    route = pyvrp_solver(inst, seconds=0.1, seed=0)
    assert max(captured) <= 14143  # unit-square diagonal scaled by 10,000
    assert validate(inst, route) > 0
    assert sorted(x for x in route if x) == list(range(1, 7))
