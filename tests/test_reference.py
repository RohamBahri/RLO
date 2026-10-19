import json
import pytest
from rlo.data import instances
from rlo.reference import load, save


def test_cost_only_archive_is_rejected():
    with pytest.raises(ValueError, match="cost-only"):
        load("benchmarks/legacy_n20.json", instances(20, "legacy", 2))


def test_reference_roundtrip(tmp_path):
    from rlo.exact import solve
    cases = instances(4, "validation", 3)
    path = tmp_path / "routes.json"
    routes = [solve(x) for x in cases]
    save(path, cases, routes, "exact DP", 4, "validation")
    costs, method = load(path, cases)
    assert len(costs) == 3 and method == "exact DP"
    with pytest.raises(FileExistsError):
        save(path, cases, routes, "DP", 4, "validation")
    record = json.loads(path.read_text())
    record["costs"][0] -= .01
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disagree"):
        load(path, cases)
    record["costs"][0] += .01
    record["fingerprints"][0] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        load(path, cases)
