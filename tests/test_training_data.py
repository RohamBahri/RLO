from itertools import islice
import numpy as np
from rlo.data import training_stream, instances, fingerprint
from rlo.problem import generate, validate
from rlo.solvers import nearest_neighbor


def test_legacy_generator_is_unchanged():
    rng = np.random.default_rng(42)
    expected = instances(20, "legacy", 5)
    actual = [generate(20, rng) for _ in range(5)]
    assert [fingerprint(x) for x in actual] == [fingerprint(x) for x in expected]


def test_training_stream_is_reproducible_diverse_and_unlabeled():
    opts = dict(sizes=(10, 20), capacity_factors=(0.6, 1.5), demand_maxes=(8, 12))
    a = list(islice(training_stream(7, **opts), 80))
    b = list(islice(training_stream(7, **opts), 80))
    assert [fingerprint(x) for x in a] == [fingerprint(x) for x in b]
    assert all(set(x) == {"coords", "demands", "capacity"} for x in a)
    assert {len(x["demands"]) - 1 for x in a} == {10, 20}
    assert len({x["capacity"] for x in a if len(x["demands"]) == 21}) > 1
    assert max(float(x["demands"].max()) for x in a) > 10
    assert all(validate(x, nearest_neighbor(x)) > 0 for x in a)


def test_invalid_generation_settings_fail():
    rng = np.random.default_rng(0)
    for args in [dict(capacity_factor=0), dict(demand_max=0)]:
        try:
            generate(20, rng, **args)
        except ValueError:
            continue
        assert False, "invalid setting was accepted"
