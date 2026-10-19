import numpy as np
import pytest
from rlo.data import instances
from rlo.reference import load


def test_frozen_legacy_reference():
    cases = instances(20, "legacy", 100)
    values, method = load("benchmarks/legacy_n20.json", cases)
    assert np.mean(values) == pytest.approx(6.383451)
    assert "NOT exact" in method
    cases[0]["coords"][0, 0] += 0.1
    with pytest.raises(ValueError, match="does not match"):
        load("benchmarks/legacy_n20.json", cases)
