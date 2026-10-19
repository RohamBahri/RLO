from itertools import permutations, product
import pytest
from rlo.data import instances
from rlo.exact import solve
from rlo.problem import length, validate
from rlo.evaluate import evaluate
from rlo.solvers import clarke_wright


def brute_force(instance):
    n = len(instance["demands"]) - 1
    best = float("inf")
    for perm in permutations(range(1, n + 1)):
        for breaks in product([False, True], repeat=n - 1):
            tour = [0]
            for i, customer in enumerate(perm):
                if i and breaks[i - 1]:
                    tour.append(0)
                tour.append(customer)
            tour.append(0)
            try:
                best = min(best, validate(instance, tour))
            except ValueError:
                pass
    return best


def test_exact_matches_bruteforce():
    for inst in instances(4, "validation", 4):
        assert validate(inst, solve(inst)) == pytest.approx(brute_force(inst))


def test_exact_dominates_heuristic():
    for inst in instances(10, "test", 3):
        assert length(inst, solve(inst)) <= validate(inst, clarke_wright(inst)) + 1e-8


def test_evaluation_marks_invalid_solutions():
    data = instances(4, "validation", 2)
    result = evaluate(data, lambda inst: [0, 1, 0], [1, 1])
    assert result["feasibility_rate"] == 0
    assert result["mean_feasible_gap_pct"] is None

def test_feasible_mean_and_paired_counts():
    from rlo.evaluate import paired
    cases = instances(4, "validation", 3)
    optimal = [validate(x, solve(x)) for x in cases]
    a = evaluate(cases, clarke_wright, optimal)
    def flaky(inst):
        return [0, 1, 0] if inst is cases[1] else clarke_wright(inst)
    b = evaluate(cases, flaky, optimal)
    assert a["mean_feasible_gap_pct"] is not None
    assert b["feasibility_rate"] == pytest.approx(2 / 3)
    assert b["mean_feasible_gap_pct"] is not None
    assert paired(a, b)["paired_count"] == 2
    assert paired(a, b)["mean_paired_cost_difference_pct"] == pytest.approx(0)
    with pytest.raises(ValueError, match="identical instances"):
        paired(a, evaluate(instances(4, "test", 3), clarke_wright))
