"""Fail-closed evaluation and paired, feasible-only CVRP comparisons."""

import time
import numpy as np
from .data import fingerprint
from .problem import validate


def evaluate(instances, solver, references=None):
    """Solver is an instance -> tour callable; invalid tours never receive a gap."""
    if references is not None and len(references) != len(instances):
        raise ValueError("reference count mismatch")
    rows = []
    for i, inst in enumerate(instances):
        start = time.perf_counter()
        try:
            cost = validate(inst, solver(inst))
            error = None
        except (ValueError, IndexError, TypeError) as exc:
            cost, error = None, str(exc)
        ref = None if references is None else float(references[i])
        if ref is not None and (not np.isfinite(ref) or ref <= 0):
            raise ValueError("reference costs must be finite and positive")
        rows.append({"index": i, "instance_id": fingerprint(inst), "cost": cost,
                     "reference": ref, "gap_pct": None if cost is None or ref is None
                     else 100 * (cost / ref - 1), "seconds": time.perf_counter() - start,
                     "error": error})
    gaps = [row["gap_pct"] for row in rows if row["gap_pct"] is not None]
    feasible = sum(row["error"] is None for row in rows)
    return {"count": len(rows), "feasible_count": feasible,
            "feasibility_rate": feasible / len(rows) if rows else 0.0,
            "reference_count": len(gaps),
            "mean_feasible_gap_pct": float(np.mean(gaps)) if gaps else None,
            "total_seconds": sum(row["seconds"] for row in rows), "instances": rows}


def paired(left, right):
    """Compare jointly feasible instances only; positive means left costs more."""
    a, b = left["instances"], right["instances"]
    if [r["instance_id"] for r in a] != [r["instance_id"] for r in b]:
        raise ValueError("paired reports must cover identical instances in the same order")
    both = [(x, y) for x, y in zip(a, b) if x["cost"] is not None and y["cost"] is not None]
    diffs = [100 * (x["cost"] / y["cost"] - 1) for x, y in both]
    gap_diffs = [x["gap_pct"] - y["gap_pct"] for x, y in both
                 if x["gap_pct"] is not None and y["gap_pct"] is not None]
    return {"paired_count": len(both), "total_count": len(a),
            "mean_paired_cost_difference_pct": float(np.mean(diffs)) if diffs else None,
            "mean_paired_gap_difference_pp": float(np.mean(gap_diffs)) if gap_diffs else None}
