"""Framework-independent, fail-closed evaluation of complete tours."""

import time
import numpy as np
from .problem import validate


def evaluate(instances, solver, references=None):
    """Solver is a callable instance -> tour. Invalid solutions cannot improve gaps."""
    if references is not None and len(references) != len(instances):
        raise ValueError("reference count mismatch")
    rows = []
    for i, inst in enumerate(instances):
        t = time.perf_counter()
        try:
            tour = solver(inst)
            cost = validate(inst, tour)
            error = None
        except (ValueError, IndexError) as exc:
            cost, error = None, str(exc)
        elapsed = time.perf_counter() - t
        ref = None if references is None else float(references[i])
        if ref is not None and ref <= 0:
            raise ValueError("reference costs must be positive")
        rows.append({"index": i, "cost": cost, "reference": ref,
                     "gap_pct": None if cost is None or ref is None else 100 * (cost / ref - 1),
                     "seconds": elapsed, "error": error})
    gaps = [row["gap_pct"] for row in rows if row["gap_pct"] is not None]
    valid = sum(row["error"] is None for row in rows)
    return {"count": len(rows), "feasibility_rate": valid / len(rows) if rows else 0,
            "mean_gap_pct": float(np.mean(gaps)) if len(gaps) == len(rows) and gaps else None,
            "total_seconds": sum(row["seconds"] for row in rows), "instances": rows}
