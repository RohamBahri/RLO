# RLO

Lean experiments in learning to solve combinatorial optimization problems. Currently, **CVRP** is the testbed; the focus is whether rollout-based policy improvement generalizes across instances.

## Start

```bash
pip install -e '.[test]'
pytest -q
python -m rlo --n 10 --solver savings --reference exact
python -m rlo --n 20 --split legacy --count 100 --reference benchmarks/legacy_n20.json
```

CVRP instances use the original 2026 generator. `legacy` reproduces its 100 seed-42 cases; `validation` and `test` are separate fixed streams. The small subset-DP solver is **exact** (up to 12 customers). Legacy and OR-Tools reference values are **feasible heuristics**, not proven optima. Evaluation checks route feasibility and measures real Euclidean distance.

The core needs only NumPy. Optional `mlx`, `cuda` (PyTorch), and `ortools` extras are available. `rlo.backend.choose()` selects MLX on Apple Silicon, CUDA PyTorch when available, or NumPy otherwise; no neural training code is included yet.

To evaluate an archived MVP5 checkpoint, install `pip install -e ".[cuda]"` (PyTorch also runs on CPU) and run `python -m rlo --n 19 --split legacy --solver mvp5 --checkpoint /path/to/mvp5.pt --reference ortools`. Checkpoints and old experimental branches stay in the external archive, not in Git.
