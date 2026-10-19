# RLO

Research on **experience-only reinforcement learning that generalizes** across instances, sizes and settings of combinatorial optimization problems. CVRP is the first testbed. See [the frozen model and experiment protocol](docs/spec.md) and [research principles](PRINCIPLES.md).

```bash
pip install -e '.[test,rl]'
pytest -q
python -m rlo --n 10 --solver savings --reference exact
```

Train the four masked RL baselines (change `--method` to `reinforce`, `rollout`, `a2c` or `pomo`):

```bash
python -m rlo.train --method a2c --sizes 10 20 30 --updates 1000 --lr 0.0001 --seed 0 --out results/a2c_0
```

Run a controlled Group A learning-rate/seed sweep (select on validation, not test):

```bash
python -m scripts.sweep --methods reinforce rollout a2c pomo --lrs 0.0001 0.0003 --seeds 0 1 2 --updates 1000
```

Evaluate unseen instances and inference budgets, without tuning on `test`:

```bash
python -m rlo.study --checkpoint results/a2c_0/last.pt --split validation --sizes 10 20 30 --k 1 16 128
```

Calibrate an archived MVP5 checkpoint independently (greedy and seeded stochastic best-of-K; routes may be infeasible):

```bash
python -m rlo.study --legacy --checkpoint /path/to/mvp5.pt --split legacy --sizes 11 13 15 17 19 --k 1 16 128
```

Experience-only KL-MBPI experiment, with optional **evaluation-only** exact diagnostics:

```bash
python -m rlo.mbpi --checkpoint results/a2c_0/last.pt --n 10 --diagnostic --updates 10 --samples 8
```

The core environment, exact small-instance solvers and benchmarks need only NumPy. Training uses **PyTorch** (CUDA on NVIDIA, MPS on Apple Silicon, CPU otherwise) to keep one verifiable training implementation; MLX-only training is not claimed. Historical MVP5 is a separate optional PyTorch adapter. Route-verified larger reference sets are not yet committed; generate them via `python -m scripts.freeze_references --n 20 --split validation --method pyvrp` after `pip install -e '.[pyvrp]'`. Archived cost-only references are historical, not exact or route-verifiable.
