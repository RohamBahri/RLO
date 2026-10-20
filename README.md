# RLO

Research on **experience-only reinforcement learning that generalizes** across instances, sizes and settings of combinatorial optimization problems. CVRP is the first testbed. See [the frozen model and experiment protocol](docs/spec.md) and [research principles](PRINCIPLES.md).

```bash
pip install -e '.[test,rl]'
pytest -q
python -m rlo --n 10 --solver savings --reference exact
```

Train the four masked RL baselines (change `--method` to `reinforce`, `rollout`, `a2c` or `pomo`):

```bash
python -m rlo.train --method a2c --sizes 10 20 30 --max-transitions 1000000 --updates 100000 --lr 0.0001 --seed 0 --out results/a2c_0
```

Run a controlled Group A learning-rate/seed sweep (select on validation, not test):

```bash
python -m scripts.sweep --methods reinforce rollout a2c pomo --lrs 0.0001 0.0003 --seeds 0 1 2 --max-transitions 1000000
pip install -e '[plots]'
python -m scripts.curves results/group_a
```

Evaluate unseen instances and inference budgets, without tuning on `test`:

```bash
python -m rlo.study --checkpoint results/a2c_0/last.pt --split validation --sizes 10 20 30 --k 1 16 128
python -m rlo.study --checkpoint results/a2c_0/last.pt --split validation --sizes 20 --k 1 4 20 --multistart
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

**Group A fairness:** The policy uses static node attention **once per episode** and a light state decoder, inverse size `1/n`, logit clipping at ±10, masked actions, a per-size REINFORCE baseline, a *frozen* rollout baseline promoted only on an independent training pool, and all-`n` POMO starts by default. Default validation is 256 instances per size. The sweep is an **initial screening**, not a convergence result: extend the strongest models until learning curves flatten within predefined compute limits. Use the untouched `test` split only after choosing settings. Previous `rlo.train` v1 checkpoints are incompatible with the new v2 architecture; archived MVP5 remains available through `--legacy`.

### Before the Group A sweep: architecture pilot

The decoder-glimpse hypothesis is not yet validated. Run identical experiments with and without `--glimpse` (default: off):

```bash
python -m scripts.sweep --methods a2c pomo --sizes 10 --widths 64 128 \
  --architectures plain glimpse --lrs 0.0003 --seeds 0 \
  --max-transitions 5000000 --out results/arch_n10
python -m scripts.sweep --methods a2c pomo --sizes 20 --widths 64 128 \
  --architectures plain glimpse --lrs 0.0003 --seeds 0 \
  --max-transitions 5000000 --out results/arch_n20
python -m scripts.curves results/arch_n10 --out results/plots_n10
```

Validation always records single-start greedy; POMO also records all-start greedy. For a **fair all-start primary** across all methods, pass both `--validation-multistart --selection-metric multistart` (more costly). Architecture should only be frozen after confirming any difference across seeds. Group A LR screening defaults to **10M transitions**, not 1M.

## Held-out references and Group A analysis (October 2026)

The canonical evaluation generator is `evaluation_instances(n, split, count, capacity_factor)`. Changed-capacity scenarios keep coordinates and demands fixed; capacity factor 1 retains historical evaluation cases. PyVRP 0.14.x search now scales integer distances by 10,000 while scoring all tours in the original Euclidean metric.

**Do not update the active `~/RLO` checkout while GPU Slurm array 6564009 is running.** Make a separate cluster checkout for CPU reference tasks:

```bash
git clone https://github.com/RohamBahri/RLO.git ~/RLO_refs
cd ~/RLO_refs
mkdir -p logs
# Install optional PyVRP 0.14.x in your active environment before submission.
sbatch --account=def-brostami-ab scripts/reference_smoke.sbatch
# Inspect successful smoke results before launching all 18 tasks:
sbatch --account=def-brostami-ab scripts/references_cpu.sbatch
```

All reference JSON files are written to `$SCRATCH/RLO_references` and store verified tours, fingerprints and capacity factors. Copy finished datasets to permanent storage. For n=10/12 references are exact; for larger n they are multiseed heuristic upper bounds. Test references can be frozen now, but cannot be used for model selection.

After Group A completes:

```bash
python -m scripts.group_a_report "$SCRATCH/RLO_runs/group_a_gpu" --out "$SCRATCH/RLO_runs/group_a_analysis"
python -m scripts.curves "$SCRATCH/RLO_runs/group_a_gpu" --out "$SCRATCH/RLO_runs/group_a_curves"
```

The report uses the final single-start greedy validation cost divided by sqrt(n), averaged across n=10/20/30 and three seeds, to select learning rates. It retains per-seed costs, optimizer update counts and training time. `rlo.study --out` now includes per-instance cost/fingerprint records for paired evaluation.