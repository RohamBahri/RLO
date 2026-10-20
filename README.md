# RLO

Experience-only reinforcement learning for combinatorial optimization, beginning with the capacitated vehicle routing problem (CVRP). The goal is **one policy trained across instances and settings that also works on unseen problems**, without external solver or heuristic demonstrations.

- [PRINCIPLES.md](PRINCIPLES.md) — research constraints, including what experience-derived KL improvement permits.
- [docs/spec.md](docs/spec.md) — canonical CVRP formulation, frozen evaluation grid, and preregistered experiment decisions.
- [benchmarks/README.md](benchmarks/README.md) — reference validity and legacy limitations.

## Setup

```bash
pip install -e '.[test,rl,plots]'
pytest -q
python -m rlo --n 10 --solver savings --reference exact
```

Training uses PyTorch (CUDA, Apple MPS, or CPU). The archived MVP5 checkpoint is evaluated through a separate adapter; it is **not** compatible with current-policy training checkpoints.

## Training and evaluation

Train one masked policy (methods: `reinforce`, `rollout`, `a2c`, `pomo`):

```bash
python -m rlo.train --method a2c --sizes 10 20 30 --width 128 \
  --max-transitions 100000 --seed 0 --out results/smoke
python -m rlo.study --checkpoint results/smoke/last.pt \
  --split validation --sizes 10 20 30 --k 1 16
```

For reproducible screens, use `python -m scripts.sweep`. For results:

```bash
python -m scripts.group_a_report results/group_a \
  --out results/group_a_analysis
python -m scripts.curves results/group_a --out results/group_a_curves
```

Choose learning rates on validation only; leave the final test split untouched until selection is complete. All reference tours are evaluated using independent unrounded Euclidean costs.

## Reference generation on Narval (CPU via Apptainer)

The native Narval Gentoo Python environment rejects standard `manylinux` PyVRP wheels. References instead run in an Apptainer Python 3.11 container. Keep the active GPU training checkout and environment untouched.

One-time setup (already completed): image at `$SCRATCH/RLO_container/python311.sif`, project clone at `$SCRATCH/RLO_container/src`, and container-created Python environment at `$SCRATCH/RLO_container/venv` with `rlo-research[pyvrp]` installed.

```bash
module load StdEnv/2023 apptainer/1.4.5
BASE="$SCRATCH/RLO_container"
git -C "$BASE/src" pull --ff-only
cd "$BASE/src"
mkdir -p logs "$SCRATCH/RLO_references"
sbatch --account=def-brostami-ab scripts/reference_smoke.sbatch
```

Check the smoke job has completed successfully (`sacct -j JOB_ID --format=JobID,State,ExitCode` and `tail logs/rlo-ref-smoke_JOB_ID.out`) before submitting:

```bash
sbatch --account=def-brostami-ab scripts/references_cpu.sbatch
```

The CPU array comprises 18 tasks (9 settings × validation/test), at most four simultaneous tasks, two CPUs each, no GPUs. Each task saves 256 independently validated feasible tours under `$SCRATCH/RLO_references`. Sizes 10 and 12 use exact subset DP. Sizes 20, 25, 30, 40, 50 use PyVRP 0.14.x (three seeds × three seconds per instance), plus n=20 capacity factors 0.5 and 1.5. These are *heuristic* references for n≥20, not proven optima. Validation is for selection; keep test results sealed until model choices are frozen.

Check completion using `sacct` and verify there are exactly 18 JSONs in `$SCRATCH/RLO_references` before removing any container assets. Smoke outputs stay isolated in `$SCRATCH/RLO_container/smoke` to avoid mixing them with the final files.

After the GPU array finishes, analyze its results with `python -m scripts.group_a_report "$SCRATCH/RLO_runs/group_a_gpu"` from the training checkout/environment.

## Repository hygiene

Keep `main` canonical. Use short-lived branches for changes, merge reviewed PRs, then delete their branches. Preserve runnable research experiments and regression tests rather than duplicating historical MVP scripts. The older cost-only CVRP-20 references are retained solely as historical records; they cannot certify route quality.
