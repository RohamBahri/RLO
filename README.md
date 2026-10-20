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

## Narval: CPU references alongside the GPU study

**While Slurm GPU array 6564009 is active, do not pull or edit `~/RLO` or change its Python environment.** Use a separate checkout and virtual environment for the new reference code:

```bash
git clone https://github.com/RohamBahri/RLO.git ~/RLO_refs
cd ~/RLO_refs
module load python/3.11
mkdir -p ~/venvs logs
virtualenv --no-download ~/venvs/rlo_refs
source ~/venvs/rlo_refs/bin/activate
pip install --no-index -e '.[test,pyvrp]'
pytest -q
python -c 'import pyvrp; print(pyvrp.__version__)'
sbatch --account=def-brostami-ab scripts/reference_smoke.sbatch
```

Inspect the smoke job's output and its `COMPLETED` state first. Only after **all** smoke instances pass:

```bash
sbatch --account=def-brostami-ab scripts/references_cpu.sbatch
```

The CPU array prepares validation and test references at n=10,12,20,25,30,40,50, plus capacity shifts at n=20. Small reference tours are exact; larger ones are **heuristic upper bounds**, not certified optima. Files are stored under `$SCRATCH/RLO_references`; copy completed datasets to persistent storage.

Once the GPU array finishes, use `python -m scripts.group_a_report "$SCRATCH/RLO_runs/group_a_gpu"` to compare completed configurations.

## Repository hygiene

Keep `main` canonical. Use short-lived branches for changes, merge reviewed PRs, then delete their branches. Preserve runnable research experiments and regression tests rather than duplicating historical MVP scripts. The older cost-only CVRP-20 references are retained solely as historical records; they cannot certify route quality.
