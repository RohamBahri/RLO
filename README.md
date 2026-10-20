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

## Reference generation on macOS

Narval's Gentoo Python packaging reports only native `linux_x86_64` wheel tags, so the official PyVRP 0.14.1 `manylinux` binary is rejected. Avoid forcing a source build or retagging the wheel. Run references on a local Mac using native PyVRP, and copy the resulting JSONs to the cluster. Do not modify the ongoing Narval GPU job's `~/RLO` checkout or environment.

```bash
git clone https://github.com/RohamBahri/RLO.git ~/RLO_refs
cd ~/RLO_refs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,pyvrp]'
python -c "from importlib.metadata import version; print(version('pyvrp'))"
python -m pytest -q

mkdir -p results/reference_smoke
for n in 10 20 50; do
  for f in 0.5 1.5; do
    python -m scripts.freeze_references --n "$n" --split validation \
      --capacity-factor "$f" --count 3 --method pyvrp \
      --seconds 0.3 --seeds 0 1 2 \
      --output "results/reference_smoke/n${n}_f${f}.json" || exit 1
  done
done
```

After the smoke test passes, freeze validation and test references with 256 instances for sizes 10,12,20,25,30,40,50 at f=1 and n=20 at f=0.5,1.5. Use exact DP for n≤12 and multiseed PyVRP for larger n, saving route-verified JSONs under `results/references`. Transfer those JSON files to `$SCRATCH/RLO_references` on Narval. Test outcomes must not influence model selection.

After the GPU array finishes, analyze it with `python -m scripts.group_a_report "$SCRATCH/RLO_runs/group_a_gpu"`.

## Repository hygiene

Keep `main` canonical. Use short-lived branches for changes, merge reviewed PRs, then delete their branches. Preserve runnable research experiments and regression tests rather than duplicating historical MVP scripts. The older cost-only CVRP-20 references are retained solely as historical records; they cannot certify route quality.
