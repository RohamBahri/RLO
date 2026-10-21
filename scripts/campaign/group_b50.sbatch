#!/usr/bin/env bash
#SBATCH --job-name=rlo-group-b50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-11%1
#SBATCH --output=logs/%x_%A_%a.out

# Group B long-run ablations: POMO base batch 16, 50M counted transitions.
# Four variants x three seeds. Control is Wave 1's POMO base-batch-16 runs.
# %1 is intentional: Wave 1 already permits 2 simultaneous GPUs.
set -euo pipefail
cd "$HOME/RLO"
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

original="$SCRATCH/RLO_runs/group_a_gpu"
expected="$(cat "$original/pomo_w128_plain_lr0.0003_seed0/commit.txt")"
for seed in 1 2; do
  [[ "$(cat "$original/pomo_w128_plain_lr0.0003_seed${seed}/commit.txt")" == "$expected" ]] || {
    echo "Original POMO commits differ" >&2; exit 2;
  }
done
actual="$(git rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || {
  echo "Wrong training checkout $actual (expected $expected)" >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked changes present; refusing to run" >&2; exit 2;
}
task="$SLURM_ARRAY_TASK_ID"
group=$((task / 3))
seed=$((task % 3))
variants=(no_distance_bias anneal_0p5 fixed_capacity n20_only)
variant="${variants[group]}"
out="$SCRATCH/RLO_runs/campaign_group_b50/${variant}_seed${seed}"
[[ ! -e "$out" ]] || {
  echo "Existing output would be overwritten: $out" >&2; exit 1;
}
mkdir -p "$out"
printf '%s\n' "$actual" > "$out/commit.txt"

extra=()
sizes=(10 20 30)
factors=(0.75 1.25)
case "$variant" in
  no_distance_bias) extra+=(--without-distance-bias) ;;
  anneal_0p5) extra+=(--anneal-to 0.5) ;;
  fixed_capacity) factors=(1.0 1.0) ;;
  n20_only) sizes=(20) ;;
  *) echo "Unknown variant $variant" >&2; exit 2 ;;
esac
echo "GROUP B $variant seed=$seed GPU required, 50M transitions"
srun python -c 'import torch; print("CUDA ready:", torch.cuda.is_available(), flush=True); assert torch.cuda.is_available()'
srun --cpu-bind=cores python -m rlo.train \
  --method pomo --sizes "${sizes[@]}" \
  --capacity-factors "${factors[@]}" --demand-maxes 10 \
  --width 128 --layers 2 --device cuda \
  --batch 16 --lr 0.0003 --seed "$seed" \
  --max-transitions 50000000 \
  --validate-every 100000000 --validate-every-transitions 2500000 \
  --validation-count 256 --selection-metric greedy \
  --out "$out" "${extra[@]}"

# Secondary all-n-start evaluation on same validation data, no test use.
for n in 10 20 30; do
  srun --cpu-bind=cores python -m rlo.study \
    --checkpoint "$out/last.pt" --split validation \
    --sizes "$n" --count 256 --k "$n" --multistart \
    --seed 0 --out "$out/multistart_n${n}.json"
done
echo "FINISHED $variant seed=$seed"
