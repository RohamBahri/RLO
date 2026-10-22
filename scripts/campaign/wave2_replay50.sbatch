#!/usr/bin/env bash
#SBATCH --job-name=rlo-wave2-replay50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/%x_%A_%a.out

# Twelve substantive runs: DDQN and known-model Dyna, nstep=1,3, 3 seeds.
# 50M FRESH CVRP physical moves per run. Extra Dyna simulations are ADDITIONAL.
# Separate real/model move counts, replay samples, and optimizer updates.
set -euo pipefail
: "${RLO_WAVE2_SHA:?Provide the pinned, tested Wave 2 source commit}"
cd "$HOME/RLO_wave2"
[[ "$(git rev-parse HEAD)" == "$RLO_WAVE2_SHA" ]] || {
  echo "Wrong source checkout" >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Dirty training source checkout" >&2; exit 2;
}
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

id="$SLURM_ARRAY_TASK_ID"
arm=$((id / 3))
seed=$((id % 3))
case "$arm" in
  0) method=ddqn; nstep=1 ;;
  1) method=dyna; nstep=1 ;;
  2) method=ddqn; nstep=3 ;;
  3) method=dyna; nstep=3 ;;
  *) echo "Unknown array task $id" >&2; exit 2 ;;
esac
out="$SCRATCH/RLO_runs/campaign_wave2_replay50/${method}_nstep${nstep}_seed${seed}"
resume=()
if [[ -e "$out" ]]; then
  if [[ "${RLO_RESUME:-0}" == 1 && -f "$out/resume.pt" ]]; then
    [[ "$(cat "$out/commit.txt")" == "$RLO_WAVE2_SHA" ]] || exit 2
    resume=(--resume "$out/resume.pt")
  else
    echo "Output exists; refusing overwrite. Use RLO_RESUME=1 for interrupted runs." >&2
    exit 2
  fi
else
  mkdir -p "$out"
  printf '%s\n' "$RLO_WAVE2_SHA" > "$out/commit.txt"
fi

# Actual CUDA Bellman update (and known-model Dyna if selected), one allocation.
srun --cpu-bind=cores python -m rlo.wave2_preflight \
  --method "$method" --n-step "$nstep" --batch 32 --replay-batch 64 \
  --width 128 --layers 2 --n 30 --device cuda

srun --cpu-bind=cores python -m rlo.wave2_train \
  --method "$method" --sizes 10 20 30 \
  --capacity-factors 0.75 1.25 --demand-maxes 10 \
  --width 128 --layers 2 --device cuda --n-step "$nstep" \
  --epsilon-anneal-fraction 0.2 \
  --batch 32 --replay-batch 64 --lr 0.0003 --seed "$seed" \
  --replay-capacity-per-size 100000 --updates-per-collection 8 \
  --target-sync-updates 1000 --max-real-transitions 50000000 \
  --validate-every-transitions 2500000 \
  --checkpoint-every-transitions 1000000 \
  --milestones 10000000 25000000 40000000 50000000 \
  --validation-count 256 --out "$out" "${resume[@]}"

echo "FINISHED method=$method nstep=$nstep seed=$seed out=$out"
