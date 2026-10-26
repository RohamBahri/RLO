#!/usr/bin/env bash
#SBATCH --job-name=rlo-wave4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-32%2
#SBATCH --signal=USR1@900
#SBATCH --output=logs/%x_%A_%a.out
set -Eeuo pipefail
: "${RLO_WAVE4_SHA:?Export the fully tested pinned Wave4 commit SHA}"
cd "$HOME/RLO_wave4"
[[ "$(git rev-parse HEAD)" == "$RLO_WAVE4_SHA" ]] || { echo WRONG_SHA >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo DIRTY_SOURCE >&2; exit 3; }
module load python/3.11
[[ -f "$HOME/venvs/rlo/bin/activate" ]] || { echo MISSING_FIR_VENV >&2; exit 3; }
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
methods=(az az_rollout vtrace_mcts gvf_mean gvf_blend self_imitation offpolicy recent_vtrace dualstream az_scratch curriculum)
task="$SLURM_ARRAY_TASK_ID"
arm=$((task / 3))
seed=$((task % 3))
(( arm < ${#methods[@]} )) || { echo INVALID_ARM >&2; exit 3; }
method="${methods[$arm]}"
out="$SCRATCH/RLO_runs/campaign_wave4/${method}_seed${seed}"
if [[ -d "$out" ]]; then
  [[ -f "$out/commit.txt" && "$(cat "$out/commit.txt")" == "$RLO_WAVE4_SHA" ]] || {
    echo "Wrong/missing commit stamp in $out" >&2; exit 3;
  }
  if [[ "${RLO_RESUME:-0}" == 1 ]]; then
    [[ -f "$out/last.pt" ]] || { echo MISSING_RESUME_CHECKPOINT >&2; exit 3; }
  else
    shopt -s nullglob dotglob
    entries=("$out"/*)
    shopt -u nullglob dotglob
    [[ ${#entries[@]} -eq 1 && ${entries[0]##*/} == commit.txt ]] || {
      echo "Refusing populated run directory: $out" >&2; exit 3;
    }
  fi
else
  [[ "${RLO_RESUME:-0}" != 1 ]] || { echo NO_RUN_TO_RESUME >&2; exit 3; }
  mkdir -p "$out"
  printf '%s\n' "$RLO_WAVE4_SHA" > "$out/commit.txt"
fi
args=()
if ((arm < 6)); then
  anchor="$SCRATCH/RLO_runs/campaign_wave4/calibrate_seed${seed}/last.pt"
  [[ -f "$anchor" ]] || { echo "Missing calibrated checkpoint $anchor" >&2; exit 3; }
  [[ "$(cat "$SCRATCH/RLO_runs/campaign_wave4/calibrate_seed${seed}/commit.txt")" == "$RLO_WAVE4_SHA" ]] || {
    echo WRONG_CALIBRATION_SOURCE >&2; exit 3;
  }
  args+=(--initial-checkpoint "$anchor" --max-moves 25000000)
else
  args+=(--max-moves 50000000)
fi
if [[ "${RLO_RESUME:-0}" == 1 ]]; then
  args+=(--resume)
else
  echo "No separate HPC smoke run. The first production update performs the CUDA gate."

fi
srun --cpu-bind=cores python -m rlo.wave4_train --method "$method" --seed "$seed" \
  --batch 16 --device cuda --out "$out" "${args[@]}"
echo WAVE4_FINISHED_OR_PAUSED
