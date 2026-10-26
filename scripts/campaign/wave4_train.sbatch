#!/usr/bin/env bash
#SBATCH --job-name=rlo-wave4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=48:00:00
#SBATCH --array=0-32%2
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "$HOME/RLO_wave4"
: "${RLO_WAVE4_SHA:?Export tested Wave4 commit SHA}"
[[ "$(git rev-parse HEAD)" == "$RLO_WAVE4_SHA" ]] || { echo 'Wrong source revision'; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Dirty checkout'; exit 3; }
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
methods=(az az_rollout vtrace_mcts gvf_mean gvf_blend self_imitation offpolicy recent_vtrace dualstream az_scratch curriculum)
task="$SLURM_ARRAY_TASK_ID"
arm=$((task/3))
seed=$((task%3))
method="${methods[$arm]}"
out="$SCRATCH/RLO_runs/campaign_wave4/${method}_seed${seed}"
mkdir -p "$out"
[[ -f "$out/last.pt" && "${RLO_RESUME:-0}" != 1 ]] && { echo 'Existing checkpoint; set RLO_RESUME=1'; exit 3; }
args=()
if ((arm<6)); then
  anchor="$SCRATCH/RLO_runs/campaign_wave4/calibrate_seed${seed}/last.pt"
  [[ -f "$anchor" ]] || { echo 'Missing calibrated anchor' >&2; exit 3; }
  args+=(--initial-checkpoint "$anchor" --max-moves 25000000)
else
  args+=(--max-moves 50000000)
fi
[[ "${RLO_RESUME:-0}" == 1 ]] && args+=(--resume)
srun --cpu-bind=cores python -c 'import torch; assert torch.cuda.is_available(); print("CUDA PREFLIGHT",torch.cuda.get_device_name(0),flush=True)'
srun --cpu-bind=cores python -m rlo.wave4_train --method "$method" --seed "$seed" \
  --batch 16 --device cuda --out "$out" "${args[@]}"
echo WAVE4_FINISHED_OR_PAUSED
