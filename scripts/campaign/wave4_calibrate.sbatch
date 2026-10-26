#!/usr/bin/env bash
#SBATCH --job-name=rlo-wave4-calibrate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
# Do not submit from Wave 3's running checkout. Use a detached, pinned worktree.
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
seed="$SLURM_ARRAY_TASK_ID"
source="$SCRATCH/RLO_runs/campaign_onpolicy50_v2/pomo_b16_lr0.0003_seed${seed}/milestone_25000000.pt"
[[ -f "$source" ]] || { echo "Missing anchor $source" >&2; exit 3; }
out="$SCRATCH/RLO_runs/campaign_wave4/calibrate_seed${seed}"
mkdir -p "$out"
[[ -f "$out/last.pt" && "${RLO_RESUME:-0}" != 1 ]] && { echo 'Refusing to restart completed calibration'; exit 3; }
flags=()
[[ "${RLO_RESUME:-0}" == 1 ]] && flags+=(--resume)
srun --cpu-bind=cores python -c 'import torch; assert torch.cuda.is_available(); print("CUDA PREFLIGHT",torch.cuda.get_device_name(0),flush=True)'
srun --cpu-bind=cores python -m rlo.wave4_train --method calibrate --seed "$seed" \
  --initial-checkpoint "$source" --expected-initial-moves 25000000 \
  --max-moves 1000000 --device cuda --batch 16 --out "$out" "${flags[@]}"
echo CALIBRATION_FINISHED_OR_PAUSED
