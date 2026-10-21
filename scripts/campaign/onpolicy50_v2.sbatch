#!/usr/bin/env bash
#SBATCH --job-name=rlo-onpolicy50v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/%x_%A_%a.out

# Long-run Wave 1 with exact resume and 10/25/40/50M milestone checkpoints.
# Submit with --export=ALL,RLO_CAMPAIGN_SHA=<tested commit>.
# Training code comes from detached $HOME/RLO_campaign.
set -euo pipefail
: "${RLO_CAMPAIGN_SHA:?Set RLO_CAMPAIGN_SHA to tested Git commit}"
cd "$HOME/RLO_campaign"
[[ "$(git rev-parse HEAD)" == "$RLO_CAMPAIGN_SHA" ]] || {
    echo "Campaign checkout does not match pinned commit" >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    echo "Tracked changes in campaign checkout" >&2; exit 2;
}
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

methods=(reinforce rollout a2c pomo)
lrs=(0.0001 0.0001 0.0003 0.0003)
batches=(32 32 32 16)
task="$SLURM_ARRAY_TASK_ID"
index=$((task / 3))
seed=$((task % 3))
method="${methods[index]}"
lr="${lrs[index]}"
batch="${batches[index]}"
out="$SCRATCH/RLO_runs/campaign_onpolicy50_v2/${method}_b${batch}_lr${lr}_seed${seed}"
resume=()
if [[ -e "$out" ]]; then
    if [[ "${RLO_RESUME:-0}" == 1 && -f "$out/last.pt" ]]; then
        resume=(--resume "$out/last.pt")
        [[ "$(cat "$out/commit.txt")" == "$RLO_CAMPAIGN_SHA" ]] || {
            echo "Output originated from a different source commit" >&2; exit 2;
        }
    else
        echo "Output exists. For an interrupted run, resubmit with RLO_RESUME=1." >&2
        exit 1
    fi
else
    mkdir -p "$out"
    printf '%s\n' "$RLO_CAMPAIGN_SHA" > "$out/commit.txt"
fi

srun python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"'
srun --cpu-bind=cores python -m rlo.campaign_train \
    --method "$method" --sizes 10 20 30 \
    --capacity-factors 0.75 1.25 --demand-maxes 10 \
    --width 128 --layers 2 --device cuda \
    --batch "$batch" --lr "$lr" --seed "$seed" \
    --max-transitions 50000000 \
    --validate-every 100000000 --validate-every-transitions 2500000 \
    --checkpoint-every-transitions 1000000 \
    --milestones 10000000 25000000 40000000 50000000 \
    --validation-count 256 --selection-metric greedy \
    --out "$out" "${resume[@]}"

echo "FINISHED method=$method seed=$seed batch=$batch target=50M"
