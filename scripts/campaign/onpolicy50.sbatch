#!/usr/bin/env bash
#SBATCH --job-name=rlo-onpolicy50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/%x_%A_%a.out

# Wave 1: Four experience-only on-policy objectives x 3 seeds at 50M.
# Execute from the original Group A GPU checkout (~/RLO).
# This script runs FRESH experiments; existing last.pt cannot be faithfully resumed.
set -euo pipefail
cd "$HOME/RLO"
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

original="$SCRATCH/RLO_runs/group_a_gpu"
shopt -s nullglob
stamps=("$original"/*/commit.txt)
if (( ${#stamps[@]} != 36 )); then
    echo "Expected 36 original Group A commit stamps, found ${#stamps[@]}" >&2
    exit 2
fi
expected="$(cat "${stamps[0]}")"
for stamp in "${stamps[@]}"; do
    if [[ "$(cat "$stamp")" != "$expected" ]]; then
        echo "Group A runs have differing source commits ($stamp)" >&2
        exit 2
    fi
done
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$expected" ]]; then
    echo "Wrong checkout! Original Group A source $expected; current $actual" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked edits found in Group A checkout; refusing to run." >&2
    exit 2
fi

methods=(reinforce rollout a2c pomo)
lrs=(0.0001 0.0001 0.0003 0.0003)
batches=(32 32 32 16)
task="$SLURM_ARRAY_TASK_ID"
index=$((task / 3))
seed=$((task % 3))
method="${methods[index]}"
lr="${lrs[index]}"
batch="${batches[index]}"
out="$SCRATCH/RLO_runs/campaign_onpolicy50/${method}_b${batch}_lr${lr}_seed${seed}"
if [[ -e "$out" ]]; then
    echo "Existing output would be overwritten: $out" >&2
    exit 1
fi

mkdir -p "$out"
printf '%s\n' "$actual" > "$out/commit.txt"
printf 'Wave 1: method=%s, lr=%s, batch=%s, seed=%s, out=%s\n' "$method" "$lr" "$batch" "$seed" "$out"
srun python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

srun --cpu-bind=cores python -m rlo.train \
    --method "$method" --sizes 10 20 30 \
    --capacity-factors 0.75 1.25 --demand-maxes 10 \
    --width 128 --layers 2 --device cuda \
    --batch "$batch" --lr "$lr" --seed "$seed" \
    --max-transitions 50000000 \
    --validate-every 100000000 --validate-every-transitions 2500000 \
    --validation-count 256 --selection-metric greedy \
    --out "$out"

# Multi-start uses same 256 validation instances, same n-start budget for each algorithm.
for n in 10 20 30; do
    srun --cpu-bind=cores python -m rlo.study \
       --checkpoint "$out/last.pt" --split validation \
       --sizes "$n" --count 256 --k "$n" --multistart \
       --seed 0 --out "$out/multistart_n${n}.json"
done
echo "FINISHED method=$method seed=$seed batch=$batch transitions=50M"
