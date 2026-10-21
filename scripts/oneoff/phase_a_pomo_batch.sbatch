#!/usr/bin/env bash
#SBATCH --job-name=rlo-pomo-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x_%A_%a.out

# Controlled Phase A2: b=8 (tasks 0-2), b=4 (tasks 3-5).
# Existing b=32 Group A runs are the control. Execute from ~/RLO.
set -euo pipefail
cd "$HOME/RLO"
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

root="$SCRATCH/RLO_runs/group_a_gpu"
expected="$(cat "$root/pomo_w128_plain_lr0.0003_seed0/commit.txt")"
for seed in 1 2; do
    [[ "$(cat "$root/pomo_w128_plain_lr0.0003_seed\${seed}/commit.txt")" == "$expected" ]] || {
        echo "Group A POMO checkpoints were produced by different commits" >&2
        exit 2
    }
done
actual="$(git rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || {
    echo "Refusing unmatched code: GPU checkout $actual vs original $expected" >&2
    exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    echo "Refusing to run with tracked modifications" >&2
    exit 2
}

i="$SLURM_ARRAY_TASK_ID"
if (( i < 3 )); then batch=8; seed="$i"; else batch=4; seed="$((i-3))"; fi
out="$SCRATCH/RLO_runs/phase_a_pomo_batch/pomo_b\${batch}_seed\${seed}"
[[ ! -e "$out" ]] || { echo "Output already exists: $out" >&2; exit 1; }
mkdir -p "$out"
echo "$actual" > "$out/commit.txt"
srun python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"'
srun --cpu-bind=cores python -m rlo.train \
    --method pomo --sizes 10 20 30 \
    --width 128 --layers 2 --device cuda \
    --batch "$batch" --lr 0.0003 --seed "$seed" \
    --max-transitions 10000000 \
    --validate-every 100000000 --validate-every-transitions 1000000 \
    --validation-count 256 --selection-metric greedy \
    --out "$out"

# Secondary all-customer-start comparison, analogous to Group A.
for n in 10 20 30; do
    srun --cpu-bind=cores python -m rlo.study \
      --checkpoint "$out/last.pt" --split validation \
      --sizes "$n" --count 256 --k "$n" --multistart \
      --seed 0 --out "$out/multistart_n\${n}.json"
done
echo "DONE POMO batch=$batch seed=$seed commit=$actual"
