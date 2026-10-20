#!/usr/bin/env bash
#SBATCH --job-name=rlo-group-a-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --array=0-35%2
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/RLO"
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# 4 algorithms x 3 learning rates x 3 seeds; one GPU for each job.
methods=(reinforce rollout a2c pomo)
lrs=(0.0001 0.0003 0.0005)
seeds=(0 1 2)
i="$SLURM_ARRAY_TASK_ID"
method="${methods[$((i / 9))]}"
lr="${lrs[$(((i / 3) % 3))]}"
seed="${seeds[$((i % 3))]}"
out="$SCRATCH/RLO_runs/group_a_gpu/${method}_w128_plain_lr${lr}_seed${seed}"
if [[ -e "$out/history.jsonl" ]]; then
  echo "Existing run at $out; refusing to overwrite" >&2
  exit 1
fi
mkdir -p "$out"
git rev-parse HEAD > "$out/commit.txt"

# Fail clearly rather than silently training on CPU.
srun python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU visible", torch.cuda.is_available(), flush=True); assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required"'

srun --cpu-bind=cores python -m rlo.train \
  --method "$method" --sizes 10 20 30 \
  --width 128 --layers 2 --device cuda \
  --batch 32 --lr "$lr" --seed "$seed" \
  --max-transitions 10000000 \
  --validate-every 100000000 \
  --validate-every-transitions 1000000 \
  --validation-count 256 --selection-metric greedy \
  --out "$out"

# Secondary inference comparison; all methods receive the same n-start budget.
for n in 10 20 30; do
  srun --cpu-bind=cores python -m rlo.study \
    --checkpoint "$out/last.pt" --split validation \
    --sizes "$n" --count 256 --k "$n" --multistart \
    --seed 0 --out "$out/multistart_n${n}.json"
done

echo "COMPLETED method=$method lr=$lr seed=$seed out=$out"
