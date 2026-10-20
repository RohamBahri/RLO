#!/usr/bin/env bash
#SBATCH --job-name=rlo-refs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-17%4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
module load StdEnv/2023 apptainer/1.4.5
BASE="$SCRATCH/RLO_container"
cd "$BASE/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

run_py() {
  srun --cpu-bind=cores apptainer exec --cleanenv \
    --bind "$SCRATCH:$SCRATCH" --pwd "$BASE/src" \
    --env OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=1 \
    "$BASE/python311.sif" "$BASE/venv/bin/python" "$@"
}

# 9 scenarios x 2 splits = 18 independent CPU tasks.
sizes=(10 12 20 25 30 40 50 20 20)
factors=(1 1 1 1 1 1 1 0.5 1.5)
scenario=$((SLURM_ARRAY_TASK_ID / 2))
split_index=$((SLURM_ARRAY_TASK_ID % 2))
splits=(validation test)
n="${sizes[$scenario]}"
factor="${factors[$scenario]}"
split="${splits[$split_index]}"
method=pyvrp
if [[ "$n" -le 12 ]]; then method=exact; fi
out="$SCRATCH/RLO_references/${split}_n${n}_f${factor}.json"
mkdir -p "$(dirname "$out")"
if [[ -e "$out" ]]; then
  echo "Reference exists; refusing overwrite: $out" >&2
  exit 1
fi

run_py -m scripts.freeze_references \
  --n "$n" --split "$split" --capacity-factor "$factor" \
  --count 256 --method "$method" --seconds 3 --seeds 0 1 2 \
  --output "$out"
echo "VERIFIED $out"
