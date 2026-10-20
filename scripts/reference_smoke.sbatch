#!/usr/bin/env bash
#SBATCH --job-name=rlo-ref-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

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
mkdir -p "$BASE/smoke"
run_py -c 'from importlib.metadata import version; print("PyVRP:", version("pyvrp"))'
for n in 10 20 50; do
  for factor in 0.5 1.5; do
    run_py -m scripts.freeze_references \
      --n "$n" --split validation --capacity-factor "$factor" \
      --count 3 --method pyvrp --seconds 0.3 --seeds 0 1 2 \
      --output "$BASE/smoke/smoke_${SLURM_JOB_ID}_n${n}_f${factor}.json"
  done
done
echo "ALL REFERENCE SMOKES PASSED"
