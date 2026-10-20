#!/usr/bin/env bash
#SBATCH --job-name=rlo-ref-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

cd "$HOME/RLO_refs"
module load python/3.11
source "$HOME/venvs/rlo_refs/bin/activate"
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
python -c 'import pyvrp; print("PyVRP available:", pyvrp.__version__)'
for n in 10 20 50; do
  for factor in 0.5 1.5; do
    srun --cpu-bind=cores python -m scripts.freeze_references \
      --n "$n" --split validation --capacity-factor "$factor" \
      --count 3 --method pyvrp --seconds 0.3 --seeds 0 1 2 \
      --output "$SCRATCH/RLO_references/smoke_n${n}_f${factor}.json"
  done
done
echo "ALL REFERENCE SMOKES PASSED"
