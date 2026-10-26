#!/usr/bin/env bash
#SBATCH --job-name=rlo-wave4-calibrate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --signal=USR1@900
#SBATCH --output=logs/%x_%A_%a.out
set -Eeuo pipefail
: "${RLO_WAVE4_SHA:?Export pinned Wave4 commit SHA}"
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
seed="$SLURM_ARRAY_TASK_ID"
# Narval POMO checkpoints are NOT automatically present on Fir.
# Transfer the three files separately and provide SHA256 values verified on Narval.
source_var="RLO_POMO_S${seed}"
sha_var="RLO_POMO_SHA256_${seed}"
anchor="${!source_var:?Export RLO_POMO_S0/S1/S2 as Fir-local source paths}"
expected="${!sha_var:?Export RLO_POMO_SHA256_0/1/2 from Narval hashes}"
[[ -f "$anchor" && "$expected" =~ ^[a-fA-F0-9]{64}$ ]] || { echo BAD_OR_MISSING_POMO_SOURCE >&2; exit 3; }
actual="$(sha256sum "$anchor" | awk '{print $1}')"
[[ "${actual,,}" == "${expected,,}" ]] || { echo "POMO_SOURCE_SHA256_MISMATCH seed=$seed" >&2; exit 3; }
out="$SCRATCH/RLO_runs/campaign_wave4/calibrate_seed${seed}"
if [[ -d "$out" ]]; then
  [[ -f "$out/commit.txt" && "$(cat "$out/commit.txt")" == "$RLO_WAVE4_SHA" ]] || {
    echo BAD_CALIBRATION_COMMIT_STAMP >&2; exit 3;
  }
  if [[ "${RLO_RESUME:-0}" == 1 ]]; then
    [[ -f "$out/last.pt" ]] || { echo MISSING_RESUME_CHECKPOINT >&2; exit 3; }
  else
    shopt -s nullglob dotglob
    entries=("$out"/*)
    shopt -u nullglob dotglob
    [[ ${#entries[@]} -eq 1 && ${entries[0]##*/} == commit.txt ]] || {
      echo REFUSING_POPULATED_CALIBRATION >&2; exit 3;
    }
  fi
else
  [[ "${RLO_RESUME:-0}" != 1 ]] || { echo NO_CALIBRATION_TO_RESUME >&2; exit 3; }
  mkdir -p "$out"
  printf '%s\n' "$RLO_WAVE4_SHA" > "$out/commit.txt"
fi
flags=()
if [[ "${RLO_RESUME:-0}" == 1 ]]; then
  flags+=(--resume)
else
  smoke_dir="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rlo_w4_calibration.XXXXXXXX")"
  trap 'rm -rf "$smoke_dir"' EXIT
  srun --cpu-bind=cores python -m rlo.wave4_train --method calibrate --seed "$seed" \
    --initial-checkpoint "$anchor" --expected-initial-moves 25000000 \
    --max-moves 1000000 --stop-after-moves 1 --batch 16 --sizes 30 \
    --validate-every 100000000 --validation-count 2 --device cuda --out "$smoke_dir"
  echo "WAVE4_CALIBRATION_GPU_PREFLIGHT_OK seed=$seed"
  rm -rf "$smoke_dir"
  trap - EXIT
fi
srun --cpu-bind=cores python -m rlo.wave4_train --method calibrate --seed "$seed" \
  --initial-checkpoint "$anchor" --expected-initial-moves 25000000 \
  --max-moves 1000000 --device cuda --batch 16 --out "$out" "${flags[@]}"
echo CALIBRATION_FINISHED_OR_PAUSED
