#!/usr/bin/env bash
#SBATCH --job-name=rlo-wave3-mbpi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x_%A_%a.out
set -Eeuo pipefail
: "${RLO_AB_SHA:?Supply tested research branch SHA}"
cd "$HOME/RLO_wave3_mbpi"
[[ "$(git rev-parse HEAD)" == "$RLO_AB_SHA" ]] || { echo WRONG_SHA; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo DIRTY_SOURCE; exit 2; }
module load python/3.11
source "$HOME/venvs/rlo/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

task="$SLURM_ARRAY_TASK_ID"
arm=$((task / 3))
seed=$((task % 3))
case "$arm" in
  0) roots=4; rollouts=8; tag=roots4_rollouts8 ;;
  1) roots=16; rollouts=4; tag=roots16_rollouts4 ;;
  *) echo "Invalid MBPI arm: $arm" >&2; exit 2 ;;
esac
var="RLO_MBPI_S${seed}"
anchor="${!var:?Supply full 25M POMO checkpoint path for each seed}"
[[ -f "$anchor" ]] || { echo "Anchor not found: $anchor"; exit 2; }
out="$SCRATCH/RLO_runs/campaign_wave3_mbpi_allocation/${tag}_seed${seed}"
resume=()
if [[ -e "$out" ]]; then
  if [[ "${RLO_RESUME:-0}" == 1 && -f "$out/last.pt" ]]; then
    [[ "$(cat "$out/commit.txt")" == "$RLO_AB_SHA" ]] || exit 2
    resume=(--resume "$out/last.pt")
  elif [[ "${RLO_RESUME:-0}" != 1 && -f "$out/commit.txt" ]]; then
    shopt -s nullglob dotglob
    entries=("$out"/*)
    shopt -u nullglob dotglob
    if [[ ${#entries[@]} != 1 || "${entries[0]##*/}" != commit.txt ||
          "$(cat "$out/commit.txt")" != "$RLO_AB_SHA" ]]; then
      echo "Nonempty or wrong-source output cannot be restarted: $out"; exit 2
    fi
    echo "REUSING_COMMIT_ONLY_DIRECTORY $out"
  else
    echo "Existing run; refusing overwrite without valid resume: $out"; exit 2
  fi
else
  mkdir -p "$out"
  printf '%s\n' "$RLO_AB_SHA" > "$out/commit.txt"
fi
echo "MBPI_ALLOCATION=$tag SEED=$seed ANCHOR=$anchor SHA=$RLO_AB_SHA"
if [[ ${#resume[@]} -eq 0 ]]; then
  smoke_dir="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mbpi_preflight.XXXXXXXX")"
  trap 'rm -rf "$smoke_dir"' EXIT
  srun --cpu-bind=cores python -m rlo.wave3_train \
    --method mbpi --initial-checkpoint "$anchor" --sizes 30 --eval-sizes 30 \
    --batch 16 --mbpi-roots "$roots" --mbpi-rollouts "$rollouts" \
    --seed "$seed" --device cuda --lr 0.0001 \
    --max-moves 25000000 --stop-after-moves 1 \
    --validate-every-moves 25000000 --validation-count 2 \
    --milestones --out "$smoke_dir"
  echo "WAVE3_MBPI_PREFLIGHT_OK tag=$tag seed=$seed"
  rm -rf "$smoke_dir"
  trap - EXIT
fi
srun --cpu-bind=cores python -m rlo.wave3_train \
  --method mbpi --initial-checkpoint "$anchor" \
  --sizes 10 20 30 --batch 16 --seed "$seed" --device cuda \
  --lr 0.0001 --mbpi-roots "$roots" --mbpi-rollouts "$rollouts" --mbpi-eta 0.2 \
  --max-moves 25000000 --validate-every-moves 1250000 \
  --checkpoint-every-moves 500000 \
  --milestones 5000000 12500000 20000000 25000000 \
  --validation-count 256 --out "$out" "${resume[@]}"
echo "FINISHED_OR_PAUSED mbpi tag=$tag seed=$seed out=$out"
