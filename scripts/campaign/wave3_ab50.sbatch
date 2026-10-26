#!/usr/bin/env bash
#SBATCH --job-name=rlo-fir-ab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --array=0-53%2
#SBATCH --output=logs/%x_%A_%a.out
set -Eeuo pipefail
: "${RLO_AB_SHA:?Supply the pinned tested Wave3 SHA}"
cd "$HOME/RLO_wave3_ab"
[[ "$(git rev-parse HEAD)" == "$RLO_AB_SHA" ]] || { echo "WRONG_SHA"; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "DIRTY_SOURCE"; exit 2; }
module load python/3.11
[[ -f "$HOME/venvs/rlo/bin/activate" ]] || { echo "MISSING_FIR_VENV"; exit 2; }
source "$HOME/venvs/rlo/bin/activate"
python -c 'import torch, numpy; print("TORCH", torch.__version__, "CUDA", torch.version.cuda)'
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

id="$SLURM_ARRAY_TASK_ID"
arm=$((id / 3))
seed=$((id % 3))
method=pomo
batch=16
extras=()
tag=
case "$arm" in
  0) method=ppo; tag=ppo_gae; extras=(--lam 0.95 --entropy 0.01) ;;
  1) method=dyna_critic; tag=dyna_value_mc ;;
  2) method=dyna_actor; tag=dyna_actor_mc ;;
  3) method=a2c; tag=a2c_matched_control ;;
  4) tag=encoder_raw; extras=(--encoder raw) ;;
  5) tag=encoder_mlp; extras=(--encoder mlp) ;;
  6) tag=encoder_codebook; extras=(--encoder codebook) ;;
  7) tag=attention_256x4_h8; extras=(--width 256 --layers 4 --heads 8) ;;
  8) tag=glimpse; extras=(--glimpse) ;;
  9) tag=dynamic_reencode; extras=(--dynamic-reencode) ;;
  10) tag=no_distance_bias; extras=(--no-distance-bias) ;;
  11) tag=multi_value_gvf; extras=(--auxiliary --aux-weight 0.1) ;;
  12) tag=hierarchy; extras=(--hierarchy) ;;
  13) tag=train_n20; extras=(--sizes 20) ;;
  14) tag=capacity_fixed; extras=(--capacity-factors 1.0 1.0) ;;
  15) tag=temperature_anneal; extras=(--temperature 1.0 --anneal-to 0.5) ;;
  16) tag=pomo_batch4; batch=4 ;;
  17) tag=pomo_batch8; batch=8 ;;
  *) echo "Invalid arm $arm"; exit 2 ;;
esac

out="$SCRATCH/RLO_runs/campaign_wave3_ab/${tag}_seed${seed}"
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

echo "LAUNCH_SHA=$RLO_AB_SHA ARM=$tag SEED=$seed HOST=$(hostname)"
# A real training update with the selected method/architecture on an A100.
if [[ ${#resume[@]} -eq 0 ]]; then
  smoke_dir="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rlo_ab_preflight.XXXXXXXX")"
  trap 'rm -rf "$smoke_dir"' EXIT
  srun --cpu-bind=cores python -m rlo.wave3_train \
    --method "$method" --sizes 30 --eval-sizes 30 --batch "$batch" --seed "$seed" \
    --device cuda --max-moves 50000000 --stop-after-moves 1 \
    --validation-count 2 --checkpoint-every-moves 1 \
    --validate-every-moves 50000000 --milestones \
    --out "$smoke_dir" "${extras[@]}"
  echo "WAVE3_PREFLIGHT_OK $tag seed=$seed"
  rm -rf "$smoke_dir"
  trap - EXIT
fi

srun --cpu-bind=cores python -m rlo.wave3_train \
  --method "$method" --sizes 10 20 30 --batch "$batch" --seed "$seed" \
  --device cuda --max-moves 50000000 --validate-every-moves 2500000 \
  --checkpoint-every-moves 1000000 \
  --milestones 10000000 25000000 40000000 50000000 \
  --validation-count 256 --out "$out" \
  "${extras[@]}" "${resume[@]}"
echo "FINISHED_OR_PAUSED $tag seed=$seed out=$out"
