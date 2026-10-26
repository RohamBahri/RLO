#!/usr/bin/env bash
set -euo pipefail
# Run only after training artifacts exist; validation only, no test unlock.
: "${1:?usage: wave4_eval.sh /path/to/policy.pt /output/directory}"
: "${2:?usage: wave4_eval.sh /path/to/policy.pt /output/directory}"
checkpoint="$1";out="$2"
mkdir -p "$out"
for spec in 'greedy 1' 'sample 8' 'sample 32' 'multistart 1' 'beam 8' 'mcts 32' 'handoff 10'; do
  read -r method k <<< "$spec"
  python -m rlo.wave4_eval --checkpoint "$checkpoint" --method "$method" --k "$k" \
    --device cpu --split validation --sizes 10 12 20 25 30 40 50 \
    --count 256 --out "$out/${method}_k${k}.json"
done
