# Wave 4 implementation (research branch, 26 October 2026)

This **overlay** adds `rlo/wave4_mcts.py`, `rlo/wave4_vtrace.py`, and `rlo/wave4_train.py` to the repository at fixed base commit `f8c315855b6d05a95d694ca52ddf851a999205e0`. It does not modify any running Wave 3 code. It does not launch the campaign.

## Methods and budgets

- Calibrate (three prerequisite runs) using **only** the frozen Wave-1 POMO actor's own sampled rollout returns. Freeze all policy parameters and fit the existing value head only; reward is negative full travel length. `calibrate` has a separate 1M-fresh-move calibration budget. Use each `last.pt` as the warm-start source for its corresponding seed. This *changes the original Wave 4 compute comparison* and the calibration cost must be included in lifetime costs.
- Warm: `az`, `az_rollout`, `vtrace_mcts`, `gvf_mean`, `gvf_blend`, `self_imitation`. Seed-matched calibrated starts, 25M additional fresh+simulated physical moves, three seeds each.
- Scratch: `offpolicy`, `recent_vtrace`, `dualstream`, `az_scratch`, `curriculum`. 50M fresh moves, except `dualstream` and `az_scratch` which use fresh+simulated moves.
- MCTS roots use 32 PUCT simulations, with the frozen policy/value network within a collection; value leaves vs rollout leaves are separate arms. Training MCTS behavior mixes 95% visits with 5% unsearched actor probabilities to preserve support. The **value-based search leaf has a calibrated but not necessarily accurate critic**.
- V-trace uses clipped ratios, terminal-aware MC special case, and recorded behavior log probabilities. Off-policy replay is episode-based with 100,000 decisions stored per problem size. `recent_vtrace` matches optimizer exposure using only current samples.
- GVF-CAPI variants are **reconstructions** (not a verified faithful historical reproduction) using a learned state-action cost-to-go network and a 4-action shortlist plus one random exploration candidate. `gvf_blend` uses 50% average and 50% top-decile cost, as a specified *new* blend until legacy source confirms the historical formula.
- The 33 configurations are 11 methods × 3 seeds; calibration is 3 additional preparatory runs. These jobs are **not** already submitted, and no training result should be claimed without CUDA trials.

## Correctness and science warnings

- This first-pass implementation retains exact interrupted/resumed RNG, optimizer, stream and replay state at *collection boundaries*, not arbitrary mid-episode/tree checkpoints. On interruption it completes the current collection and checkpoints; an allocation must allow sufficient grace time.
- MCTS counts tree edge traversals (including cached traversals) and policy-rollout physical moves separately. Cached traversal is not an actual new simulator transition, but appears on the common search-computation axis. Network evaluations and states are logged separately.
- `az` vs `vtrace_mcts` is search-target projection versus V-trace-gradient learning, not a clean unbiased off-policy comparison. `dualstream` samples 25% of visited decision roots for MCTS and uses a separately collected POMO gradient.
- `gvf_mean` / `gvf_blend` compare distinct targets, with an imperfect MBPI control because action pruning changes simulated-rollout work.
- Do not use exact optimization output, PyVRP, heuristic tours, or final test instances to train; only evaluate against them after choices are frozen.
- No built-in Wave 5 constraint environment, exact `Q^pi` diagnostic, or full beam-search benchmark is part of this initial overlay. They must not be represented as complete.

- The inference module includes greedy, independent sampling, all-start, completion-scored beam, **perturbed beam heuristic** (NOT canonical stochastic beam sampling), MCTS, and certified exact residual handoff. Exact `Q^pi` dynamic-programming diagnostics are implemented in `wave4_diag.py` for n <= 10; full projection-error experiments and matched-time PyVRP benchmarking are not yet implemented.

## Installation and execution

Use this branch as a **new checkout/worktree**, not by modifying the branches
used for the current Wave 3 jobs:

```bash
git fetch origin research/wave4-search-vtrace-20261026
git worktree add ~/RLO_wave4 origin/research/wave4-search-vtrace-20261026
cd ~/RLO_wave4
mkdir -p logs
source ~/venvs/rlo/bin/activate
python -m compileall -q rlo scripts
OMP_NUM_THREADS=1 pytest -q tests/test_wave4_core.py tests/test_wave4_smoke.py
```

A run must pass the CPU integration tests and a real H100 preflight before
submitting the substantive GPU array. The scripts expect the environment
variable `RLO_WAVE4_SHA` to equal the fully vetted immutable branch commit,
and the three Wave-1 POMO 25M milestone checkpoints at the documented paths.
Submit the three calibration jobs and verify they reached their 1M fresh-move
budget. Only then submit `scripts/campaign/wave4_train.sbatch` as the 33-arm
array. Do not submit before verifying Fir's 24-hour limits and real CUDA
throughput; the scripts deliberately do not edit the old Wave 3 worktrees.

Example (after testing, and after exporting the actual pinned SHA):

```bash
sbatch --export=ALL,RLO_WAVE4_SHA scripts/campaign/wave4_calibrate.sbatch
# After successful calibration:
sbatch --export=ALL,RLO_WAVE4_SHA scripts/campaign/wave4_train.sbatch
```

A Slurm signal triggers an orderly checkpoint at the next completed collection.
A later launch must explicitly set `RLO_RESUME=1`, use the **same SHA**, and
point to the same run directory. A task reaching the wall-time limit is not a
successful completed run: inspect the last checkpoint's `counters` and compare
its `real_moves + synthetic_moves` or `real_moves` with `max_moves`.

## Validation and outstanding engineering gates

`pytest` contains core mathematical/unit tests and 12 one-collection CPU
integration checks for the methods, plus an interrupted/resumed V-trace check.
These tests establish mechanical correctness only, never training performance.

Before claiming the 33 jobs are cluster-ready, run a representative A100/H100
preflight at `batch=16, n=30, S=32`, confirm that checkpoints resume exactly,
measure throughput and GPU memory, and check the fixed validation metric against
the existing Wave-1/3 implementation. These cluster preflights are not available
from the current runtime and cannot be inferred from the passing CPU tests.

Methods needing more work for the *complete* historical research program:
faithful original GVF-CAPI definition/source recovery; canonical stochastic
beam (we provide a named `perturbed_beam` heuristic instead); the full
projection/continuation-bias decomposition for MBPI; and matched-time classical
reference generation. Those are not silently counted as complete.
