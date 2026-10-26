# Wave 4 review corrections — October 26, 2026

This is the corrected, **not-yet-GPU-validated** implementation of the preregistered Wave 4 methods. All corrections below were fixed *before* submitting any Wave 4 training job. Do not modify or cancel the 60 running Wave 3 tasks. The branch and complete source commit must be pinned independently.

## Budget and simulation accounting (material preregistration correction)

PUCT tree search now charges only **new physical CVRP transitions** materialized during tree expansion (`SearchStats.edges`) and new policy-completion transitions (`SearchStats.rollout_moves`) to the simulated-move budget. Physical movements along already existing tree edges are *cached traversals*, recorded in `SearchStats.traversals` but not added to the simulated physical transition budget. All network calls/states and GPU elapsed time are recorded separately. This change prevents repeated walks from consuming artificial environment-move budgets.

Warm-start search, GVF-CAPI, and self-imitation have **25M additional fresh+new-simulated physical moves**, using the seed-matched POMO 25M anchor and shared 1M critic calibration cost (include original 25M and calibration in lifetime cost plots). AZ from scratch (`az_scratch`) has 50M total fresh+new-simulated physical moves. **Synchronous dual-stream (`dualstream`) has 50M FRESH physical moves**, like POMO, with its search simulations separately reported. Off-policy V-trace, recent V-trace and curriculum use 50M fresh moves. Never compare all algorithms along just a single scalar budget; report training moves, new simulator moves, cached traversals, optimizer steps, network evaluations and elapsed time, and preserve differences in sample size between methods.

## Conditional GVF-CAPI target

`gvf_mean` and `gvf_blend` train on up to **16 distinct root decisions per update**, taking **4 independent rollouts per sampled action**. This matches the *root count and per-action precision* of Wave 3's 16-roots × 4-rollouts MBPI allocation, not the total number of sampled actions/simulator transitions, since GVF intentionally shortlists only four actions. The GVF shortlist is the four smallest predicted costs among legal actions; one extra random omitted action is sampled solely to improve auxiliary GVF coverage.

For each root, compute `p_old` from the existing actor, let `S` be the four-action shortlist, and set

    target(a) = p_old(S) * softmax_a(log p_old(a) - Qhat(a)/eta)     for a in S
                p_old(a)                                         for a not in S

This leaves **all prior probability outside S unchanged** and preserves total shortlist probability; previous hard-zero targets were incorrect. The extra exploratory omitted action is never included in the KL-improvement shortlist. The non-shortlist mass conservation is regression tested. `gvf_blend` remains a **specified reconstruction**, not a verified historical implementation.

## V-trace and value calibration

- Entropy regularization is explicit `--entropy`, default **0** for matched comparisons. Opt-in entropy must be reported as a changed configuration.
- The policy/value model's POMO anchor has an **untrained value head**. Three seed-specific 1M-fresh-move `calibrate` jobs freeze the policy and fit the value head using its *own* sampled returns. They log a fixed, held-out set of n10/n20/n30 rollout-state value MSE and explained variance plus the declared gate, `--min-calibration-ev 0.0`. Calibration states and returns are **diagnostics only**, never used for critic training. The gate is a warning, not an automatic exclusion: all value-leaf methods remain in the study but are flagged if it fails. Held-out rollout targets are noisy samples, not exact `V^pi`.
- Calibrated checkpoints save these metrics. A warm-start job refuses a calibration checkpoint that has not reached its required fresh-move budget; it records a quality warning but does not exclude a fully completed checkpoint whose value gate fails. Warm-start source quality warnings are logged. New seed matched warm starts must be calibrated and use the same model/seed and SHA-pinned source.

## Fir operational safeguards (scripts included; NO Fir jobs launched)

- Source SHA and clean-tracked-worktree guards; per-run `commit.txt`, exact resume with matching stamp, and no overwrite of existing run directories.
- Fail-closed check for an activated Fir virtual environment; No separate Fir smoke or sacrificial one-update training run is included. The **first actual CUDA optimizer update of each production job** is treated as its CUDA operational gate: optimizer and gradient-finiteness checks run as part of the substantive experiment, and a success marker is printed. This respects the instruction not to spend cluster allocations on smoke testing. GPU-specific behavior remains unverified until a production run is authorized.
- Slurm notification changed to `--signal=USR1@900` because all search methods checkpoint after a complete training collection.
- The 25M POMO checkpoints are **not assumed to be on Fir**. Transfer the three original Narval source files outside running job directories, record SHA-256 on Narval, copy them to an independently verified Fir path, and export `RLO_POMO_S0/S1/S2` and `RLO_POMO_SHA256_0/1/2`. Calibration refuses wrong/missing hashes and records the source digest in its checkpoint configuration.

### Example anchor transfer preparation (execute on the relevant systems)

On Narval, for each original POMO 25M milestone:

```bash
sha256sum "$SCRATCH/RLO_runs/campaign_onpolicy50_v2/pomo_b16_lr0.0003_seed0/milestone_25000000.pt"
# Also run for seeds 1 and 2. Preserve three original hashes.
```

Copy the three checkpoints using an approved transfer mechanism to Fir. On Fir, after verifying the copied files with `sha256sum`, configure:

```bash
export RLO_POMO_S0="$SCRATCH/RLO_source_pomo/pomo_seed0_25m.pt"
export RLO_POMO_S1="$SCRATCH/RLO_source_pomo/pomo_seed1_25m.pt"
export RLO_POMO_S2="$SCRATCH/RLO_source_pomo/pomo_seed2_25m.pt"
export RLO_POMO_SHA256_0='<64-character SHA256 actually measured on Narval>'
export RLO_POMO_SHA256_1='<64-character SHA256 actually measured on Narval>'
export RLO_POMO_SHA256_2='<64-character SHA256 actually measured on Narval>'
```

Pass these values to the Slurm job with `sbatch --export=ALL,RLO_WAVE4_SHA,...` or an approved environment-file mechanism. A plain `sbatch` with an empty/export-lost source will fail closed.

## Explicit remaining scope and deviations

- AlphaZero-style distillation uses four projections on the current collection rather than a historical 50k-decision replay buffer. Search behavior remains a 95/5 MCTS-visits/policy mixture over the whole episode (full support for V-trace), not a 30%-then-greedy schedule.
- Value leaf versus rollout leaf evaluates an approximate learned value against a policy-sampled simulator completion; the latter is *not* a perfect value oracle.
- Wave 4 GVF-CAPI and blended objective are reconstructions. The `perturbed_beam` inference control is a heuristic, not canonical stochastic beam search.
- Exact small-instance `Q^pi` calculation is included, but the full projected-KL/continuation-bias MBPI diagnostic, formal proof of value-leaf calibration adequacy, and complete matched-time PyVRP comparisons remain unfinished.

## Local validation before any cluster submission

Run in a full RLO checkout with the actual modern `rlo` package, **locally on CPU**, not Fir:

```bash
python -m compileall -q rlo scripts
bash -n scripts/campaign/wave4_train.sbatch
bash -n scripts/campaign/wave4_calibrate.sbatch
OMP_NUM_THREADS=1 pytest -q tests/test_wave4_core.py tests/test_wave4_review.py \
    tests/test_wave4_smoke.py tests/test_wave4_report.py
```

One-collection CPU smoke checks each training arm, and interrupted/resumed training must exactly agree. Local unit tests alone using a tiny stub environment are not a substitute for full-repository integration tests. **Do not launch Fir test jobs merely to verify the implementation.** A future full-budget GPU run still carries unverified GPU-only performance and memory risks.
