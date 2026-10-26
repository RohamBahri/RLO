# Wave 3 — Packages A and B (October 25, 2026)

Status: implementation branch with regression tests and Narval launchers. No Wave 3 GPU results have been claimed. Main/old campaigns remain untouched.

## A: New learning algorithms

- PPO: undiscounted GAE and clipped policy objective, multiple frozen-behavior minibatch epochs with a value head and entropy option.
- Matched on-policy GAE actor–critic control for the following Dyna studies (not identical to Wave 1's original A2C-TD(lambda)).
- Dyna critic-only: sampled known-model continuations from freshly visited states; additional Monte Carlo updates change only value-head parameters, not the shared encoder or actor.
- Dyna actor+critic: same synthetic rollout source with actor-critic updates on synthetic actions sampled from the current policy. This trains the continuation objective conditional on encountered states, not the original-start distribution.
- KL-MBPI: start from the seed-matched Wave 1 POMO 25M checkpoint and perform 25M additional *total fresh + simulated* physical moves, using only self-generated action-value rollouts and KL improvement projection. The identical anchor's original 25M-to-50M continuation is the matched control. No exact-oracle or expert target enters training.

The asynchronous stale-target legacy Dyna is not claimed as replicated: the two Dyna arms are explicit synchronous, scientifically interpretable value-only and actor/value alternatives.

## B: Controlled architecture and training experiments

A shared POMO objective and environment underpin all arms:
- raw linear node encoder, pointwise residual MLP, learned soft codebook, 4-layer width256 8-head attention;
- optional masked glimpse, dynamic-state attention re-encoding, and distance-bias removal;
- auxiliary heads for realized future distance, remaining number of depot returns, and distance until next depot return (all from learner experience);
- route-close/continue high-level gate with a lower-level legal-customer pointer, together defining probabilities over the original legal actions;
- n20-only training, factor-one capacity, temperature 1.0 to 0.5, and POMO base batches 4 and 8.
The previous 50M POMO base-batch-16 runs provide the anchor, subject to a strict old/new one-step objective/gradient equivalence test.

## Resources and metrics

54 A/B runs = 18 configurations x 3 seeds; 50M fresh physical CVRP moves each. Three MBPI runs = seed-matched 25M POMO pretraining already performed + 25M total additional fresh and simulated moves per seed. Two A100 jobs run at once. Validation on the same n10/n20/n30 fixed cases (256 each), including n20-only training arms; test split is sealed. Full atomic checkpoints and exact RNG/optimizer/data-stream resume, physical move counters, synthetic move counters, SGD/replay counts and elapsed time are logged. A SIGUSR1 warning requests a graceful checkpoint rather than intentional data loss.

The launcher performs a real full-size CUDA update before the main task. Preflight failure blocks that task; existing history/checkpoints must never be overwritten.

## Safe start on Narval

Start from ~/RLO. Do not reset main or modify ~/RLO_campaign.

    cd ~/RLO
    mkdir -p logs
    git fetch origin research/wave3-packages-ab-20261025
    SHA=PINNED_SUCCESSFUL_COMMIT
    git worktree add --detach ~/RLO_wave3_ab "$SHA"
    git -C ~/RLO_wave3_ab status --short
    git -C ~/RLO_wave3_ab rev-parse HEAD
    bash -n ~/RLO_wave3_ab/scripts/campaign/wave3_ab50.sbatch
    bash -n ~/RLO_wave3_ab/scripts/campaign/wave3_mbpi25.sbatch

Ensure the worktree is clean and the exact GitHub CI-tested SHA is checked out.

    cd ~/RLO
    git show "$SHA":scripts/campaign/wave3_ab50.sbatch | sbatch --export="ALL,RLO_AB_SHA=$SHA"

For MBPI, first discover the exact POMO seed-matched sources:

    find "$SCRATCH/RLO_runs" -type f -name milestone_25000000.pt

Assign three absolute paths corresponding to the same 25M on-policy POMO experiment (seeds 0/1/2, not DDQN or other methods).

    export RLO_MBPI_S0=/absolute/path/to/pomo_seed0/milestone_25000000.pt
    export RLO_MBPI_S1=/absolute/path/to/pomo_seed1/milestone_25000000.pt
    export RLO_MBPI_S2=/absolute/path/to/pomo_seed2/milestone_25000000.pt
    git show "$SHA":scripts/campaign/wave3_mbpi25.sbatch | sbatch --export="ALL,RLO_AB_SHA=$SHA,RLO_MBPI_S0,RLO_MBPI_S1,RLO_MBPI_S2"

Monitor with sacct, squeue, and ~/RLO/logs/rlo-wave3-*.out. Preflights print WAVE3_PREFLIGHT_OK / WAVE3_MBPI_PREFLIGHT_OK. If a wall warning causes an early checkpoint, use RLO_RESUME=1 and the identical pinned source and configuration, submitting only those incomplete array indices.

## Research interpretation

Do not select from a lucky seed, retrospective best checkpoint, or final test split. Report performance separately by size and relative to fresh moves, total simulated moves, updates, and time. No training method is discarded just because another arm performs poorly. Unlike the fixed-fleet feasibility studies in the broader plan, all Package A/B runs here retain the same masked unlimited-fleet CVRP.
