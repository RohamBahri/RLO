# Wave 3 MBPI: two predeclared root allocations

This is a **separate pinned code branch** from the 54-run Packages A/B campaign. Do not update or modify a running A/B worktree.

## Scientific design

Six MBPI continuations: 2 configurations x 3 seeds. Both use the same per-seed frozen 25M POMO checkpoint, unchanged KL temperature (eta=0.2), LR=1e-4, 16 fresh cases per collection, train n10/20/30, common validation n10/20/30 (256 per size), and **25M additional total** counted CVRP physical moves (fresh + counterfactual simulation).

- Tasks 0,1,2 = roots4_rollouts8, seeds 0,1,2.
- Tasks 3,4,5 = roots16_rollouts4, seeds 0,1,2.

The 4x8 allocation estimates each legal action more precisely but directly improves fewer visited decision states per update. The 16x4 allocation trains on more root states per update, at lower Monte Carlo precision and with potentially fewer optimizer updates for the same move budget. It is **not** expected to make the same number of updates or have an identical wall time. Both are meaningful full-budget tests rather than cherry-picked best-of-run checkpoints.

The trainer samples unique visited roots whenever enough candidates exist and logs 'mbpi_roots', 'mbpi_root_candidates', 'mbpi_unique_roots', 'mbpi_actions', 'mbpi_rollouts', fresh/synthetic moves, optimizer updates, time and validation. It checks POMO method, seed, architecture and that the source checkpoint is in the [25M,25.1M) physical-move milestone interval. No optimizer labels, heuristic tours or exact Q labels enter training.

Results: `$SCRATCH/RLO_runs/campaign_wave3_mbpi_allocation/roots4_rollouts8_seed{0,1,2}` and `roots16_rollouts4_seed{0,1,2}`. Checkpoint/resume and output guards prevent accidental overwrite. **Never reuse** output folders from the old 3-run MBPI launcher.

## Prepare a separate detached worktree on Narval

Use the exact full commit SHA in the final verified run instructions.

    cd ~/RLO
    mkdir -p logs
    git fetch origin research/wave3-mbpi-allocation-20261025
    SHA=FULL_TESTED_SHA
    git worktree add --detach ~/RLO_wave3_mbpi "$SHA"
    test "$(git -C ~/RLO_wave3_mbpi rev-parse HEAD)" = "$SHA"
    test -z "$(git -C ~/RLO_wave3_mbpi status --porcelain)"
    bash -n ~/RLO_wave3_mbpi/scripts/campaign/wave3_mbpi25.sbatch

Do not use `git switch` in ~/RLO_campaign or ~/RLO_wave3_ab.

## Verify the three POMO anchors

    export RLO_MBPI_S0="$SCRATCH/RLO_runs/campaign_onpolicy50_v2/pomo_b16_lr0.0003_seed0/milestone_25000000.pt"
    export RLO_MBPI_S1="$SCRATCH/RLO_runs/campaign_onpolicy50_v2/pomo_b16_lr0.0003_seed1/milestone_25000000.pt"
    export RLO_MBPI_S2="$SCRATCH/RLO_runs/campaign_onpolicy50_v2/pomo_b16_lr0.0003_seed2/milestone_25000000.pt"
    for f in "$RLO_MBPI_S0" "$RLO_MBPI_S1" "$RLO_MBPI_S2"; do
      test -f "$f" || { echo "MISSING: $f"; break; }
      ls -lh "$f"
    done

If a path is missing, run `find "$SCRATCH/RLO_runs" -type f -name milestone_25000000.pt` and identify the **POMO** checkpoint for each seed. The launch also verifies source model metadata.

## Submit

    cd ~/RLO
    git show "$SHA":scripts/campaign/wave3_mbpi25.sbatch \
      | sbatch --export="ALL,RLO_AB_SHA=$SHA,RLO_MBPI_S0,RLO_MBPI_S1,RLO_MBPI_S2"

Each array task runs a full-size CUDA preflight for **its own roots/rollouts allocation** within the allocated A100; `WAVE3_MBPI_PREFLIGHT_OK` marks success. The run saves history, config, `last.pt`, `policy.pt`, milestone weights, and `commit.txt`. First start of a task leaves output paths unique; retry only source-matched stamp-only failures. All six should reach 25M additional moves within the wall limit or save resumable checkpoints when given Slurm's SIGUSR1.

## Monitor

    squeue -j JOBID -r
    sacct -j JOBID -X --format=JobID,State,ExitCode,Elapsed,MaxRSS
    grep -H -E "WAVE3_MBPI_PREFLIGHT_OK|FINISHED_OR_PAUSED|Traceback|Error" \
      ~/RLO/logs/rlo-wave3-mbpi_JOBID_*.out

## Resume only interrupted tasks

Do not resubmit completed tasks. Use **exactly the same SHA, anchors, parameters and output paths** with `RLO_RESUME=1`, for instance:

    git show "$SHA":scripts/campaign/wave3_mbpi25.sbatch \
      | sbatch --array=2,5 --export="ALL,RLO_AB_SHA=$SHA,RLO_RESUME=1,RLO_MBPI_S0,RLO_MBPI_S1,RLO_MBPI_S2"

The array override applies only to known incomplete task indices; substitute the actual indices.

## Interpretation

Use checkpoint-matched 25M->50M POMO continuation as control, but distinguish the **25M additional total simulations** in MBPI from the original POMO's **25M additional fresh environment moves**. Report 3-seed average and individual trajectories, checkpoint times, fresh and simulated move counts, total roots and optimizer updates; no final test split until choices are frozen. These six runs test root coverage versus rollout-estimate precision, not a theorem about MBPI.
