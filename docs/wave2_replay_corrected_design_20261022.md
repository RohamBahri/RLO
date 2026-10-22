# Wave 2 replay/Dyna corrected experimental specification — October 22, 2026

## Correction to the six-run draft (NOT VALID FOR SUBMISSION)
The original six-job `wave2_replay50.sbatch` launch plan equated 50M **real+model** physical moves. Under that rule, Dyna would collect fewer real moves, perform approximately half the optimization steps, and finish at a much larger epsilon. This confounds any claim about whether model-generated counterfactuals are useful. The draft must **not** be submitted. The revised implementation and launcher below supersede it.

## Primary scientific question
At an equal **50M fresh physical CVRP moves** (including automatically forced transitions), can simulated counterfactuals improve learning, and is 3-step return propagation helpful?

Each fresh training run uses n=10,20,30, seed 0/1/2; width-128, 2-layer distance-biased attention model; shared case-generator distribution and validation settings. No teacher trajectories, exact-solver guidance, or use of the final test split.

| Arm | n-step | Fresh moves | Additional model moves | Replay updates |
|---|---:|---:|---|---|
| DDQN | 1 | 50M | 0 | 8 per fresh collection |
| Dyna (known dynamics) | 1 | 50M | Recorded separately | scaled by new real+synthetic tuples |
| DDQN | 3 | 50M | 0 | 8 per fresh collection |
| Dyna (known dynamics) | 3 | 50M | Recorded separately | scaled by new real+synthetic tuples |

All four arms have three seeds: 12 substantive runs. For n=3, a transition's return accumulates up to three decision steps (including forced physical moves), truncates at terminal, and bootstraps from the final state. Real episodes construct these targets without counting overlapping rewards as additional fresh moves; Dyna uses its simulator for the additional steps and charges **all** physical moves generated. The uncorrected off-policy n-step target is a predeclared implementation choice: unlike 1-step DDQN, it is not in general an unbiased multi-step optimal Bellman target. Report that caveat when interpreting n-step results.

## Controlled parameters
- Common epsilon schedule on **fresh moves**: 1.0→0.1 over the first 20% (10M) of the 50M fresh-move budget, then 0.1.
- Replay capacity: 100,000 tuples **per size per real/synthetic class** (not 4,000).
- Two independent random streams for behavior/exploration and Dyna simulation, so counterfactual sampling cannot alter the real-action RNG stream.
- For each collection, DDQN performs 8 updates; Dyna's updates scale by `(new real tuples + new simulated tuples) / new real tuples`, carrying a fractional remainder. At one simulated root per real decision this gives approximately 16. With replay batch 64, expected sample reuse per newly *stored tuple* is matched, but Dyna is deliberately afforded more optimizer computation; show both quality-vs-fresh-moves and quality-vs-GPU-time plots.
- Use a target network, same Huber regression, same n-step choice and same model architecture within each matched pair. The simulator is **known dynamics**, not a learned model. No validation performance selection before training completion.
- Validation, milestones and final checkpoint at fresh-move 10/25/40/50M; report Dyna additional simulations, replay samples, optimizer updates, epsilon, per-size scores and elapsed time.
- Final sealed test stays untouched. Full-budget runs are not contingent on short-run validation performance.

## Additional comparisons after this batch
A constant **total** simulation budget question is distinct: simulated counterfactuals versus real moves at equal overall move expenditure, with **matched total replay updates**. Design that as a later controlled arm, not as the primary experiment. Do not drop legacy A2C, PPO, MBPI, E1/E2, larger representations, auxiliary GVFs, hierarchy or search.

## Correctness gate
Before a single cluster submission: all CI tests must pass including 1-step and 3-step fresh-move/resume parity for both DDQN and Dyna; exact legal masks and observations from snapshots; forced-move physical accounting; paired exploration/update formulas; and Slurm shell syntax. Inside each allocated GPU job, run a real CUDA Bellman optimizer step for its n-step algorithm. GPU elapsed-time and peak-memory performance cannot be established by CPU CI alone; the 24h Slurm walltime with checkpoints guards long jobs.
