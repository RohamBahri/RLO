# RLO Wave 2: coverage-complete method exploration (October 22, 2026)

**Status: preregistration and implementation plan, NOT a completed/runnable GPU campaign.**

## Research objective

Discover what each historically proposed or tried design principle contributes under a clean, shared CVRP protocol. The October 21–22 50M experiment is a four-objective **anchor**, not an elimination screen. Do not rule out REINFORCE, A2C, representation variants or older exploratory ideas based on the four-arm ranking. No 100M extensions in this wave. The experiment seeks understanding, not final paper-method selection.

The existing 12 successful 50M runs of REINFORCE, frozen rollout baseline, TD(lambda) A2C and all-n-start POMO (3 seeds each) are immutable controls. They need not be identically rerun; retention is not elimination. Preserve all seed/milestone checkpoints and the final test split.

## Complete coverage inventory and implementation work

Historical **tested** means prior exploratory code/results exist, but the old scores are not scientifically comparable with the current setting. **Proposed** means the older plan mentioned it without established execution. **New comparator** means worthwhile mainstream control not claimed historical.

| Track | Methods / ideas included | Historical status | 50M experimental treatment |
|---|---|---|---|
| Baseline objectives | REINFORCE with EMA; frozen rollout REINFORCE; A2C TD(lambda); all-start POMO | Modern 50M complete | Reuse 12 final and four milestones per seed as common controls |
| Critic and policy-gradient improvements | Actor-critic value-head design, TD(lambda); PPO with GAE (if implemented); entropy/temperature schedules | A2C tested; PPO a proposed/new comparator | Matched architecture and experience budget; no rejection because plain A2C lost |
| Early encoders | Raw sensory, nonlinear MLP, learned codebook | Legacy MVP1–2 tested | Rebuild compatible encoders; compare under one common objective rather than changing encoder and algorithm together |
| Attention representation | 4-layer width-256/8-head distance-biased transformer, width-128 plain, masked glimpse; static versus dynamic re-encoding, distance-bias removal | Older large model tested, glimpse pilot tested; dynamic re-encoding/without distance bias proposed | Full-budget single-axis variants; log parameter count, GPU time and memory |
| Training distribution/batch | Mixed versus n=20-only training; varied vs fixed capacity; temperature annealing; POMO batch 16 vs smaller base batch (e.g., 8) | Mixed sizes, annealing tested in legacy; POMO batch not cleanly tested | Full-budget arms, not small 10M screens; ensure n20-only compared on same n10/20/30 and extrapolation grid |
| Unmasked constraint learning | Legacy unmasked A2C learns to avoid revisits, capacity overflow and episode timeout | Legacy MVP4 and MVP5 tested | Explicit **separate constraint-learning experiment** with precisely specified invalid-action effects; do not silently combine with masked CVRP ranking |
| Off-policy, model-free | Replay-based masked graph Double-DQN with target network; optional replay actor-critic if implementable | No verified modern controlled baseline; new comparator | 50M unique/simulated experiences or explicitly declared common total-simulation axis; count replay optimizer updates separately |
| Model-based simulation / Dyna | Known-dynamics Dyna; legacy asynchronous critic-only Dyna; value-head-only Dyna; corrected actor/value updates from synthetic rollouts | Legacy MVP5 tested, inexact/stale-target concerns | Batched simulator, frozen or versioned targets, collect/simulate transition breakdown; Dyna uses known transition law, not a learned transition model |
| Rollout-based policy improvement | KL-MBPI using own sampled Q estimates, policy improvement/projection; oracle Q only diagnostic | Older single-instance experiments and current prototype | Start at common *25M* checkpoints and spend an additional *25M* total counted interactions/simulations; compare the same initial checkpoint's original 25M→50M continuation |
| Auxiliary predictions | Multi-value / GVF / Horde heads for future route length, slack, etc.; feasibility-completion GVF | Prototype/legacy proposals and experiments | Meaningful experience-defined targets on baseline problem, full-budget control without auxiliary head; true feasibility GVF separately on a fixed-fleet or other capacity-restricted variant with genuine infeasible continuations |
| Hierarchy | Route-close/continue high-level decision and low-level customer pointer; optional learned options | In historical MVP plan; not verified as completed | Same masked action space and equal overall CVRP reward; comparably sized flat control |
| Search-trained self-improvement | E1/E2, best-of-own-rollouts, AlphaZero-style MCTS/self-play targets; previously masked/unmasked AZ | Legacy explorations recorded; exact E1/E2 definitions/code not retained in current repo | Recover exact algorithm definitions from archives before implementation. Use only the learner's own simulated returns and policy search, not solver demonstration labels. Fair initialization/compute controls |
| Inference and hybrids | Greedy, equal-K sampling, all-start multistart, beam, MCTS, exact residual handoff at various remaining customers | Several explored, some only proposed | Evaluation-only; compare the SAME checkpoints and per-instance inference compute; label exact handoff a hybrid solver, not a pure learned policy |
| Evaluation baseline | Random feasible, nearest-neighbor, Clarke-Wright, exact small instances, PyVRP verified feasible references | Existing, not learning methods | No training; retain strong reference comparisons and independent cost/feasibility checking |

## One campaign, multiple parallel engineering tracks

- **Shared foundation before GPU submission:** batched masked state/rollout engine, replay buffer with immutable transition semantics, return/TD target tests, simulator accounting, checkpoint/resume parity, deterministic validation instances, legal-action / forced-action cost identity, and method registry. Correctness and ability to complete before wall-time are the only short-run gates; **never eliminate a method from a 5k-step validation screen**.
- **Learning-track GPU launch:** off-policy, Dyna, MBPI, PPO/critic extensions, hierarchy, aux heads, E1/E2/search-improvement; independent 50M-budget trials for 3 seeds, with two full-budget predeclared hyperparameter variants only where essential for a nontrivial learner. Keep one base control/config identically defined across experiments.
- **Representation/distribution-track GPU launch:** full-budget raw/MLP/codebook, large attention, glimpse, static/dynamic, distance bias, temperature, mixed/single-size, capacity and POMO batch. Where shared representation is needed, use a predeclared anchor objective (e.g., REINFORCE or POMO) and explicitly mark interaction experiments as separate; avoid an unnecessary full factorial cross-product.
- **Validation/inference track, no retraining:** load all 12 completed 50M checkpoints and relevant 25M/40M milestones; n10/12/20/25/30/40/50 factor=1, n20 factor=.5/1.5; 256 fixed validation cases per cell where feasible; greedy, equal-K, n-start and search-time tradeoffs. For all training arms use this same protocol at their fixed endpoints. Preserve paired per-instance solution costs. Never use the final test split to choose methods.
- **Constraint-learning track:** distinct reproducible restrictive environment for feasibility GVF/unmasked/hierarchy if required; provide matched no-GVF and masked controls *within that environment*. Do not compare its raw costs to unrestricted CVRP as though identical optimization tasks.

## Budgets and fairness

1. Aim for **50M budget per substantive training arm, 3 seeds**; no 100M extension yet. Reuse completed 50M controls. If warm-starting (MBPI/search-improvement), use 25M common source checkpoint + 25M additional work, with 25M→50M original continuation as matched control.
2. Report **unique fresh episode/decision transitions**, replay samples and updates, known-model rollouts, action simulations, full total simulated transitions, validation and inference overhead, and real elapsed GPU time. Distinct algorithms may need more than one fairness axis: plot quality versus unique interactions, total simulations, and time. A uniform scalar counter is not inherently comparable across all algorithms.
3. Do not change action legality, vehicle availability, reward or test data silently. Synthetic training uses the *given* CVRP dynamics, not expert solver trajectories. Classical/exact references are validation/test only.
4. Do not cherry-pick a lucky seed, best checkpoint, or best-of-K inference. Predeclare architecture, budgets, seeds 0/1/2, training distributions, and selection metrics. Retain all results and failures.
5. A failed implementation correctness check delays that arm only. Every historical method remains in the registry with status ready / porting / requires old archive / constrained-problem track / executed. No dropping methods due to this four-arm on-policy result.

## What the next analysis should teach

- Does an updated return/advantage estimator or replay genuinely improve learned policies?
- What is the incremental value of the known simulator and simulated policy improvement?
- Which representations generalize, and which differences are just parameter count or inference compute?
- Do auxiliary/structured decisions help learning, especially when constraints are genuinely restrictive?
- Does search improve training, inference only, or both, and at what total compute?
- Which methods give the strongest quality versus learning data, simulations, wall-time and generalization trade-offs?

No declaration of a main paper method until these distinct mechanisms have had controlled, material-budget tests.
