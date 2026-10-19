# Research status (October 2026)

**Completed:** CFLP learned beam search (negative exploratory result); CVRP MVP0–5 (best general checkpoint: MVP5, ~6.64% mean gap on sizes 11,13,15,17,19); experimental AlphaZero/async/GVF attempts (confounded by training instability).

**Open:** MBPI obtained ~1.1% greedy gap on one CVRP-30 instance after millions of rollouts, but has not demonstrated distribution-level learning. The goal is to test whether expensive rollout policy improvement can be amortized across unseen instances.

**Phase 1:** canonical generator, distinct fixed splits, independent feasibility/cost evaluation, exact small CVRP, immutable legacy references, optional heuristic OR-Tools. Historical checkpoints are *not* imported into this repository.

**Next:** calibrate the archived MVP5 checkpoint via greedy/best-of-K sampling; diagnose MBPI against exact CVRP-10 action values; then compare distribution training with a strong policy-gradient anchor.
