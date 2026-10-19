# Research status · October 2026

MVP5 is the best saved general policy (~6.64% mean *historical reference* gap over n=11,13,15,17,19); the archived means included rare infeasible tours. MBPI reached ~1.1% greedy gap on one n=30 instance after millions of rollouts but has no distribution-level validation. AlphaZero/GVF v3 is confounded by unstable training and uninformative feasibility targets.

**Phase 1:** canonical generator, independent feasible-tour evaluation, fixed streams, exact n≤12 solution, route-verifiable references and paired metrics. The archived n=20 OR-Tools costs (means 6.383451, 6.373003 and 6.358619) lack saved tours and used rounded integer objectives, so they are historical only. Freeze stronger n≥20 route references using PyVRP or another solver before tuning. PyVRP 0.14 uses iterated local search, not the older HGS engine.

**Next:** build one masked batched environment and exact cost-to-go from partial states, then zero-training MVP5 sampling and n=10 MBPI diagnostics.

**Research requirements:** See [PRINCIPLES.md](PRINCIPLES.md). Future training must use experience only, on diverse problem instances and settings, with explicit held-out size and parameter shifts. Prior MBPI cross-entropy distillation of search-produced action targets is **not an approved learning objective**. Use rollout returns/advantages in an RL policy-improvement update instead; preserve the historical MBPI implementation only as a diagnostic. Evaluation references and classical heuristics must never become training labels.
