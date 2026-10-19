## Research checks

- [ ] Trains a shared policy on many instances and tests on unseen instances/sizes/settings (or is evaluation-only).
- [ ] Uses only experience-based RL signals: no expert, solver, heuristic, or search pseudo-labels as supervised policy targets.
- [ ] Keeps reference solvers and ground-truth solutions out of learning updates.
- [ ] Includes a focused test and a clear reason for added complexity.
