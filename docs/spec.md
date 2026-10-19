# CVRP model and experiment protocol · frozen v1 (October 2026)

## Mathematical problem

One depot (node 0), n customers (nodes 1,...,n), coordinates in [0,1]², integer demands qᵢ ≥ 1, capacity Q. **Unlimited identical vehicles**. Minimize the sum of raw Euclidean route lengths. No time window, fixed fleet, or sequence restrictions beyond visit-once and capacity. A depot return resets load. Route order is immaterial.

Generator: coordinates i.i.d. uniform, demand uniform integers [1,demand_max]. For the variable-settings distribution, Q = max(maxᵢ qᵢ, round(f·(demand_max+1) n / (2·3.5))). With demand_max=10 and f=1, the **archived generator** instead uses Q=max(10,round(5.5n/3.5)), preserving historical test cases byte-for-byte. The base spec's f-scaled formula reduces to max(maxᵢ qᵢ,round(f·5.5n/3.5)) when demand_max=10 and f≠1. The difference between the legacy 10 floor and realized-max floor is intentional and documented.

## State and actions

An instance is fixed through an episode. Markov state s=(V,c,ℓ): visited customer set, current node, and used capacity on the current route. Depot is always treated as visited. Cumulative distance and route count are *statistics*, not state variables here. For future fixed-fleet/time-window problems, the state **must be extended**; modular constraints cannot remove that requirement.

At any nonterminal state a customer j is legal iff j not visited and ℓ+qⱼ ≤ Q. Depot 0 is legal iff c ≠ 0. Every policy decision has ≥2 legal actions; **when exactly one action is legal it is executed automatically**, including final depot return and the last possible customer choice. The first action can be forced, as required by POMO. All automatic moves still incur distance and appear in the recorded tour. Terminal iff all customers have been visited and the vehicle is at depot. `infeasible` is a terminal status reserved for extensions; it should never occur in this unlimited-fleet model.

Each transition has deterministic reward r=−||x_c−x_a||₂ and discount γ=1. Group automatic moves after a sampled action into the same sampled-step reward. The accumulated cost must equal an independent `validate(tour)` evaluation.

## Observations and architecture

Per node, in this order: **x, y, q/Q, is_depot, visited, is_current** (6 sensors). Two global sensors, stored once: **ℓ/Q and n**. Pairwise attention bias uses **raw Euclidean distance** with no √2 rescaling. No customer-index embeddings, positional encoding, or identifier-dependent operations. Self-attention and symmetric pooling make the policy permutation-equivariant; test this property directly. Fixed node features in the old MVP5 PyTorch adapter are historical only and must not be mixed with the new architecture.

One size per batch, no padding. Model and training can support different sizes by taking different same-sized batches. Capacity is initially the only constraint module, providing legality, load update, and features. New constraint types may require additional state and observation dimensions.

## Experimental protocol

**Group A:** moving-average REINFORCE, greedy-rollout-baseline REINFORCE, A2C-TD(λ), POMO-style policy gradients. One shared architecture and masked environment. LR sweep *separately* for each method; three or more seeds; record curves versus total transitions and wall time; extend the top two toward a predeclared validation plateau or compute cap.

**Group B:** ablate distance bias, temperature annealing, single/mixed-size training, capacity/demand diversity, one factor at a time. Masking stays on; historical unmasked methods are not directly comparable.

**Groups C/D:** same-distribution new instances, unseen intermediate and larger sizes, parameter shifts; greedy and seeded best-of-K cost-versus-inference-compute curves. Historical MVP5 is diagnostic only; report feasibility independently. Save all settings, seeds, version/commit identifiers, per-instance results and train/evaluation time. Validation is for selection; `test` is used only after configurations are frozen.

**MBPI:** after Group A anchors, use rollout Q estimates under a *declared continuation policy*, with KL-constrained policy improvement from the policy's own simulated returns. n≤10 exact residual action values are **diagnostics**, never training labels. Compare against the strongest Group A method with equal transitions and time; include search-rollout costs.

Reference solutions: exact DP for small complete problems and exact residual n≤10; route-verified heuristic references for larger n. Never call a heuristic reference optimal. Stronger fixed validation/test references for n≥20 remain to be generated using the optional reference builder.
