from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution


def select_parents(
    population: list[Solution],
    rng: RandomNumberGenerator,
    cost_evaluator: CostEvaluator,
    k: int = 2,
) -> tuple[Solution, Solution]:
    """
    Design Hypothesis:
    Parent 1 selection uses a quality-driven tournament with a mild feasibility
    tie-breaker to secure an exploiting, high-fitness anchor. Parent 2 is selected
    via a quality-diversity tournament over candidates distinct from Parent 1,
    scoring each candidate by both normalized penalised cost and fast customer-route
    Jaccard dissimilarity relative to Parent 1. This prevents wasted crossover on
    near-identical clones without degrading parent quality into unproductive tail
    solutions. Redundant cost and feasibility evaluations are cached on demand.

    Reflection:
    - Pitfalls addressed:
      * Kp.1 & Kp.2: Avoids diversity collapse from pure cost elitism while
        preventing fitness degradation caused by forcing tail-strata selection.
      * Kp.3: Applies a gentle feasibility preference (1% tie-breaker) rather
        than hard filtering, preserving valuable near-feasible structures.
      * Kt.1 & Kt.2: Uses canonical client set extraction `set(v for v in r if v > 0)`
        and set-theoretic maximum route overlap instead of fragile positional route index matching.
      * Kt.3: Lazy on-demand caching of cost/feasibility over small tournaments
        (size 4) prevents $O(n)$ or $O(n^2)$ overhead on large populations.
    - Remaining limitations:
      Route-set Jaccard distance measures cluster overlap but is insensitive to
      intra-route visit sequencing differences (e.g., reversed or 2-opt permutations).
    - Follow-up revisions:
      1. Incorporate directed or undirected edge (broken pairs) difference to capture
         within-route sequence diversity alongside cluster overlap.
      2. Dynamically adapt the diversity weight (0.45) based on the search state
         (e.g., higher during diversification phases, lower during intensification).
    """
    n = len(population)
    if n < 2:
        raise ValueError("Need at least two solutions")
    if k != 2:
        raise ValueError("Only k=2 is supported")
    if n == 2:
        return population[0], population[1]

    cost_cache: dict[int, float] = {}
    feas_cache: dict[int, bool] = {}

    def get_cost(idx: int) -> float:
        if idx not in cost_cache:
            cost_cache[idx] = cost_evaluator.penalised_cost(population[idx])
        return cost_cache[idx]

    def get_feas(idx: int) -> bool:
        if idx not in feas_cache:
            feas_cache[idx] = population[idx].is_feasible()
        return feas_cache[idx]

    def get_effective_cost(idx: int) -> float:
        # Controlled feasibility preference: 1% tie-breaker penalty if infeasible
        return get_cost(idx) * (1.0 if get_feas(idx) else 1.01)

    # --- Parent 1 Selection: Cost-focused tournament with feasibility preference ---
    t1_size = min(4, n)
    p1_candidates: list[int] = []
    while len(p1_candidates) < t1_size:
        idx = rng.randint(n)
        if idx not in p1_candidates:
            p1_candidates.append(idx)

    # Break exact ties with small uniform jitter
    p1_idx = min(
        p1_candidates,
        key=lambda i: get_effective_cost(i) + rng.rand() * 1e-6,
    )
    parent1 = population[p1_idx]

    # Pre-extract non-empty client sets for Parent 1 routes
    p1_routes: list[set[int]] = []
    for route in parent1.routes():
        clients = set(v for v in route if v > 0)
        if clients:
            p1_routes.append(clients)

    # --- Parent 2 Selection: Quality + Route Diversity Tournament ---
    t2_size = min(4, n - 1)
    p2_candidates: list[int] = []
    while len(p2_candidates) < t2_size:
        idx = rng.randint(n)
        if idx != p1_idx and idx not in p2_candidates:
            p2_candidates.append(idx)

    def route_diversity_to_p1(cand_idx: int) -> float:
        if not p1_routes:
            return 1.0
        cand_routes: list[set[int]] = []
        for route in population[cand_idx].routes():
            clients = set(v for v in route if v > 0)
            if clients:
                cand_routes.append(clients)
        if not cand_routes:
            return 1.0

        total_sim = 0.0
        for r1 in p1_routes:
            max_jacc = 0.0
            for r2 in cand_routes:
                inter = len(r1 & r2)
                if inter > 0:
                    jacc = inter / len(r1 | r2)
                    if jacc > max_jacc:
                        max_jacc = jacc
                        if max_jacc == 1.0:
                            break
            total_sim += max_jacc
        return 1.0 - (total_sim / len(p1_routes))

    cand_costs = [get_effective_cost(i) for i in p2_candidates]
    min_c = min(cand_costs)
    max_c = max(cand_costs)
    c_range = max_c - min_c

    best_p2_score = -float("inf")
    p2_idx = p2_candidates[0]

    for i, idx in enumerate(p2_candidates):
        cost_score = 1.0 if c_range < 1e-6 else 1.0 - (cand_costs[i] - min_c) / c_range
        div_score = route_diversity_to_p1(idx)
        # Quality is primary (up to 1.0), diversity provides complementary boost (up to 0.45)
        total_score = cost_score + 0.45 * div_score + rng.rand() * 1e-6
        if total_score > best_p2_score:
            best_p2_score = total_score
            p2_idx = idx

    return parent1, population[p2_idx]