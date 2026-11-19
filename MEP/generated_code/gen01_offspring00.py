def select_parents(population, rng, cost_evaluator, k=2):
    """Select two parents balancing exploitation, feasibility, and structural diversity.

    Phase II — Design Hypothesis:
    Parent 1 selection uses a cost-and-feasibility tournament to ensure high-quality
    exploitation while permitting controlled infeasible selection (probabilistic bypass).
    Parent 2 is chosen from a second tournament scored by biased rank aggregation
    (combining cost rank and route-structure Jaccard diversity relative to Parent 1).
    This structural diversification prevents mating near-identical solutions and avoids
    wasting crossover operations without requiring expensive all-pairs comparisons.

    Phase IV — Reflection:
    - Kp/Kt Addressed:
      * Mitigates Kp.1 (diversity loss) by rewarding route-level Jaccard diversity in Parent 2.
      * Mitigates Kp.2 (random selection waste) by retaining tournament selection pressure.
      * Mitigates Kp.3 / Ks.3 (feasibility conflict) via probabilistic feasibility filtering
        with penalised cost tie-breaking, gracefully handling empty strata.
      * Respects Kt.1/Kt.2/Kt.3: operates on route client sets via `set(v for v in route if v > 0)`
        without full-population O(N^2) comparisons.
    - Remaining limitation / failure mode:
      * Route set Jaccard similarity measures customer grouping within routes but does not
        account for sequence/edge ordering inside the routes or sub-tours.
    - Concrete follow-up revisions:
      1. Incorporate directed edge (broken pairs) difference alongside route-set Jaccard overlap.
      2. Dynamically adapt the diversity weight lambda based on the solver's current search phase
         or population diversity metric.
    """
    n = len(population)
    if n < 2:
        raise ValueError("Need at least two solutions")
    if k != 2:
        raise ValueError("Only k=2 is supported")

    cost_cache = {}
    feas_cache = {}
    route_cache = {}

    def get_cost(idx):
        if idx not in cost_cache:
            cost_cache[idx] = cost_evaluator.penalised_cost(population[idx])
        return cost_cache[idx]

    def get_feas(idx):
        if idx not in feas_cache:
            feas_cache[idx] = population[idx].is_feasible()
        return feas_cache[idx]

    def get_routes(idx):
        if idx not in route_cache:
            sol = population[idx]
            route_cache[idx] = [
                set(v for v in route if v > 0)
                for route in sol.routes()
            ]
            route_cache[idx] = [r for r in route_cache[idx] if len(r) > 0]
        return route_cache[idx]

    def sample_tournament(pool_size, count, exclude=None):
        avail = pool_size - (1 if exclude is not None and 0 <= exclude < pool_size else 0)
        target = min(count, avail)
        selected = []
        while len(selected) < target:
            idx = rng.randint(pool_size)
            if idx != exclude and idx not in selected:
                selected.append(idx)
        return selected

    def compute_diversity(idx1, idx2):
        r1 = get_routes(idx1)
        r2 = get_routes(idx2)
        if not r1 or not r2:
            return 1.0
        r1_lens = [len(r) for r in r1]
        total_sim = 0.0
        for cr in r2:
            len_cr = len(cr)
            if len_cr == 0:
                continue
            max_sim = 0.0
            for pr, len_pr in zip(r1, r1_lens):
                inter = len(cr & pr)
                if inter > 0:
                    sim = inter / (len_cr + len_pr - inter)
                    if sim > max_sim:
                        max_sim = sim
                        if max_sim >= 0.999:
                            break
            total_sim += max_sim
        return 1.0 - (total_sim / len(r2))

    # Tournament 1 for Parent 1
    t1_size = min(3, n)
    t1_candidates = sample_tournament(n, t1_size)
    allow_infeasible_p1 = rng.rand() < 0.20

    def p1_rank_key(idx):
        cost = get_cost(idx)
        if allow_infeasible_p1:
            return (0, cost)
        return (0 if get_feas(idx) else 1, cost)

    p1_idx = min(t1_candidates, key=p1_rank_key)

    # Tournament 2 for Parent 2 (distinct index, diversity-aware)
    t2_size = min(4, n - 1)
    t2_candidates = sample_tournament(n, t2_size, exclude=p1_idx)
    allow_infeasible_p2 = rng.rand() < 0.20

    def p2_cost_key(idx):
        cost = get_cost(idx)
        if allow_infeasible_p2:
            return (0, cost)
        return (0 if get_feas(idx) else 1, cost)

    sorted_by_cost = sorted(t2_candidates, key=p2_cost_key)
    cost_ranks = {idx: rank for rank, idx in enumerate(sorted_by_cost)}

    diversities = {idx: compute_diversity(p1_idx, idx) for idx in t2_candidates}
    sorted_by_div = sorted(t2_candidates, key=lambda idx: diversities[idx], reverse=True)
    div_ranks = {idx: rank for rank, idx in enumerate(sorted_by_div)}

    def combined_rank_score(idx):
        # Balanced rank sum: exploitation (cost) + exploration (diversity relative to p1)
        return (cost_ranks[idx] + div_ranks[idx], get_cost(idx))

    p2_idx = min(t2_candidates, key=combined_rank_score)

    # Ensure distinct objects if duplicates exist in population
    if population[p2_idx] is population[p1_idx]:
        for idx in range(n):
            if population[idx] is not population[p1_idx]:
                p2_idx = idx
                break

    return population[p1_idx], population[p2_idx]