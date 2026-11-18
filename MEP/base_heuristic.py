"""
Paper-aligned baseline for select_parents.

This is a faithful, API-adapted reconstruction of the best selector in
Appendix C of Malik et al. (2026), "PyVRP+: LLM-Driven Metacognitive
Heuristic Evolution for Hybrid Genetic Search in Vehicle Routing Problems".

The only intentional compatibility adaptation is route-client extraction:
PyVRP versions may expose route elements as ScheduledActivity objects
rather than plain integers.
"""

BASE_HEURISTIC_CODE = '''def select_parents(population, rng, cost_evaluator, k=2):
    """
    Hybrid Adaptive Stratified Diversity-Pressure Parent Selector.

    REASON: Pure cost ranking over-exploits; uniform random selection wastes
    crossover budget. The design hypothesis is to select an elite feasible
    first parent, then choose a complementary second parent using structural
    dissimilarity and cost/feasibility signals.

    REFLECT: Stratification can become coarse for tiny populations, and the
    lightweight route-overlap distance is an approximation of full edge
    distance. The implementation includes fallbacks for empty strata.
    """
    n = len(population)
    if n < 2:
        raise ValueError("select_parents requires at least two solutions")
    if k != 2:
        raise ValueError("Only k=2 is supported")

    # Cache objective and feasibility once per call.
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]
    feasible = [sol.is_feasible() for sol in population]
    sorted_indices = sorted(range(n), key=lambda i: costs[i])

    # Cost strata: elite = top 1/6, mid = next through 2/3, tail = remainder.
    elite_cut = max(1, n // 6)
    mid_cut = max(elite_cut + 1, n * 2 // 3)
    elite_indices = sorted_indices[:elite_cut]
    mid_indices = sorted_indices[elite_cut:mid_cut]
    tail_indices = sorted_indices[mid_cut:]

    def tournament_select(indices, t_size=7, allow_infeasible_prob=0.2):
        if not indices:
            indices = sorted_indices
        if len(indices) <= t_size:
            candidates = list(indices)
        else:
            candidates = [
                indices[rng.randint(len(indices))]
                for _ in range(t_size)
            ]

        filtered = []
        for idx in candidates:
            if feasible[idx] or rng.rand() < allow_infeasible_prob:
                filtered.append(idx)
        if not filtered:
            filtered = candidates
        return min(filtered, key=lambda i: costs[i])

    parent1_idx = tournament_select(elite_indices)
    parent1 = population[parent1_idx]
    parent1_rank = sorted_indices.index(parent1_idx)

    # A strong feasible elite parent is paired with a more exploratory parent.
    if feasible[parent1_idx] and parent1_rank < elite_cut:
        parent2_strata = mid_indices + tail_indices
    else:
        parent2_strata = elite_indices + mid_indices
    if not parent2_strata:
        parent2_strata = sorted_indices

    # Sample without replacement, avoiding a duplicate parent where possible.
    max_sample_size = min(20 if n > 500 else 30, len(parent2_strata))
    sampled = []
    seen = set()
    attempts = 0
    while len(sampled) < max_sample_size and attempts < max_sample_size * 2:
        idx = parent2_strata[rng.randint(len(parent2_strata))]
        if idx != parent1_idx and idx not in seen:
            sampled.append(idx)
            seen.add(idx)
        attempts += 1

    if not sampled:
        sampled = [idx for idx in parent2_strata if idx != parent1_idx]
    if not sampled:
        sampled = [idx for idx in sorted_indices if idx != parent1_idx]
    if not sampled:
        # Defensive fallback; n >= 2 guarantees this should not be needed.
        return population[parent1_idx], population[(parent1_idx + 1) % n]

    # Normalize cost scores within the sampled candidate set.
    sampled_costs = [costs[i] for i in sampled]
    min_cost = min(sampled_costs)
    max_cost = max(sampled_costs)
    cost_range = max_cost - min_cost
    if cost_range == 0:
        cost_range = 1.0

    def encode_solution_simple(solution):
        all_clients = set()
        route_clients = []
        for route in solution.routes():
            clients = set()
            for activity in route:
                # PyVRP 0.14+ iterates Activity(type, idx), not client integers.
                # Older releases/forks may expose integer visits or a .client field.
                client_id = getattr(activity, "client", None)
                if not isinstance(client_id, int):
                    is_client = getattr(activity, "is_client", None)
                    if callable(is_client) and is_client():
                        client_id = getattr(activity, "idx", None)
                    elif isinstance(activity, int):
                        # Legacy route iteration used 0 as the depot sentinel.
                        client_id = activity if activity > 0 else None
                if isinstance(client_id, int) and client_id >= 0:
                    clients.add(client_id)
            route_clients.append(clients)
            all_clients.update(clients)
        return all_clients, route_clients

    def simple_structural_distance(enc1, enc2):
        all1, routes1 = enc1
        all2, routes2 = enc2
        union = len(all1 | all2)
        global_similarity = len(all1 & all2) / union if union else 0.0

        # Route order is not a stable identity; compare each route to its
        # best-overlapping route in the other solution (symmetric average).
        def directional_route_similarity(left, right):
            if not left:
                return 0.0
            scores = []
            for route_a in left:
                best = 0.0
                for route_b in right:
                    denom = len(route_a | route_b)
                    score = len(route_a & route_b) / denom if denom else 0.0
                    if score > best:
                        best = score
                scores.append(best)
            return sum(scores) / len(scores)

        route_similarity = 0.5 * (
            directional_route_similarity(routes1, routes2)
            + directional_route_similarity(routes2, routes1)
        )
        similarity = 0.7 * global_similarity + 0.3 * route_similarity
        return 1.0 - similarity

    if n > 500:
        # Large populations: avoid structural encoding.
        best_score = float("-inf")
        parent2_idx = None
        for idx in sampled:
            cost_score = 1.0 - (costs[idx] - min_cost) / cost_range
            feasibility_bonus = 1.0 if feasible[idx] else 0.0
            diversity_bonus = (
                0.5 if abs(costs[idx] - costs[parent1_idx]) > cost_range * 0.1
                else 0.0
            )
            score = (
                0.60 * cost_score
                + 0.30 * feasibility_bonus
                + 0.10 * diversity_bonus
            )
            score += (rng.rand() - 0.5) * 0.02
            if score > best_score:
                best_score, parent2_idx = score, idx
    else:
        # Small/medium populations: lazy structural encoding.
        parent1_encoding = encode_solution_simple(parent1)
        best_score = float("-inf")
        parent2_idx = None
        for idx in sampled:
            structural_distance = simple_structural_distance(
                parent1_encoding, encode_solution_simple(population[idx])
            )
            cost_score = 1.0 - (costs[idx] - min_cost) / cost_range
            feasibility_bonus = 1.0 if feasible[idx] else 0.0
            score = (
                0.55 * structural_distance
                + 0.30 * cost_score
                + 0.15 * feasibility_bonus
            )
            score += (rng.rand() - 0.5) * 0.02
            if score > best_score:
                best_score, parent2_idx = score, idx

    if parent2_idx is None:
        parent2_idx = tournament_select(parent2_strata)
    if parent2_idx == parent1_idx:
        alternatives = [i for i in range(n) if i != parent1_idx]
        parent2_idx = alternatives[0]

    return population[parent1_idx], population[parent2_idx]
'''

BASE_HEURISTIC_DESCRIPTION = (
    "Appendix-C Hybrid Adaptive Stratified Diversity-Pressure selector: "
    "cost strata, feasibility-biased tournament for parent 1, and adaptive "
    "cost/feasibility/structural-diversity scoring for parent 2."
)

# Additional transparent seed heuristics diversify the initial MEP population.
# These are implementation baselines, not claimed to be the authors' exact seeds.
RANDOM_HEURISTIC_CODE = '''def select_parents(population, rng, cost_evaluator, k=2):
    """Uniform random distinct-parent baseline."""
    n = len(population)
    if n < 2:
        raise ValueError("Need at least two solutions")
    if k != 2:
        raise ValueError("Only k=2 is supported")
    i = rng.randint(n)
    j = rng.randint(n - 1)
    if j >= i:
        j += 1
    return population[i], population[j]
'''

TOURNAMENT_HEURISTIC_CODE = '''def select_parents(population, rng, cost_evaluator, k=2):
    """Cost-biased tournament baseline."""
    n = len(population)
    if n < 2:
        raise ValueError("Need at least two solutions")
    if k != 2:
        raise ValueError("Only k=2 is supported")
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]
    chosen = []
    for _ in range(2):
        candidates = set()
        while len(candidates) < min(4, n):
            candidates.add(rng.randint(n))
        ranked = sorted(candidates, key=lambda i: costs[i])
        winner = ranked[0]
        if winner in chosen and len(ranked) > 1:
            winner = ranked[1]
        chosen.append(winner)
    if chosen[0] == chosen[1]:
        chosen[1] = (chosen[0] + 1) % n
    return population[chosen[0]], population[chosen[1]]
'''

INITIAL_HEURISTICS = [
    (BASE_HEURISTIC_CODE, BASE_HEURISTIC_DESCRIPTION),
    (TOURNAMENT_HEURISTIC_CODE, "Cost-biased tournament baseline."),
    (RANDOM_HEURISTIC_CODE, "Uniform random distinct-parent baseline."),
]
