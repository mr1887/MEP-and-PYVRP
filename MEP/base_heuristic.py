"""Reference implementation transcribed from Appendix C (Figures 4–7).

The selector follows the paper's algorithm and coefficients. The only
compatibility note is that this PyVRP fork uses legacy integer route visits,
where 0 denotes the depot; the appendix's v > 0 extraction is preserved.
"""
BASE_HEURISTIC_CODE = '''from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution

def select_parents(population: list[Solution], rng: RandomNumberGenerator,
                   cost_evaluator: CostEvaluator, k: int = 2) -> tuple[Solution, Solution]:
    """
    Hybrid Adaptive Stratified Diversity-Pressure Parent Selector.
    Appendix C: cost/feasibility caching, 1/6–2/3–1/3 strata, feasibility-
    biased tournament for parent 1, complementary strata for parent 2, and
    adaptive diversity/cost/feasibility scoring.
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."
    assert k == 2, "Only k=2 supported."
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]
    feasible = [sol.is_feasible() for sol in population]
    sorted_indices = sorted(range(n), key=lambda i: costs[i])
    elite_cut = max(1, n // 6)
    mid_cut = max(elite_cut + 1, n * 2 // 3)
    elite_indices = sorted_indices[:elite_cut]
    mid_indices = sorted_indices[elite_cut:mid_cut]
    tail_indices = sorted_indices[mid_cut:]

    def encode_solution_simple(solution: Solution):
        all_clients = set()
        route_clients = []
        for route in solution.routes():
            clients_in_route = set()
            for v in route:
                if hasattr(v, "is_client"):
                    if v.is_client():
                        idx = getattr(v, "idx", None)
                        if idx is None and hasattr(v, "client"):
                            idx = v.client
                        if idx is not None and idx > 0:
                            clients_in_route.add(idx)
                elif hasattr(v, "idx"):
                    if v.idx > 0:
                        clients_in_route.add(v.idx)
                elif isinstance(v, (int, float)):
                    if v > 0:
                        clients_in_route.add(int(v))
            route_clients.append(clients_in_route)
            all_clients.update(clients_in_route)
        return all_clients, route_clients

    def simple_structural_distance(enc1, enc2):
        all_clients1, route_clients1 = enc1
        all_clients2, route_clients2 = enc2
        if not all_clients1 and not all_clients2:
            return 0.0
        intersection = len(all_clients1 & all_clients2)
        union = len(all_clients1 | all_clients2)
        if union == 0:
            return 0.0
        global_similarity = intersection / union
        min_routes = min(len(route_clients1), len(route_clients2))
        if min_routes == 0:
            return 1.0 - global_similarity
        route_overlaps = []
        for i in range(min_routes):
            if not route_clients1[i] and not route_clients2[i]:
                continue
            r_intersection = len(route_clients1[i] & route_clients2[i])
            r_union = len(route_clients1[i] | route_clients2[i])
            if r_union > 0:
                route_overlaps.append(r_intersection / r_union)
        if route_overlaps:
            avg_route_similarity = sum(route_overlaps) / len(route_overlaps)
        else:
            avg_route_similarity = 0.0
        combined_similarity = 0.7 * global_similarity + 0.3 * avg_route_similarity
        return 1.0 - combined_similarity

    def tournament_select(indices, t_size=7, allow_infeasible_prob=0.2):
        if len(indices) <= t_size:
            candidates = indices[:]
        else:
            candidates = [indices[rng.randint(len(indices))] for _ in range(t_size)]
        filtered = []
        for idx in candidates:
            if feasible[idx]:
                filtered.append(idx)
            elif rng.rand() < allow_infeasible_prob:
                filtered.append(idx)
        if not filtered:
            filtered = candidates
        return min(filtered, key=lambda i: costs[i])

    parent1_idx = tournament_select(elite_indices)
    parent1 = population[parent1_idx]
    parent1_feas = feasible[parent1_idx]
    p1_cost_rank = sorted_indices.index(parent1_idx)
    if parent1_feas and p1_cost_rank < elite_cut:
        parent2_strata = mid_indices + tail_indices
    else:
        parent2_strata = elite_indices + mid_indices
    if not parent2_strata:
        parent2_strata = sorted_indices

    max_sample_size = min(20, len(parent2_strata)) if n > 500 else min(30, len(parent2_strata))
    sampled_parent2_indices = []
    seen = set()
    attempts = 0
    while len(sampled_parent2_indices) < max_sample_size and attempts < max_sample_size * 2:
        candidate = parent2_strata[rng.randint(len(parent2_strata))]
        if candidate != parent1_idx and candidate not in seen:
            sampled_parent2_indices.append(candidate)
            seen.add(candidate)
        attempts += 1
    if not sampled_parent2_indices:
        for idx in parent2_strata:
            if idx != parent1_idx:
                sampled_parent2_indices.append(idx)
                break
        else:
            sampled_parent2_indices.append(parent2_strata[0])

    if n > 500:
        best_score = -float('inf')
        parent2_idx = None
        p2_costs = [costs[i] for i in sampled_parent2_indices]
        p2_min_cost = min(p2_costs)
        p2_max_cost = max(p2_costs) if max(p2_costs) > p2_min_cost else p2_min_cost + 1
        for idx in sampled_parent2_indices:
            if idx == parent1_idx:
                continue
            cand_cost = costs[idx]
            cand_feas = feasible[idx]
            cost_score = 1.0 - (cand_cost - p2_min_cost) / (p2_max_cost - p2_min_cost)
            feas_bonus = 1.0 if cand_feas else 0.0
            diversity_bonus = 0.5 if abs(costs[idx] - costs[parent1_idx]) > (p2_max_cost - p2_min_cost) * 0.1 else 0.0
            score = 0.6 * cost_score + 0.3 * feas_bonus + 0.1 * diversity_bonus
            rand_val = rng.rand() if hasattr(rng, "rand") else (rng.random() if hasattr(rng, "random") else 0.5)
            score += (rand_val - 0.5) * 0.02
            if score > best_score:
                best_score, parent2_idx = score, idx
    else:
        p1_enc = encode_solution_simple(population[parent1_idx])
        p2_costs = [costs[i] for i in sampled_parent2_indices]
        p2_min_cost = min(p2_costs)
        p2_max_cost = max(p2_costs) if max(p2_costs) > p2_min_cost else p2_min_cost + 1
        w_struct, w_cost, w_feas = 0.55, 0.3, 0.15
        best_score = -float('inf')
        parent2_idx = None
        for idx in sampled_parent2_indices:
            if idx == parent1_idx:
                continue
            cand_cost = costs[idx]
            cand_feas = feasible[idx]
            cand_enc = encode_solution_simple(population[idx])
            struct_dist = simple_structural_distance(p1_enc, cand_enc)
            cost_score = 1.0 - (cand_cost - p2_min_cost) / (p2_max_cost - p2_min_cost)
            feas_bonus = 1.0 if cand_feas else 0.0
            score = w_struct * struct_dist + w_cost * cost_score + w_feas * feas_bonus
            rand_val = rng.rand() if hasattr(rng, "rand") else (rng.random() if hasattr(rng, "random") else 0.5)
            score += (rand_val - 0.5) * 0.02
            if score > best_score:
                best_score, parent2_idx = score, idx

    if parent2_idx is None:
        parent2_idx = tournament_select(parent2_strata)
    return population[parent1_idx], population[parent2_idx]
'''

BASE_HEURISTIC_DESCRIPTION = (
    "Appendix C reference selector (Figures 4–7), transcribed with the "
    "paper's strata, tournament, sampling limits, weights and route encoding."
)

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
