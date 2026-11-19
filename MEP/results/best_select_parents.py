def select_parents(population, rng, cost_evaluator, k=2):
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
