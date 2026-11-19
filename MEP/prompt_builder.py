"""
Structured MEP prompt for evolving the HGS select_parents component.

The sequence follows Appendix B of Malik et al. (2026):
Role/Planning -> Context -> Reasoning -> Implementation -> Reflection.
The final response is constrained to one extractable Python code block.
"""

def build_evolution_prompt(
    parent1_code: str,
    parent1_fitness: float,
    parent2_code: str,
    parent2_fitness: float,
    generation: int,
    best_fitness_so_far: float,
    context_notes: str = "",
) -> str:
    """Build the Reason-Act-Reflect prompt for one offspring heuristic."""

    return f"""
ROLE
You are an AI optimization researcher specializing in Vehicle Routing
Problems and Hybrid Genetic Search (HGS). You are evolving ONLY the
parent-selection component used to choose two Solution objects before
crossover. Do not redesign the rest of HGS.

PHASE I — DOMAIN-AWARE PLANNING
Before inspecting the parent code, explicitly consider:
- Kp (common pitfalls):
  1. Excessive selection pressure causes loss of population diversity.
  2. Uniform random selection ignores solution quality and wastes crossover.
  3. Feasibility and penalised cost can conflict; filtering all infeasible
     solutions may remove useful diversity.
- Ks (mitigation strategies):
  1. Combine partial elitism with diversity-aware selection.
  2. Use bounded tournaments or cost strata rather than always taking the
     two lowest-cost solutions.
  3. Cache penalised costs and feasibility; allow controlled infeasible
     candidates where appropriate.
- Kt (VRP-specific traps):
  1. Solutions are route structures, not flat numeric chromosomes.
  2. Route iteration APIs can expose activity objects rather than integers.
  3. Small populations can produce empty strata; large populations make
     pairwise structural comparison expensive.
Use these as strategic guidance, not as claims that a candidate will improve.

CONTEXT
The target is a Python function integrated into an HGS solver. The function
selects two existing VRP solutions for crossover. Lower objective cost is
better, but parent selection must balance exploitation and exploration.

Parent selector 1:
Current mean cost/fitness = {parent1_fitness:.6f}
```python
{parent1_code.strip()}
```

Parent selector 2:
Current mean cost/fitness = {parent2_fitness:.6f}
```python
{parent2_code.strip()}
```

Evolution state:
- Generation: {generation}
- Best fitness observed: {best_fitness_so_far:.6f}
- Additional evaluation feedback: {context_notes or "No additional feedback."}

PHASE II — REASON
Analyze both parents:
1. Identify their exploration/exploitation biases.
2. Explain their weaknesses against Kp and Kt.
3. State one concise, testable design hypothesis for the offspring.
Do not generate code until the diagnosis and hypothesis are clear.

PHASE III — ACT
Implement a new selector with exactly this signature:
def select_parents(population, rng, cost_evaluator, k=2):

Requirements:
- population contains PyVRP Solution objects.
- rng is PyVRP RandomNumberGenerator; use rng.randint(n) for [0,n)
  and rng.rand() for a random float in [0,1).
- cost_evaluator.penalised_cost(solution) measures penalised solution cost.
- Return exactly a tuple (parent1, parent2) containing two objects from
  the input population. Select distinct objects whenever n >= 2.
- Do not mutate population or its solutions.
- Prioritize feasible solutions but permit controlled infeasible selection.
- Account for small populations and empty strata.
- Cache repeated costs/feasibility checks.
- Keep selection overhead low; avoid all-pairs comparisons.
- Use only Python built-ins and the supplied PyVRP objects.
- Do not redefine PyVRP classes or add external dependencies.
- This repository pins the author's PyVRP fork.
  * To access routes, always call `solution.routes()` (a Solution object is NOT directly iterable and does NOT have `get_routes()`).
  * Each `route` in `solution.routes()` can be iterated over.
  * Its route iterator yields integer visits and uses 0 as the depot sentinel.
    For route encoding, use `set(v for v in route if v > 0)`. Do not alter the client indexing.
- Do not assume a candidate improves fitness merely because its design
  sounds sophisticated.

PHASE IV — REFLECT
Put a concise reflection in the generated function's docstring:
- Which Kp/Kt pitfalls does the code address?
- What limitation or new failure mode remains?
- Name two concrete follow-up revisions.

OUTPUT CONTRACT
Return exactly ONE Python fenced code block containing only:
1. Necessary imports from pyvrp._pyvrp, if needed.
2. The complete select_parents function.
Do not put prose, REASONING labels, or REFLECTION labels outside the code
block. Include the design hypothesis and reflection in the function docstring.
The code must be directly extractable and executable.
""".strip()
