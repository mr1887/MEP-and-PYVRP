"""
MEP loop: LLM evolves select_parents; every candidate is evaluated by the
author's PyVRP HGS fork, not by a separately reimplemented GA.
"""
import os
import sys
import time
import datetime

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from base_heuristic import INITIAL_HEURISTICS
from prompt_builder import build_evolution_prompt
from llm_client import generate_heuristic
from code_sandbox import compile_heuristic, quick_smoke_test
from fitness import evaluate_heuristic, load_vrp_instance
from population import Population, Heuristic


def save_code_to_file(code: str, generation: int, index: int):
    os.makedirs(config.GENERATED_CODE_DIR, exist_ok=True)
    filepath = os.path.join(config.GENERATED_CODE_DIR, f"gen{generation:02d}_offspring{index:02d}.py")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(code)
    return filepath


def print_banner():
    print("""
+==============================================================+
|       LLM-DRIVEN PARENT SELECTOR EVOLUTION FOR VRP            |
|       Candidate evaluation: author's PyVRP HGS fork           |
+==============================================================+
""")


def run_evolution():
    print_banner()
    start_time = time.time()
    if not config.GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY in your environment.")
        return

    print("\n[Phase I] Checking VRP benchmark instances...")
    for name in config.VRP_INSTANCES:
        data = load_vrp_instance(name)
        print(f"  OK {name}: {data.num_clients} clients, {data.num_vehicles} vehicles")

    print("\n[Phase I] Evaluating initial parent selectors with HGS...")
    pop = Population(max_size=config.POPULATION_SIZE)
    for seed_idx, (seed_code, description) in enumerate(INITIAL_HEURISTICS):
        func, error = compile_heuristic(seed_code)
        if func is None:
            print(f"  Seed {seed_idx + 1} rejected: {error}")
            continue
        print(f"  Seed {seed_idx + 1}/{len(INITIAL_HEURISTICS)}: running HGS...")
        try:
            fitness, details = evaluate_heuristic(seed_code)
        except Exception as exc:
            print(f"  Seed {seed_idx + 1} HGS evaluation failed: {exc}")
            continue
        if fitness == float("inf"):
            print(f"  Seed {seed_idx + 1}: no feasible HGS run; skipped.")
            continue
        pop.add(Heuristic(code=seed_code, fitness=fitness, generation_born=0,
                          description=description, eval_details=details))
        print(f"  Seed {seed_idx + 1}: mean HGS cost = {fitness:,.2f}")

    if pop.size() < 2:
        raise RuntimeError(
            "Fewer than two valid seed selectors. Check the author-fork installation, "
            "selector hook, VRP data and HGS runtime."
        )

    print(f"\nInitial selector population: {pop.size()} candidates")
    context_notes = ""
    print(f"\n[Phase II] {config.NUM_GENERATIONS} generations × "
          f"{config.OFFSPRING_PER_GENERATION} offspring; "
          f"HGS={config.HGS_TIME_LIMIT}s × {config.NUM_EVAL_RUNS} seeds × "
          f"{len(config.VRP_INSTANCES)} instances")

    for gen in range(1, config.NUM_GENERATIONS + 1):
        gen_start = time.time()
        print(f"\n{'=' * 58}\nGENERATION {gen}/{config.NUM_GENERATIONS}\n{'=' * 58}")
        offspring_added = 0
        for idx in range(config.OFFSPRING_PER_GENERATION):
            print(f"\n[Gen {gen}] Offspring {idx + 1}/{config.OFFSPRING_PER_GENERATION}")
            p1, p2 = pop.select_two_parents(rng_seed=gen * 100 + idx)
            prompt = build_evolution_prompt(
                parent1_code=p1.code, parent1_fitness=p1.fitness,
                parent2_code=p2.code, parent2_fitness=p2.fitness,
                generation=gen, best_fitness_so_far=pop.get_best_fitness(),
                context_notes=context_notes,
            )
            code, reflection = generate_heuristic(prompt)
            if not code:
                print("  LLM did not return a candidate.")
                context_notes = "Previous LLM call failed to produce valid code."
                continue
            path = save_code_to_file(code, gen, idx)
            func, error = compile_heuristic(code)
            if func is None:
                print(f"  Candidate rejected: {error}")
                context_notes = f"Previous candidate failed validation: {error}"
                continue
            try:
                fitness, details = evaluate_heuristic(code)
            except Exception as exc:
                print(f"  HGS evaluation failed: {exc}")
                context_notes = f"Previous candidate failed during HGS evaluation: {exc}"
                continue
            if fitness == float("inf"):
                print("  Candidate produced no feasible HGS run; not added.")
                context_notes = "No feasible HGS result; reconsider feasibility handling."
                continue

            previous_best = pop.get_best_fitness()
            pop.add(Heuristic(code=code, fitness=fitness, generation_born=gen,
                              description=(reflection or f"Gen {gen} offspring {idx}")[:200],
                              eval_details=details))
            offspring_added += 1
            context_notes = (
                f"Mean HGS objective {fitness:.3f}; "
                f"previous best {previous_best:.3f}. "
                f"Reflection: {(reflection or '')[:200]}"
            )
            print(f"  HGS mean objective = {fitness:,.2f}")

        if offspring_added:
            pop.select_survivors()
        print(pop.summary())
        print(f"Generation runtime: {time.time() - gen_start:.1f}s")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        pop.save_to_file(os.path.join(config.RESULTS_DIR, "checkpoint.json"))

    best = pop.get_best()
    elapsed = time.time() - start_time
    print("\nBEST SELECTOR")
    print(f"Mean HGS objective: {best.fitness:,.2f}")
    print(f"Generation born: {best.generation_born}")
    print(f"Description: {best.description}")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "best_select_parents.py"), "w", encoding="utf-8") as file:
        file.write(best.code)
    pop.save_to_file(os.path.join(config.RESULTS_DIR, "final_population.json"))
    print(f"Completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return best


if __name__ == "__main__":
    run_evolution()
