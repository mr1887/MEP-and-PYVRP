from __future__ import annotations

import ast
import importlib
import inspect
import sys
import random
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator, Optional

from pyvrp._pyvrp import PopulationParams, SubPopulation

# if TYPE_CHECKING:
from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution


class Population:
    """
    Creates a Population instance.

    Parameters
    ----------
    diversity_op
        Operator to use to determine pairwise diversity between solutions. Have
        a look at :mod:`pyvrp.diversity` for available operators.
    params
        Population parameters. If not provided, a default will be used.
    custom_selector
        Path to Python module containing LLM-generated selection logic.
    """

    def __init__(
        self,
        diversity_op: Callable[[Solution, Solution], float],
        params: PopulationParams | None = None,
        custom_selector: Optional[Path] = None,
    ):
        self._op = diversity_op
        self._params = params if params is not None else PopulationParams()

        self._selector = self._default_selector
        if custom_selector:
            self._selector = self._load_custom_selector(custom_selector)

        self._feas = SubPopulation(diversity_op, self._params)
        self._infeas = SubPopulation(diversity_op, self._params)
        
    def _load_custom_selector(self, path: Path) -> Callable:
        """Load and validate LLM-generated selection function."""
        module = self._safe_import(path)
        
        if not hasattr(module, "select_parents"):
            raise ValueError("LLM module must contain 'select_parents' function")

        func = module.select_parents
        self._validate_signature(func)
        return func

    def _validate_signature(self, func: Callable):
        """Ensure function has correct signature."""
        sig = inspect.signature(func)
        expected = [
            "population", 
            "rng", 
            "cost_evaluator",
            "k"
        ]
        
        if list(sig.parameters.keys()) != expected:
            raise ValueError(f"Invalid signature. Expected {expected}")

    def _safe_import(self, path: Path) -> ModuleType:
        """Safely import module with restricted environment."""
        spec = importlib.util.spec_from_file_location("llm_selector", path)
        module = importlib.util.module_from_spec(spec)
        
        # Restricted environment
        restricted_globals = {
            "__builtins__": {
                "min": min,
                "max": max,
                "len": len,
                "range": range,
                "sorted": sorted,
                "Exception": Exception,
                "list": list,  # Add built-in list type
                "tuple": tuple,  # Add other needed types
                "type": type,
                "int": int,
                "random": random,
            },
            "Solution": Solution,
            "CostEvaluator": CostEvaluator,
            "RandomNumberGenerator": RandomNumberGenerator,
        }
        
        module.__dict__.update(restricted_globals)
        self._validate_ast(path.read_text())
        
        spec.loader.exec_module(module)
        return module

    def _validate_ast(self, code: str):
        """Ensure code contains only allowed constructs."""
        tree = ast.parse(code)

        allowed_builtins = (ast.Name, ast.Attribute, ast.Subscript)
        prohibited = (
            ast.Import, ast.ImportFrom,
            # ast.Try, ast.While, ast.With,
            # ast.ListComp, ast.Global, ast.Nonlocal,
        )
        
        for node in ast.walk(tree):
            if isinstance(node, prohibited):
                raise ValueError(f"Prohibited construct: {type(node).__name__}")

            # # Allow subscript annotations like list[Solution]
            # if isinstance(node, ast.Subscript):
            #     if not isinstance(node.value, allowed_builtins):
            #         raise ValueError("Complex type annotations prohibited")

    def __iter__(self) -> Generator[Solution, None, None]:
        """
        Iterates over the solutions contained in this population.
        """
        for item in self._feas:
            yield item.solution

        for item in self._infeas:
            yield item.solution

    def __len__(self) -> int:
        """
        Returns the current population size.
        """
        return len(self._feas) + len(self._infeas)

    def _update_fitness(self, cost_evaluator: CostEvaluator):
        """
        Updates the biased fitness values for the subpopulations.

        Parameters
        ----------
        cost_evaluator
            CostEvaluator to use for computing the fitness.
        """
        self._feas.update_fitness(cost_evaluator)
        self._infeas.update_fitness(cost_evaluator)

    def num_feasible(self) -> int:
        """
        Returns the number of feasible solutions in the population.
        """
        return len(self._feas)

    def num_infeasible(self) -> int:
        """
        Returns the number of infeasible solutions in the population.
        """
        return len(self._infeas)

    def add(self, solution: Solution, cost_evaluator: CostEvaluator):
        """
        Inserts the given solution in the appropriate feasible or infeasible
        (sub)population.

        .. note::

           Survivor selection is automatically triggered when the subpopulation
           reaches its maximum size, given by
           :attr:`~pyvrp.Population.PopulationParams.max_pop_size`.

        Parameters
        ----------
        solution
            Solution to add to the population.
        cost_evaluator
            CostEvaluator to use to compute the cost.
        """
        # Note: the CostEvaluator is required here since adding a solution
        # may trigger a purge which needs to compute the biased fitness which
        # requires computing the cost.
        if solution.is_feasible():
            # Note: the feasible subpopulation actually does not depend
            # on the penalty values but we use the same implementation.
            self._feas.add(solution, cost_evaluator)
        else:
            self._infeas.add(solution, cost_evaluator)

    def clear(self):
        """
        Clears the population by removing all solutions currently in the
        population.
        """
        self._feas = SubPopulation(self._op, self._params)
        self._infeas = SubPopulation(self._op, self._params)

    def tournament(
        self,
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int = 2,
    ) -> Solution:
        """
        Selects a solution from this population by k-ary tournament, based
        on the (internal) fitness values of the selected solutions.

        Parameters
        ----------
        rng
            Random number generator.
        cost_evaluator
            Cost evaluator to use when computing the fitness.
        k
            The number of solutions to draw for the tournament. Defaults to
            two, which results in a binary tournament.

        Returns
        -------
        Solution
            The selected solution.
        """
        self._update_fitness(cost_evaluator)
        return self._tournament(rng, k)

    def _tournament(self, rng: RandomNumberGenerator, k: int) -> Solution:
        if k <= 0:
            raise ValueError(f"Expected k > 0; got k = {k}.")

        def select():
            num_feas = len(self._feas)
            idx = rng.randint(len(self))

            if idx < num_feas:
                return self._feas[idx]

            return self._infeas[idx - num_feas]

        items = [select() for _ in range(k)]
        fittest = min(items, key=lambda item: item.fitness)
        return fittest.solution

    def _default_selector(
        self, 
        rng: RandomNumberGenerator, 
        cost_evaluator: CostEvaluator, 
        k: int = 2
    ) -> tuple[Solution, Solution]:
        """Original tournament selection logic."""
        """
        Selects two (if possible non-identical) parents by tournament, subject
        to a diversity restriction.

        Parameters
        ----------
        rng
            Random number generator.
        cost_evaluator
            Cost evaluator to use when computing the fitness.
        k
            The number of solutions to draw for the tournament. Defaults to
            two, which results in a binary tournament.

        Returns
        -------
        tuple
            A solution pair (parents).
        """
        self._update_fitness(cost_evaluator)
        first = self._tournament(rng, k)
        second = self._tournament(rng, k)

        diversity = self._op(first, second)
        lb = self._params.lb_diversity
        ub = self._params.ub_diversity

        tries = 1
        while not (lb <= diversity <= ub) and tries <= 10:
            tries += 1
            second = self._tournament(rng, k)
            diversity = self._op(first, second)

        return first, second

    def select(
        self,
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int = 2,
    ) -> tuple[Solution, Solution]:
        """Main entry point for parent selection."""
        if self._selector == self._default_selector:
            return self._default_selector(rng, cost_evaluator, k)
        
        # For LLM selector: combine feasible/infeasible populations
        population = list(self)
        return self._selector(population, rng, cost_evaluator, k)