"""Runtime adapter for the custom selector hook in the author's PyVRP fork.

The fork's original loader rejects all imports and omits common builtins used
by the paper's Appendix-C selector. This patch changes only selector loading;
HGS still controls the population, crossover, local search and survivor logic.
"""
from __future__ import annotations
import ast
from pathlib import Path


def install_selector_loader_patch():
    try:
        from pyvrp.Population import Population as HgsPopulation
        from Pyvrp.pyvrp.cpp import CostEvaluator, RandomNumberGenerator, Solution
    except Exception as exc:
        raise RuntimeError(
            "Installed pyvrp is not the author's HGS fork. Install requirements.txt."
        ) from exc

    def load_custom_selector(self, path):
        path = Path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        funcs = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "select_parents"]
        if len(funcs) != 1:
            raise ValueError("Selector file must define exactly one select_parents function.")
        # Restrict imports to the PyVRP namespace. This is compatibility
        # validation, NOT a security sandbox: Python exec is not secure.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.split(".")[0] != "pyvrp" for alias in node.names):
                    raise ValueError("Only imports from pyvrp are allowed.")
            elif isinstance(node, ast.ImportFrom):
                if not node.module or node.module.split(".")[0] != "pyvrp":
                    raise ValueError("Only imports from pyvrp are allowed.")
        namespace = {"__builtins__": __builtins__, "Solution": Solution,
                     "CostEvaluator": CostEvaluator,
                     "RandomNumberGenerator": RandomNumberGenerator}
        exec(compile(tree, str(path), "exec"), namespace)
        func = namespace.get("select_parents")
        if not callable(func):
            raise ValueError("Selector module did not define callable select_parents.")
        import inspect
        expected = ["population", "rng", "cost_evaluator", "k"]
        if list(inspect.signature(func).parameters) != expected:
            raise ValueError(f"Expected signature {expected}; got {inspect.signature(func)}")
        return func

    HgsPopulation._load_custom_selector = load_custom_selector
    return HgsPopulation
