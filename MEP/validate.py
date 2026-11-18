"""Validation and compilation for LLM-generated HGS parent selectors.

exec() is not a security sandbox. Only execute code from a trusted workflow.
The definitive integration check is performed by invoking the selector in HGS.
"""
from __future__ import annotations
import ast
import traceback
from typing import Optional, Tuple


def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error on line {exc.lineno}: {exc.msg}"

    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "select_parents"]
    if len(funcs) != 1:
        return False, "Code must define exactly one module-level select_parents function."
    args = funcs[0].args
    names = [arg.arg for arg in args.args]
    if names != ["population", "rng", "cost_evaluator", "k"]:
        return False, "Signature must be select_parents(population, rng, cost_evaluator, k=2)."
    if len(args.defaults) != 1:
        return False, "The k parameter must have exactly one default value."

    # Imports execute at module-load time, so reject non-PyVRP imports here
    # as well as in the HGS loader. This is validation, not a security boundary.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] != "pyvrp" for alias in node.names):
                return False, "Only imports from pyvrp are allowed."
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] != "pyvrp":
                return False, "Only imports from pyvrp are allowed."
    return True, "OK"


def compile_heuristic(code: str) -> Tuple[Optional[object], str]:
    valid, error = validate_code(code)
    if not valid:
        return None, error
    try:
        namespace = {"__builtins__": __builtins__}
        exec(compile(code, "<llm-select-parents>", "exec"), namespace)
        func = namespace.get("select_parents")
        return (func, "OK") if callable(func) else (None, "select_parents is not callable.")
    except Exception:
        return None, traceback.format_exc()


def quick_smoke_test(func, *args, **kwargs):
    """The definitive smoke test is a real HGS evaluation in fitness.py."""
    return (callable(func),
            "Callable check passed; actual HGS invocation occurs during evaluation.")
