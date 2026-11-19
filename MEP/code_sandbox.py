"""Syntax/signature validation for generated parent selectors.

This validator is not a security sandbox. The bundled HGS fork imports and
executes selector code in-process; only execute trusted LLM output.
"""
from __future__ import annotations

import ast
from typing import Optional, Tuple


def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error on line {exc.lineno}: {exc.msg}"

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "select_parents"
    ]
    if len(functions) != 1:
        return False, "Code must define exactly one module-level select_parents."

    args = functions[0].args
    names = [arg.arg for arg in args.args]
    if names != ["population", "rng", "cost_evaluator", "k"]:
        return False, "Expected select_parents(population, rng, cost_evaluator, k=2)."
    if len(args.defaults) != 1:
        return False, "Parameter k must have a default value."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name != "pyvrp._pyvrp" for alias in node.names):
                return False, "Only imports from pyvrp._pyvrp are allowed."
        elif isinstance(node, ast.ImportFrom):
            if node.module != "pyvrp._pyvrp":
                return False, "Only imports from pyvrp._pyvrp are allowed."

    return True, "OK"


def compile_heuristic(code: str) -> Tuple[Optional[object], str]:
    valid, message = validate_code(code)
    if not valid:
        return None, message
    namespace = {}
    try:
        exec(compile(code, "<mep-select-parents>", "exec"), namespace)
    except Exception as exc:
        return None, f"Compilation failed: {type(exc).__name__}: {exc}"
    function = namespace.get("select_parents")
    if not callable(function):
        return None, "select_parents is not callable."
    return function, "OK"


def quick_smoke_test(func, *args, **kwargs):
    return callable(func), "HGS invocation is verified by the integration smoke test."
