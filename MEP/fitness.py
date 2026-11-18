"""Evaluate an LLM-generated parent selector by running the author's HGS."""
from __future__ import annotations
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyvrp
from pyvrp.solve import SolveParams, solve
from pyvrp.stop import MaxRuntime
import config
from hgs_backend import install_selector_loader_patch


def load_vrp_instance(filename: str):
    path = Path(config.DATA_DIR) / filename
    if not path.is_file():
        raise FileNotFoundError(f"VRP instance not found: {path}")
    return pyvrp.read(str(path), round_func="trunc")


def evaluate_single_run(selector_code: str, data, seed: int, time_limit: float) -> float:
    """Run one genuine HGS instance with the candidate selector injected."""
    install_selector_loader_patch()
    selector_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as fh:
            fh.write(selector_code)
            selector_path = Path(fh.name)
        params = SolveParams(custom_selector_path=selector_path)
        result = solve(data, stop=MaxRuntime(time_limit), seed=seed,
                       collect_stats=False, display=False, params=params)
        return float(result.cost()) if result.is_feasible() else float("inf")
    finally:
        if selector_path is not None:
            selector_path.unlink(missing_ok=True)


def evaluate_heuristic(selector_code, instances: Optional[List[str]] = None) -> Tuple[float, Dict]:
    """Mean feasible HGS objective cost over configured instances and seeds."""
    if instances is None:
        instances = config.VRP_INSTANCES
    instance_means, details = [], {}
    for name in instances:
        print(f"    [HGS] Evaluating {name}...")
        data = load_vrp_instance(name)
        costs = []
        for run_idx in range(config.NUM_EVAL_RUNS):
            seed = config.BASE_RANDOM_SEED + run_idx
            cost = evaluate_single_run(selector_code, data, seed, config.HGS_TIME_LIMIT)
            costs.append(cost)
            print(f"      seed={seed}: objective={cost:,.2f}")
        finite = [value for value in costs if value != float("inf")]
        mean_cost = statistics.mean(finite) if finite else float("inf")
        instance_means.append(mean_cost)
        details[name] = {"costs": costs, "mean": mean_cost,
                         "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
                         "feasible_runs": len(finite), "total_runs": len(costs)}
    overall = (statistics.mean(instance_means)
               if all(v != float("inf") for v in instance_means) else float("inf"))
    return overall, details
