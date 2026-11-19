"""Evaluate a candidate selector by running the author's unmodified HGS."""
from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyvrp
from pyvrp.solve import SolveParams, solve
from pyvrp.stop import MaxRuntime

import config


def load_vrp_instance(filename: str):
    path = Path(config.DATA_DIR) / filename
    if not path.is_file():
        raise FileNotFoundError(f"VRP instance not found: {path}")
    return pyvrp.read(str(path), round_func="trunc")


def evaluate_single_run(selector_code: str, data, seed: int, time_limit: float) -> float:
    """Run HGS with the candidate passed through SolveParams.custom_selector."""
    selector_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8", delete=False
        ) as fh:
            fh.write(selector_code)
            selector_path = Path(fh.name)

        params = SolveParams(custom_selector_path=selector_path)
        result = solve(
            data,
            stop=MaxRuntime(time_limit),
            seed=seed,
            collect_stats=False,
            display=False,
            params=params,
        )
        return float(result.cost()) if result.is_feasible() else float("inf")
    finally:
        if selector_path is not None:
            selector_path.unlink(missing_ok=True)


def evaluate_heuristic(
    selector_code: str, instances: Optional[List[str]] = None
) -> Tuple[float, Dict]:
    """Return mean feasible objective cost over instances and independent seeds."""
    if instances is None:
        instances = config.VRP_INSTANCES

    per_instance_means = []
    details = {}
    for name in instances:
        print(f"    [HGS] Evaluating {name}...")
        data = load_vrp_instance(name)
        costs = []
        for run_idx in range(config.NUM_EVAL_RUNS):
            seed = config.BASE_RANDOM_SEED + run_idx
            value = evaluate_single_run(
                selector_code, data, seed, config.HGS_TIME_LIMIT
            )
            costs.append(value)
            shown = "infeasible" if value == float("inf") else f"{value:,.2f}"
            print(f"      seed={seed}: objective={shown}")

        feasible_costs = [value for value in costs if value != float("inf")]
        mean_cost = (
            statistics.mean(feasible_costs)
            if len(feasible_costs) == len(costs)
            else float("inf")
        )
        per_instance_means.append(mean_cost)
        details[name] = {
            "costs": costs,
            "mean": mean_cost,
            "std": statistics.stdev(feasible_costs)
            if len(feasible_costs) > 1 else 0.0,
            "feasible_runs": len(feasible_costs),
            "total_runs": len(costs),
        }

    # Equal weight per instance, matching an average over benchmark instances.
    overall = (
        statistics.mean(per_instance_means)
        if all(value != float("inf") for value in per_instance_means)
        else float("inf")
    )
    return overall, details
