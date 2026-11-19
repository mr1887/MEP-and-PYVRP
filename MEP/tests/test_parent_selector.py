"""Fast unit tests for selector contracts and PyVRP Activity API compatibility.

These tests use small doubles so they do not need a compiled PyVRP installation.
Run with: python -m unittest discover -s tests -v
"""
import unittest

from base_heuristic import INITIAL_HEURISTICS
from validate import compile_heuristic


class FakeActivity:
    def __init__(self, idx, is_client=True):
        self.idx = idx
        self._is_client = is_client

    def is_client(self):
        return self._is_client


class FakeSolution:
    def __init__(self, idx):
        self.idx = idx
        self._feasible = idx % 3 != 0
        # PyVRP 0.14-style activity objects; include a depot-like activity.
        self._routes = [
            [FakeActivity(0, False), FakeActivity(idx), FakeActivity((idx + 1) % 12)]
        ]

    def is_feasible(self):
        return self._feasible

    def routes(self):
        return self._routes


class FakeCostEvaluator:
    def penalised_cost(self, solution):
        return float(1000 + solution.idx * 10)


class FakeRNG:
    def __init__(self):
        self.counter = 0

    def randint(self, high):
        value = self.counter % high
        self.counter += 1
        return value

    def rand(self):
        value = (self.counter % 100) / 100.0
        self.counter += 1
        return value

    def rand(self):
        self.counter += 1
        return (self.counter % 100) / 100.0


class ParentSelectorTests(unittest.TestCase):
    def setUp(self):
        self.population = [FakeSolution(i) for i in range(12)]
        self.cost_evaluator = FakeCostEvaluator()

    def test_all_initial_seeds_compile_and_return_distinct_population_members(self):
        self.assertGreaterEqual(len(INITIAL_HEURISTICS), 2)
        for code, description in INITIAL_HEURISTICS:
            with self.subTest(seed=description):
                selector, error = compile_heuristic(code)
                self.assertIsNotNone(selector, error)
                p1, p2 = selector(
                    self.population, FakeRNG(), self.cost_evaluator
                )
                self.assertIsNot(p1, p2)
                self.assertIn(p1, self.population)
                self.assertIn(p2, self.population)

    def test_appendix_c_selector_handles_pyvrp_014_activity_objects(self):
        code, _ = INITIAL_HEURISTICS[0]
        selector, error = compile_heuristic(code)
        self.assertIsNotNone(selector, error)
        for seed in range(20):
            p1, p2 = selector(
                self.population, FakeRNG(), self.cost_evaluator
            )
            self.assertIsNot(p1, p2)
            self.assertIn(p1, self.population)
            self.assertIn(p2, self.population)


if __name__ == "__main__":
    unittest.main()
