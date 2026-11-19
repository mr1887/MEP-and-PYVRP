# MEP + PyVRP+ reproduction

Repository structure:
- `PyVRP/`: bundled source of the author's PyVRP 0.11-era HGS fork.
- `MEP/`: the group's MEP loop, prompt, LLM client, heuristic population and evaluator.

## Reproduction target

This implementation evolves the `select_parents` HGS component. The Appendix-C
reference selector is transcribed in `base_heuristic.py` (Figures 4–7, paper
pages 13–16). It preserves the paper's strata, 7-way feasibility-biased
tournament, 20% infeasible admission probability, complementary parent-2
strata, 20/30 candidate sampling, 500-solution threshold, scoring weights,
0.02 random perturbation and route-set encoding.

The paper also evolves `select_survivors` and `update_penalties`. Those are
not yet included in this MEP experiment, so do not describe this repository as
a complete reproduction of all PyVRP+ components or benchmark tables.

## Installation (Windows)

Use 64-bit Python 3.11 or 3.12 and a 64-bit MSVC developer shell. The PyVRP
fork builds native C++ extensions; the x64 compiler must be selected.

From the repository root:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .\\PyVRP
python -m pip install -r .\\MEP\\requirements.txt
```

The local editable install is intentional: it ensures Python imports the
bundled author's fork, not a separately installed PyPI PyVRP release.

## Configure and run

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
cd MEP
python -m unittest discover -s tests -v
python evolve.py
```

For a short smoke run:

```powershell
$env:NUM_GENERATIONS="1"
$env:OFFSPRING_PER_GENERATION="1"
$env:NUM_EVAL_RUNS="2"
$env:HGS_TIME_LIMIT="2"
python evolve.py
```

Defaults are 10 generations, 10 offspring per generation, a MEP heuristic
population of 5, 5 HGS seeds per instance and 10 seconds per HGS run. These
settings reflect the paper's stated evolutionary counts where available, but
the paper's full experimental protocol uses much broader benchmarks and
separate repeated runs. The default data folder currently contains only one
CVRP instance.

## Important implementation notes

- The HGS hook is `SolveParams(custom_selector_path=...)`; `solve()` creates
  the author's `Population`, whose `select()` invokes the supplied function.
- The selector must return two Solution objects from the supplied population.
- The custom loader now permits the paper's `from pyvrp._pyvrp import ...`
  import and uses normal Python builtins. This is an execution compatibility
  change, not a security sandbox: generated code runs in-process.
- The appendix code uses legacy integer route visits (0 = depot) in this
  fork. The reference selector preserves that API assumption.
- HGS's survivor selection and penalty adaptation remain the author's defaults.
  Only parent selection is evolved in this version.
- Fitness is mean feasible HGS objective cost (lower is better). To match the
  paper's sign convention exactly, the equivalent MEP maximization fitness is
  its negative; the current MEP population minimizes stored cost directly.
- Do not compare the one-instance smoke-run output with the paper's aggregate
  six-variant results.
