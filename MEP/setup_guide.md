# Setup guide: MEP + author PyVRP HGS

## 1. Prerequisites

- 64-bit Python 3.11 or 3.12.
- Windows: Visual Studio Build Tools with Desktop development with C++, MSVC x64/x64 tools and Windows SDK.
- Run installation from an x64 Developer PowerShell / Developer Command Prompt.
- A Gemini API key.

## 2. Install from the repository root

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .\\PyVRP
python -m pip install -r .\\MEP\\requirements.txt
```

Check that Python imports the bundled fork:

```powershell
python -c "import pyvrp; print(pyvrp.__file__)"
``

The printed path should point into this repository's PyVRP source tree / editable
installation. Do not install a different PyVRP version from PyPI over it.

## 3. Run a short integration experiment

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
$env:NUM_GENERATIONS="1"
$env:OFFSPRING_PER_GENERATION="1"
$env:NUM_EVAL_RUNS="2"
$env:HGS_TIME_LIMIT="2"
cd MEP
python evolve.py
```

The evaluator writes a temporary selector module and passes its path through
`SolveParams(custom_selector_path=...)`. The fork's `Population.select()`
then calls the candidate during HGS parent selection. If the import, signature,
return type or route API is incompatible, HGS evaluation fails and the
candidate is rejected.

## 4. Full configured run

Unset smoke-test variables or set the paper-style counts:

```powershell
$env:NUM_GENERATIONS="10"
$env:OFFSPRING_PER_GENERATION="10"
$env:NUM_EVAL_RUNS="5"
$env:HGS_TIME_LIMIT="10"
python evolve.py
```

Current default benchmark data includes one CVRP instance. Add and document
more instances before making multi-variant claims.

## 5. Scope and scientific reporting

The current reproduction evolves only `select_parents`. The paper also
evolves `select_survivors` and `update_penalties`, and reports results over
six VRP variants with 60 instances per variant. This repository is therefore
a partial reproduction focused on the Appendix-C parent selector.

The Appendix-C reference code is transcribed in `base_heuristic.py`. It
preserves the pseudocode/code details on paper pages 13–16, including the
population threshold 500, sample caps 20/30, tournament size 7, infeasible
admission probability 0.2, score weights 0.60/0.30/0.10 and 0.55/0.30/0.15,
and random perturbation amplitude 0.02.

The custom selector loader executes Python code in-process. AST checks and
function-signature validation are not a security sandbox. Do not run
untrusted code.
