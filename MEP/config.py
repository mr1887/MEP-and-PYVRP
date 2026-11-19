"""Configuration for the MEP experiments."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Automatically load environment variables from .env if present
for env_path in [ROOT_DIR / ".env", ROOT_DIR.parent / ".env"]:
    if env_path.is_file():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Paper case study: 10 generations, 10 offspring per generation, retain 5.
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "10"))
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "5"))
OFFSPRING_PER_GENERATION = int(os.getenv("OFFSPRING_PER_GENERATION", "10"))

DATA_DIR = str(ROOT_DIR / "data")
VRP_INSTANCES = [
    item.strip()
    for item in os.getenv("VRP_INSTANCES", "X-n101-k25.vrp").split(",")
    if item.strip()
]
HGS_TIME_LIMIT = float(os.getenv("HGS_TIME_LIMIT", "10"))
NUM_EVAL_RUNS = int(os.getenv("NUM_EVAL_RUNS", "5"))
BASE_RANDOM_SEED = int(os.getenv("BASE_RANDOM_SEED", "42"))
GENERATED_CODE_DIR = str(ROOT_DIR / "generated_code")
RESULTS_DIR = str(ROOT_DIR / "results")

if NUM_GENERATIONS < 1:
    raise ValueError("NUM_GENERATIONS must be >= 1")
if POPULATION_SIZE < 2:
    raise ValueError("POPULATION_SIZE must be >= 2")
if OFFSPRING_PER_GENERATION < 1:
    raise ValueError("OFFSPRING_PER_GENERATION must be >= 1")
if HGS_TIME_LIMIT <= 0:
    raise ValueError("HGS_TIME_LIMIT must be > 0")
if NUM_EVAL_RUNS < 2:
    raise ValueError("NUM_EVAL_RUNS must be >= 2")
if not VRP_INSTANCES:
    raise ValueError("VRP_INSTANCES must contain at least one .vrp file")
