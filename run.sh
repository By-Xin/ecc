#!/usr/bin/env bash
# =========================================================
# Hybrid-NEAT + DQN  DEMO  &  FULL EXPERIMENT PIPELINE
# =========================================================
set -euo pipefail

echo ">>> (1/3) Installing required Python packages ..."
python - <<'PY'
import subprocess, sys, pkg_resources, json

req = {
    "torch==2.2":    "torch",
    "gymnasium==0.29": "gymnasium",
    "neat-python==0.92": "neat",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm"
}
for spec, mod in req.items():
    try:
        pkg_resources.get_distribution(mod)
    except pkg_resources.DistributionNotFound:
        print(f"Installing {spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
PY
echo ">>> Packages ready."

# ---------------------------------------------------------
echo -e "\n>>> (2/3) Running SMOKE TEST (2 generations, Hybrid only) ..."
python - <<'PY'
from src.neat_algorithms import eval_genomes_hybrid, run_training
winner, _, _, _ = run_training(num_generations=2,
                               eval_fn=eval_genomes_hybrid,
                               run_seed=123,
                               algorithm_name="SmokeTest")
print(f"Smoke-test finished.  Best fitness = {winner.fitness:.2f}")
assert winner.fitness > 0.0, "Smoke-test failed: fitness not positive."
PY
echo ">>> Smoke-test passed."

# ---------------------------------------------------------
echo -e "\n>>> (3/3) Running FULL EXPERIMENT (30 generations × 5 seeds × 3 algs) ..."
mkdir -p results

for ALG in Hybrid Meta Vanilla; do
  for SEED in 1 2 3 4 5; do
    echo "    → Algorithm=${ALG}  Seed=${SEED}"
    export ALG SEED
    python - <<'PY'
import os, json
from pathlib import Path
from src.neat_algorithms import (
    eval_genomes_hybrid, eval_genomes_meta, eval_genomes_vanilla, run_training
)

alg  = os.environ["ALG"]
seed = int(os.environ["SEED"])
gens = 30

fn = {"Hybrid":  eval_genomes_hybrid,
      "Meta":    eval_genomes_meta,
      "Vanilla": eval_genomes_vanilla}[alg]

_, _, _, best_rewards = run_training(gens, fn, seed, alg)
Path("results").mkdir(exist_ok=True)
with open(f"results/{alg}_seed{seed}.json", "w") as f:
    json.dump(best_rewards, f)
PY
  done
done

echo -e "\n>>> Experiment completed.  All reward curves stored under ./results/"
echo    "    File pattern: results/<Algorithm>_seed<idx>.json"

