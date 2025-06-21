#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic‑C UCB Experiment on MountainCar‑v0 with NEAT
===================================================

This script compares a grid of fixed UCB exploration constants *C* with a
Thompson‑Sampling (TS) "meta‑policy" that *dynamically* switches between two C
values.  For each configuration we run multiple random seeds, record the final
(best‑genome) fitness at generation ``GENERATIONS`` and produce:

* A **boxplot** comparing the distributions of final fitness across Cs.
* Non‑parametric **significance tests** (Kruskal‑Wallis across all groups and
  pairwise Mann‑Whitney U versus the TS meta‑policy).
* All training loops use ``tqdm`` progress bars.

The code keeps other hyper‑parameters untouched relative to your previous
``ucb_global`` baseline and relies on ``ThreadPoolExecutor`` for per‑genome
parallelism.  Runs are executed sequentially to keep progress bars clean.

Dependencies
------------
* Python 3.9+
* gymnasium (≥0.29)  or legacy gym
* neat‑python
* numpy, matplotlib, tqdm, scipy

If ``scipy`` is absent the script still runs (significance tests are skipped
with a warning).
"""

from __future__ import annotations

import logging
import math
import os
import random
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import neat  # type: ignore
import numpy as np
from neat import nn  # type: ignore
from tqdm.auto import tqdm

# Optional stats
try:
    from scipy.stats import kruskal, mannwhitneyu  # type: ignore

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not found ‑‑ significance tests will be skipped.")

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- Hyper‑parameters -----------------------------------------
GAME: str = "MountainCar-v0"
CONFIG_PATH: str = "./config"
GENERATIONS: int = 200
EPISODES: int = 5
EP_STEPS: int = 100
RUNS: int = 5  # seeds per configuration
NUM_THREADS: int = os.cpu_count() or 1  # thread workers per process

# Grid of fixed‑C values
C_GRID: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
# Dynamic‑C meta policy will switch between these two using Thompson Sampling
C_DYNAMIC_CHOICES: Tuple[float, float] = (0.5, 2.0)

# Output directory structure
OUT_DIR = Path("results_ucb_dynamic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Thompson‑Sampling bandit for dynamic C -------------------
class GaussianTSBandit:
    """Two‑arm Gaussian Thompson Sampling with unknown mean, known var≈1."""

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.n = np.zeros(k, dtype=int)
        self.mean = np.zeros(k, dtype=float)

    def select(self) -> int:
        sigma = 1.0 / np.sqrt(self.n + 1)
        sample = np.random.normal(self.mean, sigma)
        return int(np.argmax(sample))

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.mean[arm] += (reward - self.mean[arm]) / self.n[arm]

# ---------------- Single‑genome evaluation (thread‑parallel) ---------------

def evaluate_single_genome(args):
    """Thread‑safe evaluation of a genome under a *fixed* C for this call."""
    (
        genome_id,
        genome,
        config,
        global_ucb,
        c_val,
    ) = args

    net = nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(GAME)
    rewards: List[float] = []

    for _ in range(EPISODES):
        obs, _ = env.reset()
        total, step, terminated = 0.0, 0, False
        while not terminated and step < EP_STEPS:
            step += 1
            counts = global_ucb["counts"].copy()
            t = global_ucb["t"]
            bonus = c_val * np.sqrt(np.log(t + step + 1) / (counts + 1))
            qvals = np.asarray(net.activate(obs))
            action = int(np.argmax(qvals + bonus))

            obs, reward, term, trunc, _ = env.step(action)
            terminated = term or trunc
            total += reward
        rewards.append(total)

    env.close()
    return genome_id, float(np.mean(rewards))

# ---------------- Per‑generation evaluator factory ------------------------

def make_parallel_evaluator(global_ucb: Dict[str, np.ndarray], c_val: float):
    """Return a NEAT‑compatible (genomes, config) -> None evaluator."""

    def eval_genomes(genomes, config):
        tasks = [
            (gid, g, config, global_ucb, c_val) for gid, g in genomes  # type: ignore
        ]
        best_fitness = float("-inf")
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            for genome_id, fitness in pool.map(evaluate_single_genome, tasks):
                for gid, genome in genomes:  # type: ignore
                    if gid == genome_id:
                        genome.fitness = fitness
                        best_fitness = max(best_fitness, fitness)
                        break
        # Global time update (simple proxy, counts unchanged)
        global_ucb["t"] += len(genomes) * EPISODES * EP_STEPS
        return best_fitness

    return eval_genomes

# ---------------- Single training run (one seed) ---------------------------

def run_once(seed: int, c_fixed: float | None = None) -> Tuple[np.ndarray, float]:
    """Run NEAT with either a *fixed* C or a dynamic TS meta‑policy.

    Returns (best_curve, final_best_fitness).
    """
    np.random.seed(42 + seed)
    random.seed(42 + seed)

    best_curve = np.empty(GENERATIONS, dtype=float)

    # Global UCB state (shared across all genomes / generations within a run)
    env_tmp = gym.make(GAME)
    n_actions = env_tmp.action_space.n
    env_tmp.close()
    global_ucb = {"counts": np.zeros(n_actions, int), "t": 0}

    # Thompson‑Sampling bandit for dynamic choice
    if c_fixed is None:
        bandit = GaussianTSBandit(k=len(C_DYNAMIC_CHOICES))
    else:
        bandit = None

    config = neat.Config(  # type: ignore
        neat.DefaultGenome,  # type: ignore
        neat.DefaultReproduction,  # type: ignore
        neat.DefaultSpeciesSet,  # type: ignore
        neat.DefaultStagnation,  # type: ignore
        CONFIG_PATH,
    )
    pop = neat.Population(config)  # type: ignore

    # Progress bar over generations
    gen_iter = tqdm(range(GENERATIONS), desc=f"Seed {seed}", leave=False)
    for gen in gen_iter:
        # Choose C for **this** generation
        if c_fixed is not None:
            c_val = c_fixed
            arm_idx = None
        else:
            arm_idx = bandit.select()  # type: ignore[arg-type]
            c_val = C_DYNAMIC_CHOICES[arm_idx]

        eval_fn = make_parallel_evaluator(global_ucb, c_val)
        pop.run(eval_fn, 1)
        best_fitness = pop.best_genome.fitness  # type: ignore[attr-defined]
        best_curve[gen] = best_fitness

        # Reward bandit if dynamic
        if arm_idx is not None and bandit is not None:
            bandit.update(arm_idx, best_fitness)

        # Update tqdm postfix every 50 generations for speed
        if gen % 50 == 0 or gen == GENERATIONS - 1:
            gen_iter.set_postfix(best=f"{best_fitness:.1f}")

    return best_curve, best_curve[-1]

# ---------------- Main experiment loop -------------------------------------

def main() -> None:
    config_names: List[str] = [f"C={c}" for c in C_GRID] + [
        "TS‑dynamic",
    ]
    n_cfg = len(config_names)

    # Containers: config -> list of final fitness
    final_scores: Dict[str, List[float]] = defaultdict(list)
    best_curves: Dict[str, List[np.ndarray]] = defaultdict(list)

    for cfg_idx, cfg_name in enumerate(tqdm(config_names, desc="Configs")):
        if cfg_name == "TS‑dynamic":
            c_val: float | None = None  # signal dynamic mode
        else:
            c_val = float(cfg_name.split("=")[1])

        # Sequential seeds with progress bar
        for seed in tqdm(range(RUNS), desc=f"{cfg_name} Seeds", leave=False):
            curve, final_fit = run_once(seed, c_fixed=c_val)
            final_scores[cfg_name].append(final_fit)
            best_curves[cfg_name].append(curve)

            # Persist per‑run data (optional, easy restart)
            np.save(
                OUT_DIR / f"curve_{cfg_name.replace('=', '')}_seed{seed}.npy", curve
            )

    # -------------- Visualization: boxplot ---------------------------------
    plt.figure(figsize=(10, 6))
    data = [final_scores[cfg] for cfg in config_names]
    plt.boxplot(data, labels=config_names, showfliers=True)
    plt.title(
        f"UCB Global final fitness comparison\n({RUNS} seeds × {GENERATIONS} generations)"
    )
    plt.ylabel("Final Best Fitness (Gen %d)" % GENERATIONS)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    boxplot_path = OUT_DIR / "ucb_C_boxplot.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Boxplot saved to {boxplot_path}")

    # -------------- Significance tests -------------------------------------
    if not SCIPY_AVAILABLE:
        print("SciPy unavailable – skipping significance tests.")
        return

    # Kruskal‑Wallis across all configurations
    all_groups = [np.asarray(final_scores[cfg]) for cfg in config_names]
    stat_kw, p_kw = kruskal(*all_groups)
    print(f"Kruskal‑Wallis H={stat_kw:.3f}, p={p_kw:.3e}")

    # Pairwise Mann‑Whitney U against TS‑dynamic
    ts_scores = np.asarray(final_scores["TS‑dynamic"])
    for cfg in config_names:
        if cfg == "TS‑dynamic":
            continue
        u_stat, p_val = mannwhitneyu(ts_scores, final_scores[cfg], alternative="two-sided")
        print(f"TS‑dynamic vs {cfg}: U={u_stat:.1f}, p={p_val:.3e}")

    print("Done.")


if __name__ == "__main__":
    main()
