# compare_strategies.py
"""Compare four exploration strategies on MountainCar-v0 with NEAT:
1. eps              – cross‑generation ε‑greedy (ε linearly decays per generation)
2. ucb_global       – UCB with a *global* Q/count table shared across generations

Each strategy is run `RUNS` times, each run evolves `GENERATIONS` generations.
We plot average best and average mean fitness curves with std‑dev shading.
All raw curves are saved to .npy for further analysis.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gym
import neat
import numpy as np
import matplotlib.pyplot as plt
from neat import nn

# ---------------- Hyper‑parameters -----------------------------------------
GAME = "MountainCar-v0"
CONFIG_PATH = "./config"
GENERATIONS = 50
EPISODES = 5
EP_STEPS = 200  # MountainCar默认最大步数是200
RUNS = 8

# ε‑greedy schedule
eps_start, eps_end = 1.0, 0.05
eps_decay_gen = 2*(eps_start - eps_end) / (GENERATIONS - 1)

# UCB constant
C_VAL = 0.5

# Output directory
enabled_dir = "results"
os.makedirs(enabled_dir, exist_ok=True)

# ---------------- Strategy Evaluators --------------------------------------

def evaluator_factory(strategy: str, eps_gen: float = None,
                      global_ucb=None):
    if strategy == "eps":
        assert eps_gen is not None
        def eval_genomes(genomes, config):
            env = gym.make(GAME)
            for _, genome in genomes:
                net = nn.FeedForwardNetwork.create(genome, config)
                rews = []
                for _ in range(EPISODES):
                    obs, _ = env.reset()
                    total, t, done = 0, 0, False
                    while not done and t < EP_STEPS:
                        t += 1
                        if np.random.rand() < eps_gen:
                            action = env.action_space.sample()
                        else:
                            action = int(np.argmax(net.activate(obs)))
                        obs, r, term, trunc, _ = env.step(action)
                        done = term or trunc
                        total += r
                    rews.append(total)
                genome.fitness = np.mean(rews)
            env.close()
        return eval_genomes

    if strategy == "ucb_global":
        assert global_ucb is not None
        counts_g, qvals_g, t_glob = global_ucb
        def eval_genomes(genomes, config):
            nonlocal t_glob
            env = gym.make(GAME)
            for _, genome in genomes:
                net = nn.FeedForwardNetwork.create(genome, config)
                rews = []
                for _ in range(EPISODES):
                    obs, _ = env.reset()
                    total, step, done = 0, 0, False
                    while not done and step < EP_STEPS:
                        step += 1
                        t_glob += 1
                        bonus = C_VAL * np.sqrt(np.log(t_glob + 1) / (counts_g + 1))
                        scores = np.array(net.activate(obs)) + bonus
                        action = int(np.argmax(scores))
                        obs, r, term, trunc, _ = env.step(action)
                        done = term or trunc
                        total += r
                        counts_g[action] += 1
                        qvals_g[action] += (r - qvals_g[action]) / counts_g[action]
                    rews.append(total)
                genome.fitness = np.mean(rews)
            env.close()
        return eval_genomes

    raise ValueError("Unknown strategy")

# ---------------- Run once --------------------------------------------------

def run_once(strategy: str, run_idx: int):
    best_curve = np.zeros(GENERATIONS)
    mean_curve = np.zeros(GENERATIONS)

    eps_g = eps_start
    # global UCB stats
    env_tmp = gym.make(GAME)
    n_act = env_tmp.action_space.n
    env_tmp.close()
    counts_g = np.zeros(n_act, int)
    qvals_g  = np.zeros(n_act, float)
    t_glob   = 0

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    pop = neat.Population(config)

    for gen in range(GENERATIONS):
        if strategy == "greedy":
            eval_fn = evaluator_factory("greedy")
        elif strategy == "eps":
            eval_fn = evaluator_factory("eps", eps_gen=eps_g)
        elif strategy == "ucb_fixed":
            eval_fn = evaluator_factory("ucb_fixed")
        elif strategy == "ucb_global":
            eval_fn = evaluator_factory("ucb_global", global_ucb=[counts_g, qvals_g, t_glob])
        else:
            raise ValueError

        pop.run(eval_fn, 1)

        best_curve[gen] = pop.best_genome.fitness
        fits = [g.fitness for g in pop.population.values() if g.fitness is not None]
        mean_curve[gen] = np.mean(fits)

        if strategy == "eps":
            eps_g = max(eps_end, eps_g - eps_decay_gen)

    # save per-run data
    np.save(os.path.join(enabled_dir, f"{strategy}_run{run_idx}_best.npy"), best_curve)
    np.save(os.path.join(enabled_dir, f"{strategy}_run{run_idx}_mean.npy"), mean_curve)
    np.savetxt(os.path.join(enabled_dir, f"{strategy}_run{run_idx}_best.csv"), best_curve, delimiter=',', header='best_curve', comments='')
    np.savetxt(os.path.join(enabled_dir, f"{strategy}_run{run_idx}_mean.csv"), mean_curve, delimiter=',', header='mean_curve', comments='')

    return best_curve, mean_curve

# ---------------- Main loop -------------------------------------------------

def main():
    strategies = ["eps", "ucb_global"]
    gens = np.arange(1, GENERATIONS+1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for strat in strategies:
        all_best = np.zeros((RUNS, GENERATIONS))
        all_mean = np.zeros((RUNS, GENERATIONS))
        for r in range(RUNS):
            print(f"[{strat}] run {r+1}/{RUNS}")
            best, mean = run_once(strat, r)
            all_best[r] = best
            all_mean[r] = mean

        avg_best, std_best = all_best.mean(0), all_best.std(0)
        avg_mean, std_mean = all_mean.mean(0), all_mean.std(0)

        # plot
        axes[0].plot(gens, avg_best, label=f"{strat} best")
        axes[0].fill_between(gens, avg_best-std_best, avg_best+std_best, alpha=0.2)
        axes[1].plot(gens, avg_mean, label=f"{strat} mean")
        axes[1].fill_between(gens, avg_mean-std_mean, avg_mean+std_mean, alpha=0.2)

        # save aggregated data
        np.save(os.path.join(enabled_dir, f"{strat}_avg_best.npy"), avg_best)
        np.save(os.path.join(enabled_dir, f"{strat}_avg_mean.npy"), avg_mean)
        # np.savetxt(os.path.join(enabled_dir, f"{strat}_avg_best.csv"), avg_best, delimiter=',', header='avg_best', comments='')
        # np.savetxt(os.path.join(enabled_dir, f"{strat}_avg_mean.csv"), avg_mean, delimiter=',', header='avg_mean', comments='')

    axes[0].set(title="Average Best Fitness ±1σ", xlabel="Generation", ylabel="Fitness")
    axes[1].set(title="Average Mean Fitness ±1σ", xlabel="Generation", ylabel="Fitness")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True)

    plt.tight_layout()
    out_png = os.path.join(enabled_dir, "strategy_comparison.png")
    plt.savefig(out_png)
    print(f"Saved comparison plot to {out_png}")
    plt.show()

if __name__ == "__main__":
    main()

