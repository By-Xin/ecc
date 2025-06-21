#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two exploration strategies on MountainCar-v0 with NEAT:

1. eps        – ε-greedy (ε linearly decays per generation)
2. ucb_global – UCB with a global Q/count table shared *across* generations

For each strategy we run `RUNS` independent seeds; each run evolves
`GENERATIONS` generations. 结果曲线以均值±1σ 可视化并保存到 disk。
"""

import os
import gym
import neat
import numpy as np
import matplotlib.pyplot as plt
from neat import nn

# ---------------- Hyper-parameters -----------------------------------------
GAME          = "MountainCar-v0"
CONFIG_PATH   = "./config"
GENERATIONS   = 50
EPISODES      = 10
EP_STEPS      = 200            # 最大步长（MountainCar 默认 200）
RUNS          = 10

# ε-greedy schedule
eps_start, eps_end = 1.0, 0.05
eps_decay_gen      = (eps_start - eps_end) / (GENERATIONS - 1)   # 线性衰减

# UCB constant
C_VAL = 0.5

# Output directory
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

def make_eps_evaluator(eps_gen: float):
    """按给定 ε 生成评估器；每代更新 ε 时重新构造。"""
    assert 0.0 <= eps_gen <= 1.0
    def eval_genomes(genomes, config):
        env = gym.make(GAME)
        for _, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, config)
            rewards = []
            for _ in range(EPISODES):
                obs, _ = env.reset()
                total, step, done = 0.0, 0, False
                while (not done) and (step < EP_STEPS):
                    step += 1
                    if np.random.rand() < eps_gen:
                        act = env.action_space.sample()
                    else:
                        act = int(np.argmax(net.activate(obs)))
                    obs, r, term, trunc, _ = env.step(act)
                    done   = term or trunc
                    total += r
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()
    return eval_genomes


def make_ucb_evaluator(global_ucb: dict):
    """
    global_ucb = {"counts": ndarray(n_act, int),
                  "qvals" : ndarray(n_act, float),
                  "t"     : int}
    """
    counts_g = global_ucb["counts"]
    qvals_g  = global_ucb["qvals"]

    def eval_genomes(genomes, config):
        # —— 每代开始：拷贝一份局部统计，避免同代个体互相污染 ——
        counts = counts_g.copy()
        qvals  = qvals_g.copy()
        t      = global_ucb["t"]

        env = gym.make(GAME)
        for _, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, config)
            rewards = []
            for _ in range(EPISODES):
                obs, _ = env.reset()
                total, step, done = 0.0, 0, False
                while (not done) and (step < EP_STEPS):
                    step += 1
                    t    += 1                       # 全局时间步递增
                    bonus = C_VAL * np.sqrt(np.log(t + 1) / (counts + 1))
                    act   = int(np.argmax(np.array(net.activate(obs)) + bonus))
                    obs, r, term, trunc, _ = env.step(act)
                    done   = term or trunc
                    total += r

                    # 更新局部统计
                    counts[act] += 1
                    qvals[act]  += (r - qvals[act]) / counts[act]
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()

        # —— 代末回写到全局统计 ——
        counts_g[:]        = counts
        qvals_g[:]         = qvals
        global_ucb["t"]    = t

    return eval_genomes

# ---------------------------------------------------------------------------#
#                           单次实验（一个随机种子）                          #
# ---------------------------------------------------------------------------#
def run_once(strategy: str, run_idx: int):
    best_curve, mean_curve = np.zeros(GENERATIONS), np.zeros(GENERATIONS)

    # —— 初始化 NEAT 种群 ——
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    pop = neat.Population(config)

    # —— 根据策略准备评估函数 ——
    if strategy == "eps":
        eps_g     = eps_start
        eval_fn   = make_eps_evaluator(eps_g)       # 每代会替换
    elif strategy == "ucb_global":
        # 全局统计容器（可变对象）
        env_tmp   = gym.make(GAME); n_act = env_tmp.action_space.n; env_tmp.close()
        global_ucb = {
            "counts": np.zeros(n_act, dtype=int),
            "qvals" : np.zeros(n_act, dtype=float),
            "t"     : 0
        }
        eval_fn = make_ucb_evaluator(global_ucb)    # 只创建一次
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # —— 演化 GENERATIONS 代 ——
    for gen in range(GENERATIONS):
        # ε-greedy 每代更新 ε 后重建评估器
        if strategy == "eps":
            eval_fn = make_eps_evaluator(eps_g)

        pop.run(eval_fn, 1)                         # 演化 1 代

        best_curve[gen] = pop.best_genome.fitness
        fits            = [g.fitness for g in pop.population.values()
                           if g.fitness is not None]
        mean_curve[gen] = np.mean(fits)

        if strategy == "eps":
            eps_g = max(eps_end, eps_g - eps_decay_gen)

    # —— 保存每次 run 的曲线 —— 
    np.save(os.path.join(out_dir, f"{strategy}_run{run_idx}_best.npy"), best_curve)
    np.save(os.path.join(out_dir, f"{strategy}_run{run_idx}_mean.npy"), mean_curve)

    return best_curve, mean_curve

# ---------------------------------------------------------------------------#
#                                主循环                                      #
# ---------------------------------------------------------------------------#
def main():
    strategies = ["eps", "ucb_global"]
    gens       = np.arange(1, GENERATIONS + 1)
    fig, axes  = plt.subplots(1, 2, figsize=(15, 6))

    for strat in strategies:
        all_best = np.zeros((RUNS, GENERATIONS))
        all_mean = np.zeros((RUNS, GENERATIONS))
        for r in range(RUNS):
            print(f"[{strat}] run {r + 1}/{RUNS}")
            best, mean   = run_once(strat, r)
            all_best[r]  = best
            all_mean[r]  = mean

        avg_best, std_best = all_best.mean(0), all_best.std(0)
        avg_mean, std_mean = all_mean.mean(0), all_mean.std(0)

        axes[0].plot(gens, avg_best, label=f"{strat} best")
        axes[0].fill_between(gens, avg_best - std_best, avg_best + std_best, alpha=0.2)

        axes[1].plot(gens, avg_mean, label=f"{strat} mean")
        axes[1].fill_between(gens, avg_mean - std_mean, avg_mean + std_mean, alpha=0.2)

        # 保存聚合后的均值曲线
        np.save(os.path.join(out_dir, f"{strat}_avg_best.npy"), avg_best)
        np.save(os.path.join(out_dir, f"{strat}_avg_mean.npy"), avg_mean)

    axes[0].set(title="Average Best Fitness ±1σ",  xlabel="Generation", ylabel="Fitness")
    axes[1].set(title="Average Mean Fitness ±1σ",  xlabel="Generation", ylabel="Fitness")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "strategy_comparison.png")
    plt.savefig(out_png)
    print(f"Saved comparison plot to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
