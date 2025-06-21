#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare three exploration strategies on MountainCar-v0 with NEAT.

    eps          – cross-generation ε-greedy (ε linearly decays)
    ucb_global   – UCB with a global Q/count table shared across generations
    hybrid       – 每个 episode 以概率 p_g 选 UCB，否则选 ε-greedy；
                   p_g 在 GENERATIONS 内线性退火

For 每种策略 we run RUNS 独立随机种子，每 run 演化 GENERATIONS 代。
输出：
    • Average best fitness ±1σ
    • Average mean fitness ±1σ
    • Success rate (best ≥ −110) ±1σ
所有 raw/aggregate 曲线保存到 ./results
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
EPISODES      = 5
EP_STEPS      = 200
RUNS          = 8

# ε-greedy schedule
eps_start, eps_end     = 1.0, 0.05
eps_decay_gen          = (eps_start - eps_end) / (GENERATIONS - 1)






# 尝试一下
eps_decay_gen_sqrt          = (eps_start - eps_end) / (np.sqrt(GENERATIONS) - 1)
eps_decay_gen_sq = (eps_start - eps_end) / (GENERATIONS**2 - 1)
eps_decay_gen_poly = (eps_start - eps_end) / (GENERATIONS - 1)*log(GENERATIONS-1)







# hybrid: p(UCB) schedule
p_start, p_end         = 0.9, 0.2
p_decay_gen            = (p_start - p_end) / (GENERATIONS - 1)

# UCB constant
C_VAL = 0.5

# Output directory
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# ---------------- 评估器工厂 -------------------------------------------------
def make_eps_evaluator(eps_gen: float):
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
    counts_g = global_ucb["counts"]
    qvals_g  = global_ucb["qvals"]

    def eval_genomes(genomes, config):
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
                    t    += 1
                    bonus = C_VAL * np.sqrt(np.log(t + 1) / (counts + 1))
                    act   = int(np.argmax(np.array(net.activate(obs)) + bonus))
                    obs, r, term, trunc, _ = env.step(act)
                    done   = term or trunc
                    total += r
                    counts[act] += 1
                    qvals[act]  += (r - qvals[act]) / counts[act]
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()

        counts_g[:]        = counts
        qvals_g[:]         = qvals
        global_ucb["t"]    = t
    return eval_genomes


def make_hybrid_evaluator(global_ucb: dict,
                          eps_gen: float,
                          p_use_ucb: float):
    """episode 级决策：Bernoulli(p_use_ucb) 选 UCB，否则 ε-greedy"""
    counts_g = global_ucb["counts"]
    qvals_g  = global_ucb["qvals"]

    def eval_genomes(genomes, config):
        counts = counts_g.copy()
        qvals  = qvals_g.copy()
        t      = global_ucb["t"]

        env = gym.make(GAME)
        for _, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, config)
            rewards = []
            for _ in range(EPISODES):
                use_ucb = np.random.rand() < p_use_ucb
                obs, _  = env.reset()
                total, step, done = 0.0, 0, False
                while (not done) and (step < EP_STEPS):
                    step += 1
                    if use_ucb:
                        t += 1
                        bonus = C_VAL * np.sqrt(np.log(t + 1) / (counts + 1))
                        act   = int(np.argmax(np.array(net.activate(obs)) + bonus))
                    else:
                        if np.random.rand() < eps_gen:
                            act = env.action_space.sample()
                        else:
                            act = int(np.argmax(net.activate(obs)))
                    obs, r, term, trunc, _ = env.step(act)
                    done   = term or trunc
                    total += r
                    if use_ucb:
                        counts[act] += 1
                        qvals[act]  += (r - qvals[act]) / counts[act]
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()

        counts_g[:]      = counts
        qvals_g[:]       = qvals
        global_ucb["t"]  = t
    return eval_genomes

# ---------------- 单次实验 ---------------------------------------------------
def run_once(strategy: str, run_idx: int):
    best_curve     = np.zeros(GENERATIONS)
    mean_curve     = np.zeros(GENERATIONS)
    success_curve  = np.zeros(GENERATIONS)  # best ≥ −110 → 成功

    eps_g = eps_start
    p_g   = p_start

    # 全局 UCB 容器（供 ucb_global & hybrid 共用）
    env_tmp = gym.make(GAME)
    n_act   = env_tmp.action_space.n
    env_tmp.close()
    global_ucb = {
        "counts": np.zeros(n_act, dtype=int),
        "qvals" : np.zeros(n_act, dtype=float),
        "t"     : 0
    }

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    pop = neat.Population(config)

    # UCB 评估器可提前生成；eps/hybrid 需随代更新
    ucb_eval = make_ucb_evaluator(global_ucb)

    for gen in range(GENERATIONS):
        if strategy == "eps":
            eval_fn = make_eps_evaluator(eps_g)
        elif strategy == "ucb_global":
            eval_fn = ucb_eval
        elif strategy == "hybrid":
            eval_fn = make_hybrid_evaluator(global_ucb, eps_g, p_g)
        else:
            raise ValueError("unknown strategy")

        pop.run(eval_fn, 1)

        best_curve[gen]    = pop.best_genome.fitness
        fits               = [g.fitness for g in pop.population.values()
                              if g.fitness is not None]
        mean_curve[gen]    = np.mean(fits)
        success_curve[gen] = 1.0 if best_curve[gen] >= -110 else 0.0

        # 更新调度
        eps_g = max(eps_end, eps_g - eps_decay_gen)
        p_g   = max(p_end,   p_g   - p_decay_gen)

    # 保存
    np.save(os.path.join(out_dir, f"{strategy}_run{run_idx}_best.npy"), best_curve)
    np.save(os.path.join(out_dir, f"{strategy}_run{run_idx}_mean.npy"), mean_curve)
    np.save(os.path.join(out_dir, f"{strategy}_run{run_idx}_succ.npy"), success_curve)
    return best_curve, mean_curve, success_curve

# ---------------- 主循环 ----------------------------------------------------
def main():
    strategies = ["eps", "ucb_global", "hybrid"]
    gens       = np.arange(1, GENERATIONS + 1)
    fig, axes  = plt.subplots(1, 3, figsize=(22, 6))

    for strat in strategies:
        all_best = np.zeros((RUNS, GENERATIONS))
        all_mean = np.zeros((RUNS, GENERATIONS))
        all_succ = np.zeros((RUNS, GENERATIONS))
        for r in range(RUNS):
            print(f"[{strat}] run {r + 1}/{RUNS}")
            best, mean, succ = run_once(strat, r)
            all_best[r]      = best
            all_mean[r]      = mean
            all_succ[r]      = succ

        # 聚合
        avg_best, std_best = all_best.mean(0), all_best.std(0)
        avg_mean, std_mean = all_mean.mean(0), all_mean.std(0)
        avg_succ           = all_succ.mean(0)
        std_succ           = all_succ.std(0)

        # 可视化
        axes[0].plot(gens, avg_best, label=f"{strat} best")
        axes[0].fill_between(gens,
                             avg_best - std_best,
                             avg_best + std_best,
                             alpha=0.2)

        axes[1].plot(gens, avg_mean, label=f"{strat} mean")
        axes[1].fill_between(gens,
                             avg_mean - std_mean,
                             avg_mean + std_mean,
                             alpha=0.2)

        axes[2].plot(gens, avg_succ, label=f"{strat} succ-rate")
        axes[2].fill_between(gens,
                             np.clip(avg_succ - std_succ, 0, 1),
                             np.clip(avg_succ + std_succ, 0, 1),
                             alpha=0.2)

        # 保存聚合数据
        np.save(os.path.join(out_dir, f"{strat}_avg_best.npy"), avg_best)
        np.save(os.path.join(out_dir, f"{strat}_avg_mean.npy"), avg_mean)
        np.save(os.path.join(out_dir, f"{strat}_avg_succ.npy"), avg_succ)

    # 图标题与标签
    axes[0].set(title="Average Best Fitness ±1σ",
                xlabel="Generation",
                ylabel="Fitness")
    axes[1].set(title="Average Mean Fitness ±1σ",
                xlabel="Generation",
                ylabel="Fitness")
    axes[2].set(title="Success Rate (best ≥ −110) ±1σ",
                xlabel="Generation",
                ylabel="Rate")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "strategy_comparison_full.png")
    plt.savefig(out_png)
    print(f"Saved comparison plot to {out_png}")
    plt.show()

if __name__ == "__main__":
    main()
