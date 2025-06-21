#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare four exploration strategies on MountainCar-v0 with NEAT

    eps            – ε-greedy (ε linearly decays)
    ucb_global     – global-UCB across generations
    hybrid         – p-schedule (退火) 切换 UCB/ε
    hybrid_ts      – episode-wise Thompson-Sampling bandit 选择 UCB 或 ε

每种策略跑 RUNS seeds、GENERATIONS generations，
输出 Best / Mean / Success-Rate / p(A) 四联图。
"""

import os, math, random
import gym
import neat
import numpy as np
import matplotlib.pyplot as plt
from neat import nn

# ---------------- Hyper-parameters -----------------------------------------
GAME            = "MountainCar-v0"
CONFIG_PATH     = "./config"
GENERATIONS     = 100           # ← 延长训练
EPISODES        = 5
EP_STEPS        = 200
RUNS            = 8

# ε-greedy schedule
eps_start, eps_end   = 1.0, 0.05
eps_decay_gen        = (eps_start - eps_end) / (GENERATIONS - 1)

# hybrid (退火) schedule
p_start, p_end       = 0.9, 0.2
p_decay_gen          = (p_start - p_end) / (GENERATIONS - 1)

# UCB constant
C_VAL = 0.5

# 输出目录
out_dir = "results_ts"
os.makedirs(out_dir, exist_ok=True)

# ---------------- Thompson-Sampling Bandit ---------------------------------
class GaussianTSBandit:
    """两臂正态 TS；先验 N(0,1)，观测方差近似 1。"""
    def __init__(self):
        self.n      = np.zeros(2, dtype=int)   # 计数
        self.mean   = np.zeros(2, dtype=float) # 样本均值

    def select(self) -> int:
        sigma = 1.0 / np.sqrt(self.n + 1)      # 近似后验方差
        sample = np.random.normal(self.mean, sigma)
        return int(np.argmax(sample))

    def update(self, arm: int, reward: float):
        self.n[arm]  += 1
        delta         = reward - self.mean[arm]
        self.mean[arm]+= delta / self.n[arm]

    @property
    def p_use_ucb(self) -> float:
        """软指标：根据当前均值返回 P(A> B)。"""
        diff = self.mean[0] - self.mean[1]
        var  = 1.0/(self.n[0]+1) + 1.0/(self.n[1]+1)
        return 0.5 * (1 + math.erf(diff / math.sqrt(2*var)))

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
                while not done and step < EP_STEPS:
                    step += 1
                    if np.random.rand() < eps_gen:
                        act = env.action_space.sample()
                    else:
                        act = int(np.argmax(net.activate(obs)))
                    obs, r, term, trunc, _ = env.step(act)
                    done, total = term or trunc, total + r
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()
    return eval_genomes


def make_ucb_evaluator(global_ucb: dict):
    counts_g, qvals_g = global_ucb["counts"], global_ucb["qvals"]
    def eval_genomes(genomes, config):
        counts, qvals = counts_g.copy(), qvals_g.copy()
        t = global_ucb["t"]
        env = gym.make(GAME)
        for _, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, config)
            rewards = []
            for _ in range(EPISODES):
                obs, _ = env.reset()
                total, step, done = 0.0, 0, False
                while not done and step < EP_STEPS:
                    step += 1; t += 1
                    bonus = C_VAL * np.sqrt(np.log(t + 1) / (counts + 1))
                    act   = int(np.argmax(np.array(net.activate(obs)) + bonus))
                    obs, r, term, trunc, _ = env.step(act)
                    done, total = term or trunc, total + r
                    counts[act] += 1
                    qvals[act]  += (r - qvals[act]) / counts[act]
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()
        counts_g[:] = counts; qvals_g[:] = qvals; global_ucb["t"] = t
    return eval_genomes


def make_hybrid_evaluator(global_ucb: dict,
                          eps_gen: float,
                          p_use_ucb: float):
    counts_g, qvals_g = global_ucb["counts"], global_ucb["qvals"]
    def eval_genomes(genomes, config):
        counts, qvals = counts_g.copy(), qvals_g.copy()
        t = global_ucb["t"]
        env = gym.make(GAME)
        for _, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, config)
            rewards = []
            for _ in range(EPISODES):
                use_ucb = np.random.rand() < p_use_ucb
                obs, _  = env.reset()
                total, step, done = 0.0, 0, False
                while not done and step < EP_STEPS:
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
                    done, total = term or trunc, total + r
                    if use_ucb:
                        counts[act] += 1
                        qvals[act]  += (r - qvals[act]) / counts[act]
                rewards.append(total)
            genome.fitness = float(np.mean(rewards))
        env.close()
        counts_g[:] = counts; qvals_g[:] = qvals; global_ucb["t"] = t
    return eval_genomes


def make_ts_evaluator(global_ucb: dict,
                      eps_gen: float,
                      bandit: GaussianTSBandit):
    counts_g, qvals_g = global_ucb["counts"], global_ucb["qvals"]
    def eval_genomes(genomes, config):
        counts, qvals = counts_g.copy(), qvals_g.copy()
        t = global_ucb["t"]
        env = gym.make(GAME)
        for _, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, config)
            rewards = []
            for _ in range(EPISODES):
                arm = bandit.select()          # 0 → UCB, 1 → ε
                use_ucb = (arm == 0)
                obs, _  = env.reset()
                total, step, done = 0.0, 0, False
                while not done and step < EP_STEPS:
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
                    done, total = term or trunc, total + r
                    if use_ucb:
                        counts[act] += 1
                        qvals[act]  += (r - qvals[act]) / counts[act]
                rewards.append(total)
                bandit.update(arm, total)       # ← 更新上层 Bandit
            genome.fitness = float(np.mean(rewards))
        env.close()
        counts_g[:] = counts; qvals_g[:] = qvals; global_ucb["t"] = t
    return eval_genomes

# ---------------- 单次实验 ---------------------------------------------------
def run_once(strategy: str, run_idx: int):
    best_curve, mean_curve, succ_curve = (np.zeros(GENERATIONS) for _ in range(3))
    p_curve = np.zeros(GENERATIONS)     # 仅 hybrid_ts 填充

    eps_g, p_g = eps_start, p_start

    env_tmp = gym.make(GAME)
    n_act   = env_tmp.action_space.n
    env_tmp.close()
    global_ucb = {"counts": np.zeros(n_act, int),
                  "qvals" : np.zeros(n_act, float),
                  "t"     : 0}
    bandit = GaussianTSBandit()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    pop = neat.Population(config)
    ucb_eval = make_ucb_evaluator(global_ucb)      # 共用

    for gen in range(GENERATIONS):
        if strategy == "eps":
            eval_fn = make_eps_evaluator(eps_g)
        elif strategy == "ucb_global":
            eval_fn = ucb_eval
        elif strategy == "hybrid":
            eval_fn = make_hybrid_evaluator(global_ucb, eps_g, p_g)
        elif strategy == "hybrid_ts":
            eval_fn = make_ts_evaluator(global_ucb, eps_g, bandit)
            p_curve[gen] = bandit.p_use_ucb
        else:
            raise ValueError

        pop.run(eval_fn, 1)

        best_curve[gen] = pop.best_genome.fitness
        mean_curve[gen] = np.mean([g.fitness for g in pop.population.values()
                                   if g.fitness is not None])
        succ_curve[gen] = 1.0 if best_curve[gen] >= -110 else 0.0

        # 更新调度
        eps_g = max(eps_end, eps_g - eps_decay_gen)
        p_g   = max(p_end,   p_g   - p_decay_gen)

    # 存盘
    base = f"{strategy}_run{run_idx}"
    np.save(os.path.join(out_dir, f"{base}_best.npy"), best_curve)
    np.save(os.path.join(out_dir, f"{base}_mean.npy"), mean_curve)
    np.save(os.path.join(out_dir, f"{base}_succ.npy"), succ_curve)
    if strategy == "hybrid_ts":
        np.save(os.path.join(out_dir, f"{base}_p.npy"), p_curve)
    return best_curve, mean_curve, succ_curve, p_curve

# ---------------- 主循环 ----------------------------------------------------
def main():
    strategies = ["eps", "ucb_global", "hybrid", "hybrid_ts"]
    gens = np.arange(1, GENERATIONS + 1)
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    for strat in strategies:
        all_best = np.zeros((RUNS, GENERATIONS))
        all_mean = np.zeros((RUNS, GENERATIONS))
        all_succ = np.zeros((RUNS, GENERATIONS))
        all_p    = np.zeros((RUNS, GENERATIONS))
        for r in range(RUNS):
            print(f"[{strat}] run {r+1}/{RUNS}")
            best, mean, succ, p_traj = run_once(strat, r)
            all_best[r], all_mean[r], all_succ[r] = best, mean, succ
            all_p[r] = p_traj

        avg_best, std_best = all_best.mean(0), all_best.std(0)
        avg_mean, std_mean = all_mean.mean(0), all_mean.std(0)
        avg_succ, std_succ = all_succ.mean(0), all_succ.std(0)

        axes[0].plot(gens, avg_best, label=f"{strat} best")
        axes[0].fill_between(gens, avg_best - std_best, avg_best + std_best, alpha=0.2)
        axes[1].plot(gens, avg_mean, label=f"{strat} mean")
        axes[1].fill_between(gens, avg_mean - std_mean, avg_mean + std_mean, alpha=0.2)
        axes[2].plot(gens, avg_succ, label=f"{strat} succ")
        axes[2].fill_between(gens,
                             np.clip(avg_succ - std_succ, 0, 1),
                             np.clip(avg_succ + std_succ, 0, 1),
                             alpha=0.2)
        if strat == "hybrid_ts":
            avg_p, std_p = all_p.mean(0), all_p.std(0)
            axes[3].plot(gens, avg_p, label="hybrid_ts P(UCB)")
            axes[3].fill_between(gens,
                                 np.clip(avg_p - std_p, 0, 1),
                                 np.clip(avg_p + std_p, 0, 1),
                                 alpha=0.2)

        # 聚合数据存盘
        np.save(os.path.join(out_dir, f"{strat}_avg_best.npy"), avg_best)
        np.save(os.path.join(out_dir, f"{strat}_avg_mean.npy"), avg_mean)
        np.save(os.path.join(out_dir, f"{strat}_avg_succ.npy"), avg_succ)
        if strat == "hybrid_ts":
            np.save(os.path.join(out_dir, f"{strat}_avg_p.npy"), avg_p)

    # 轴标签
    axes[0].set(title="Avg Best Fitness ±1σ", xlabel="Generation", ylabel="Fitness")
    axes[1].set(title="Avg Mean Fitness ±1σ", xlabel="Generation", ylabel="Fitness")
    axes[2].set(title="Success Rate (≥-110)", xlabel="Generation", ylabel="Rate")
    axes[3].set(title="Prob of UCB", xlabel="Generation", ylabel="P(UCB)")

    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True)

    plt.tight_layout()
    fname = os.path.join(out_dir, "strategy_comparison_TS.png")
    plt.savefig(fname)
    print(f"Saved plot to {fname}")
    plt.show()

if __name__ == "__main__":
    main()
