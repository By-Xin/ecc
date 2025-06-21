"""
compare_algorithms.py
---------------------
对比三种 CartPole 学习方案：
1. Tabular Q-learning（离散 ε-greedy）
2. NEAT baseline（纯进化）
3. Advanced Hybrid (DQN + NEAT)

命令行参数：
    --runs          每种算法独立随机种子次数     (默认 3)
    --episodes      Q-learning / Hybrid 训练回合  (默认 5000)
    --generations   NEAT baseline 进化代数        (默认 50)
"""

import argparse, random, math, os, pickle, time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import gym

from advanced_hybrid_cartpole import HybridTrainer, set_seed

# ------------------ ① Tabular Q-learning ------------------
def run_q_learning(episodes: int, seed: int) -> np.ndarray:
    set_seed(seed)
    env = gym.make("CartPole-v1"); obs_dim = env.observation_space.shape[0]
    # 每个维度用固定离散区间
    bins = [
        np.linspace(-4.8, 4.8, 10),          # cart pos
        np.linspace(-5.0, 5.0, 10),          # cart vel
        np.linspace(-0.418, 0.418, 10),      # pole angle
        np.linspace(-5.0, 5.0, 10)           # pole ang vel
    ]
    q_table = np.zeros([9]*obs_dim + [2])    # (9,9,9,9,2)

    def discretize(obs):
        idx = []
        for i, b in enumerate(bins):
            idx.append(int(np.digitize(obs[i], b)))
        return tuple(idx)

    alpha, gamma = 0.1, 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 4000
    rewards = []
    steps = 0
    for ep in range(episodes):
        obs, _ = env.reset(); done = False; total = 0
        while not done:
            eps = eps_end + (eps_start - eps_end)*math.exp(-steps/eps_decay)
            idx = discretize(obs)
            if random.random() < eps:
                a = random.randrange(2)
            else:
                a = int(np.argmax(q_table[idx]))
            nxt, r, term, trunc, _ = env.step(a)
            idx2 = discretize(nxt)
            best_next = np.max(q_table[idx2])
            q_table[idx][a] += alpha*(r + gamma*best_next - q_table[idx][a])
            obs = nxt; total += r; done = term or trunc
            steps += 1
        rewards.append(total)
    return np.array(rewards)

# ------------------ ② NEAT baseline ------------------
def run_neat_baseline(generations: int, seed: int, cfg_path: str = "./config") -> np.ndarray:
    import neat, gym
    set_seed(seed)
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
    pop = neat.Population(cfg)
    env = gym.make("CartPole-v1")
    best_scores = []
    for g in range(generations):
        for genome in pop.population.values():
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            obs, _ = env.reset(); done = False; tot = 0
            while not done:
                act = int(np.argmax(net.activate(obs)))
                obs, r, term, trunc, _ = env.step(act)
                tot += r; done = term or trunc
            genome.fitness = tot
        best = max(pop.population.values(), key=lambda x: x.fitness)
        best_scores.append(best.fitness)
        pop.population = pop.reproduction.reproduce(cfg, pop.species, cfg.pop_size, pop.generation)
        pop.species.speciate(cfg, pop.population, pop.generation)
        pop.generation += 1
    return np.array(best_scores)

# ------------------ ③ Hybrid (DQN + NEAT) ---------------
def run_adv_hybrid(episodes: int, seed: int) -> np.ndarray:
    trainer = HybridTrainer(episodes=episodes, seed=seed)
    return np.array(trainer.run())

# ------------------ ④ 多次运行汇总 ------------------------
def multiple_runs(fn, runs: int, *args) -> Tuple[np.ndarray, np.ndarray]:
    curves = []
    for i in range(runs):
        print(f"   Seed {i} ...")
        curves.append(fn(*args, seed=i))
    # 对齐长度
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    arr = np.stack(curves, axis=0)
    return arr.mean(0), arr.std(0)

# ------------------ ⑤ 绘图 -------------------------------
def plot_results(q_mean, q_std, neat_mean, neat_std, hy_mean, hy_std):
    plt.figure(figsize=(8,5))
    x1 = np.arange(len(q_mean))
    plt.plot(x1, q_mean, label="Q-learning", color="tab:blue")
    plt.fill_between(x1, q_mean-q_std, q_mean+q_std, alpha=0.2, color="tab:blue")

    x2 = np.linspace(0, len(q_mean)-1, len(neat_mean))
    plt.plot(x2, neat_mean, label="NEAT baseline", color="tab:orange")
    plt.fill_between(x2, neat_mean-neat_std, neat_mean+neat_std, alpha=0.2, color="tab:orange")

    plt.plot(x1, hy_mean, label="Hybrid (DQN+NEAT)", color="tab:green")
    plt.fill_between(x1, hy_mean-hy_std, hy_mean+hy_std, alpha=0.2, color="tab:green")

    plt.xlabel("Episode (Q / Hybrid)  — mapped Generation (NEAT)")
    plt.ylabel("Total reward")
    plt.title("CartPole: mean ±1σ over runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("adv_comparison.png", dpi=300)
    print("Saved figure to adv_comparison.png")

# ------------------ ⑥ CLI -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()

    print("=== Running Tabular Q-learning ===")
    q_mean, q_std = multiple_runs(run_q_learning, args.runs, args.episodes)
    print("=== Running NEAT baseline ===")
    neat_mean, neat_std = multiple_runs(run_neat_baseline, args.runs, args.generations)
    print("=== Running Hybrid (DQN+NEAT) ===")
    hy_mean, hy_std = multiple_runs(run_adv_hybrid, args.runs, args.episodes)

    plot_results(q_mean, q_std, neat_mean, neat_std, hy_mean, hy_std)


if __name__ == "__main__":
    main()

