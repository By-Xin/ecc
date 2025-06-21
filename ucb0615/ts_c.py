#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thompson Sampling策略选择不同C值的UCB在MountainCar-v0上的对比实验
使用Beta分布进行Thompson Sampling，在两个UCB策略之间自适应选择
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

import os, math, random
import gym
import neat
import numpy as np
import matplotlib.pyplot as plt
from neat import nn
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ---------------- Hyper-parameters -----------------------------------------
GAME            = "MountainCar-v0"
CONFIG_PATH     = "./config"
GENERATIONS     = 100
EPISODES        = 5
EP_STEPS        = 100
RUNS            = 3
NUM_THREADS     = mp.cpu_count()

# UCB constants - 两个不同的探索强度
C_VAL_LOW = 0.3   # 保守探索
C_VAL_HIGH = 1.0  # 激进探索

# Thompson Sampling 参数
REWARD_THRESHOLD = -120  # 奖励阈值，超过此值认为策略成功
ALPHA_INIT = 1.0  # Beta分布初始alpha参数
BETA_INIT = 1.0   # Beta分布初始beta参数

# 输出目录
out_dir = "results_thompson_ucb_parallel"
os.makedirs(out_dir, exist_ok=True)

# ---------------- Thompson Sampling UCB策略选择器 --------------------------
class ThompsonUCBSelector:
    def __init__(self, c_low=C_VAL_LOW, c_high=C_VAL_HIGH, threshold=REWARD_THRESHOLD):
        self.c_low = c_low
        self.c_high = c_high
        self.threshold = threshold
        
        # Beta分布参数：[low_strategy, high_strategy]
        self.alpha = np.array([ALPHA_INIT, ALPHA_INIT])
        self.beta = np.array([BETA_INIT, BETA_INIT])
        
        # 统计信息
        self.strategy_counts = np.zeros(2)
        self.strategy_rewards = [[], []]
        self.selection_history = []
        
    def select_strategy(self):
        """使用Thompson Sampling选择UCB策略"""
        # 从Beta分布采样
        samples = np.random.beta(self.alpha, self.beta)
        strategy_idx = np.argmax(samples)
        
        self.strategy_counts[strategy_idx] += 1
        self.selection_history.append(strategy_idx)
        
        return strategy_idx, [self.c_low, self.c_high][strategy_idx]
    
    def update_strategy(self, strategy_idx, reward):
        """根据奖励更新Thompson Sampling参数"""
        self.strategy_rewards[strategy_idx].append(reward)
        
        # 根据奖励是否超过阈值更新Beta分布参数
        if reward >= self.threshold:
            self.alpha[strategy_idx] += 1  # 成功
        else:
            self.beta[strategy_idx] += 1   # 失败
    
    def get_strategy_probabilities(self):
        """计算当前策略选择概率"""
        return self.alpha / (self.alpha + self.beta)

# ---------------- 并行化的单个genome评估函数 -------------------------------
def evaluate_single_genome_thompson(args):
    """使用Thompson Sampling策略评估单个genome"""
    genome_id, genome, config, selector, env_actions = args
    net = nn.FeedForwardNetwork.create(genome, config)
    rewards = []
    episode_strategies = []

    env = gym.make(GAME)
    for episode in range(EPISODES):
        # 为每个episode选择UCB策略
        strategy_idx, c_val = selector.select_strategy()
        episode_strategies.append((strategy_idx, c_val))
        
        obs, _ = env.reset()
        total, step, done = 0.0, 0, False
        
        # 局部UCB计数（每个episode重置）
        local_counts = np.zeros(env_actions)
        
        while not done and step < EP_STEPS:
            step += 1
            # 使用选定的UCB策略
            bonus = c_val * np.sqrt(np.log(step + 1) / (local_counts + 1))
            qvals = np.array(net.activate(obs))
            act = int(np.argmax(qvals + bonus))
            
            local_counts[act] += 1
            obs, r, term, trunc, _ = env.step(act)
            done, total = term or trunc, total + r

        rewards.append(total)
        # 更新Thompson Sampling参数
        selector.update_strategy(strategy_idx, total)

    env.close()
    fitness = float(np.mean(rewards))
    strategy_info = {
        'strategies': episode_strategies,
        'rewards': rewards,
        'fitness': fitness
    }
    return genome_id, fitness, strategy_info

# ---------------- 并行化的评估器工厂 -----------------------------------------
def make_thompson_evaluator(selector, env_actions):
    def eval_genomes(genomes, config):
        best_fitness = float('-inf')
        tasks = [(gid, g, config, selector, env_actions) for gid, g in genomes]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            results = list(executor.map(evaluate_single_genome_thompson, tasks))

        generation_strategies = []
        for genome_id, fitness, strategy_info in results:
            generation_strategies.append(strategy_info)
            for gid, genome in genomes:
                if gid == genome_id:
                    genome.fitness = fitness
                    best_fitness = max(best_fitness, fitness)
                    break

        return best_fitness, generation_strategies

    return eval_genomes

# ---------------- 并行化的单次实验 -------------------------------------------
def run_thompson_experiment(run_idx):
    """Thompson Sampling UCB策略选择实验"""
    np.random.seed(42 + run_idx)
    random.seed(42 + run_idx)

    # 初始化环境信息
    env_tmp = gym.make(GAME)
    n_act = env_tmp.action_space.n
    env_tmp.close()

    # 初始化Thompson Sampling选择器
    selector = ThompsonUCBSelector()
    
    # 记录数组
    best_curve = np.zeros(GENERATIONS)
    mean_curve = np.zeros(GENERATIONS)
    succ_curve = np.zeros(GENERATIONS)
    strategy_prob_curves = np.zeros((GENERATIONS, 2))  # [low_c, high_c]概率
    strategy_usage_curves = np.zeros((GENERATIONS, 2))  # 策略使用次数

    # NEAT 配置与种群
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    pop = neat.Population(config)

    for gen in range(GENERATIONS):
        eval_fn = make_thompson_evaluator(selector, n_act)
        pop.run(eval_fn, 1)

        # 记录指标
        best_genome = max(pop.population.values(), key=lambda g: g.fitness if g.fitness else float('-inf'))
        all_fitness = [g.fitness for g in pop.population.values() if g.fitness is not None]
        
        best_curve[gen] = best_genome.fitness
        mean_curve[gen] = np.mean(all_fitness)
        succ_curve[gen] = 1.0 if best_curve[gen] >= -110 else 0.0
        
        # 记录Thompson Sampling策略信息
        strategy_prob_curves[gen] = selector.get_strategy_probabilities()
        strategy_usage_curves[gen] = selector.strategy_counts.copy()
        
        # 重置每代的计数（可选）
        # selector.strategy_counts = np.zeros(2)

    return (run_idx, best_curve, mean_curve, succ_curve, 
            strategy_prob_curves, strategy_usage_curves, selector)

# ---------------- 主循环 ----------------------------------------------------
def main():
    gens = np.arange(1, GENERATIONS + 1)
    
    # 结果汇总数组
    all_best = np.zeros((RUNS, GENERATIONS))
    all_mean = np.zeros((RUNS, GENERATIONS))
    all_succ = np.zeros((RUNS, GENERATIONS))
    all_strategy_probs = np.zeros((RUNS, GENERATIONS, 2))
    all_strategy_usage = np.zeros((RUNS, GENERATIONS, 2))
    
    print(f"使用 {NUM_THREADS} 个线程进行Thompson Sampling UCB策略选择实验")
    print(f"UCB参数: C_low={C_VAL_LOW}, C_high={C_VAL_HIGH}")
    print(f"奖励阈值: {REWARD_THRESHOLD}")

    # 并行运行多个seeds
    with ProcessPoolExecutor(max_workers=min(RUNS, NUM_THREADS)) as executor:
        futures = [executor.submit(run_thompson_experiment, r) for r in range(RUNS)]
        for future in as_completed(futures):
            (run_idx, best, mean, succ, strategy_probs, 
             strategy_usage, final_selector) = future.result()
            print(f"  完成 run {run_idx+1}/{RUNS}")
            
            all_best[run_idx] = best
            all_mean[run_idx] = mean
            all_succ[run_idx] = succ
            all_strategy_probs[run_idx] = strategy_probs
            all_strategy_usage[run_idx] = strategy_usage
            
            # 打印最终策略偏好
            final_probs = final_selector.get_strategy_probabilities()
            print(f"    最终策略概率: Low_C({C_VAL_LOW})={final_probs[0]:.3f}, "
                  f"High_C({C_VAL_HIGH})={final_probs[1]:.3f}")

    # 计算均值和标准差
    avg_best, std_best = all_best.mean(0), all_best.std(0)
    avg_mean, std_mean = all_mean.mean(0), all_mean.std(0)
    avg_succ, std_succ = all_succ.mean(0), all_succ.std(0)
    avg_strategy_probs = all_strategy_probs.mean(0)
    std_strategy_probs = all_strategy_probs.std(0)

    # 创建综合可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 性能指标对比
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(gens, avg_best, 'b-', label=f"Thompson UCB Best", linewidth=2)
    plt.fill_between(gens, avg_best - std_best, avg_best + std_best, alpha=0.2)
    plt.axhline(y=-110, color='r', linestyle='--', alpha=0.7, label='Success Threshold')
    plt.title("Best Fitness Evolution", fontsize=12, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    plt.plot(gens, avg_mean, 'g-', label="Thompson UCB Mean", linewidth=2)
    plt.fill_between(gens, avg_mean - std_mean, avg_mean + std_mean, alpha=0.2)
    plt.title("Mean Fitness Evolution", fontsize=12, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    plt.plot(gens, avg_succ, 'orange', label="Success Rate", linewidth=2)
    plt.fill_between(gens,
                     np.clip(avg_succ - std_succ, 0, 1),
                     np.clip(avg_succ + std_succ, 0, 1),
                     alpha=0.2)
    plt.title("Success Rate (≥ -110)", fontsize=12, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Thompson Sampling策略选择概率
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(gens, avg_strategy_probs[:, 0], 'b-', 
             label=f'Conservative UCB (C={C_VAL_LOW})', linewidth=2)
    plt.plot(gens, avg_strategy_probs[:, 1], 'r-', 
             label=f'Aggressive UCB (C={C_VAL_HIGH})', linewidth=2)
    plt.fill_between(gens, 
                     avg_strategy_probs[:, 0] - std_strategy_probs[:, 0],
                     avg_strategy_probs[:, 0] + std_strategy_probs[:, 0],
                     alpha=0.2, color='blue')
    plt.fill_between(gens,
                     avg_strategy_probs[:, 1] - std_strategy_probs[:, 1], 
                     avg_strategy_probs[:, 1] + std_strategy_probs[:, 1],
                     alpha=0.2, color='red')
    plt.title("Thompson Sampling Strategy Probabilities", fontsize=12, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Selection Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 3. 策略使用累计次数
    ax5 = plt.subplot(2, 3, 5)
    avg_usage = all_strategy_usage.mean(0)
    plt.plot(gens, avg_usage[:, 0], 'b-', 
             label=f'Conservative UCB Usage', linewidth=2)
    plt.plot(gens, avg_usage[:, 1], 'r-', 
             label=f'Aggressive UCB Usage', linewidth=2)
    plt.title("Cumulative Strategy Usage", fontsize=12, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Usage Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 策略选择比例饼图（最终代）
    ax6 = plt.subplot(2, 3, 6)
    final_usage = avg_usage[-1]
    total_usage = final_usage.sum()
    if total_usage > 0:
        percentages = final_usage / total_usage * 100
        labels = [f'Conservative\n(C={C_VAL_LOW})\n{percentages[0]:.1f}%',
                  f'Aggressive\n(C={C_VAL_HIGH})\n{percentages[1]:.1f}%']
        colors = ['lightblue', 'lightcoral']
        plt.pie(final_usage, labels=labels, colors=colors, autopct='', startangle=90)
        plt.title("Final Strategy Usage Distribution", fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # 保存图像
    fname = os.path.join(out_dir, "thompson_ucb_analysis.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n详细分析结果已保存到 {fname}")
    
    # 保存数值结果
    results_fname = os.path.join(out_dir, "thompson_ucb_results.npz")
    np.savez(results_fname,
             generations=gens,
             best_fitness=all_best,
             mean_fitness=all_mean,
             success_rate=all_succ,
             strategy_probabilities=all_strategy_probs,
             strategy_usage=all_strategy_usage,
             c_low=C_VAL_LOW,
             c_high=C_VAL_HIGH,
             threshold=REWARD_THRESHOLD)
    print(f"数值结果已保存到 {results_fname}")
    
    # 打印总结统计
    print("\n=== 实验总结 ===")
    print(f"最终代最佳适应度: {avg_best[-1]:.2f} ± {std_best[-1]:.2f}")
    print(f"最终代平均适应度: {avg_mean[-1]:.2f} ± {std_mean[-1]:.2f}")
    print(f"最终代成功率: {avg_succ[-1]:.2f} ± {std_succ[-1]:.2f}")
    print(f"最终保守策略概率: {avg_strategy_probs[-1, 0]:.3f} ± {std_strategy_probs[-1, 0]:.3f}")
    print(f"最终激进策略概率: {avg_strategy_probs[-1, 1]:.3f} ± {std_strategy_probs[-1, 1]:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()