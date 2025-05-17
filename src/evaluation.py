"""
评估模块
负责：
  • Meta-NEAT 局部适应 (local_adaptation)
  • 终局评估 evaluate_genome
  • 轻度基因组变异 mutate_genome
"""

import copy
import random

import gym
import neat
import numpy as np
from tqdm.auto import tqdm

from src.config import (
    EP_STEP,
    BASE_LOCAL_TRIALS,
    MAX_LOCAL_TRIALS,
    local_search_stats,
    collect_statistics,
    VERBOSE,               # ★ 修复: 显式导入 VERBOSE
)
from src.utils import log_print
from src.environment import evaluate_single_genome


# -------------------------------------------------- #
def mutate_genome(genome: neat.DefaultGenome, config: neat.Config):
    """轻度变异：仅微调连接权重。"""
    for cg in genome.connections.values():
        if random.random() < 0.6:
            cg.weight += random.gauss(0, 0.1)


# -------------------------------------------------- #
def local_adaptation(genome, config, env, gen_progress: float = 0.5):
    """Meta-NEAT 内循环：对单个 genome 进行泊松分布次数的局部搜索。

    返回 (best_genome, best_reward)
    """
    log_print("开始局部适应")
    best_genome = copy.deepcopy(genome)
    best_reward = evaluate_single_genome(best_genome, config, env)
    log_print(f"基准奖励: {best_reward}")

    # 自适应 lambda
    if gen_progress < 0.3:
        lambda_param = BASE_LOCAL_TRIALS * 1.2
    elif gen_progress > 0.7:
        lambda_param = BASE_LOCAL_TRIALS * 0.8
    else:
        lambda_param = BASE_LOCAL_TRIALS

    # 泊松采样局部搜索次数（设上下限）
    local_trials = np.random.poisson(lambda_param)
    local_trials = max(1, min(local_trials, MAX_LOCAL_TRIALS))
    log_print(f"当前局部搜索次数: {local_trials} (lambda={lambda_param:.2f})")

    # 进度条（仅在 VERBOSE = True 时显示）
    iterator = range(local_trials)
    if VERBOSE:
        iterator = tqdm(iterator, desc="局部适应试验", leave=False)

    improvements = 0
    for i in iterator:
        log_print(f"局部适应试验 {i + 1}/{local_trials}")
        mutated = copy.deepcopy(genome)
        mutate_genome(mutated, config)
        reward = evaluate_single_genome(mutated, config, env)
        log_print(f"变异后奖励: {reward}")

        if reward > best_reward:
            best_genome, best_reward = mutated, reward
            improvements += 1
            log_print(f"更新最佳奖励: {best_reward}")

    # 统计信息汇总
    if collect_statistics:
        local_search_stats["trials"].append(local_trials)
        local_search_stats["improvements"].append(improvements)
        ratio = improvements / local_trials if local_trials else 0
        local_search_stats["improvement_ratios"].append(ratio)

    log_print(
        f"局部适应完成。总尝试: {local_trials}, 有效改进: {improvements}"
    )
    return best_genome, best_reward


# -------------------------------------------------- #
def evaluate_genome(winner, config, num_eval_episodes: int = 5) -> float:
    """在标准环境下对最终 winner 多次评估，返回平均奖励。"""
    from src.config import GAME  # 避免循环引用

    env = gym.make(GAME)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    rewards = []

    iterator = range(num_eval_episodes)
    if not VERBOSE:
        iterator = tqdm(iterator, desc="最终评估", leave=False)

    for _ in iterator:
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action = int(np.argmax(net.activate(obs)))
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards))

