#!/usr/bin/env python
"""
compare_algorithms.py
比较 Hybrid 与 Vanilla-NEAT 在 CartPole-v1 上的收敛曲线（多随机种子均值 ± 置信区间）。
"""

from __future__ import annotations
import json, os, random, pathlib, numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.neat_algorithms import (
    eval_genomes_hybrid, eval_genomes_vanilla, run_training
)

# ============ 可调参数 ============ #
NUM_GENERATIONS = 100         # 训练代数
SEEDS           = range(20)   # 随机种子列表
OUT_DIR         = pathlib.Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ============ 容器 ============ #
records: dict[str, list[list[float]]] = {
    "Hybrid":  [],
    "Vanilla": []
}

# ============ 主循环 ============ #
for alg_name, eval_fn in [("Hybrid", eval_genomes_hybrid),
                          ("Vanilla", eval_genomes_vanilla)]:
    for seed in SEEDS:
        print(f"[{alg_name}]  seed={seed}")
        # 运行一次进化；返回每代最佳 raw reward
        _, _, _, best_rewards = run_training(
            NUM_GENERATIONS, eval_fn, run_seed=seed, algorithm_name=alg_name
        )
        records[alg_name].append(best_rewards)
        # 持久化单条曲线，便于后续其他统计
        with open(OUT_DIR / f"{alg_name}_seed{seed}.json", "w") as f:
            json.dump(best_rewards, f)

# ============ 统计量 ============ #
def compute_stats(arr: np.ndarray):
    """
    arr.shape == (num_runs, num_generations)
    返回 mean, std, ci (95%).
    """
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0, ddof=1)
    ci   = 1.96 * std / np.sqrt(arr.shape[0])
    return mean, std, ci

stats = {}
for alg_name, curves in records.items():
    arr                   = np.array(curves)           # shape (runs, gens)
    stats[alg_name]       = compute_stats(arr)
    np.save(OUT_DIR / f"{alg_name}_stats.npy", arr)    # 原始数组备份

# ============ 绘图 ============ #
plt.figure(figsize=(8, 5))
x = np.arange(1, NUM_GENERATIONS + 1)

for alg_name, (mean, std, ci) in stats.items():
    line, = plt.plot(x, mean, label=alg_name, linewidth=2)
    plt.fill_between(x, mean - ci, mean + ci, alpha=0.25)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Hybrid vs. Vanilla-NEAT on CartPole-v1")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

fig_path = OUT_DIR / "fitness_comparison.png"
plt.savefig(fig_path, dpi=300)
print(f"\nFigure saved to: {fig_path.absolute()}")
plt.show()

