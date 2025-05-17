"""
示例：对比三种算法在 CartPole-v1 上的表现。
运行:  python main_hybrid.py
"""
from src.neat_algorithms import (
    eval_genomes_meta, eval_genomes_vanilla,
    eval_genomes_hybrid, run_training
)

if __name__ == "__main__":
    gens, seed = 10, 2025

    print("\n=== Hybrid (NEAT + DQN) ===")
    _, _, _, hyb = run_training(gens, eval_genomes_hybrid, seed, "Hybrid")

    print("\n=== Meta-NEAT ===")
    _, _, _, meta = run_training(gens, eval_genomes_meta, seed, "Meta-NEAT")

    print("\n=== Vanilla-NEAT ===")
    _, _, _, van = run_training(gens, eval_genomes_vanilla, seed, "Vanilla-NEAT")

    # 若需绘图，可在此调用 visualization.py

