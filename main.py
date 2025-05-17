from src.neat_algorithms import (
    eval_genomes_meta, eval_genomes_vanilla,
    eval_genomes_hybrid, run_training
)

if __name__ == "__main__":
    gens = 15
    seed = 123
    print("=== Running Hybrid (NEAT + DQN) ===")
    _, _, _, hyb_fit = run_training(gens, eval_genomes_hybrid, seed, "Hybrid")

    print("\n=== Running Pure Meta-NEAT ===")
    _, _, _, meta_fit = run_training(gens, eval_genomes_meta, seed, "Meta-NEAT")

    print("\n=== Running Pure Vanilla-NEAT ===")
    _, _, _, van_fit = run_training(gens, eval_genomes_vanilla, seed, "Vanilla-NEAT")

    # 可视化
    from src.visualization import plot_training_progression
    plot_training_progression([hyb_fit], [van_fit])

