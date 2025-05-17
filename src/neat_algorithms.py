"""
NEAT 训练与评估模块
扩展：新增 Hybrid (NEAT + DQN) 评估函数 eval_genomes_hybrid
并在 run_training 中自动注入 RL Agent 与 Env。
"""
import neat, random, numpy as np
from tqdm.auto import tqdm

from src.config  import EP_STEP, NUM_GENERATIONS, VERBOSE
from src.utils   import log_print
from src.environment import (
    make_env, evaluate_single_genome,
    evaluate_single_genome_hybrid          # ★ 新
)
from src.evaluation import local_adaptation
from src.rl_agent   import DQNAgent        # ★ 新

# ---------------- Meta-NEAT ---------------- #
def eval_genomes_meta(genomes, config,
                      gen=0, total_gens=NUM_GENERATIONS):
    env          = make_env()
    gen_progress = gen / total_gens
    log_print(f"== 评估第 {gen+1}/{total_gens} 代 (进度 {gen_progress:.2f}) ==")

    iterator = genomes if VERBOSE else tqdm(genomes, leave=False,
                                            desc=f"Meta-NEAT Gen {gen+1}")
    for gid, genome in iterator:
        _, best_rwd = local_adaptation(genome, config, env, gen_progress)
        genome.fitness = best_rwd / float(EP_STEP)
    env.close()


# ---------------- Vanilla-NEAT -------------- #
def eval_genomes_vanilla(genomes, config):
    env = make_env()
    iterator = genomes if VERBOSE else tqdm(genomes,
                                            desc="Vanilla-NEAT", leave=False)
    for gid, genome in iterator:
        reward = evaluate_single_genome(genome, config, env)
        genome.fitness = reward / float(EP_STEP)
    env.close()


# ---------------- Hybrid (NEAT + DQN) ------- #
def eval_genomes_hybrid(genomes, config,
                        rl_agent: DQNAgent, env):
    """
    代内评估：NEAT 个体 × 同一个共享 DQNAgent。
    """
    iterator = genomes if VERBOSE else tqdm(genomes,
                                            desc="Hybrid-NEAT", leave=False)
    for gid, genome in iterator:
        reward = evaluate_single_genome_hybrid(genome, config, rl_agent, env)
        genome.fitness = reward / float(EP_STEP)


# ---------------- 训练主函数 ---------------- #
def run_training(num_generations: int,
                 eval_fn, run_seed: int,
                 algorithm_name: str = ""):
    """
    返回
        winner, neat_config, stats, generation_best_rewards
    """
    from src.config import CONFIG_PATH

    # —— 固定随机性 —— #
    random.seed(run_seed); np.random.seed(run_seed)

    neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           CONFIG_PATH)
    pop = neat.Population(neat_cfg)
    pop.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter(); pop.add_reporter(stats)

    # ---------- 评估函数包装 ---------- #
    if eval_fn == eval_genomes_meta:
        def wrapper(genomes, cfg):
            cur_gen = pop.generation
            return eval_fn(genomes, cfg, cur_gen, num_generations)
        eval_wrapper = wrapper

    elif eval_fn == eval_genomes_hybrid:
        # 为一次完整 run 共享同一 DQNAgent & Env
        rl_agent = DQNAgent()
        env      = make_env()
        def wrapper(genomes, cfg):
            return eval_fn(genomes, cfg, rl_agent, env)
        eval_wrapper = wrapper
    else:
        eval_wrapper = eval_fn      # Vanilla-NEAT

    # ---------- 进度条 ---------- #
    pbar_desc = algorithm_name or "Evolution"
    with tqdm(total=num_generations, desc=pbar_desc, leave=True) as pbar:
        class _Reporter(neat.reporting.BaseReporter):
            def end_generation(self, *a): pbar.update(1)
        pop.add_reporter(_Reporter())

        winner = pop.run(eval_wrapper, num_generations)

    # —— 收集每代最佳奖励 —— #
    gen_best = [g.fitness * EP_STEP for g in stats.most_fit_genomes]
    return winner, neat_cfg, stats, gen_best

