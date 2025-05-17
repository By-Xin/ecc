"""
环境相关模块
包含环境创建和交互的函数
"""
import gym, neat, numpy as np
from src.config  import EP_STEP, USE_SOFTMAX
from src.utils   import softmax

# —— Hybrid 相关 —— #
from src.rl_agent        import DQNAgent
from src.hyper_heuristic import choose_policy

# --------------------------------------------------- #
def make_env(seed: int | None = None) -> gym.Env:
    """创建 CartPole-v1 环境；可指定随机种子。"""
    from src.config import GAME
    env = gym.make(GAME)
    if seed is not None:
        env.reset(seed=seed)
    return env

# --------------------------------------------------- #
def evaluate_single_genome(genome, config, env) -> float:
    """
    Vanilla 单次评估：NEAT 网络独立完成一个 episode。
    """
    from src.config import TAU

    net, total_reward = neat.nn.FeedForwardNetwork.create(genome, config), 0.0
    state, _ = env.reset()
    for _ in range(EP_STEP):
        out = net.activate(state)
        if USE_SOFTMAX:
            probs  = softmax(out, TAU)
            action = np.random.choice(len(probs), p=probs)
        else:
            action = int(np.argmax(out))
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated: break
    return total_reward


# --------------------------------------------------- #
def evaluate_single_genome_hybrid(genome, neat_config,
                                  rl_agent: DQNAgent,
                                  env) -> float:
    """
    Hybrid 评估：episode 内每一步伯努利决定使用
    （1）已有 NEAT 网络，或
    （2）在线学习的 DQNAgent。
    """
    from src.config import TAU

    net       = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    state, _  = env.reset()
    total_rwd = 0.0

    for _ in range(EP_STEP):
        # -------- 决策 -------- #
        if choose_policy() == 'RL':
            action = rl_agent.select_action(state)
        else:
            out    = net.activate(state)
            if USE_SOFTMAX:
                probs  = softmax(out, TAU)
                action = np.random.choice(len(probs), p=probs)
            else:
                action = int(np.argmax(out))

        # -------- 交互 -------- #
        nxt_state, reward, done, truncated, _ = env.step(action)
        rl_agent.store(state, action, reward, nxt_state, done or truncated)
        rl_agent.update()
        rl_agent.soft_update()

        total_rwd += reward
        state      = nxt_state
        if done or truncated: break
    return total_rwd

