import gym
import neat
import numpy as np
import random
import math
import pickle

# -----------------------
# Q‑Learning agent (tabular)
# -----------------------
class QAgent:
    """Tabular Q‑learning agent with ε‑greedy exploration."""
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995,
                 bins=(30, 30, 50, 50),
                 win_size=(0.25, 0.25, 0.01, 0.1)):
        self.env = env
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bins = bins
        self.win_size = np.array(win_size, dtype=np.float32)
        # 初始化 Q‑table (state bins × action)
        self.q_table = np.random.uniform(0, 1, size=self.bins + (env.action_space.n,))

    # ------ 辅助函数 ------
    def _discretize(self, state):
        """与 epsilon_decay_test.py 中 get_discrete_state 保持一致."""
        return tuple(((state / self.win_size) + np.array([15, 10, 1, 10])).astype(int))

    # ------ 与环境交互 ------
    def _select_action(self, ds):
        if random.random() > self.epsilon:
            return int(np.argmax(self.q_table[ds]))
        return self.env.action_space.sample()

    def run_episode(self):
        state, _ = self.env.reset()
        ds = self._discretize(state)
        done = False
        total_reward = 0.0

        while not done:
            action = self._select_action(ds)
            nxt, r, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            nds = self._discretize(nxt)

            # Q‑learning 更新
            if not done:
                max_nxt = np.max(self.q_table[nds])
                td_target = r + self.discount * max_nxt
            else:
                td_target = r
            self.q_table[ds + (action,)] += self.lr * (td_target - self.q_table[ds + (action,)])
            ds = nds
            total_reward += r

        # ε 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return total_reward

# -----------------------
# NEAT agent (best genome)
# -----------------------
class NEATAgent:
    """维护一套 NEAT 种群, 周期性进化并使用当前最佳 genome."""
    def __init__(self, config_path: str, pop_size: int = 50):
        self.config = neat.Config(neat.DefaultGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  config_path)
        self.pop = neat.Population(self.config)
        self.best_genome = None
        self.env = gym.make('CartPole-v1')
        # 为了与外部调度配合, 采用手动一步到位的世代式进化

    def _evaluate_genome(self, genome):
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = int(np.argmax(net.activate(obs)))
            obs, r, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += r
        genome.fitness = total_reward
        return total_reward

    def evolve_one_generation(self):
        """完整评估当前种群并进行一次繁殖, 更新 best_genome."""
        genomes = list(self.pop.population.items())
        for gid, genome in genomes:
            self._evaluate_genome(genome)
        # 调用 NEAT 内部回调完成繁殖与物种更新
        self.pop.reporters.post_evaluate(self.config, self.pop.population, self.pop.species, None)
        self.pop.population = self.pop.reproduction.reproduce(self.config, self.pop.species,
                                                              self.config.pop_size,
                                                              self.pop.generation)
        self.pop.species.speciate(self.config, self.pop.population, self.pop.generation)
        self.pop.generation += 1
        # 更新最佳 genome
        self.best_genome = max(self.pop.population.values(), key=lambda g: g.fitness)

    # --- 与环境交互 (仅推断, 不学习) ---
    def run_episode(self):
        if self.best_genome is None:
            self.evolve_one_generation()
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = int(np.argmax(net.activate(obs)))
            obs, r, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += r
        return total_reward

# -----------------------
# Hybrid training loop
# -----------------------

def hybrid_train(total_episodes: int = 10000,
                 p_q: float = 0.7,
                 neat_interval: int = 100,
                 neat_config_path: str = './config'):
    """并行混合训练: episode 级概率调度 + 周期性 NEAT 世代进化"""
    env_q = gym.make('CartPole-v1')
    q_agent = QAgent(env_q)
    neat_agent = NEATAgent(neat_config_path)

    all_rewards = []
    for epi in range(1, total_episodes + 1):
        if random.random() < p_q:
            reward = q_agent.run_episode()
        else:
            reward = neat_agent.run_episode()
        all_rewards.append(reward)

        # 每隔 neat_interval 个 episode, 让 NEAT 进化一代
        if epi % neat_interval == 0:
            neat_agent.evolve_one_generation()

        # 日志输出
        if epi % 500 == 0:
            recent_mean = np.mean(all_rewards[-100:])
            print(f"Episode {epi:>5} | recent‑100 mean reward = {recent_mean:.1f} | ε = {q_agent.epsilon:.3f}")

    # 训练完毕, 保存成果
    np.save('hybrid_q_table.npy', q_agent.q_table)
    if neat_agent.best_genome is not None:
        with open('hybrid_best_genome.pkl', 'wb') as f:
            pickle.dump(neat_agent.best_genome, f)
    return all_rewards

if __name__ == '__main__':
    hybrid_train()

