"""
Advanced Hybrid CartPole
------------------------
· 小型 DQN（函数逼近 Q-learning，含经验回放与目标网络）
· NEAT 进化（仅离线进化，无个体内学习）
· 动态调度：p_q 从 0.3 线性升至 0.8（warm-up 1000 episode）
· 每 20 episode 进化一代 NEAT
· 知识迁移：若 NEAT 最优分数 > 当前 Q 最优 + 50，则额外跑 10 个 NEAT episode
保存文件名：advanced_hybrid_cartpole.py
"""

# ---------------------------------------------------------------------
# 依赖
# ---------------------------------------------------------------------
import random, math, collections, pickle
from typing import Tuple, List

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import neat

# ---------------------------------------------------------------------
# 工具：固定随机种子
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------
# DQN (MLP + 经验回放 + 目标网络)
# ---------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition: Tuple):
        self.buffer.append(transition)

    def __len__(self):  # 支持 len(buffer)
        return len(self.buffer)

    def sample(self, batch_size: int):
        s, a, r, s2, d = zip(*random.sample(self.buffer, batch_size))
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))


class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 eps_start: float = 1.0,
                 eps_final: float = 0.05,
                 eps_decay: int = 5_000,
                 target_update: int = 500):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.gamma = gamma
        self.batch = batch_size
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.steps = 0
        self.eps_start, self.eps_final, self.eps_decay = eps_start, eps_final, eps_decay
        self.target_update = target_update

    # ε-greedy 概率
    def epsilon(self):
        return self.eps_final + (self.eps_start - self.eps_final) * math.exp(-self.steps / self.eps_decay)

    def act(self, state: np.ndarray) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.randrange(2)
        with torch.no_grad():
            q_vals = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q_vals))

    def remember(self, *tr):  # (s,a,r,s2,done)
        self.replay.push(tr)

    def update(self):
        if len(self.replay) < self.batch:  # 样本不足
            return
        s, a, r, s2, d = self.replay.sample(self.batch)
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            target = r + self.gamma * q_next * (1 - d)
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# ---------------------------------------------------------------------
# NEAT 代理（纯进化）
# ---------------------------------------------------------------------
class NEATAgent:
    def __init__(self, cfg_path: str):
        self.cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
        self.pop = neat.Population(self.cfg)
        self.pop.add_reporter(neat.StdOutReporter(False))
        self.stats = neat.StatisticsReporter(); self.pop.add_reporter(self.stats)
        self.env = gym.make("CartPole-v1")
        self.best_genome = None

    def _score(self, genome) -> float:
        net = neat.nn.FeedForwardNetwork.create(genome, self.cfg)
        obs, _ = self.env.reset(); total = 0; done = False
        while not done:
            action = int(np.argmax(net.activate(obs)))
            obs, r, term, trunc, _ = self.env.step(action)
            total += r; done = term or trunc
        genome.fitness = total
        return total

    def evolve_one_generation(self) -> float:
        for g in self.pop.population.values():
            self._score(g)
        self.best_genome = max(self.pop.population.values(), key=lambda g: g.fitness)
        self.pop.reporters.post_evaluate(self.cfg, self.pop.population, self.pop.species, self.best_genome)
        # reproduce
        self.pop.population = self.pop.reproduction.reproduce(
            self.cfg, self.pop.species, self.cfg.pop_size, self.pop.generation
        )
        self.pop.species.speciate(self.cfg, self.pop.population, self.pop.generation)
        self.pop.generation += 1
        return self.best_genome.fitness

    def act(self, obs: np.ndarray) -> int:
        if self.best_genome is None:
            raise RuntimeError("NEAT 尚未进化，请先调用 evolve_one_generation()")
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.cfg)
        return int(np.argmax(net.activate(obs)))

# ---------------------------------------------------------------------
# 混合训练器
# ---------------------------------------------------------------------
class HybridTrainer:
    def __init__(self,
                 cfg_path: str = "./config",
                 episodes: int = 10_000,
                 warmup: int = 1_000,
                 p_start: float = 0.3,
                 p_final: float = 0.8,
                 neat_interval: int = 20,
                 transfer_gap: float = 50.0,
                 seed: int = 0):
        set_seed(seed)
        self.env = gym.make("CartPole-v1")
        self.q = DQNAgent(self.env.observation_space.shape[0], 2)
        self.neat = NEATAgent(cfg_path)
        self.EPISODES, self.WARM = episodes, warmup
        self.P0, self.P1 = p_start, p_final
        self.NINT = neat_interval
        self.GAP = transfer_gap
        self.best_q = -float("inf")
        self.rewards: List[float] = []

    # 动态调度概率
    def p_q(self, ep):
        return self.P1 if ep >= self.WARM else self.P0 + (self.P1 - self.P0) * ep / self.WARM

    # ----------------运行一个 DQN episode-----------------
    def _episode_q(self) -> float:
        obs, _ = self.env.reset(); done = False; total = 0
        while not done:
            a = self.q.act(obs)
            nxt, r, term, trunc, _ = self.env.step(a)
            self.q.remember(obs, a, r, nxt, term or trunc)
            self.q.update()
            obs = nxt; total += r; done = term or trunc
        self.best_q = max(self.best_q, total)
        return total

    # ----------------运行一个 NEAT episode----------------
    def _episode_neat(self, push_to_replay: bool = False) -> float:
        if self.neat.best_genome is None:
            self.neat.evolve_one_generation()
        obs, _ = self.env.reset(); done = False; total = 0
        while not done:
            a = self.neat.act(obs)
            nxt, r, term, trunc, _ = self.env.step(a)
            if push_to_replay:
                self.q.remember(obs, a, r, nxt, term or trunc)
            obs = nxt; total += r; done = term or trunc
        return total

    # ----------------主训练循环-------------------------
    def run(self) -> List[float]:
        for ep in range(1, self.EPISODES + 1):
            if random.random() < self.p_q(ep) and ep > 1:
                reward = self._episode_q()
            else:
                reward = self._episode_neat(push_to_replay=True)
            self.rewards.append(reward)

            # 定期进化 NEAT
            if ep % self.NINT == 0:
                best_fit = self.neat.evolve_one_generation()
                # 知识迁移：NEAT 明显更好时，追加经验
                if best_fit > self.best_q + self.GAP:
                    for _ in range(10):
                        self._episode_neat(push_to_replay=True)

            if ep % 500 == 0:
                recent_mean = np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
                print(f"Episode {ep:5d} | mean-100 {recent_mean:6.1f} | ε {self.q.epsilon():.2f}")

        # 保存模型与结果
        torch.save(self.q.q_net.state_dict(), "adv_hybrid_qnet.pt")
        if self.neat.best_genome:
            pickle.dump(self.neat.best_genome, open("adv_hybrid_best_genome.pkl", "wb"))
        np.save("adv_hybrid_rewards.npy", np.array(self.rewards))
        return self.rewards

# ---------------------------------------------------------------------
# 快速 Smoke Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    trainer = HybridTrainer(episodes=2000, seed=42)  # demo 跑 2000 回合
    trainer.run()

