import os
import random
import gym
import neat
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

"""
Hybrid training script: NEAT‑UCB × DQN.
At every decision step the agent chooses between two action providers:
  A) DQN (gradient‑based, value learning with replay)
  B) NEAT network evaluated with UCB exploration bonus (structural search)

The choice is Bernoulli with probability P_DQN. The original UCB logic is
reused unchanged when the NEAT branch is active. The DQN branch shares the
environment interaction, storing all transitions in a replay buffer.  After
a genome is evaluated, NEAT receives fitness equal to the undiscounted return.

This is a minimal, single‑file demonstration and is *not* optimised for speed.
"""

# =====================================
# Configuration
# =====================================
GAME = "CartPole-v1"
CONFIG_PATH = "./config"
GENERATIONS = 50                # NEAT generations
CHECKPOINT_INTERVAL = 5
CHECKPOINT_DIR = "./checkpoints"
C_UCB = 0.1                      # UCB exploration parameter
P_DQN = 0.7                      # Probability of using the DQN branch per step

# DQN hyper‑parameters
HIDDEN = 64
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
LR = 1e-3
TARGET_SYNC = 500               # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5_000               # steps
GAMMA = 0.99

# Statistics (for plotting)
best_fitness_history = []
mean_fitness_history = []
std_fitness_history = []

# =====================================
# DQN implementation (minimal)
# =====================================
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buf)

# Global DQN components (initialised later to know state/action dims)
dqn_policy = None
q_target = None
replay = None
optimizer = None
step_count = 0

def init_dqn(obs_dim, n_actions):
    global dqn_policy, q_target, replay, optimizer
    dqn_policy = QNet(obs_dim, n_actions)
    q_target = QNet(obs_dim, n_actions)
    q_target.load_state_dict(dqn_policy.state_dict())
    replay = ReplayBuffer(REPLAY_CAPACITY)
    optimizer = optim.Adam(dqn_policy.parameters(), lr=LR)

def select_dqn_action(obs):
    global step_count
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step_count / EPS_DECAY)
    step_count += 1
    if random.random() < eps:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = dqn_policy(obs_t)
        return int(torch.argmax(q_values).item())

def train_dqn():
    if len(replay) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
    q_vals = dqn_policy(states).gather(1, actions)
    with torch.no_grad():
        next_q = q_target(next_states).max(1, keepdim=True)[0]
        target = rewards + GAMMA * (1 - dones) * next_q
    loss = nn.functional.mse_loss(q_vals, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step_count % TARGET_SYNC == 0:
        q_target.load_state_dict(dqn_policy.state_dict())

# =====================================
# Hybrid evaluation for NEAT genomes
# =====================================

def eval_genomes(genomes, config):
    env = gym.make(GAME)
    global best_fitness_history, mean_fitness_history, std_fitness_history
    fitness_vals = []

    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, _ = env.reset()
        done = False
        genome.fitness = 0.0
        t = 0
        # UCB statistics (per episode)
        action_counts = np.zeros(n_actions, dtype=np.int32)
        action_values = np.zeros(n_actions, dtype=np.float32)

        while not done:
            t += 1
            # Decide which branch to use
            use_dqn = random.random() < P_DQN
            if use_dqn:
                action = select_dqn_action(obs)
            else:
                # NEAT preference + UCB bonus
                prefs = net.activate(obs)
                bonus = C_UCB * np.sqrt(np.log(t + 1) / (action_counts + 1))
                ucb_scores = np.array(prefs) + bonus
                action = int(np.argmax(ucb_scores))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            genome.fitness += reward

            # DQN learns from *all* transitions (regardless of branch)
            replay.push(obs, action, reward, next_obs, float(done))
            train_dqn()

            # Update UCB statistics *if* NEAT branch used (keeps semantics)
            if not use_dqn:
                action_counts[action] += 1
                prev_val = action_values[action]
                action_values[action] += (reward - prev_val) / action_counts[action]

            obs = next_obs

        fitness_vals.append(genome.fitness)

    env.close()

    # Track statistics across population
    best_fitness_history.append(np.max(fitness_vals))
    mean_fitness_history.append(np.mean(fitness_vals))
    std_fitness_history.append(np.std(fitness_vals))

# =====================================
# Main routine
# =====================================
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    cfg.fitness_threshold = float("inf")  # disable early stopping

    # Preliminary env to infer dimensions for DQN
    tmp_env = gym.make(GAME)
    obs_dim = tmp_env.observation_space.shape[0]
    n_actions = tmp_env.action_space.n
    tmp_env.close()
    init_dqn(obs_dim, n_actions)

    pop = neat.Population(cfg)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(CHECKPOINT_INTERVAL,
                                       filename_prefix=f"{CHECKPOINT_DIR}/neat-checkpoint-"))

    # Evolutionary run
    winner = pop.run(eval_genomes, GENERATIONS)

    # Save champion genome and DQN weights
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    torch.save(dqn_policy.state_dict(), "dqn_weights.pt")
    print("Training complete. Artifacts saved: 'best_genome.pkl', 'dqn_weights.pt'.")

    # Plot fitness curves
    gens = np.arange(1, len(best_fitness_history) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(gens, best_fitness_history, marker="o", label="Best Fitness")
    plt.plot(gens, mean_fitness_history, marker="s", label="Mean Fitness")
    plt.fill_between(
        gens,
        np.array(mean_fitness_history) - np.array(std_fitness_history),
        np.array(mean_fitness_history) + np.array(std_fitness_history),
        alpha=0.2, label="Fitness StdDev"
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Hybrid NEAT×DQN Fitness over Generations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_fitness_plot.png")
    plt.show()

