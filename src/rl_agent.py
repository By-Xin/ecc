import random, collections, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from src.config import (
    RL_HIDDEN, RL_LR, RL_GAMMA, RL_EPS_START,
    RL_EPS_END, RL_EPS_DECAY, BUFFER_SIZE, BATCH_SIZE
)

class _QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=RL_HIDDEN):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.layers(x)

class DQNAgent:
    """Minimal DQN for CartPole (no Double-DQN/PER to keep dependencies light)."""
    def __init__(self, state_dim=4, action_dim=2):
        self.q_net = _QNetwork(state_dim, action_dim)
        self.target_net = _QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=RL_LR)
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.steps = 0

    # ---------- interaction ---------- #
    def select_action(self, state: np.ndarray) -> int:
        eps = RL_EPS_END + (RL_EPS_START - RL_EPS_END) * \
              np.exp(-1. * self.steps / RL_EPS_DECAY)
        self.steps += 1
        if random.random() < eps:
            return random.randrange(2)                 # explore
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            return int(self.q_net(state_t).argmax().item())  # exploit

    def store(self, *transition):
        """transition: (s, a, r, s', done)"""
        self.buffer.append(transition)

    # ---------- learning ---------- #
    def _sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1)
        )

    def update(self):
        if len(self.buffer) < BATCH_SIZE: return
        s, a, r, s2, d = self._sample()
        q_pred = self.q_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1, keepdim=True)[0]
            q_target = r + RL_GAMMA * q_next * (1 - d)
        loss = nn.functional.mse_loss(q_pred, q_target)
        self.opt.zero_grad(), loss.backward(), self.opt.step()

    def soft_update(self, tau=0.005):
        """Polyak averaging."""
        for tgt, src in zip(self.target_net.parameters(), self.q_net.parameters()):
            tgt.data.mul_(1 - tau).add_(src.data * tau)

