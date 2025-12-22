import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# =========================
# Policy Network
# =========================
class RequestPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 动作均值（CPU, MEM）
        self.mean = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(1, 2))

        # Value
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        std = torch.exp(self.log_std)
        value = self.value(h)
        return mean, std, value


# =========================
# PPO Agent (Offline)
# =========================
class RequestPPOAgent:
    def __init__(
            self,
            state_dim,
            lr=3e-4,
            gamma=0.99,
            clip_eps=0.2,
            epochs=5,
            batch_size=64,
            device="cpu"
    ):
        self.policy = RequestPolicy(state_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    # --------- action ----------
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, std, value = self.policy(state)

        dist = Normal(mean, std)
        action = torch.tanh(dist.sample())
        log_prob = dist.log_prob(action).sum(-1)

        return (
            action.detach().cpu().numpy(),
            log_prob.detach(),
            value.detach()
        )

    # --------- store ----------
    def store(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    # --------- train ----------
    def learn(self):
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).squeeze(-1).to(self.device)

        # returns / advantage（无 episode，直接 baseline）
        returns = rewards
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            idx = torch.randperm(len(states))

            for i in range(0, len(states), self.batch_size):
                batch = idx[i:i+self.batch_size]

                mean, std, value = self.policy(states[batch])
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(actions[batch]).sum(-1)
                ratio = torch.exp(new_log_probs - old_log_probs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[batch]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (value.squeeze(-1) - returns[batch]).pow(2).mean()

                loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.reset_buffer()
