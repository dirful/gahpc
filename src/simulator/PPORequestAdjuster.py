import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPORequestAdjuster(nn.Module):
    """
    Offline PPO for resource request adjustment
    Action space:
        0: no-op
        1: cpu +10%
        2: cpu -10%
        3: mem +10%
        4: mem -10%
    """

    def __init__(self, latent_dim, lr=3e-4):
        super().__init__()

        self.state_dim = latent_dim + 3  # z + RDS + cpu_req + mem_req
        self.action_dim = 5

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(64, self.action_dim)
        self.value_head = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps_clip = 0.2

    # -------------------------
    # Core PPO methods
    # -------------------------

    def forward(self, state):
        h = self.encoder(state)
        return self.policy_head(h), self.value_head(h)

    def select_action(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    @staticmethod
    def apply_action(cpu, mem, action):
        if action == 1:
            cpu *= 1.1
        elif action == 2:
            cpu *= 0.9
        elif action == 3:
            mem *= 1.1
        elif action == 4:
            mem *= 0.9
        return cpu, mem

    @staticmethod
    def compute_reward(cpu_req, mem_req, cpu_use, mem_use,
                       alpha=1.0, beta=2.0):
        waste = max(0, cpu_req - cpu_use) + max(0, mem_req - mem_use)
        violation = max(0, cpu_use - cpu_req) + max(0, mem_use - mem_req)
        return -(alpha * waste + beta * violation)

    # -------------------------
    # Training entry
    # -------------------------

    def train_offline(self, dataset, epochs=30):
        """
        dataset item format:
        (z, rds, cpu_req, mem_req, cpu_use, mem_use)
        """

        for ep in range(epochs):
            total_reward = 0.0

            for item in dataset:
                z, rds, cpu_req, mem_req, cpu_use, mem_use = item

                state = torch.tensor(
                    list(z) + [rds, cpu_req, mem_req],
                    dtype=torch.float32
                )

                action, logp_old, value = self.select_action(state)
                new_cpu, new_mem = self.apply_action(cpu_req, mem_req, action.item())

                reward = self.compute_reward(new_cpu, new_mem, cpu_use, mem_use)
                total_reward += reward

                advantage = reward - value.item()

                logits, value_new = self.forward(state)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(action)

                ratio = torch.exp(logp - logp_old.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                policy_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * (value_new.squeeze() - reward) ** 2

                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"[PPO] Epoch {ep + 1:02d} | AvgReward = {total_reward / len(dataset):.4f}")
