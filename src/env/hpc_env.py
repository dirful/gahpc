# src/agent/ppo_agent.py (保留你原来结构, 重点改 train_episode 接口)
import torch
import torch.nn as nn
from log.logger import get_logger

logger = get_logger(__name__)

class PPOAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, obs):
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32)
            out = self.policy(t).item()
            return int(out > 0)

    def train_episode(self, simulator, max_steps=200):
        env = simulator.env
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            action = self.act(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
            steps += 1
        logger.info("[PPO] Episode finished reward=%.3f steps=%d", total, steps)
        return total
