# src/agent/ppo_agent_fixed.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from collections import deque
from log.logger import get_logger

logger = get_logger(__name__)

class ActorCriticFixed(nn.Module):
    def __init__(self, state_dim=30, action_dim=4, hidden_dim=64):
        super(ActorCriticFixed, self).__init__()

        logger.info(f"ActorCriticFixed: state_dim={state_dim}, action_dim={action_dim}")

        # 共享特征层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.shared_layers(state)
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        value = self.critic(features)
        return action_mean, action_std, value

    def act(self, state):
        """采样动作（tanh 约束）并返回 action, log_prob, value（numpy）"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(next(self.parameters()).device)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action_mean, action_std, value = self.forward(state)
            base_dist = Normal(action_mean, action_std)
            tanh_transform = TanhTransform(cache_size=1)
            dist = TransformedDistribution(base_dist, tanh_transform)
            action = dist.rsample()  # reparameterized sample
            log_prob = dist.log_prob(action).sum(-1)

            action_np = action.cpu().numpy().flatten()
            logp_np = log_prob.cpu().numpy().astype(np.float32)
            val_np = value.cpu().numpy().squeeze().astype(np.float32)

        return action_np, logp_np, val_np

class PPOAgentFixed:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dim = getattr(cfg, 'state_dim', 30)
        self.action_dim = getattr(cfg, 'action_dim', 4)
        self.hidden_dim = getattr(cfg, 'ppo_hidden_dim', 64)

        logger.info(f"PPOAgentFixed init: state_dim={self.state_dim}, action_dim={self.action_dim}")

        self.policy = ActorCriticFixed(self.state_dim, self.action_dim, self.hidden_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=getattr(cfg, 'ppo_lr', 3e-4))

        self.gamma = getattr(cfg, 'ppo_gamma', 0.99)
        self.clip_epsilon = getattr(cfg, 'ppo_clip', 0.2)
        self.ppo_epochs = getattr(cfg, 'ppo_epochs', 4)
        self.minibatch_size = getattr(cfg, 'ppo_minibatch_size', 64)
        self.gae_lambda = getattr(cfg, 'gae_lambda', 0.95)

        # memory buffers (store numpy)
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        logger.info("PPOAgentFixed initialized")

    def _extract_state_features(self, state):
        """从各种状态格式提取固定长度特征（30维）"""
        # 与原版保持一致（确保30维）
        import numpy as _np
        if isinstance(state, (_np.ndarray, list)):
            arr = _np.array(state, dtype=_np.float32).flatten()
            padded = _np.zeros(self.state_dim, dtype=_np.float32)
            padded[:min(len(arr), self.state_dim)] = arr[:self.state_dim]
            return padded
        elif isinstance(state, dict):
            features = []
            nodes = state.get('nodes', [])
            for node in nodes[:3]:
                features.extend([
                    node.get('cpu_cores', 0) / 100.0,
                    node.get('memory_gb', 0) / 1000.0,
                    node.get('current_cpu_util', 0),
                    node.get('current_mem_util', 0)
                ])
            sys_load = state.get('system_load', {})
            features.extend([
                sys_load.get('avg_cpu_utilization', 0),
                sys_load.get('avg_memory_utilization', 0),
                sys_load.get('queue_length', 0) / 100.0
            ])
            job_queue = state.get('job_queue', [])
            features.append(len(job_queue) / 50.0)
            while len(features) < self.state_dim:
                features.append(0.0)
            return _np.array(features[:self.state_dim], dtype=_np.float32)
        elif isinstance(state, torch.Tensor):
            arr = state.cpu().numpy().flatten()
            padded = _np.zeros(self.state_dim, dtype=_np.float32)
            padded[:min(len(arr), self.state_dim)] = arr[:self.state_dim]
            return padded
        else:
            try:
                arr = np.array(state, dtype=np.float32).flatten()
                padded = np.zeros(self.state_dim, dtype=np.float32)
                padded[:min(len(arr), self.state_dim)] = arr[:self.state_dim]
                return padded
            except Exception:
                logger.warning(f"Cannot parse state type: {type(state)}")
                return np.zeros(self.state_dim, dtype=np.float32)

    def act(self, state):
        """外部调用：返回动作、log_prob、value（均为 numpy）"""
        try:
            state_features = self._extract_state_features(state)
            action, log_prob, value = self.policy.act(state_features)
            # 确保动作为长度 action_dim
            if isinstance(action, np.ndarray):
                if action.size != self.action_dim:
                    if action.size < self.action_dim:
                        action = np.pad(action, (0, self.action_dim - action.size), mode='constant')
                    else:
                        action = action[:self.action_dim]
            return action, log_prob, value
        except Exception as e:
            logger.error(f"PPO Agent act failed: {e}")
            return np.zeros(self.action_dim, dtype=np.float32), 0.0, 0.0

    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        """存储原始（numpy）样本，后续统一转 tensor"""
        state_features = self._extract_state_features(state)
        self.states.append(state_features.astype(np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))
        # value/log_prob 可能为 scalar 或 array
        self.values.append(float(value) if value is not None else 0.0)
        self.log_probs.append(float(log_prob) if log_prob is not None else 0.0)
        self.dones.append(bool(done))

    def _compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """计算 GAE 优势和 returns，输入 numpy arrays"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def learn(self):
        """PPO 更新：多 epoch + mini-batch"""
        if len(self.states) < 32:
            return

        try:
            device = self.device
            states = torch.FloatTensor(np.array(self.states)).to(device)
            actions = torch.FloatTensor(np.array(self.actions)).to(device)
            old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
            values_np = np.array(self.values, dtype=np.float32)
            rewards_np = np.array(self.rewards, dtype=np.float32)
            dones_np = np.array(self.dones, dtype=np.float32)

            # bootstrap next_value using last state
            with torch.no_grad():
                last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(device)
                _, _, next_value_t = self.policy(last_state)
                next_value = next_value_t.cpu().numpy().squeeze()

            advantages_np, returns_np = self._compute_gae(
                rewards_np, values_np, dones_np, next_value,
                gamma=self.gamma, lam=self.gae_lambda
            )

            advantages = torch.FloatTensor(advantages_np).to(device)
            returns = torch.FloatTensor(returns_np).to(device)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            data_size = states.shape[0]
            indices = np.arange(data_size)

            for epoch in range(self.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, data_size, self.minibatch_size):
                    mb_idx = indices[start:start + self.minibatch_size]
                    mb_states = states[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_logp = old_log_probs[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_ret = returns[mb_idx]

                    action_mean, action_std, values_pred = self.policy(mb_states)
                    base_dist = Normal(action_mean, action_std)
                    dist = TransformedDistribution(base_dist, TanhTransform(cache_size=1))
                    new_logp = dist.log_prob(mb_actions).sum(-1)

                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * (mb_ret - values_pred.squeeze()).pow(2).mean()

                    loss = actor_loss + 0.5 * critic_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

            logger.debug(f"PPO learn done: dataset={data_size}, epochs={self.ppo_epochs}")
        except Exception as e:
            logger.error(f"PPO learn failed: {e}")
        finally:
            self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def train_episode(self, simulator, max_steps=50):
        """在模拟器上跑一个 episode 并学习一次"""
        if not hasattr(simulator, 'reset') or not hasattr(simulator, 'step'):
            logger.error("Simulator missing reset/step")
            return 0.0

        state = simulator.reset()
        episode_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            action, log_prob, value = self.act(state)
            next_state, reward, done, _ = simulator.step(action)
            self.remember(state, action, reward, next_state, done, log_prob=log_prob, value=value)
            state = next_state
            episode_reward += float(reward)
            step_count += 1

        if len(self.states) > 0:
            self.learn()

        logger.info(f"[PPOFixed] Episode reward: {episode_reward:.2f}, steps: {step_count}")
        return episode_reward
