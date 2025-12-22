# src/agent/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from collections import deque
import random
from log.logger import get_logger

logger = get_logger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # 记录维度
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor网络（策略网络）
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic网络（价值网络）
        self.critic = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self._initialize_weights()

        logger.info(f"ActorCritic网络初始化: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """前向传播"""
        # 确保输入维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加batch维度

        features = self.shared_layers(state)

        # Actor输出
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Critic输出
        value = self.critic(features)

        return action_mean, action_std, value

    def act(self, state):
        """选择动作"""
        with torch.no_grad():
            # 确保state是tensor
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)

            # 添加batch维度
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action_mean, action_std, value = self.forward(state)

            # 创建正态分布
            dist = Normal(action_mean, action_std)

            # 采样动作
            action = dist.sample()

            # 计算log概率
            log_prob = dist.log_prob(action).sum(-1)

            # 对连续动作进行clip
            action = torch.tanh(action)  # 限制在[-1, 1]之间

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def evaluate(self, state, action):
        """评估动作"""
        # 确保维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        action_mean, action_std, value = self.forward(state)

        # 创建正态分布
        dist = Normal(action_mean, action_std)

        # 计算log概率和熵
        action_log_prob = dist.log_prob(action).sum(-1)
        dist_entropy = dist.entropy().mean()

        return action_log_prob, torch.squeeze(value), dist_entropy

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []

    def store(self, state, action, reward, value, log_prob, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)

    def get_all(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones),
            np.array(self.next_states)
        )

class PPOAgent:
    def __init__(self, cfg):
        self.cfg = cfg

        # 获取状态和动作维度
        state_dim = getattr(cfg, 'state_dim', 30)  # 默认为30，匹配模拟器
        action_dim = getattr(cfg, 'action_dim', 4)  # 4个连续动作

        logger.info(f"PPO Agent初始化: state_dim={state_dim}, action_dim={action_dim}")

        # 网络
        self.policy = ActorCritic(state_dim, action_dim,
                                  getattr(cfg, 'ppo_hidden_dim', 64))

        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=cfg.ppo_lr)

        # PPO参数
        self.gamma = cfg.ppo_gamma
        self.clip_epsilon = cfg.ppo_clip
        self.epochs = getattr(cfg, 'ppo_epochs', 10)
        self.batch_size = getattr(cfg, 'ppo_batch_size', 32)

        # 记忆缓冲区
        self.memory = Memory()
        self.replay_buffer = deque(maxlen=getattr(cfg, 'replay_buffer_size', 10000))

        # 训练状态
        self.episode_rewards = []
        self.total_steps = 0

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        logger.info(f"PPO Agent初始化完成，设备: {self.device}")

    def act(self, state):
        """选择动作"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        action, log_prob, value = self.policy.act(state)

        # 确保动作在合理范围内
        if isinstance(action, np.ndarray):
            action = np.clip(action, -1.0, 1.0)

        return action[0], log_prob[0], value[0]

    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        """存储经验"""
        # 如果log_prob和value未提供，重新计算
        if log_prob is None or value is None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_mean, action_std, value_tensor = self.policy(state_tensor)
                dist = Normal(action_mean, action_std)
                log_prob_tensor = dist.log_prob(torch.FloatTensor(action).unsqueeze(0).to(self.device))

            log_prob = log_prob_tensor.item()
            value = value_tensor.item()

        # 存储到记忆
        self.memory.store(state, action, reward, value, log_prob, done, next_state)

        # 也存储到回放缓冲区
        self.replay_buffer.append((state, action, reward, next_state, done))

        self.total_steps += 1

    def compute_gae(self, rewards, values, dones, next_values):
        """计算广义优势估计（GAE）"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gamma * last_advantage * (1 - dones[t])

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def learn(self):
        """从记忆中学"""
        if len(self.memory.states) < self.batch_size:
            return

        # 获取所有记忆
        states, actions, rewards, values, log_probs, dones, next_states = self.memory.get_all()

        # 计算下一个状态的值
        with torch.no_grad():
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            _, _, next_values = self.policy(next_states_tensor)
            next_values = next_values.cpu().numpy()

        # 计算GAE和returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)

        # 转换数据为tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # PPO更新
        for _ in range(self.epochs):
            # 随机采样批次
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # 获取批次数据
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                # 评估当前策略
                new_log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # 计算概率比
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 计算PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (values - batch_returns).pow(2).mean()

                # 总损失
                entropy_bonus = -0.01 * entropy  # 鼓励探索
                loss = actor_loss + 0.5 * critic_loss + entropy_bonus

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()

        # 清空记忆
        self.memory.clear()

    def train_episode(self, simulator):
        """训练一个episode"""
        if not hasattr(simulator, 'reset') or not hasattr(simulator, 'step'):
            logger.error("Simulator缺少必要方法")
            return 0

        state = simulator.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 100:  # 防止无限循环
            # 选择动作
            action, log_prob, value = self.act(state)

            # 执行动作
            next_state, reward, done, _ = simulator.step(action)

            # 存储经验
            self.remember(state, action, reward, next_state, done, log_prob, value)

            # 更新状态
            state = next_state
            episode_reward += reward
            step_count += 1

            # 定期学习
            if step_count % 10 == 0:
                self.learn()

        # 最终学习
        if len(self.memory.states) > 0:
            self.learn()

        # 记录奖励
        self.episode_rewards.append(episode_reward)

        logger.info(f"[PPO] Episode {len(self.episode_rewards)}: "
                    f"Reward={episode_reward:.2f}, Steps={step_count}")

        return episode_reward

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'total_steps': self.total_steps
        }, path)
        logger.info(f"模型保存到 {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.total_steps = checkpoint['total_steps']
        logger.info(f"模型从 {path} 加载")