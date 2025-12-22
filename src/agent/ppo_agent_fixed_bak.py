# src/agent/ppo_agent_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
from log.logger import get_logger

logger = get_logger(__name__)

class ActorCriticFixed(nn.Module):
    def __init__(self, state_dim=30, action_dim=4, hidden_dim=64):
        super(ActorCriticFixed, self).__init__()

        logger.info(f"ActorCriticFixed: state_dim={state_dim}, action_dim={action_dim}")

        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor网络
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic网络
        self.critic = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

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
        """选择动作 - 返回4维动作"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)

            if state.dim() == 1:
                state = state.unsqueeze(0)

            action_mean, action_std, value = self.forward(state)

            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            # 使用tanh将动作限制在[-1, 1]范围内
            action = torch.tanh(action)

        # 返回展平的numpy数组
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy(), value.cpu().numpy()

class PPOAgentFixed:
    def __init__(self, cfg):
        self.cfg = cfg

        # 强制使用30维状态
        state_dim = 30
        action_dim = 4

        logger.info(f"PPOAgentFixed: state_dim={state_dim} (强制), action_dim={action_dim}")

        self.policy = ActorCriticFixed(state_dim, action_dim,
                                       getattr(cfg, 'ppo_hidden_dim', 64))

        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.ppo_lr)

        self.gamma = cfg.ppo_gamma
        self.clip_epsilon = cfg.ppo_clip

        # 记忆
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        logger.info(f"PPOAgentFixed初始化完成")

    def _extract_state_features(self, state):
        """从各种状态格式中提取特征向量"""

        # 如果已经是numpy数组或列表，直接使用
        if isinstance(state, (np.ndarray, list)):
            state_array = np.array(state, dtype=np.float32)
            return state_array.flatten()[:30]  # 确保30维

        # 如果是字典，提取关键特征
        elif isinstance(state, dict):
            features = []

            # 提取节点特征
            nodes = state.get('nodes', [])
            for node in nodes[:3]:  # 最多取3个节点
                features.extend([
                    node.get('cpu_cores', 0) / 100.0,
                    node.get('memory_gb', 0) / 1000.0,
                    node.get('current_cpu_util', 0),
                    node.get('current_mem_util', 0)
                ])

            # 提取系统负载特征
            system_load = state.get('system_load', {})
            features.extend([
                system_load.get('avg_cpu_utilization', 0),
                system_load.get('avg_memory_utilization', 0),
                system_load.get('queue_length', 0) / 100.0
            ])

            # 提取作业队列特征
            job_queue = state.get('job_queue', [])
            features.append(len(job_queue) / 50.0)

            # 补充到30维
            while len(features) < 30:
                features.append(0.0)

            return np.array(features[:30], dtype=np.float32)

        # 如果是Tensor，转换为numpy
        elif isinstance(state, torch.Tensor):
            state_array = state.cpu().numpy()
            return state_array.flatten()[:30]

        # 其他类型，尝试转换
        else:
            try:
                state_array = np.array(state, dtype=np.float32)
                return state_array.flatten()[:30]
            except:
                # 如果转换失败，返回零向量
                logger.warning(f"无法解析状态类型: {type(state)}")
                return np.zeros(30, dtype=np.float32)

    def act(self, state):
        """选择动作 - 返回4维动作"""
        try:
            # 提取状态特征
            state_features = self._extract_state_features(state)

            # 确保状态是Tensor
            if not isinstance(state_features, torch.Tensor):
                state_features = torch.FloatTensor(state_features).to(self.device)

            # 调用策略网络的act方法
            action, log_prob, value = self.policy.act(state_features)

            # 确保动作是4维
            if isinstance(action, np.ndarray):
                if len(action) != 4:
                    logger.warning(f"动作维度 {len(action)} 不是4，进行填充")
                    if len(action) < 4:
                        action = np.pad(action, (0, 4 - len(action)), mode='constant')
                    else:
                        action = action[:4]

            return action, log_prob, value

        except Exception as e:
            logger.error(f"PPO Agent act失败: {e}")
            # 返回默认的4维动作
            return np.zeros(4), 0.0, 0.0

    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        """存储经验"""

        # 提取状态特征
        state_features = self._extract_state_features(state)
        next_state_features = self._extract_state_features(next_state)

        # 如果log_prob和value未提供，使用默认值
        if log_prob is None:
            log_prob = 0.0
        if value is None:
            value = 0.0

        self.states.append(state_features)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def learn(self):
        if len(self.states) < 32:
            return

        try:
            # 转换为Tensor
            states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)

            # 计算returns
            returns = []
            R = 0
            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            returns_tensor = torch.FloatTensor(returns).to(self.device)

            # 前向传播
            action_mean, action_std, values = self.policy(states_tensor)
            dist = Normal(action_mean, action_std)

            # 计算损失
            new_log_probs = dist.log_prob(actions_tensor).sum(-1)
            old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages = returns_tensor - torch.FloatTensor(self.values).to(self.device)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (values.squeeze() - returns_tensor).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            logger.debug(f"学习完成: actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f}")

        except Exception as e:
            logger.error(f"学习失败: {e}")
        finally:
            # 清空记忆
            self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def train_episode(self, simulator):
        """训练一个episode"""
        if not hasattr(simulator, 'reset'):
            logger.error("Simulator缺少reset方法")
            return 0

        state = simulator.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 50:
            action, log_prob, value = self.act(state)
            next_state, reward, done, _ = simulator.step(action)

            self.remember(state, action, reward, next_state, done, log_prob, value)

            state = next_state
            episode_reward += reward
            step_count += 1

        if len(self.states) > 0:
            self.learn()

        logger.info(f"[PPOFixed] Episode 奖励: {episode_reward:.2f}, 步数: {step_count}")

        return episode_reward