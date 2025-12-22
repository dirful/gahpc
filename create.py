# create.py
# -*- coding: utf-8 -*-

"""
create.py
自动生成 HPC + PPO + LLM 项目脚手架
"""

import os

def module_header(title):
    return f'"""\n{title}\nAutomatically generated module.\n"""'

LOADERS_PY = module_header("Data Loaders") + """
import os
import pandas as pd

class DataLoader:
    \"\"\"
    通用数据加载器（支持 CSV / Parquet）
    可扩展用于 HPC workload 的 job_logs / task_usage 等数据。
    \"\"\"

    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_csv(self, filename: str):
        path = os.path.join(self.base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)

    def load_parquet(self, filename: str):
        path = os.path.join(self.base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_parquet(path)

"""

PREPROCESS_PY = module_header("Data Preprocessing") + """
import pandas as pd
import numpy as np

class Preprocessor:
    \"\"\"
    HPC workload 数据清洗与时间对齐模块。
    \"\"\"

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype in [float, int]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("unknown")
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[float, int])
        df[numeric.columns] = (numeric - numeric.mean()) / (numeric.std() + 1e-8)
        return df
"""

HPC_ENV_PY = module_header("HPC PPO Environment") + """
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class HPCEnv(gym.Env):
    \"\"\"
    HPC 调度仿真环境：
    - state: job 队列状态、节点负载
    - action: 调度策略选择
    - reward: 根据等待时间、吞吐量、负载均衡计算
    \"\"\"

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_jobs: int = 100):
        super().__init__()
        self.max_jobs = max_jobs
        self.state_dim = 16

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.random.uniform(-1, 1, size=(self.state_dim,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        obs = np.random.uniform(-1, 1, size=(16,)).astype(np.float32)
        reward = -np.random.rand()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}
"""

PPO_AGENT_PY = module_header("PPO Agent") + """
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, state_dim=16, action_dim=4, lr=3e-4):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

"""

FILES = {
    "src/data/loaders.py": LOADERS_PY,
    "src/data/preprocess.py": PREPROCESS_PY,
    "src/env/hpc_env.py": HPC_ENV_PY,
    "src/agent/ppo_agent.py": PPO_AGENT_PY,
}

DIRS = [
    "src/data",
    "src/env",
    "src/agent",
    "src/generator",
    "src/llm",
    "src/hpc",
    "src/utils",
]

def ensure_dirs():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        init_file = os.path.join(d, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("")

def write_files():
    for path, content in FILES.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

def main():
    print("Generating HPC project scaffold...")
    ensure_dirs()
    write_files()
    print("Done.")

if __name__ == "__main__":
    main()
