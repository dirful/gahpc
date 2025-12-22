import numpy as np
from log.logger import get_logger

logger = get_logger(__name__)

class SchedulerEnv:
    """
    简化 HPC 环境：
    - observation: 当前等待任务的特征向量
    - action: 选择哪个任务执行
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.queue = []
        self.time = 0
        self.done = False
        return self._obs()

    def _obs(self):
        if len(self.queue) == 0:
            return np.zeros(4)
        return np.array(self.queue[0])

    def step(self, action):
        reward = 0
        if len(self.queue) > 0:
            job = self.queue.pop(0)
            reward = -abs(job[3])  # 例如 duration 越大 reward 越低

        self.time += 1
        done = self.time > 200
        return self._obs(), reward, done, {}

    def add_jobs(self, jobs):
        for j in jobs:
            self.queue.append(j)
