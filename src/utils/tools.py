import math

import numpy as np
import random
from pathlib import Path
import warnings

import torch

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_and_fix_jobs(jobs, cfg, stats):
    fixed = []
    for job in jobs:
        cpu = job.get("cpu", np.nan)
        mem = job.get("mem", np.nan)
        duration = job.get("duration", np.nan)

        # 修复 NaN
        if math.isnan(cpu) or cpu <= 0:
            cpu = max(0.01, stats["cpu_mean"])

        if math.isnan(mem) or mem <= 0:
            mem = max(1, stats["mem_mean"])

        if math.isnan(duration) or duration <= 0:
            duration = max(1, stats["duration_mean"])

        job["cpu"] = cpu
        job["mem"] = mem
        job["duration"] = duration
        fixed.append(job)

    return fixed


def ensure_outdir(path: str):
    """确保输出目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
