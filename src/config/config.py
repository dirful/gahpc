import argparse
from typing import List

import torch


class Config:
    """ 全局配置，可直接改这里 """

    def __init__(self):
        # ===== 数据模式配置 =====
        self.data_mode = 'job_stats'  # 'job_stats' 或 'time_series'
        self.feature_level = 'job'    # 'job' 或 'task'

        # ===== Database =====
        self.db_host = "localhost"
        self.db_port = 3307
        self.db_user = "root"
        self.db_password = "123456"
        self.db_name = "xiyoudata"

        # 自动构造数据库 URL
        self.db_url = (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        # ===== 数据加载配置 =====
        self.sample_limit = 50000      # 数据采样限制
        self.max_jobs_ts = 500         # 时间序列最大job数
        self.normalize = False         # 是否标准化特征

        # ===== 时间序列配置 =====
        self.seq_len = 24              # 序列长度
        self.window_seconds = 300      # 窗口大小（秒）
        self.min_windows = 6           # 最小窗口数

        # ===== 特征配置 =====
        if self.data_mode == 'job_stats':
            self.feature_list = [
                'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                'mem_mean', 'mem_std', 'mem_max',
                'machines_count', 'task_count',
                'duration_sec', 'cpu_intensity', 'mem_intensity',
                'task_density', 'cpu_cv', 'mem_cv'
            ]
        else:  # time_series 模式
            # 时间序列特征将在运行时确定
            self.feature_list = None

        # ===== GAN 模块 =====
        self.enable_gan = True
        self.gan_epochs = 5
        self.gan_latent_dim = 64
        self.gan_hidden_dim = 128
        self.gan_lr = 1e-4             # 学习率
        self.gan_batch_size = 64       # 批次大小
        self.gan_sample_n = 8          # GAN采样数量

        # 根据数据模式确定输出维度
        if self.data_mode == 'job_stats':
            self.gan_out_dim = 15       # job特征维度
        else:
            self.gan_out_dim = 4      # cpu, mem, disk_io, duration

        # ===== PPO 模块 =====
        self.enable_ppo = True
        self.episodes = 10
        self.ppo_lr = 3e-4
        self.ppo_gamma = 0.99
        self.ppo_clip = 0.2

        # PPO网络配置
        self.ppo_hidden_dim = 64
        self.state_dim = 30  # 匹配模拟器状态维度
        self.action_dim = 4  # 连续动作维度

        # ===== LLM 模块 =====
        self.enable_llm = True
        self.llm_model = "yxchia/qwen2.5-1.5b-instruct:Q8_0"
        self.ollama_host = "http://localhost:11434"
        self.llm_num_generate = 10     # LLM生成job数量
        self.llm_temperature = 0.7     # LLM温度参数
        self.llm_max_tokens = 500      # LLM最大token数

        # ===== 环境配置 =====
        self.time_window = 5           # 时间窗口
        self.num_nodes = 10            # 集群节点数
        self.max_jobs_per_node = 5     # 每个节点最大job数
        self.node_cpu_capacity = 1.0   # 节点CPU容量
        self.node_mem_capacity = 1.0   # 节点内存容量

        # ===== Validation thresholds (safety caps) =====
        self.max_cpu = 1.0
        self.min_cpu = 0.0
        self.max_mem = 128.0           # 根据你的集群单机内存单位调整
        self.min_mem = 0.0
        self.max_disk_io = 1000.0      # 磁盘IO上限
        self.min_disk_io = 0.0
        self.max_duration = 24 * 3600  # 秒
        self.min_duration = 1.0

        # ===== 通用 =====
        self.random_seed = 42
        self.log_level = "INFO"        # 日志级别
        self.save_model = True         # 是否保存模型
        self.model_save_dir = "models" # 模型保存目录

        # ===== 训练配置 =====
        self.batch_size = 32           # 训练批次大小
        self.train_split = 0.8         # 训练集比例
        self.learning_rate = 1e-3      # 通用学习率

        # 模拟器配置
        self.num_nodes = 10
        self.max_jobs_per_node = 5
        self.node_cpu_capacity = 1.0
        self.node_mem_capacity = 1.0
        self.max_steps = 100

        # 验证阈值
        self.min_disk_io = 0.0
        self.max_disk_io = 10.0
        self.min_duration = 1.0

        # LLM配置（如果LLMClient支持）
        self.llm_temperature = 0.7
        self.llm_max_tokens = 500



def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(
        description='Deep embedding + nonlinear clustering pipeline for TimeGAN features'
    )

    # 数据相关参数
    p.add_argument('--window-agg', type=str, default=None,
                   help='Path to precomputed window_agg (csv or parquet)')
    p.add_argument('--db-uri', type=str,
                   default='mysql+pymysql://root:123456@localhost:3307/xiyoudata',
                   help='SQLAlchemy DB URI to read task_usage')
    p.add_argument('--window-seconds', type=int, default=300,
                   help='Window size in seconds for aggregating task_usage into windows')
    p.add_argument('--time-unit', type=float, default=1000.0,
                   help='Divide DB timestamps by this to get seconds (e.g., 1000 if ms)')

    # 特征相关参数
    p.add_argument('--features', type=str,
                   default='cpu_rate_mean,canonical_memory_usage_mean',
                   help='Comma separated feature columns to use for sequences')
    p.add_argument('--max-seq-len', type=int, default=None)
    p.add_argument('--sample-jobs', type=int, default=2000,
                   help='Limit number of jobs used for AE training (random sample)')

    # 模型相关参数
    p.add_argument('--ae-type', type=str, default='lstm', choices=['lstm', 'gru'])
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--hidden-dim', type=int, default=64)

    # 训练相关参数
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', type=str, default='cpu')

    # 分析相关参数
    p.add_argument('--umap-components', type=int, default=2)
    p.add_argument('--min-cluster-size', type=int, default=20)
    p.add_argument('--n-clusters', type=int, default=8)

    # 输出相关参数
    p.add_argument('--outdir', type=str, default='./out_deep_pipeline')

    return p.parse_args()

def get_features_list(features_str: str) -> List[str]:
    """将逗号分隔的特征字符串转换为列表"""
    return [f.strip() for f in features_str.split(',')]

def setup_device(requested_device: str) -> str:
    """设置计算设备（CPU/GPU）"""
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        return 'cpu'
    return requested_device