# src/generator/gan_adapter.py

import numpy as np
from log.logger import get_logger

logger = get_logger(__name__)

class GANJobFeatureAdapter:
    """将GAN输出适配为完整的job-level特征"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.required_job_features = [
            'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
            'mem_mean', 'mem_std', 'mem_max',
            'machines_count', 'task_count',
            'duration_sec', 'cpu_intensity', 'mem_intensity',
            'task_density', 'cpu_cv', 'mem_cv'
        ]

    def adapt_gan_samples(self, gan_samples, real_stats=None):
        """
        将GAN样本适配为完整的job-level特征

        Args:
            gan_samples: GAN生成的样本列表
            real_stats: 真实数据的统计信息（用于归一化/反归一化）
        """
        if not gan_samples:
            return []

        adapted_samples = []

        for sample in gan_samples:
            adapted = {}

            # 1. 如果GAN已经生成了完整特征，直接使用
            if all(feat in sample for feat in self.required_job_features):
                for feat in self.required_job_features:
                    adapted[feat] = sample[feat]

            # 2. 如果只有基本特征，需要扩展
            else:
                # 从基本特征推导
                cpu = sample.get('cpu', 0.5)
                mem = sample.get('mem', 0.3) * 10000  # 假设mem是0-1，转换为MB
                duration = sample.get('duration', 300.0)

                # 生成完整特征
                adapted['cpu_mean'] = float(cpu)
                adapted['cpu_std'] = float(cpu * 0.3)  # 假设std是mean的30%
                adapted['cpu_max'] = float(min(1.0, cpu + adapted['cpu_std'] * 1.5))
                adapted['cpu_95th'] = float(cpu + adapted['cpu_std'] * 1.645)

                adapted['mem_mean'] = float(mem)
                adapted['mem_std'] = float(mem * 0.2)
                adapted['mem_max'] = float(mem + adapted['mem_std'] * 2)

                adapted['machines_count'] = int(max(1, np.random.poisson(3)))
                adapted['task_count'] = int(max(1, np.random.poisson(25)))
                adapted['duration_sec'] = float(duration)

                # 计算派生特征
                adapted['cpu_intensity'] = adapted['cpu_mean'] * adapted['duration_sec']
                adapted['mem_intensity'] = adapted['mem_mean'] * adapted['duration_sec']
                adapted['task_density'] = adapted['task_count'] / (adapted['duration_sec'] + 1e-6)
                adapted['cpu_cv'] = adapted['cpu_std'] / (adapted['cpu_mean'] + 1e-8)
                adapted['mem_cv'] = adapted['mem_std'] / (adapted['mem_mean'] + 1e-8)

            # 3. 应用统计归一化（如果提供了真实统计）
            if real_stats:
                adapted = self._apply_statistical_normalization(adapted, real_stats)

            adapted_samples.append(adapted)

        return adapted_samples

    def _apply_statistical_normalization(self, sample, stats):
        """根据真实统计信息调整样本"""
        # 这里可以实现更复杂的统计匹配
        # 例如：确保样本的分布与真实数据相似

        # 简单的范围调整
        for feat in sample:
            if feat in stats:
                feat_stats = stats[feat]
                if isinstance(feat_stats, dict):
                    # 确保值在合理范围内
                    min_val = feat_stats.get('min', 0)
                    max_val = feat_stats.get('max', 1)
                    mean_val = feat_stats.get('mean', 0.5)

                    # 简单的截断
                    sample[feat] = np.clip(sample[feat], min_val, max_val)

        return sample