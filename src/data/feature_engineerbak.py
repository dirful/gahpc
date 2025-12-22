# src/data/feature_engineer.py
import numpy as np
import pandas as pd
from log.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_level = getattr(cfg, 'feature_level', 'job')

    def transform(self, df: pd.DataFrame, data_mode=None):
        """根据配置的特征级别进行转换"""
        if data_mode is None:
            data_mode = getattr(self.cfg, 'data_mode', 'job_stats')

        if data_mode == 'time_series':
            # 时间序列数据不需要转换
            return df
        elif self.feature_level == 'job':
            return self._transform_job(df)
        else:
            return self._transform_task(df)

    def _transform_job(self, df: pd.DataFrame):
        """转换 job-level 特征"""
        df = df.copy()

        # 确保所有 job 特征都存在
        job_features = [
            'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
            'mem_mean', 'mem_std', 'mem_max',
            'machines_count', 'task_count',
            'duration_sec', 'cpu_intensity', 'mem_intensity',
            'task_density', 'cpu_cv', 'mem_cv'
        ]

        # 添加缺失的特征
        for feat in job_features:
            if feat not in df.columns:
                logger.warning(f"添加缺失的 job 特征: {feat}")
                df[feat] = 0.0

        # 处理缺失值
        df.fillna(0, inplace=True)

        # 标准化数值特征（可选）
        if hasattr(self.cfg, 'normalize') and self.cfg.normalize:
            df = self._normalize_job_features(df)

        # 只保留 job 特征
        keep_cols = [c for c in job_features if c in df.columns]
        return df[keep_cols]

    def _transform_task(self, df: pd.DataFrame):
        """原有的 task-level 转换逻辑"""
        df = df.copy()
        df.fillna(0, inplace=True)

        # 标准化字段名：cpu, mem, disk_io, duration
        if 'sampled_cpu_usage' in df.columns:
            df['cpu'] = df['sampled_cpu_usage'].fillna(df.get('cpu_rate', 0))
        if 'canonical_memory_usage' in df.columns:
            df['mem'] = df['canonical_memory_usage']
        if 'disk_io_time' in df.columns:
            df['disk_io'] = df['disk_io_time']
        if 'start_time' in df.columns and 'end_time' in df.columns:
            df['duration'] = df['end_time'] - df['start_time']

        # keep minimal columns
        keep = [c for c in ['job_id','cpu','mem','disk_io','duration','machine_id']
                if c in df.columns]
        return df[keep]

    def _normalize_job_features(self, df: pd.DataFrame):
        """标准化 job 特征"""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 排除可能不需要标准化的列
        exclude_cols = ['job_id', 'machine_id', 'task_index']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df

    def get_stats(self, df: pd.DataFrame):
        """返回用于 LLM Prompt 的分布性统计"""
        stats = {}

        data_mode = getattr(self.cfg, 'data_mode', 'job_stats')

        if data_mode == 'time_series':
            # 对于时间序列数据，提取主要特征的统计
            features_to_check = ['cpu_avg', 'mem_avg', 'cpu_mean', 'mem_mean', 'cpu', 'mem']
        elif self.feature_level == 'job':
            # job-level 统计
            features_to_check = [
                'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                'mem_mean', 'mem_std', 'mem_max',
                'duration_sec', 'cpu_intensity', 'mem_intensity'
            ]
        else:
            # task-level 统计
            features_to_check = ['cpu','mem','disk_io','duration']

        # 实际存在的特征
        features = [f for f in features_to_check if f in df.columns]

        for feat in features:
            if feat in df.columns:
                s = df[feat].dropna()
                if len(s) > 0:
                    stats[feat] = {
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "p50": float(s.quantile(0.5)),
                        "p90": float(s.quantile(0.9))
                    }
                else:
                    stats[feat] = {
                        "mean": 0.0, "std": 0.0, "min": 0.0,
                        "max": 0.0, "p50": 0.0, "p90": 0.0
                    }

        # 如果没有任何统计信息，添加默认值
        if not stats:
            logger.warning("没有提取到统计信息，使用默认值")
            stats = {
                'cpu': {"mean": 0.5, "std": 0.2, "min": 0.0, "max": 1.0, "p50": 0.5, "p90": 0.8},
                'mem': {"mean": 0.3, "std": 0.15, "min": 0.0, "max": 0.8, "p50": 0.3, "p90": 0.5},
                'disk_io': {"mean": 0.2, "std": 0.1, "min": 0.0, "max": 0.5, "p50": 0.2, "p90": 0.35},
                'duration': {"mean": 300, "std": 200, "min": 1, "max": 1800, "p50": 300, "p90": 600}
            }

        logger.info(f"计算了 {len(stats)} 个特征的统计信息")
        return stats

    def to_training_matrix(self, df: pd.DataFrame):
        """把 DataFrame 转成训练矩阵"""
        data_mode = getattr(self.cfg, 'data_mode', 'job_stats')

        if data_mode == 'time_series':
            # 时间序列模式直接返回数值
            return df.select_dtypes(include=[np.number]).values.astype(np.float32)
        elif self.feature_level == 'job':
            # job-level 特征
            cols = [c for c in [
                'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                'mem_mean', 'mem_std', 'mem_max',
                'machines_count', 'task_count',
                'duration_sec', 'cpu_intensity', 'mem_intensity',
                'task_density', 'cpu_cv', 'mem_cv'
            ] if c in df.columns]
        else:
            # task-level 特征
            cols = [c for c in ['cpu','mem','disk_io','duration']
                    if c in df.columns]

        return df[cols].values.astype(np.float32)