# src/data/feature_engineer.py
import numpy as np
import pandas as pd
from log.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_level = getattr(cfg, 'feature_level', 'job')
        self.scaler = None

    def transform(self, df: pd.DataFrame, data_mode=None):
        if data_mode is None:
            data_mode = getattr(self.cfg, 'data_mode', 'job_stats')

        # 基础清洗（CPI / instructions）
        df = self.clean_cpi_and_instructions(df)

        if data_mode == 'time_series':
            return df
        elif self.feature_level == 'job':
            return self._transform_job(df)
        else:
            return self._transform_task(df)

    def clean_cpi_and_instructions(self, df: pd.DataFrame):
        """清理 CPI/instructions 异常样本，clip 大 CPI，移除 instructions<=0"""
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        # instructions 清洗：若存在 instructions 列，丢弃 <=0
        if 'instructions' in df.columns:
            bad = (df['instructions'] <= 0) | (df['instructions'].isna())
            if bad.any():
                logger.info(f"Removing {bad.sum()} rows with instructions<=0")
                df = df.loc[~bad]

        # CPI 处理：clip
        if 'cpi' in df.columns:
            upper = getattr(self.cfg, 'cpi_clip_upper', 200.0)
            df['cpi'] = df['cpi'].clip(lower=0.0, upper=upper)

        # CPU/MEM 极端值 clip（避免长尾破坏训练）
        if 'cpu_mean' in df.columns:
            df['cpu_mean'] = df['cpu_mean'].clip(lower=0.0, upper=1.0)
        if 'mem_mean' in df.columns:
            df['mem_mean'] = df['mem_mean'].clip(lower=0.0, upper=1.0)

        return df

    def _transform_job(self, df: pd.DataFrame):
        df = df.copy()
        job_features = [
            'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
            'mem_mean', 'mem_std', 'mem_max',
            'machines_count', 'task_count',
            'duration_sec', 'cpu_intensity', 'mem_intensity',
            'task_density', 'cpu_cv', 'mem_cv'
        ]

        for feat in job_features:
            if feat not in df.columns:
                logger.debug(f"Adding missing job feature: {feat}")
                df[feat] = 0.0

        df.fillna(0.0, inplace=True)

        if getattr(self.cfg, 'normalize', False):
            df = self._normalize_job_features(df)

        keep_cols = [c for c in job_features if c in df.columns]
        return df[keep_cols]

    def _transform_task(self, df: pd.DataFrame):
        df = df.copy()
        df.fillna(0.0, inplace=True)

        if 'sampled_cpu_usage' in df.columns:
            df['cpu'] = df['sampled_cpu_usage'].fillna(df.get('cpu_rate', 0))
        if 'canonical_memory_usage' in df.columns:
            df['mem'] = df['canonical_memory_usage']
        if 'disk_io_time' in df.columns:
            df['disk_io'] = df['disk_io_time']
        if 'start_time' in df.columns and 'end_time' in df.columns:
            df['duration'] = df['end_time'] - df['start_time']

        keep = [c for c in ['job_id', 'cpu', 'mem', 'disk_io', 'duration', 'machine_id'] if c in df.columns]
        return df[keep]

    def _normalize_job_features(self, df: pd.DataFrame):
        """支持 standard / robust / minmax"""
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        mode = getattr(self.cfg, 'normalize_mode', 'standard')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['job_id', 'machine_id', 'task_index']
        cols = [c for c in numeric_cols if c not in exclude]
        if not cols:
            return df

        if mode == 'standard':
            self.scaler = StandardScaler()
        elif mode == 'robust':
            self.scaler = RobustScaler()
        elif mode == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        df[cols] = self.scaler.fit_transform(df[cols])

        if getattr(self.cfg, 'save_scaler', False):
            try:
                import joblib
                joblib.dump(self.scaler, getattr(self.cfg, 'scaler_path', './scaler.pkl'))
                logger.info("Saved scaler to disk")
            except Exception as e:
                logger.warning(f"Failed to save scaler: {e}")

        return df

    def get_stats(self, df: pd.DataFrame):
        """返回更丰富的统计量，便于 LLM 提示使用"""
        stats = {}
        data_mode = getattr(self.cfg, 'data_mode', 'job_stats')

        if data_mode == 'time_series':
            features_to_check = ['cpu_avg', 'mem_avg', 'cpu_mean', 'mem_mean', 'cpu', 'mem']
        elif self.feature_level == 'job':
            features_to_check = [
                'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                'mem_mean', 'mem_std', 'mem_max',
                'duration_sec', 'cpu_intensity', 'mem_intensity'
            ]
        else:
            features_to_check = ['cpu', 'mem', 'disk_io', 'duration']

        features = [f for f in features_to_check if f in df.columns]

        for feat in features:
            s = df[feat].dropna()
            if len(s) > 0:
                stats[feat] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p25": float(s.quantile(0.25)),
                    "p50": float(s.quantile(0.5)),
                    "p75": float(s.quantile(0.75)),
                    "p90": float(s.quantile(0.9)),
                    "p99": float(s.quantile(0.99)),
                    "count": int(len(s)),
                    "na": int(s.isna().sum())
                }
            else:
                stats[feat] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "count": 0, "na": 0}

        if not stats:
            logger.warning("No stats extracted; using defaults")
            stats = {
                'cpu': {"mean": 0.5, "std": 0.2, "min": 0.0, "max": 1.0, "p50": 0.5, "p90": 0.8, "count": 0, "na": 0},
                'mem': {"mean": 0.3, "std": 0.15, "min": 0.0, "max": 0.8, "p50": 0.3, "p90": 0.5, "count": 0, "na": 0},
                'duration': {"mean": 300, "std": 200, "min": 1, "max": 1800, "p50": 300, "p90": 600, "count": 0, "na": 0}
            }

        logger.info(f"Calculated {len(stats)} stats")
        return stats

    def to_training_matrix(self, df: pd.DataFrame):
        data_mode = getattr(self.cfg, 'data_mode', 'job_stats')

        if data_mode == 'time_series':
            return df.select_dtypes(include=[np.number]).values.astype(np.float32)
        elif self.feature_level == 'job':
            cols = [c for c in [
                'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                'mem_mean', 'mem_std', 'mem_max',
                'machines_count', 'task_count',
                'duration_sec', 'cpu_intensity', 'mem_intensity',
                'task_density', 'cpu_cv', 'mem_cv'
            ] if c in df.columns]
        else:
            cols = [c for c in ['cpu', 'mem', 'disk_io', 'duration'] if c in df.columns]

        mat = df[cols].values.astype(np.float32)
        return mat
