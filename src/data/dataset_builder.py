# src/data/dataset_builder.py
import numpy as np
import pandas as pd
from log.logger import get_logger

logger = get_logger(__name__)

class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_sequences(self, df: pd.DataFrame):
        """构建序列数据 - 支持序列和非序列"""

        # 获取特征列表
        if hasattr(self.cfg, 'feature_list') and self.cfg.feature_list:
            features = self.cfg.feature_list
        else:
            # 如果没有配置特征列表，使用所有数值列
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        if not features:
            logger.error("没有可用的特征列")
            return np.zeros((0, 1, 0))

        # 确保特征列存在
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"缺失特征列: {missing}")
            for col in missing:
                df[col] = 0.0

        # 选择特征列
        X = df[features].values

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)

        # 检查是否需要构建序列
        seq_len = getattr(self.cfg, 'seq_len', 1)

        if seq_len > 1:
            # 构建序列数据
            N = len(X)
            if N >= seq_len:
                # 方法1：直接分割
                num_sequences = N // seq_len
                if num_sequences > 0:
                    X_seq = X[:num_sequences * seq_len].reshape(num_sequences, seq_len, -1)
                    logger.info(f"构建序列: {X_seq.shape}")
                    return X_seq
                else:
                    # 数据比序列长度长，但不够一个完整序列
                    logger.warning(f"数据长度 {N} 不足以构建序列长度 {seq_len}")
                    # 填充到至少一个序列
                    padding = np.zeros((seq_len - N, X.shape[1]))
                    X_padded = np.vstack([X, padding])
                    return X_padded.reshape(1, seq_len, -1)
            else:
                # 数据比序列长度短，填充
                logger.warning(f"数据长度 {N} < 序列长度 {seq_len}，填充")
                padding = np.zeros((seq_len - N, X.shape[1]))
                X_padded = np.vstack([X, padding])
                return X_padded.reshape(1, seq_len, -1)
        else:
            # 非序列数据，添加一个时间维度 (batch_size, 1, features)
            logger.info(f"非序列数据: {X.shape} -> {(X.shape[0], 1, X.shape[1])}")
            return X.reshape(-1, 1, X.shape[1])

    def build_numpy_from_sequences(self, sequences):
        """从序列数据构建GAN训练用的2D numpy数组"""
        if sequences is None or len(sequences) == 0:
            return np.array([])

        if len(sequences.shape) == 3:
            # 如果是3D序列数据，展平成2D
            N, T, F = sequences.shape
            return sequences.reshape(N * T, F)
        elif len(sequences.shape) == 2:
            # 已经是2D数据
            return sequences
        else:
            logger.warning(f"未知的数据形状: {sequences.shape}")
            return np.array([])

    def _create_sliding_windows(self, data, window_size=None):
        """创建滑动窗口"""
        if window_size is None:
            window_size = getattr(self.cfg, 'seq_len', 24)

        n_samples, n_features = data.shape
        windows = []

        for i in range(0, n_samples - window_size + 1, window_size // 2):
            window = data[i:i + window_size]
            if len(window) == window_size:
                windows.append(window)

        if windows:
            return np.array(windows)
        else:
            # 如果数据太少，填充到至少一个窗口
            if len(data) < window_size:
                padding = np.zeros((window_size - len(data), n_features))
                data_padded = np.vstack([data, padding])
                return data_padded.reshape(1, window_size, n_features)
            else:
                return data[:window_size].reshape(1, window_size, n_features)