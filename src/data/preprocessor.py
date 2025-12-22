#!/usr/bin/env python3
"""
preprocessor.py - 序列构建和预处理
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

def build_per_job_sequences(
        window_agg: pd.DataFrame,
        features: List[str],
        max_len: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """为每个作业构建时间序列"""
    groups = window_agg.groupby('job_id')
    sequences = []
    job_ids = []

    for job_id, g in groups:
        g_sorted = g.sort_values('window_index')
        # 选择存在的特征列，否则用零填充
        cols = [c for c in features if c in g_sorted.columns]
        arr = g_sorted[cols].values.astype(float) if len(cols) > 0 else np.zeros((len(g_sorted), len(features)))

        if max_len is not None:
            if arr.shape[0] > max_len:
                arr = arr[:max_len]
            elif arr.shape[0] < max_len:
                pad = np.full((max_len - arr.shape[0], arr.shape[1]), np.nan)
                arr = np.vstack([arr, pad])

        sequences.append(arr)
        job_ids.append(job_id)

    return sequences, job_ids

def normalize_sequences(
        seqs: List[np.ndarray],
        scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler, np.ndarray]:
    """标准化序列数据"""
    max_len = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    stacked = np.zeros((len(seqs), max_len, D), dtype=float)
    mask = np.zeros_like(stacked, dtype=bool)

    # 堆叠序列并创建掩码
    for i, s in enumerate(seqs):
        T = s.shape[0]
        stacked[i, :T, :] = np.nan_to_num(s, nan=0.0)
        mask[i, :T, :] = ~np.isnan(s)

    # 计算标量
    flat = np.vstack([s[~np.isnan(s).any(axis=1)]
                      for s in seqs if s.size > 0])
    if flat.shape[0] == 0:
        flat = np.zeros((0, D))

    if scaler is None:
        scaler = StandardScaler()
        if flat.shape[0] > 0:
            scaler.fit(flat)
        else:
            scaler.mean_ = np.zeros(D)
            scaler.scale_ = np.ones(D)

    # 应用标准化
    for i in range(stacked.shape[0]):
        stacked[i] = (stacked[i] - scaler.mean_) / (scaler.scale_ + 1e-9)

    return stacked, scaler, mask

def create_feature_matrix(sequences: List[np.ndarray], job_ids: List[int]) -> pd.DataFrame:
    """创建特征矩阵（每作业的聚合统计）"""
    fm_rows = []

    for i, jid in enumerate(job_ids):
        s = sequences[i]
        row = {
            'job_id': jid,
            'cpu_mean': np.nanmean(s[:, 0]) if s.size > 0 else 0,
            'cpu_std': np.nanstd(s[:, 0]) if s.size > 0 else 0,
            'mem_mean': np.nanmean(s[:, 1]) if s.shape[1] > 1 else 0,
            'mem_std': np.nanstd(s[:, 1]) if s.shape[1] > 1 else 0,
            'total_windows': np.sum(~np.isnan(s[:, 0]))
        }
        fm_rows.append(row)

    return pd.DataFrame(fm_rows).set_index('job_id')

def sample_sequences(
        sequences: List[np.ndarray],
        job_ids: List[int],
        sample_size: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """对序列进行采样"""
    if sample_size is not None and len(sequences) > sample_size:
        sel = np.random.choice(len(sequences), sample_size, replace=False)
        sequences = [sequences[i] for i in sel]
        job_ids = [job_ids[i] for i in sel]

    return sequences, job_ids