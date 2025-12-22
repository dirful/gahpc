#!/usr/bin/env python3
"""
database.py - 数据库加载和窗口聚合
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import Optional

def load_task_usage_200k(db_uri: str, sample_limit: int = 20000) -> pd.DataFrame:
    """从数据库加载约200k行任务使用数据"""
    engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)

    query = f"""
    SELECT
        start_time,
        end_time,
        job_id,
        task_index,
        machine_id,
        cpu_rate,
        canonical_memory_usage,
        assigned_memory_usage,
        unmapped_page_cache,
        total_page_cache,
        maximum_memory_usage,
        disk_io_time,
        local_disk_space_usage,
        maximum_cpu_rate,
        maximum_disk_io_time,
        cycles_per_instruction,
        memory_accesses_per_instruction,
        sample_portion,
        aggregation_type,
        sampled_cpu_usage
    FROM task_usage
    ORDER BY start_time
    LIMIT {int(sample_limit)};
    """

    print(f"Executing SQL to load up to {sample_limit} rows from task_usage...")
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} rows")
    return df

def prepare_window_agg_from_task_usage(
        df_task_usage: pd.DataFrame,
        window_seconds: int = 300,
        time_unit: float = 1000.0
) -> pd.DataFrame:
    """将原始任务使用数据聚合为每作业每窗口的统计信息"""
    df = df_task_usage.copy()
    # 确保开始时间为数值类型
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    # 创建窗口索引
    df['window_ts'] = (df['start_time'] / time_unit).astype('Int64')
    df['window_index'] = (df['window_ts'] // int(window_seconds)).astype('Int64')

    # 定义聚合函数
    def p95(x):
        return np.percentile(x.dropna(), 95) if x.dropna().size > 0 else np.nan

    def p90(x):
        return np.percentile(x.dropna(), 90) if x.dropna().size > 0 else np.nan

    def p50(x):
        return np.percentile(x.dropna(), 50) if x.dropna().size > 0 else np.nan

    # 定义聚合映射
    agg_map = {
        'cpu_rate': ['mean', 'median', 'max', p95],
        'sampled_cpu_usage': ['mean', 'median', 'max', p95],
        'maximum_cpu_rate': ['mean', 'max'],
        'canonical_memory_usage': ['mean', 'median', 'max', p90, p50],
        'assigned_memory_usage': ['mean', 'max'],
        'maximum_memory_usage': ['mean', 'max', p95],
        'disk_io_time': ['mean', 'max', p90],
        'maximum_disk_io_time': ['mean', 'max'],
        'total_page_cache': ['mean', 'max'],
        'cycles_per_instruction': ['mean', 'median', 'max']
    }

    # 只保留存在的列
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}

    # 执行分组聚合
    group_cols = ['job_id', 'window_index']
    agg_df = df.groupby(group_cols).agg(agg_map).reset_index()

    # 扁平化多级列索引
    agg_df.columns = ['_'.join([str(c) for c in col if c != ''])
                      if isinstance(col, tuple) else col
                      for col in agg_df.columns]

    return agg_df

def load_window_agg(file_path: str) -> pd.DataFrame:
    """从文件加载窗口聚合数据"""
    print(f"Loading window_agg from file: {file_path}")
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path)