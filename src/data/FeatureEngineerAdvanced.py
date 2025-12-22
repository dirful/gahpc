"""
feature_engineer_enhanced.py

增强版 FeatureEngineer（v1 + v2 + v3 + v4）
- 基于你提供的 MySQL 表（task_usage, task_events, job_events, machine_events, machine_attributes, task_constraints）
- 自动生成基础统计、变化率、burst、phase、周期性、竞争度、事件 embedding
- v2: Autoencoder embedding（可选：使用 Keras，如不可用回退到 PCA）
- v3: DAG/依赖特征（基于 job_events / logical_job_name_hash 的启发式构建）
- v4: RL/调度专用特征（cluster pressure, queue length, slowdown, machine utilization）

使用说明：
- 将 db_uri 指向你的 MySQL（例如 mysql+pymysql://user:pwd@host:3307/xiyoudata）
- 直接调用 FeatureEngineer(db_uri).build_all_features(limit=...) 来运行

注意：此脚本尽量减少强依赖。若系统安装了 tensorflow/keras，会使用 Keras 做 autoencoder；否则使用 sklearn.decomposition.PCA
"""

import os
import time
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

logger = logging.getLogger("FeatureEngineer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# try keras for autoencoder
USE_KERAS = False
try:
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    USE_KERAS = True
except Exception:
    USE_KERAS = False


class FeatureEngineerEnhanced:
    def __init__(self, db_uri: str, sample_limit: int = 500000):
        self.engine = create_engine(db_uri)
        self.sample_limit = sample_limit

    # ------------------------- 数据加载 -------------------------
    def _sql_read(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def load_task_usage(self, start_ts: Optional[int] = None, end_ts: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        limit = limit or self.sample_limit
        where = []
        if start_ts is not None:
            where.append(f"start_time >= {start_ts}")
        if end_ts is not None:
            where.append(f"start_time <= {end_ts}")
        where_sql = " AND ".join(where)
        if where_sql:
            where_sql = "WHERE " + where_sql
        sql = f"""
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
        {where_sql}
        LIMIT {limit}
        """
        logger.info("Loading task_usage from DB...")
        return self._sql_read(sql)

    def load_task_events(self) -> pd.DataFrame:
        sql = "SELECT * FROM task_events"
        logger.info("Loading task_events from DB...")
        return self._sql_read(sql)

    def load_job_events(self) -> pd.DataFrame:
        sql = "SELECT * FROM job_events"
        logger.info("Loading job_events from DB...")
        return self._sql_read(sql)

    def load_machine_events(self) -> pd.DataFrame:
        sql = "SELECT * FROM machine_events"
        logger.info("Loading machine_events from DB...")
        return self._sql_read(sql)

    # ------------------------- 基础特征 -------------------------
    @staticmethod
    def _safe_div(a, b):
        return a / (b + 1e-9)

    def add_basic_features(self, usage: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding basic features...")
        df = usage.copy()
        df["duration_sec"] = (df["end_time"] - df["start_time"]) / 1000.0
        df["cpu_mean_est"] = df["cpu_rate"].fillna(0)
        df["mem_mean_est"] = df["canonical_memory_usage"].fillna(0)
        df["cpu_util_ratio"] = self._safe_div(df["cpu_rate"], df.get("maximum_cpu_rate", df["cpu_rate"]))
        df["mem_util_ratio"] = self._safe_div(df["canonical_memory_usage"], df.get("maximum_memory_usage", df["canonical_memory_usage"] if "maximum_memory_usage" in df else df["canonical_memory_usage"]))
        return df

    # ------------------------- 衍生特征：变化率/差分/rolling -------------------------
    def add_derivative_and_rolling(self, df: pd.DataFrame, group_cols=("job_id", "task_index"), time_col="start_time", window=3) -> pd.DataFrame:
        logger.info("Adding derivative and rolling features...")
        df = df.sort_values(list(group_cols) + [time_col]).reset_index(drop=True)
        for col in ["cpu_rate", "canonical_memory_usage", "disk_io_time"]:
            if col not in df.columns:
                continue
            diff_col = f"{col}_diff"
            df[diff_col] = df.groupby(list(group_cols))[col].diff().fillna(0)
            # relative diff
            df[f"{col}_rel_diff"] = self._safe_div(df[diff_col], df[col].replace(0, np.nan)).fillna(0)
            # rolling statistics per group
            df[f"{col}_rolling_mean_{window}"] = df.groupby(list(group_cols))[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
            df[f"{col}_rolling_std_{window}"] = df.groupby(list(group_cols))[col].transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
        return df

    # ------------------------- 突发/burst 特征 -------------------------
    def add_burst_features(self, df: pd.DataFrame, z_thresh=2.5) -> pd.DataFrame:
        logger.info("Adding burst features...")
        out = df
        for col in ["cpu_rate", "disk_io_time"]:
            if col not in out.columns:
                continue
            diff = out.get(f"{col}_diff", out[col].diff().fillna(0))
            median = diff.abs().median()
            mad = (diff.abs() - median).abs().median() + 1e-9
            z = (diff - median) / mad
            out[f"{col}_burst"] = (z.abs() > z_thresh).astype(int)
        return out

    # ------------------------- Phase 聚类 -------------------------
    def add_phase_clustering(self, df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        logger.info("Adding phase clustering (KMeans)...")
        feats = [c for c in ["cpu_rate", "canonical_memory_usage", "disk_io_time"] if c in df.columns]
        if not feats:
            return df
        data = df[feats].fillna(0).values
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data)
            df["phase_label"] = labels
        except Exception as e:
            logger.warning("Phase clustering failed: %s", e)
            df["phase_label"] = 0
        return df

    # ------------------------- 周期性编码 -------------------------
    def add_periodic_encoding(self, df: pd.DataFrame, time_col: str = "start_time") -> pd.DataFrame:
        logger.info("Adding periodic time encoding (hour/day)...")
        # start_time is in seconds or milliseconds? user had BIGINT timestamps; task_usage uses start_time in original schema - assume seconds or milliseconds? We'll detect.
        ts_sample = int(df[time_col].iloc[0]) if len(df) > 0 else 0
        # heuristic: if timestamp > 1e12 then ms, else s
        if ts_sample > 1e12:
            factor = 1000.0
        else:
            factor = 1.0
        # convert to POSIX hours/days
        df["ts_s"] = (df[time_col] / factor).astype(int)
        df["hour"] = ((df["ts_s"] // 3600) % 24).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = ((df["ts_s"] // 86400) % 7).astype(int)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    # ------------------------- 同机竞争/内容度 -------------------------
    def add_machine_contention(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding machine contention features...")
        # count tasks per machine per timeslot (coarse by start_time)
        df["task_count_on_machine"] = df.groupby(["machine_id", "start_time"])['task_index'].transform('count')
        df["machine_cpu_sum"] = df.groupby(["machine_id", "start_time"])['cpu_rate'].transform('sum')
        df["machine_mem_sum"] = df.groupby(["machine_id", "start_time"])['canonical_memory_usage'].transform('sum')
        # normalized pressure
        df["machine_cpu_pressure_norm"] = df["machine_cpu_sum"] / (df["task_count_on_machine"] + 1e-9)
        return df

    # ------------------------- 事件 embedding（task_events/job_events） -------------------------
    def add_event_counts(self, df: pd.DataFrame, task_events: pd.DataFrame, job_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        logger.info("Adding event count embeddings...")
        te = task_events.copy()
        # group by job_id, task_index and event_type
        te_counts = te.groupby(["job_id", "task_index", "event_type"]).size().unstack(fill_value=0)
        te_counts = te_counts.add_prefix("task_event_").reset_index()
        df = df.merge(te_counts, on=["job_id", "task_index"], how="left")
        df.fillna(0, inplace=True)
        # job-level events
        if job_events is not None:
            je = job_events.copy()
            je_counts = je.groupby(["job_id", "event_type"]).size().unstack(fill_value=0)
            je_counts = je_counts.add_prefix("job_event_").reset_index()
            df = df.merge(je_counts, on=["job_id"], how="left").fillna(0)
        return df

    # ------------------------- v2: Autoencoder embedding / PCA fallback -------------------------
    def compute_autoencoder_embedding(self, df: pd.DataFrame, feature_cols: list, latent_dim: int = 16, epochs: int = 20, batch_size: int = 256) -> Tuple[pd.DataFrame, np.ndarray]:
        logger.info("Computing autoencoder embeddings (if keras present) otherwise PCA")
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if USE_KERAS:
            try:
                input_dim = Xs.shape[1]
                inp = Input(shape=(input_dim,))
                h = Dense(max(64, latent_dim*4), activation='relu')(inp)
                h = Dense(max(32, latent_dim*2), activation='relu')(h)
                code = Dense(latent_dim, activation='linear', name='code')(h)
                h2 = Dense(max(32, latent_dim*2), activation='relu')(code)
                h2 = Dense(max(64, latent_dim*4), activation='relu')(h2)
                out = Dense(input_dim, activation='linear')(h2)
                ae = Model(inputs=inp, outputs=out)
                ae.compile(optimizer=Adam(1e-3), loss='mse')
                ae.fit(Xs, Xs, epochs=epochs, batch_size=batch_size, verbose=1)
                encoder = Model(inputs=inp, outputs=code)
                codes = encoder.predict(Xs, batch_size=batch_size)
                codes_df = pd.DataFrame(codes, index=df.index, columns=[f'ae_{i}' for i in range(codes.shape[1])])
                return pd.concat([df.reset_index(drop=True), codes_df.reset_index(drop=True)], axis=1), scaler
            except Exception as e:
                logger.warning("Keras AE failed (%s), falling back to PCA", e)
        # PCA fallback
        pca = PCA(n_components=min(latent_dim, Xs.shape[1]-1))
        codes = pca.fit_transform(Xs)
        codes_df = pd.DataFrame(codes, index=df.index, columns=[f'pca_{i}' for i in range(codes.shape[1])])
        logger.info("PCA embedding done (shape=%s)", codes.shape)
        return pd.concat([df.reset_index(drop=True), codes_df.reset_index(drop=True)], axis=1), scaler

    # ------------------------- v3: DAG / 依赖特征（启发式） -------------------------
    def build_job_dag_features(self, df_usage: pd.DataFrame, job_events: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building heuristic DAG features from job_events and task_usage...")
        je = job_events.copy()
        # heuristic: jobs sharing the same logical_job_name_hash are considered same workflow family
        if 'logical_job_name_hash' in je.columns:
            group = je.groupby('logical_job_name_hash')
            # compute job submission order by earliest time
            first_time = je.groupby('job_id')['time'].min().reset_index().rename(columns={'time':'first_time'})
            # merge
            job_meta = first_time
            # predecessor count: count of jobs with same logical name and earlier first_time
            job_meta = job_meta.merge(je[['job_id','logical_job_name_hash']].drop_duplicates(), on='job_id', how='left')
            job_meta['pred_count'] = job_meta.apply(lambda r: sum((job_meta['logical_job_name_hash']==r['logical_job_name_hash']) & (job_meta['first_time']<r['first_time'])), axis=1)
            # join to usage
            df = df_usage.merge(job_meta[['job_id','pred_count']], on='job_id', how='left')
            df['pred_count'] = df['pred_count'].fillna(0).astype(int)
        else:
            df = df_usage.copy()
            df['pred_count'] = 0
        return df

    # ------------------------- v4: RL/调度专用特征 -------------------------
    def add_rl_features(self, df_usage: pd.DataFrame, machine_events: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding RL-specific features (cluster pressure, queue length, slowdown)...")
        df = df_usage.copy()
        # cluster pressure = total cpu_rate across all tasks at same timeslot
        df['cluster_cpu_sum'] = df.groupby('start_time')['cpu_rate'].transform('sum')
        df['cluster_mem_sum'] = df.groupby('start_time')['canonical_memory_usage'].transform('sum')
        # queue length heuristic: number of jobs with start_time > current and same submission window (requires job_events) - approximate with count jobs per start_time
        df['jobs_at_time'] = df.groupby('start_time')['job_id'].transform('nunique')
        # slowdown = actual_duration / requested_duration (use memory_request or cpu_request from task_events via join)
        # We'll try to load task_events and merge cpu_request/memory_request
        try:
            te = self.load_task_events()
            te_small = te[['job_id','task_index','cpu_request','memory_request']].drop_duplicates()
            df = df.merge(te_small, on=['job_id','task_index'], how='left')
            df['cpu_slowdown'] = (df['duration_sec'] / (df['cpu_request'] + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            logger.warning('Could not load task_events for RL features: %s', e)
            df['cpu_slowdown'] = 0.0
        # machine utilization estimate: cpu_sum / machine cpus (from machine_events)
        try:
            me = machine_events.copy()
        except Exception:
            try:
                me = self.load_machine_events()
            except Exception:
                me = pd.DataFrame()
        if not me.empty and 'cpus' in me.columns:
            # take latest cpus per machine
            me_latest = me.sort_values('time').drop_duplicates('machine_id', keep='last')[['machine_id','cpus']]
            df = df.merge(me_latest, on='machine_id', how='left')
            df['machine_util_est'] = df['machine_cpu_sum'] / (df['cpus'] + 1e-9)
        else:
            df['machine_util_est'] = df['machine_cpu_sum']
        return df

    # ------------------------- SQL 生成工具：窗口聚合 -------------------------
    def generate_window_aggregation_sql(self, window_seconds:int = 300, fields:Optional[list]=None, table_name: str = 'task_usage') -> str:
        fields = fields or ['cpu_rate','canonical_memory_usage','disk_io_time']
        aggs = []
        for f in fields:
            aggs.append(f"AVG({f}) AS {f}_avg")
            aggs.append(f"STD({f}) AS {f}_std")
            aggs.append(f"MAX({f}) AS {f}_max")
            aggs.append(f"MIN({f}) AS {f}_min")
        agg_sql = ',\n                '.join(aggs)
        sql = f"""
        SELECT
            job_id,
            FLOOR((start_time/1000)/{window_seconds}) AS window_index,
            {agg_sql},
            COUNT(*) AS n_tasks
        FROM {table_name}
        GROUP BY job_id, window_index
        ORDER BY job_id, window_index
        """
        return sql

    # ------------------------- 主流程整合接口 -------------------------
    def build_all_features(self, limit: Optional[int] = None) -> pd.DataFrame:
        logger.info("Start building all features pipeline...")
        limit = limit or self.sample_limit
        usage = self.load_task_usage(limit=limit)
        task_events = self.load_task_events()
        job_events = self.load_job_events()
        machine_events = self.load_machine_events()

        # v1: basic
        usage = self.add_basic_features(usage)
        usage = self.add_derivative_and_rolling(usage)
        usage = self.add_burst_features(usage)
        usage = self.add_phase_clustering(usage)
        usage = self.add_periodic_encoding(usage)
        usage = self.add_machine_contention(usage)
        usage = self.add_event_counts(usage, task_events, job_events)

        # v2: autoencoder embedding (choose feature cols automatically)
        auto_cols = [c for c in usage.columns if c not in ['start_time','end_time','job_id','task_index','machine_id']]
        try:
            usage, _scaler = self.compute_autoencoder_embedding(usage, auto_cols, latent_dim=16, epochs=10)
        except Exception as e:
            logger.warning('AE embedding failed: %s', e)

        # v3: DAG features
        usage = self.build_job_dag_features(usage, job_events)

        # v4: RL features
        usage = self.add_rl_features(usage, machine_events)

        usage = usage.fillna(0)
        logger.info("Feature pipeline finished. Result shape: %s", usage.shape)
        return usage


if __name__ == '__main__':
    DB_URI = os.environ.get('DB_URI', 'mysql+pymysql://root:123456@127.0.0.1:3307/xiyoudata')
    fe = FeatureEngineerEnhanced(DB_URI)
    df = fe.build_all_features(limit=200000)
    print(df.head())
