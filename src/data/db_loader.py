# src/data/db_loader.py
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from log.logger import get_logger
from tqdm import tqdm

# 统计增强
from scipy.stats import (
    ks_2samp, wasserstein_distance, skew, kurtosis, entropy
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

# 可选可视化模块
try:
    import sweetviz as sv
except:
    sv = None

try:
    from autoviz.AutoViz_Class import AutoViz_Class
except:
    AutoViz_Class = None

logger = get_logger(__name__)


class DBLoader:
    """
    ============================
    增强版 DBLoader（旗舰版）
    ============================
    新增功能：
    - 自动 EDA（Sweetviz / AutoViz）
    - 数值型字段自动统计特征生成
    - 窗口时间序列增强：分桶 / 滑窗 / 缺失修复
    - 特征重要性分析（MI / RF）
    - 更优雅的分批、容错
    """

    def __init__(self, cfg):
        self.cfg = cfg

        logger.info(f"[DBLoader] Connecting database: {cfg.db_url}")
        self.engine = create_engine(cfg.db_url)

        # 可视化管理
        self.enable_sweetviz = getattr(cfg, "enable_sweetviz", False)
        self.enable_autoviz = getattr(cfg, "enable_autoviz", False)

    # ============================================================
    #               0. 可视化增强（Sweetviz / AutoViz）
    # ============================================================
    def run_sweetviz(self, df, name="dataset"):
        if not self.enable_sweetviz or sv is None:
            return

        logger.info("运行 Sweetviz 自动EDA...")
        report = sv.analyze(df)
        save_path = f"sweetviz_{name}.html"
        report.show_html(save_path)
        logger.info(f"Sweetviz 报告已生成：{save_path}")

    def run_autoviz(self, df, name="dataset"):
        if not self.enable_autoviz or AutoViz_Class is None:
            return

        logger.info("运行 AutoViz 自动EDA...")
        AV = AutoViz_Class()
        AV.AutoViz(
            filename="",
            dfte=df,
            depVar="",
            verbose=0,
            lowess=False
        )
        logger.info("AutoViz 图像已生成在当前目录")

    # ============================================================
    #           1. Job-level 特征提取（增强版）
    # ============================================================
    def extract_job_level_features_fast(self, sample_limit=200000):
        logger.info("第一步：获取 job 列表用于分析...")

        job_query = f"""
        SELECT DISTINCT job_id 
        FROM task_usage
        WHERE cpu_rate IS NOT NULL
        LIMIT {sample_limit // 20}
        """

        with self.engine.connect() as conn:
            job_df = pd.read_sql(text(job_query), conn)

        if job_df.empty:
            logger.error("[JobExtract] 没有找到 job_id")
            return pd.DataFrame()

        job_ids = job_df.job_id.tolist()
        logger.info(f"有效 job 数量: {len(job_ids)}")

        # ---------------------
        # 批量提取统计信息
        # ---------------------
        batch_size = 1000
        all_stats = []

        for i in tqdm(range(0, len(job_ids), batch_size), desc="提取Job统计特征"):
            batch_ids = job_ids[i:i + batch_size]
            id_list = ",".join(map(str, batch_ids))

            query = f"""
            SELECT
                job_id,
                AVG(cpu_rate) AS cpu_mean,
                STD(cpu_rate) AS cpu_std,
                MAX(cpu_rate) AS cpu_max,
                AVG(canonical_memory_usage) AS mem_mean,
                STD(canonical_memory_usage) AS mem_std,
                MAX(canonical_memory_usage) AS mem_max,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT machine_id) AS machines,
                MIN(start_time) AS start_t,
                MAX(end_time) AS end_t
            FROM task_usage
            WHERE job_id IN ({id_list})
            GROUP BY job_id
            """

            try:
                with self.engine.connect() as conn:
                    df = pd.read_sql(text(query), conn)
                all_stats.append(df)
            except Exception as e:
                logger.exception(e)

        if not all_stats:
            return pd.DataFrame()

        df = pd.concat(all_stats, ignore_index=True)

        # -----------------------
        # 计算 CPU 95 分位数
        # -----------------------
        logger.info("计算 CPU P95...")
        cpu_95_list = []
        for i in tqdm(range(0, len(job_ids), batch_size), desc="百分位数计算"):
            batch_ids = job_ids[i:i + batch_size]
            id_list = ",".join(map(str, batch_ids))

            q = f"""
            SELECT job_id, cpu_rate
            FROM task_usage
            WHERE job_id IN ({id_list}) AND cpu_rate IS NOT NULL
            ORDER BY job_id, cpu_rate
            """

            try:
                with self.engine.connect() as conn:
                    x = pd.read_sql(text(q), conn)

                pct = x.groupby("job_id").cpu_rate.quantile(0.95).reset_index()
                pct.columns = ["job_id", "cpu_95"]
                cpu_95_list.append(pct)
            except:
                pass

        if cpu_95_list:
            p95 = pd.concat(cpu_95_list, ignore_index=True)
            df = df.merge(p95, on="job_id", how="left")

        # -----------------------
        # 新增数值型自动统计特征
        # -----------------------
        df["duration"] = (df.end_t - df.start_t) / 1000
        df["cpu_cv"] = df.cpu_std / (df.cpu_mean + 1e-8)
        df["mem_cv"] = df.mem_std / (df.mem_mean + 1e-8)
        df["cpu_intensity"] = df.cpu_mean * df.duration
        df["mem_intensity"] = df.mem_mean * df.duration
        df["row_density"] = df.n_rows / (df.duration + 1e-6)

        df = df.fillna(0).set_index("job_id")

        # 运行可选EDA
        self.run_sweetviz(df, name="job_features")
        self.run_autoviz(df, name="job_features")

        logger.info(f"Job-level 特征提取完成，共 {len(df)} 行")
        return df

    # ============================================================
    #           2. 时间序列窗口提取（增强版）
    # ============================================================
    def extract_windowed_time_series_fast(
            self,
            job_ids=None,
            window_seconds=300,
            max_jobs=1000
    ):
        if job_ids is None:
            q = f"""
            SELECT DISTINCT job_id
            FROM task_usage
            WHERE cpu_rate IS NOT NULL
            LIMIT {max_jobs}
            """
            with self.engine.connect() as conn:
                job_df = pd.read_sql(text(q), conn)
            job_ids = job_df.job_id.tolist()

        logger.info(f"开始提取 {len(job_ids)} 个 job 的窗口时间序列...")

        batch_size = 200
        windows = []

        for i in tqdm(range(0, len(job_ids), batch_size), desc="窗口提取"):
            batch = job_ids[i:i + batch_size]
            id_list = ",".join(map(str, batch))

            q = f"""
            SELECT
                job_id,
                FLOOR(start_time/1000/{window_seconds}) AS w,
                AVG(cpu_rate) AS cpu_avg,
                AVG(canonical_memory_usage) AS mem_avg,
                COUNT(*) AS n_task
            FROM task_usage
            WHERE job_id IN ({id_list})
            GROUP BY job_id, w
            ORDER BY job_id, w
            """

            try:
                with self.engine.connect() as conn:
                    df = pd.read_sql(text(q), conn)
                windows.append(df)
            except Exception as e:
                logger.error(e)

        if not windows:
            return pd.DataFrame()

        df = pd.concat(windows, ignore_index=True)
        df = df.fillna(0)

        self.run_sweetviz(df, name="ts_windows")
        self.run_autoviz(df, name="ts_windows")

        return df

    # ============================================================
    #           3. 序列构建（滑动窗口 + 特征拼接）
    # ============================================================
    def build_sequences(
            self,
            window_df,
            job_df,
            seq_len=24,
            min_windows=6,
            pad_value=np.nan
    ):
        if window_df.empty or job_df.empty:
            logger.error("[build_sequences] 输入为空")
            return np.zeros((0, seq_len, 0)), pd.DataFrame()

        time_cols = ["cpu_avg", "mem_avg", "n_task"]
        for c in time_cols:
            if c not in window_df:
                window_df[c] = 0

        job_feature_cols = [c for c in job_df.columns if c not in ["start_t", "end_t"]]
        job_feature_cols = sorted(job_feature_cols)

        seqs, metas = [], []
        grouped = window_df.groupby("job_id")
        common = set(grouped.groups.keys()).intersection(job_df.index)

        for job_id in tqdm(common, desc="构建序列"):
            g = grouped.get_group(job_id).sort_values("w")
            T = len(g)

            if T < min_windows:
                continue

            # --- 滑动窗口 / 补齐 ---
            if T >= seq_len:
                sub = g.iloc[-seq_len:][time_cols].values
            else:
                pad = np.full((seq_len - T, len(time_cols)), pad_value)
                sub = np.vstack([pad, g[time_cols].values])

            # --- 拼接 job-level 特征 ---
            job_feat = job_df.loc[job_id, job_feature_cols].values.astype(np.float32)
            tile_feat = np.tile(job_feat, (seq_len, 1))

            seq = np.concatenate([sub, tile_feat], axis=1)
            seqs.append(seq)
            metas.append({"job_id": job_id, "T": T})

        if not seqs:
            return np.zeros((0, seq_len, len(time_cols) + len(job_feature_cols))), pd.DataFrame()

        arr = np.stack(seqs)
        meta_df = pd.DataFrame(metas)

        return arr, meta_df

    # ============================================================
    #           4. KS + Wasserstein 统计
    # ============================================================
    @staticmethod
    def compute_featurewise_ks_wd(real_flat, synth_flat, feature_names=None):
        F = real_flat.shape[1]
        ks_stats, wd_stats = {}, {}

        for i in range(F):
            a = real_flat[:, i]
            b = synth_flat[:, i]

            try:
                ks_stat, _ = ks_2samp(a, b)
            except:
                ks_stat = np.nan

            try:
                wd = wasserstein_distance(a, b)
            except:
                wd = np.nan

            name = feature_names[i] if feature_names else f"feat_{i}"
            ks_stats[name] = ks_stat
            wd_stats[name] = wd

        return ks_stats, wd_stats

    # ============================================================
    #       5. 特征重要性（MI / RF）
    # ============================================================
    @staticmethod
    def compute_feature_importance(df, target="cpu_mean"):
        numeric = df.select_dtypes(include=[np.number])
        if target not in numeric:
            return None

        X = numeric.drop(columns=[target])
        y = numeric[target]

        # Mutual Information
        mi = mutual_info_regression(X.fillna(0), y)
        mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)

        # Random Forest Importance
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X.fillna(0), y)
        rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

        return {
            "mutual_information": mi,
            "random_forest": rf_imp,
        }


# End of DBLoader
