import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import pymysql
from pymysql.err import OperationalError, ProgrammingError
import time
import random
warnings.filterwarnings("ignore")

# ===================== 1. MySQL连接（核心优化：随机抽样+时间范围） =====================
class HPCMySQLConnector:
    def __init__(
            self,
            host: str = "localhost",
            port: int = 3307,
            user: str = "root",
            password: str = "123456",
            database: str = "xiyoudata",
            sample_size: int = 50000,  # 随机抽样总数
            timeout: int = 600
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.sample_size = sample_size  # 改为抽样总数，而非分批大小
        self.timeout = timeout
        self.conn = None

    def connect(self) -> bool:
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
                write_timeout=self.timeout
            )
            cursor = self.conn.cursor()
            cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {self.timeout * 1000};")
            cursor.close()
            print(f"✅ 成功连接MySQL数据库: {self.host}:{self.port}/{self.database}")
            print(f"✅ 设置会话查询超时为{self.timeout}秒，随机抽样{self.sample_size}行")
            return True
        except OperationalError as e:
            print(f"❌ 数据库连接失败: {e}")
            try:
                self.conn = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    charset="utf8mb4"
                )
                print(f"✅ 降级连接成功（关闭超时配置）")
                return True
            except:
                return False
        except ProgrammingError as e:
            print(f"❌ 超时配置失败（忽略）: {e}")
            return True

    def random_sample_table(self, table_name: str) -> pd.DataFrame:
        """核心优化：随机抽样，确保覆盖不同时间窗口"""
        if not self.conn:
            raise ValueError("请先调用connect()连接数据库")

        # 只加载必要字段
        field_mapping = {
            "task_usage": "job_id, task_index, start_time, cpu_rate, canonical_memory_usage, disk_io_time, maximum_cpu_rate, sampled_cpu_usage, machine_id",
            "task_events": "job_id, task_index, priority, cpu_request, memory_request, disk_space_request",
            "job_events": "job_id",
            "machine_events": "*",
            "machine_attributes": "*",
            "task_constraints": "*"
        }
        select_cols = field_mapping.get(table_name, "*")

        try:
            # ========== 关键优化1：先获取时间范围 ==========
            if table_name == "task_usage" and "start_time" in select_cols:
                cursor = self.conn.cursor()
                # 获取时间范围
                cursor.execute(f"SELECT MIN(start_time), MAX(start_time) FROM {table_name}")
                min_time, max_time = cursor.fetchone()
                print(f"🔍 {table_name} 时间范围: [{min_time}, {max_time}]")

                # 按时间分片抽样（确保覆盖不同时间段）
                time_bins = 10  # 分成10个时间片
                bin_size = (max_time - min_time) // time_bins if max_time > min_time else 1
                df_list = []

                for bin_idx in range(time_bins):
                    bin_start = min_time + bin_idx * bin_size
                    bin_end = bin_start + bin_size
                    # 每个时间片抽样部分数据
                    sample_per_bin = self.sample_size // time_bins

                    query = f"""
                        SELECT {select_cols} 
                        FROM {table_name} 
                        WHERE start_time >= {bin_start} AND start_time < {bin_end}
                        ORDER BY RAND()  # 随机排序
                        LIMIT {sample_per_bin}
                    """
                    batch_df = pd.read_sql(query, self.conn)
                    if len(batch_df) > 0:
                        df_list.append(batch_df)
                        print(f"🔄 {table_name} 时间片{bin_idx}: 抽样{len(batch_df)}行 [{bin_start}, {bin_end}]")

                # 合并所有时间片数据
                if df_list:
                    final_df = pd.concat(df_list, ignore_index=True)
                    # 去重并截断到目标抽样数
                    final_df = final_df.drop_duplicates(subset=["job_id", "task_index", "start_time"])
                    final_df = final_df.head(self.sample_size)
                else:
                    # 备用方案：全表随机抽样
                    query = f"""
                        SELECT {select_cols} 
                        FROM {table_name} 
                        ORDER BY RAND()
                        LIMIT {self.sample_size}
                    """
                    final_df = pd.read_sql(query, self.conn)
            else:
                # 非task_usage表直接随机抽样
                query = f"""
                    SELECT {select_cols} 
                    FROM {table_name} 
                    ORDER BY RAND()
                    LIMIT {self.sample_size}
                """
                final_df = pd.read_sql(query, self.conn)

            # ========== 关键优化2：检查时间戳多样性 ==========
            if table_name == "task_usage" and "start_time" in final_df.columns:
                unique_times = final_df["start_time"].nunique()
                print(f"✅ {table_name} 抽样完成：共{len(final_df)}行，唯一时间戳数={unique_times}")
                if unique_times < 5:
                    print(f"⚠️ {table_name} 时间戳多样性不足，补充随机数据")
                    # 补充更多随机数据
                    extra_query = f"""
                        SELECT {select_cols} 
                        FROM {table_name} 
                        ORDER BY RAND()
                        LIMIT {self.sample_size // 2}
                    """
                    extra_df = pd.read_sql(extra_query, self.conn)
                    final_df = pd.concat([final_df, extra_df], ignore_index=True).drop_duplicates()
                    final_df = final_df.head(self.sample_size)
                    print(f"✅ {table_name} 补充后：共{len(final_df)}行，唯一时间戳数={final_df['start_time'].nunique()}")
            else:
                print(f"✅ 加载表 {table_name} 成功，共 {len(final_df)} 行")

            return final_df

        except Exception as e:
            print(f"❌ 随机抽样{table_name}失败: {e}")
            # 降级方案：按偏移量加载（兼容旧逻辑）
            return self.batch_load_table(table_name, limit=self.sample_size)

    def batch_load_table(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """降级方案：分批加载（保留旧逻辑）"""
        field_mapping = {
            "task_usage": "job_id, task_index, start_time, cpu_rate, canonical_memory_usage, disk_io_time, maximum_cpu_rate, sampled_cpu_usage, machine_id",
            "task_events": "job_id, task_index, priority, cpu_request, memory_request, disk_space_request",
            "job_events": "job_id",
            "machine_events": "*",
            "machine_attributes": "*",
            "task_constraints": "*"
        }
        select_cols = field_mapping.get(table_name, "*")

        df_list = []
        offset = 0
        max_rows = limit if limit else float('inf')

        while True:
            query = f"""
                SELECT {select_cols} 
                FROM {table_name} 
                ORDER BY (SELECT NULL)
                LIMIT 5000 OFFSET {offset}
            """
            try:
                cursor = self.conn.cursor(pymysql.cursors.SSDictCursor)
                cursor.execute(query)
                batch_data = cursor.fetchall()
                cursor.close()

                if not batch_data:
                    break

                batch_df = pd.DataFrame(batch_data)
                df_list.append(batch_df)
                offset += 5000
                print(f"🔄 加载{table_name}：已加载{offset}行（当前批次{len(batch_df)}行）")

                if offset >= max_rows:
                    break

                time.sleep(0.05)
            except Exception as e:
                print(f"❌ 分批加载{table_name}失败（偏移量{offset}）: {e}")
                break

        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            if limit and len(final_df) > limit:
                final_df = final_df.head(limit)
            print(f"✅ 加载表 {table_name} 成功，共 {len(final_df)} 行")
            return final_df
        else:
            print(f"❌ 加载表 {table_name} 失败：无数据")
            return pd.DataFrame()

    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """加载所有表（task_usage用随机抽样，其他表用分批加载）"""
        tables = [
            "job_events", "task_events", "machine_events",
            "machine_attributes", "task_constraints", "task_usage"
        ]
        hpc_data = {}

        # 先加载小表
        for table in ["job_events", "machine_events", "task_constraints", "task_events", "machine_attributes"]:
            hpc_data[table] = self.batch_load_table(table, limit=self.sample_size)

        # task_usage用随机抽样（核心）
        hpc_data["task_usage"] = self.random_sample_table("task_usage")

        return hpc_data

    def close(self):
        if self.conn:
            self.conn.close()
            print("✅ 数据库连接已关闭")

# ===================== 2. 时序窗口构建（恢复正常逻辑，移除虚拟窗口） =====================
class TimeWindowProcessor:
    def __init__(
            self,
            window_size: int = 10,
            slide_step: int = 5,
            seq_len: int = 50  # 恢复正常时序长度
    ):
        self.window_size = window_size
        self.slide_step = slide_step
        self.seq_len = seq_len

    def create_time_windows(self, df: pd.DataFrame, time_col: str = "start_time") -> pd.DataFrame:
        if len(df) == 0:
            print("⚠️ 空DataFrame，跳过时间窗口构建")
            return df

        # 检查时间列
        if time_col not in df.columns:
            print(f"⚠️ 缺少时间列{time_col}，可用列：{df.columns.tolist()}")
            time_candidates = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
            if time_candidates:
                time_col = time_candidates[0]
                print(f"✅ 自动匹配时间列：{time_col}")
            else:
                print("❌ 无可用时间列，返回原数据")
                return df

        # 处理时间列
        df = df.copy()
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])

        if len(df) == 0:
            print("⚠️ 时间列无有效数值，返回空数据")
            return df

        # 正常生成时间窗口（恢复原逻辑）
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        time_span = max_time - min_time
        print(f"🔍 时间范围: [{min_time}, {max_time}], 跨度: {time_span}秒")

        window_bins = np.arange(min_time, max_time + self.window_size, self.slide_step)
        df["window_id"] = pd.cut(
            df[time_col],
            bins=window_bins,
            labels=False,
            include_lowest=True
        )
        df = df.dropna(subset=["window_id"])
        df["window_id"] = df["window_id"].astype(int)

        window_count = df["window_id"].nunique()
        print(f"✅ 生成时间窗口完成：总窗口数={window_count}, 覆盖时间跨度={time_span}秒")
        return df

    def build_task_time_series(self, task_usage: pd.DataFrame) -> Dict[Tuple[int, int], np.ndarray]:
        if len(task_usage) == 0:
            print("⚠️ task_usage为空，返回空时序数据")
            return {}

        # 定义核心特征
        feat_cols = [
            "cpu_rate", "canonical_memory_usage", "disk_io_time",
            "maximum_cpu_rate", "sampled_cpu_usage"
        ]
        # 检查特征列
        missing_feats = [col for col in feat_cols if col not in task_usage.columns]
        if missing_feats:
            print(f"⚠️ 缺少时序特征列：{missing_feats}，仅使用存在的列")
            feat_cols = [col for col in feat_cols if col in task_usage.columns]
            if not feat_cols:
                print("❌ 无可用时序特征列，返回空数据")
                return {}

        # 检查关键列
        for col in ["job_id", "task_index"]:
            if col not in task_usage.columns:
                print(f"❌ 缺少关键列{col}，无法构建时序数据")
                return {}

        # 生成窗口
        task_usage = self.create_time_windows(task_usage)
        if len(task_usage) == 0:
            return {}

        # 窗口聚合
        agg_dict = {col: "mean" for col in feat_cols}
        try:
            task_window_agg = task_usage.groupby(["job_id", "task_index", "window_id"]).agg(agg_dict).reset_index()
        except Exception as e:
            print(f"❌ 窗口聚合失败: {e}")
            return {}

        # 构建时序序列
        task_series = {}
        task_groups = task_window_agg.groupby(["job_id", "task_index"])

        for (job_id, task_index), group in task_groups:
            # 按窗口排序
            group_sorted = group.sort_values("window_id")
            group_feats = group_sorted[feat_cols].values

            # 只保留时序长度足够的Task
            if len(group_feats) >= self.seq_len:
                task_series[(job_id, task_index)] = group_feats[:self.seq_len]

        print(f"✅ Task时序序列构建完成：有效Task数={len(task_series)}, 时序长度={self.seq_len}")
        return task_series

# ===================== 3. 数据预处理（恢复正常逻辑） =====================
class HPCDataPreprocessor:
    def __init__(
            self,
            seq_len: int = 50,
            window_size: int = 10,
            slide_step: int = 5
    ):
        self.seq_len = seq_len
        self.window_processor = TimeWindowProcessor(
            window_size=window_size,
            slide_step=slide_step,
            seq_len=seq_len
        )
        self.feature_cols = [
            "cpu_rate", "canonical_memory_usage", "disk_io_time",
            "maximum_cpu_rate", "sampled_cpu_usage"
        ]
        self.static_cols = [
            "priority", "cpu_request", "memory_request", "disk_space_request"
        ]
        self.scalers = {}

    def normalize_series(self, series: np.ndarray, col_name: str = None) -> np.ndarray:
        mean = series.mean()
        std = series.std() + 1e-8
        if col_name:
            self.scalers[col_name] = (mean, std)
        return (series - mean) / std

    def process_task_data(self, hpc_data: Dict[str, pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        print("\n--- 数据基础统计 ---")
        for table_name, df in hpc_data.items():
            print(f"{table_name}: 总行数={len(df)}, 非空行数={len(df.dropna())}")

        task_usage = hpc_data["task_usage"].copy() if "task_usage" in hpc_data else pd.DataFrame()
        task_events = hpc_data["task_events"].copy() if "task_events" in hpc_data else pd.DataFrame()
        job_events = hpc_data["job_events"].copy() if "job_events" in hpc_data else pd.DataFrame()

        if len(task_usage) == 0:
            raise ValueError("task_usage表为空，无法继续处理！")

        # 移除虚拟时序相关代码，恢复正常逻辑
        print("\n--- 构建Task时序窗口 ---")
        task_series_dict = self.window_processor.build_task_time_series(task_usage)
        if len(task_series_dict) == 0:
            raise ValueError("无有效时序数据！请检查时间窗口配置或数据")

        print("\n--- 静态特征处理 ---")
        static_feat_available = len(task_events) > 0
        if static_feat_available:
            for col in self.static_cols:
                if col in task_events.columns:
                    mean_val = task_events[col].mean()
                    task_events[col] = task_events[col].fillna(mean_val)
                    print(f"填充task_events.{col}缺失值，均值={mean_val:.4f}")
        else:
            print("⚠️ task_events为空，静态特征使用默认值")

        task_list = []
        task_metas = []
        job_ids = job_events["job_id"].dropna().unique() if len(job_events) > 0 else []
        job_id_map = {jid: idx for idx, jid in enumerate(job_ids)} if len(job_ids) > 0 else {}

        processed_count = 0
        skipped_count = 0

        for (job_id, task_index), ts_data in task_series_dict.items():
            try:
                ts_data_norm = np.zeros_like(ts_data)
                for i, col in enumerate(self.feature_cols[:ts_data.shape[1]]):
                    ts_data_norm[:, i] = self.normalize_series(ts_data[:, i], col)

                static_data = np.zeros(len(self.static_cols))
                if static_feat_available and "job_id" in task_events.columns and "task_index" in task_events.columns:
                    static_match = task_events[
                        (task_events["job_id"] == job_id) & (task_events["task_index"] == task_index)
                        ]
                    if len(static_match) > 0:
                        for i, col in enumerate(self.static_cols):
                            if col in static_match.columns:
                                static_data[i] = static_match[col].iloc[0]
                        static_data = self.normalize_series(static_data)

                static_repeated = np.tile(static_data, (self.seq_len, 1))
                task_feat = np.concatenate([ts_data_norm, static_repeated], axis=1)
                task_list.append(task_feat)

                machine_id = task_usage[
                    (task_usage["job_id"] == job_id) & (task_usage["task_index"] == task_index)
                    ]["machine_id"].iloc[0] if ("machine_id" in task_usage.columns and len(task_usage) > 0) else -1
                job_id_mapped = job_id_map.get(job_id, -1)

                task_metas.append({
                    "job_id": job_id_mapped,
                    "task_index": task_index,
                    "machine_id": machine_id,
                    "priority": static_data[0],
                    "cpu_request": static_data[1],
                    "memory_request": static_data[2],
                    "disk_request": static_data[3],
                    "raw_job_id": job_id
                })
                processed_count += 1
            except Exception as e:
                skipped_count += 1
                continue

        print(f"\n--- Task处理统计 ---")
        print(f"成功处理Task数: {processed_count}")
        print(f"跳过Task数: {skipped_count}")

        if processed_count == 0:
            raise ValueError(f"无有效Task数据！成功处理={processed_count}, 跳过={skipped_count}")

        all_job_ids = [meta["job_id"] for meta in task_metas]
        unique_job_ids = sorted(list(set(all_job_ids)))
        num_jobs_final = len(unique_job_ids)
        job_id_final_map = {jid: idx for idx, jid in enumerate(unique_job_ids)}

        job_mask = torch.zeros(num_jobs_final, len(task_list))
        for task_idx, meta in enumerate(task_metas):
            job_idx = job_id_final_map[meta["job_id"]]
            job_mask[job_idx, task_idx] = 1.0

        model_input = torch.tensor(np.array(task_list), dtype=torch.float32)
        print(f"\n--- 预处理结果 ---")
        print(f"模型输入形状: {model_input.shape} [num_tasks, seq_len, feat_dim]")
        print(f"Job-Task矩阵形状: {job_mask.shape} [num_jobs, num_tasks]")
        return model_input, job_mask, task_metas

# ===================== 4. Transformer模型（恢复正常复杂度） =====================
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_feat_dim: int,
            d_model: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            seq_len: int = 50
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_feat_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, job_mask: torch.Tensor = None, aggregate_job: bool = False) -> torch.Tensor:
        x = self.input_proj(x) + self.pos_encoding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = x.mean(dim=1)

        if aggregate_job and job_mask is not None:
            x = torch.matmul(job_mask, x) / job_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return self.layer_norm(x)

class HPCAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 128, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

class TransAE(nn.Module):
    def __init__(
            self,
            input_feat_dim: int,
            seq_len: int = 50,
            d_model: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            latent_dim: int = 32,
            dropout: float = 0.1
    ):
        super().__init__()
        self.transformer = TransformerEncoder(
            input_feat_dim=input_feat_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len
        )
        self.ae = HPCAutoencoder(input_dim=d_model, latent_dim=latent_dim, dropout=dropout)

    def forward(
            self,
            x: torch.Tensor,
            job_mask: torch.Tensor = None,
            aggregate_job: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trans_feat = self.transformer(x, job_mask, aggregate_job=aggregate_job)
        recon_feat, latent_feat = self.ae(trans_feat)
        return recon_feat, latent_feat, trans_feat

# ===================== 5. 自定义聚类（无修改） =====================
class HPCCustomKMeans:
    def __init__(self, n_clusters: int = 5, hpc_weight: float = 0.2):
        self.n_clusters = n_clusters
        self.hpc_weight = hpc_weight
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.centroids = None
        self.labels = None

    def custom_distance(self, feat1: np.ndarray, feat2: np.ndarray, meta1: Dict, meta2: Dict) -> float:
        feat_dist = np.linalg.norm(feat1 - feat2)
        res1 = np.array([meta1["cpu_request"], meta1["memory_request"], meta1["disk_request"]])
        res2 = np.array([meta2["cpu_request"], meta2["memory_request"], meta2["disk_request"]])
        res_dist = 1 - np.dot(res1, res2) / (np.linalg.norm(res1)*np.linalg.norm(res2) + 1e-8)
        job_dist = 0 if meta1["job_id"] == meta2["job_id"] else 1.0
        return 0.5*feat_dist + 0.3*res_dist + 0.2*job_dist

    def fit(self, latent_feat: np.ndarray, task_metas: List[Dict]):
        self.kmeans.fit(latent_feat)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

        max_iter = 10
        for _ in range(max_iter):
            new_labels = []
            for i in range(len(latent_feat)):
                distances = []
                for j in range(self.n_clusters):
                    cluster_meta = [task_metas[k] for k in range(len(self.labels)) if self.labels[k] == j]
                    if not cluster_meta:
                        centroid_feat = self.centroids[j]
                    else:
                        centroid_feat = np.mean([latent_feat[k] for k in range(len(self.labels)) if self.labels[k] == j], axis=0)
                    distances.append(self.custom_distance(latent_feat[i], centroid_feat, task_metas[i], cluster_meta[0] if cluster_meta else task_metas[i]))
                new_labels.append(np.argmin(distances))

            new_centroids = []
            for c in range(self.n_clusters):
                cluster_feats = latent_feat[np.array(new_labels) == c]
                if len(cluster_feats) == 0:
                    new_centroids.append(self.centroids[c])
                else:
                    new_centroids.append(np.mean(cluster_feats, axis=0))

            if np.array_equal(self.labels, new_labels):
                break
            self.labels = new_labels
            self.centroids = np.array(new_centroids)

    def evaluate(self, latent_feat: np.ndarray, task_metas: List[Dict]) -> Dict:
        intra_res_consist = []
        for c in range(self.n_clusters):
            cluster_metas = [task_metas[i] for i in range(len(self.labels)) if self.labels[i] == c]
            if len(cluster_metas) < 2:
                intra_res_consist.append(1.0)
                continue
            cpu_var = np.var([m["cpu_request"] for m in cluster_metas])
            mem_var = np.var([m["memory_request"] for m in cluster_metas])
            intra_res_consist.append(1 - (cpu_var + mem_var) / 2)

        sil_score = silhouette_score(latent_feat, self.labels) if len(np.unique(self.labels)) > 1 else 0.0

        job_cluster_ratio = []
        for job_id in np.unique([m["job_id"] for m in task_metas]):
            job_tasks = [i for i, m in enumerate(task_metas) if m["job_id"] == job_id]
            if len(job_tasks) == 0:
                continue
            job_labels = [self.labels[i] for i in job_tasks]
            max_count = max([job_labels.count(l) for l in np.unique(job_labels)])
            job_cluster_ratio.append(max_count / len(job_labels))

        return {
            "intra_resource_consistency": np.mean(intra_res_consist),
            "silhouette_score": sil_score,
            "job_cohesion": np.mean(job_cluster_ratio) if job_cluster_ratio else 0.0,
            "total_score": 0.4*np.mean(intra_res_consist) + 0.3*sil_score + 0.3*np.mean(job_cluster_ratio)
        }

# ===================== 6. 主流程（恢复正常配置） =====================
def main():
    # ===================== 核心配置（恢复正常） =====================
    WINDOW_SIZE = 10
    SLIDE_STEP = 5
    SEQ_LEN = 50           # 恢复正常时序长度
    LATENT_DIM = 32        # 恢复正常特征维度
    D_MODEL = 128          # 恢复正常模型复杂度
    NUM_CLUSTERS = 5       # 恢复正常聚类数
    EPOCHS = 10            # 恢复正常训练轮数
    SAMPLE_SIZE = 100000   # 随机抽样10万行（覆盖更多时间窗口）
    DB_HOST = "localhost"
    BATCH_SIZE = 256       # 恢复正常批次大小
    DB_TIMEOUT = 600       # 恢复正常超时时间

    # ===================== 1. 连接MySQL并加载数据 =====================
    print("=== 1. 连接MySQL数据库 ===")
    connector = HPCMySQLConnector(
        host=DB_HOST,
        port=3307,
        user="root",
        password="123456",
        database="xiyoudata",
        sample_size=SAMPLE_SIZE,
        timeout=DB_TIMEOUT
    )
    if not connector.connect():
        print("❌ 数据库连接失败，程序退出")
        return

    print("\n=== 2. 随机抽样加载HPC数据表（10万行） ===")
    hpc_data = connector.load_all_tables()
    connector.close()

    # ===================== 2. 数据预处理 =====================
    print("\n=== 3. 数据预处理（时序窗口+特征整合） ===")
    preprocessor = HPCDataPreprocessor(
        seq_len=SEQ_LEN,
        window_size=WINDOW_SIZE,
        slide_step=SLIDE_STEP
    )
    try:
        model_input, job_mask, task_metas = preprocessor.process_task_data(hpc_data)
    except ValueError as e:
        print(f"❌ 数据预处理失败: {e}")
        return

    # ===================== 3. 初始化模型 =====================
    print("\n=== 4. 初始化模型与分批训练 ===")
    input_feat_dim = model_input.shape[-1]
    model = TransAE(
        input_feat_dim=input_feat_dim,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        num_heads=4,
        num_layers=2,
        latent_dim=LATENT_DIM,
        dropout=0.1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    num_samples = model_input.shape[0]
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"总样本数: {num_samples}, 批次大小: {BATCH_SIZE}, 总批次: {num_batches}")

    # ===================== 4. 模型训练 =====================
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, num_samples)
            batch_input = model_input[start_idx:end_idx]

            recon_feat, latent_feat, trans_feat = model(batch_input, aggregate_job=False)
            loss = loss_fn(recon_feat, trans_feat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * (end_idx - start_idx)

        avg_loss = epoch_loss / num_samples
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    # ===================== 5. 提取特征 =====================
    print("\n=== 5. 提取低维特征 ===")
    model.eval()
    latent_feat_list = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, num_samples)
            batch_input = model_input[start_idx:end_idx]

            _, latent_feat, _ = model(batch_input, aggregate_job=False)
            latent_feat_list.append(latent_feat.cpu().numpy())

    if latent_feat_list:
        task_latent_np = np.concatenate(latent_feat_list, axis=0)
        print(f"✅ Task级低维特征形状: {task_latent_np.shape}")

        # Job级聚合
        if job_mask.shape[1] == len(task_latent_np):
            task_latent_tensor = torch.tensor(task_latent_np, dtype=torch.float32)
            with torch.no_grad():
                job_latent_tensor = torch.matmul(job_mask, task_latent_tensor) / job_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
            job_latent_np = job_latent_tensor.cpu().numpy()
            print(f"✅ Job级低维特征形状: {job_latent_np.shape}")
            cluster_feat = job_latent_np
        else:
            cluster_feat = task_latent_np
            print("⚠️ 无法聚合Job级特征，使用Task级特征聚类")

        # ===================== 6. 自定义聚类 =====================
        print("\n=== 6. 自定义HPC聚类 ===")
        if len(cluster_feat) >= NUM_CLUSTERS:
            # PCA降维（可选）
            if cluster_feat.shape[1] > 50:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50, random_state=42)
                cluster_feat = pca.fit_transform(cluster_feat)
                print(f"PCA降维后特征形状: {cluster_feat.shape}")

            hpc_kmeans = HPCCustomKMeans(n_clusters=NUM_CLUSTERS)
            hpc_kmeans.fit(cluster_feat, task_metas)

            # 聚类评价
            metrics = hpc_kmeans.evaluate(cluster_feat, task_metas)
            print("✅ 聚类评价指标:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # 结果分析
            print("\n=== 7. 聚类结果分析 ===")
            for cluster_id in range(NUM_CLUSTERS):
                cluster_tasks = [task_metas[i] for i in range(len(hpc_kmeans.labels)) if hpc_kmeans.labels[i] == cluster_id]
                if not cluster_tasks:
                    continue
                avg_cpu = np.mean([t["cpu_request"] for t in cluster_tasks])
                avg_mem = np.mean([t["memory_request"] for t in cluster_tasks])
                job_count = len(set([t["raw_job_id"] for t in cluster_tasks]))
                print(f"聚类 {cluster_id}:")
                print(f"  包含Task数: {len(cluster_tasks)}")
                print(f"  涉及Job数: {job_count}")
                print(f"  平均CPU请求: {avg_cpu:.4f}")
                print(f"  平均内存请求: {avg_mem:.4f}")
        else:
            print("⚠️ 数据量不足，跳过聚类")
    else:
        print("⚠️ 无低维特征，跳过聚类")

    print("\n=== 流程执行完成 ===")

if __name__ == "__main__":
    main()
