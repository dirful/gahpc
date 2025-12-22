# ==================== HPC 长任务聚类 · 完全修复版（已解决维度不匹配）===================
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import pymysql
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 12
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True

# ========================== 配置区 ==========================
SAMPLE_SIZE      = 800000
WINDOW_SIZE_MIN  = 60
SLIDE_STEP_MIN   = 15
MIN_SEQ_LEN      = 6
BATCH_SIZE       = 256
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# ========================== 1. 数据库 ==========================
class Connector:
    def __init__(self):
        self.conn = pymysql.connect(
            host="localhost", port=3307, user="root",
            password="123456", database="xiyoudata", charset="utf8mb4"
        )
        print("数据库连接成功")

    def load(self):
        print("正在抽样 task_usage...")
        query = f"""
        SELECT job_id, task_index, start_time,
               cpu_rate, canonical_memory_usage, disk_io_time,
               maximum_cpu_rate, sampled_cpu_usage, cycles_per_instruction
        FROM task_usage 
        ORDER BY RAND() 
        LIMIT {SAMPLE_SIZE}
        """
        usage = pd.read_sql(query, self.conn)

        print("正在抽样 task_events...")
        query2 = f"""
        SELECT DISTINCT job_id, task_index, cpu_request, memory_request, priority, disk_space_request
        FROM task_events
        ORDER BY RAND()
        LIMIT {int(SAMPLE_SIZE * 2.5)}
        """
        events = pd.read_sql(query2, self.conn)
        self.conn.close()
        print(f"抽样完成：task_usage {len(usage)} 行，task_events {len(events)} 行")
        return usage, events

# ========================== 2. 时间窗口 ==========================
class TimeProcessor:
    def __init__(self):
        self.ws = WINDOW_SIZE_MIN
        self.ss = SLIDE_STEP_MIN
        self.seq_len = MIN_SEQ_LEN

    def build_sequences(self, df: pd.DataFrame, feats: list):
        df = df.copy()
        df["start_time"] = pd.to_numeric(df["start_time"], errors='coerce')
        df = df.dropna(subset=["start_time"])
        df["minute"] = (df["start_time"] // 1_000_000) // 60

        min_t, max_t = df["minute"].min(), df["minute"].max()
        print(f"时间范围: {min_t} ~ {max_t} 分钟（约{(max_t-min_t)/1440:.2f} 天）")

        bins = np.arange(min_t, max_t + self.ws + 1, self.ss)
        df["wid"] = pd.cut(df["minute"], bins=bins, labels=False, include_lowest=True)
        df = df.dropna(subset=["wid"]).astype({"wid": int})

        agg = df.groupby(["job_id","task_index","wid"])[feats].mean().reset_index()

        seq_dict = {}        # key: (job_id, task_index, window_start_idx) → sequence
        task_static_map = {} # (job_id, task_index) → static features

        for (jid, tid), g in agg.groupby(["job_id", "task_index"]):
            g = g.sort_values("wid")
            windows = g["wid"].values
            values  = g[feats].values

            if len(values) < self.seq_len:
                continue

            # 滑动窗口生成多条序列
            for start in range(0, len(values) - self.seq_len + 1, 3):  # step=3 控制数量
                seq = values[start:start + self.seq_len]
                key = (jid, tid, start)          # 唯一标识
                seq_dict[key] = seq
                task_static_map[key] = (jid, tid)  # 记录属于哪个 task

        print(f"成功构建 {len(seq_dict)} 条长任务序列")
        return seq_dict, task_static_map

# ========================== 3. Transformer AE ==========================
class TransAE(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=8, num_layers=3, latent_dim=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, activation="gelu", dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Linear(d_model, feat_dim)   # 每 timestep 重建原始维度

    def forward(self, x):
        proj_x = self.proj(x)
        enc = self.transformer(proj_x)
        z = self.to_latent(enc.mean(dim=1))
        rec = self.decoder(enc)
        return rec, z, enc

# ========================== 主流程 ==========================
def main():
    conn = Connector()
    usage_df, events_df = conn.load()
    tp = TimeProcessor()

    # 动态特征
    feats = ["cpu_rate","canonical_memory_usage","disk_io_time","maximum_cpu_rate","sampled_cpu_usage"]
    if 'cycles_per_instruction' in usage_df.columns and usage_df['cycles_per_instruction'].notna().any():
        feats.append('cycles_per_instruction')
        print("已加入 cycles_per_instruction 作为第6维动态特征")
    else:
        print("未检测到有效 CPI 字段")

    seq_dict, task_static_map = tp.build_sequences(usage_df, feats)
    if len(seq_dict) == 0:
        raise RuntimeError("无长任务序列！")

    sequences = np.array(list(seq_dict.values()), dtype=np.float32)   # [N, T, D_dyn]

    # CPI 特殊归一化
    if 'cycles_per_instruction' in feats:
        idx = feats.index('cycles_per_instruction')
        cpi = sequences[:, :, idx]
        cpi_log = np.log1p(cpi)
        cpi_scaled = RobustScaler().fit_transform(cpi_log.reshape(-1, 1)).reshape(cpi.shape)
        sequences[:, :, idx] = np.clip(cpi_scaled, -5, 5)

    # 全局标准化动态特征
    scaler_dyn = StandardScaler()
    sequences = scaler_dyn.fit_transform(sequences.reshape(-1, len(feats))).reshape(sequences.shape)

    # ==================== 关键修复：静态特征对齐 ====================
    static_cols = ["priority","cpu_request","memory_request","disk_space_request"]
    # 先建 (job_id, task_index) → static 的映射
    events_df = events_df.drop_duplicates(subset=["job_id", "task_index"])
    static_dict = {(row.job_id, row.task_index): row[static_cols].values for _, row in events_df.iterrows()}

    static_list = []
    valid_keys = []
    for key, seq in seq_dict.items():
        jid, tid, _ = key
        static = static_dict.get((jid, tid))
        if static is not None:
            static_list.append(static)
            valid_keys.append(key)
        # else: 丢弃无静态的任务（极少）

    print(f"静态特征匹配率: {len(static_list)}/{len(seq_dict)}")

    # 只保留有静态特征的序列
    sequences = np.array([seq_dict[k] for k in valid_keys])
    # 重新归一化（因为丢了一些样本）
    sequences = scaler_dyn.transform(sequences.reshape(-1, len(feats))).reshape(sequences.shape)

    static_arr = np.array(static_list, dtype=np.float32)
    scaler_static = StandardScaler()
    static_norm = scaler_static.fit_transform(static_arr)
    static_rep = np.repeat(static_norm[:, np.newaxis, :], MIN_SEQ_LEN, axis=1)  # [N, T, 4]

    # 拼接
    X_np = np.concatenate([sequences, static_rep], axis=2)   # [N, T, D]
    print(f"最终输入形状: {X_np.shape}  (样本数={X_np.shape[0]})")

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    dataset = data.TensorDataset(X_tensor)
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型 & 训练
    feat_dim = X_np.shape[2]
    model = TransAE(feat_dim=feat_dim, latent_dim=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE)
            rec, z, _ = model(x)
            loss = criterion(rec, x)
            loss += 1e-4 * z.abs().mean()   # L1 正则

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} │ Loss: {total_loss/len(loader):.6f}")

    # 提取表征
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            _, z, _ = model(x)
            latents.append(z.cpu().numpy())
    latent_np = np.concatenate(latents)

    # 聚类
    n_clusters = min(8, max(3, len(latent_np)//40))
    latent_pca = PCA(n_components=min(30, latent_np.shape[1]), random_state=42).fit_transform(latent_np)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(latent_pca)
    sil = silhouette_score(latent_pca, labels)
    print(f"聚类完成 → {n_clusters} 类，Silhouette = {sil:.4f}")

    # t-SNE 可视化
    tsne = TSNE(n_components=2, perplexity=min(50, len(latent_np)-1), random_state=42, init='pca')
    embed = tsne.fit_transform(latent_pca)

    plt.figure(figsize=(12,9))
    sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=labels, palette="tab10", s=70, alpha=0.9)
    plt.title(f"HPC 长任务聚类 | N={len(latent_np)} | K={n_clusters} | Sil={sil:.3f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f"hpc_cluster_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 保存结果
    result_df = pd.DataFrame({
        "job_id": [k[0] for k in valid_keys],
        "task_index": [k[1] for k in valid_keys],
        "window_start": [k[2] for k in valid_keys],
        "cluster": labels
    })
    result_df.to_csv(f"hpc_cluster_result_{timestamp}.csv", index=False)
    print("全部完成！结果已保存。")

if __name__ == "__main__":
    main()