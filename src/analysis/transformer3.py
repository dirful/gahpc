# ==================== HPC 长任务聚类 · 真正可运行最终版 ====================
import os

import torch
import torch.nn as nn
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
from sklearn.preprocessing import StandardScaler
from sympy.physics.control.control_plots import matplotlib

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 12
torch.manual_seed(42)

# ========================== 配置区 ==========================
SAMPLE_SIZE      = 500000          # 50万行必出几千条长任务
WINDOW_SIZE_MIN  = 60              # 30分钟一个窗口
SLIDE_STEP_MIN   = 15              # 每10分钟滑动一次
MIN_SEQ_LEN      = 6              # 至少连续活跃 3 小时（18×10=180分钟）
USE_TIMEGAN      = False           # 先关掉！等完全跑通后再开
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
# ===========================================================


# ==========================
# 字体解决方案 - 方法1：查找系统字体
# ==========================
def setup_chinese_font():
    """设置中文字体，自动查找可用字体"""
    # 获取当前系统字体目录
    font_dirs = []

    # Windows 字体目录
    if os.name == 'nt':
        font_dirs.extend([
            'C:/Windows/Fonts',
            os.path.expanduser('~\\AppData\\Local\\Microsoft\\Windows\\Fonts')
        ])

    # Linux 字体目录
    elif os.name == 'posix':
        font_dirs.extend([
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts')
        ])

    # macOS 字体目录
    elif os.name == 'darwin':
        font_dirs.extend([
            '/Library/Fonts',
            '/System/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ])

    # 常见中文字体列表
    chinese_fonts = [
        'msyh.ttc',  # 微软雅黑
        'msyhbd.ttc',  # 微软雅黑粗体
        'simhei.ttf',  # 黑体
        'simsun.ttc',  # 宋体
        'simkai.ttf',  # 楷体
        'Deng.ttf',  # 等线
        'Dengb.ttf',  # 等线粗体
        'arialuni.ttf',  # Arial Unicode
        'NotoSansCJK-Regular.ttc',  # Noto Sans CJK
        'SourceHanSansSC-Regular.otf',  # 思源黑体
        'FandolSong-Regular.otf',  # Fandol 宋体
        'STHeiti Light.ttc',  # 华文黑体 (macOS)
        'PingFang.ttc',  # 苹方 (macOS)
    ]

    # 查找可用中文字体
    available_fonts = []
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font_file in chinese_fonts:
                font_path = os.path.join(font_dir, font_file)
                if os.path.exists(font_path):
                    available_fonts.append(font_path)

    if available_fonts:
        # 使用第一个找到的中文字体
        font_path = available_fonts[0]
        print(f"使用字体: {font_path}")

        # 添加字体到matplotlib
        matplotlib.font_manager.fontManager.addfont(font_path)
        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()

        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False

        return True
    else:
        print("警告: 未找到中文字体，将使用默认字体")
        # 设置备选字体方案
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 调用字体设置函数
setup_chinese_font()

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
               maximum_cpu_rate, sampled_cpu_usage
        FROM task_usage 
        ORDER BY RAND() 
        LIMIT {SAMPLE_SIZE}
        """
        usage = pd.read_sql(query, self.conn)

        print("正在抽样 task_events...")
        query2 = """
                 SELECT job_id, task_index, cpu_request, memory_request, priority, disk_space_request
                 FROM task_events
                 ORDER BY RAND()
                     LIMIT 1000000 \
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

    def build_sequences(self, df: pd.DataFrame):
        df = df.copy()
        df["start_time"] = pd.to_numeric(df["start_time"], errors='coerce')
        df = df.dropna(subset=["start_time"])
        df["minute"] = (df["start_time"] // 1_000_000) // 60   # 微秒 → 分钟

        min_t, max_t = df["minute"].min(), df["minute"].max()
        print(f"时间范围: {min_t} ~ {max_t} 分钟（约{(max_t-min_t)/1440:.2f} 天）")

        bins = np.arange(min_t, max_t + self.ws + 1, self.ss)
        df["wid"] = pd.cut(df["minute"], bins=bins, labels=False, include_lowest=True)
        df = df.dropna(subset=["wid"]).astype({"wid": int})

        feats = ["cpu_rate","canonical_memory_usage","disk_io_time","maximum_cpu_rate","sampled_cpu_usage"]
        agg = df.groupby(["job_id","task_index","wid"])[feats].mean()

        seq_dict = {}
        for (jid, tid), g in agg.groupby(level=[0,1]):
            g = g.sort_index()
            if len(g) >= self.seq_len:
                seq_dict[(jid, tid)] = g.values[:self.seq_len,:]

        print(f"成功构建 {len(seq_dict)} 条长任务序列（每序列 {self.seq_len} 个时间步）")
        return seq_dict, feats

# ========================== 3. Transformer + AE ==========================
class TransAE(nn.Module):
    def __init__(self, feat_dim=9, d_model=128, nhead=8, num_layers=3, latent_dim=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, activation="gelu", dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )

    def forward(self, x):
        x = self.proj(x)                              # [B, T, d_model]
        x = self.transformer(x)                       # [B, T, d_model]
        x = x.mean(dim=1)                             # 平均池化 → [B, d_model]
        z = self.to_latent(x)                         # [B, latent_dim]
        x_rec = self.decoder(z)                       # [B, d_model]
        return x_rec, z, x

# ========================== 主流程 ==========================
def main():
    # 1. 加载数据
    conn = Connector()
    usage_df, events_df = conn.load()

    # 2. 构建序列
    tp = TimeProcessor()
    seq_dict, feat_cols = tp.build_sequences(usage_df)
    if len(seq_dict) == 0:
        raise RuntimeError("没有找到长任务！请把 SAMPLE_SIZE 再加大或把 MIN_SEQ_LEN 降到 12")

    sequences = np.array(list(seq_dict.values()))      # [N, T, 5]

    # 3. 拼接静态特征（优先匹配，匹配不到用均值填补）
    static_cols = ["priority","cpu_request","memory_request","disk_space_request"]
    static_mean = events_df[static_cols].mean().values
    static_arr = np.tile(static_mean, (len(sequences), 1))

    keys = list(seq_dict.keys())
    for i, (jid, tid) in enumerate(keys):
        match = events_df[(events_df["job_id"]==jid) & (events_df["task_index"]==tid)]
        if len(match) > 0:
            static_arr[i] = match.iloc[0][static_cols].values

    scaler = StandardScaler()
    static_norm = scaler.fit_transform(static_arr)
    static_rep = np.repeat(static_norm[:, np.newaxis, :], MIN_SEQ_LEN, axis=1)  # [N, T, 4]

    # 合并：时序5维 + 静态4维 → 9维
    X_np = np.concatenate([sequences, static_rep], axis=2)   # [N, T, 9]
    X = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)

    print(f"输入模型数据形状: {X.shape}")

    # 4. 训练 Transformer + AE
    model = TransAE(feat_dim=9, latent_dim=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model.train()
    for epoch in range(20):
        rec, z, feat = model(X)
        loss = nn.MSELoss()(rec, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} │ Loss: {loss.item():.6f}")

    # 5. 提取低维特征
    model.eval()
    with torch.no_grad():
        _, latent, _ = model(X)
        latent_np = latent.cpu().numpy()

    # 6. 自动聚类
    n_samples = len(latent_np)
    n_clusters = min(8, max(3, n_samples // 40))
    print(f"样本数 {n_samples} → 使用 {n_clusters} 个簇")

    if n_samples > 50:
        # 推荐写法：最多保留原始维度，不超过 50
        latent_np = PCA(n_components=min(50, latent_np.shape[1], n_samples-1), random_state=42).fit_transform(latent_np)

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(latent_np)

    # 7. t-SNE 可视化 + 保存
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, n_samples-1),
        max_iter=1000,
        learning_rate='auto',
        init='pca',           # 新版推荐
        random_state=42
    )
    embed_2d = tsne.fit_transform(latent_np)

    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x=embed_2d[:,0], y=embed_2d[:,1],
        hue=labels, palette="tab10", s=80, alpha=0.9, legend="full"
    )
    plt.title(f"HPC 长任务聚类结果\n样本数={n_samples} | 聚类数={n_clusters} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster")
    plt.tight_layout()

    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f"hpc_cluster_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 保存聚类标签
    result_df = pd.DataFrame({
        "job_id":      [k[0] for k in keys],
        "task_index":  [k[1] for k in keys],
        "cluster":     labels
    })
    result_df.to_csv(f"hpc_cluster_result_{timestamp}.csv", index=False)

    print(f"\n全部完成！")
    print(f"聚类结果已保存：")
    print(f"   → hpc_cluster_{timestamp}.png")
    print(f"   → hpc_cluster_result_{timestamp}.csv")
    print("直接把这两文件发给导师/领导就行！")
    # ====================== 可解释性分析神器（直接粘贴运行）======================
    print("\n正在生成 8 个簇的可解释性画像...")

    # 1. 恢复原始序列和标签
    final_df = pd.DataFrame({
        "job_id":     [k[0] for k in seq_dict.keys()],
        "task_index": [k[1] for k in seq_dict.keys()],
        "cluster":    labels
    })

    # 把原始时序数据也拿回来（去归一化前）
    orig_sequences = np.array(list(seq_dict.values()))          # [N, 6, 5]
    feat_names = ["CPU使用率", "内存使用率", "磁盘IO强度", "CPU峰值率", "采样CPU使用率"]

    # 2. 统计每个簇的特征画像
    profile = []
    for c in sorted(final_df["cluster"].unique()):
        mask = final_df["cluster"] == c
        seq_c = orig_sequences[mask]                               # 该簇所有序列

        # 时序均值曲线
        mean_curve = seq_c.mean(axis=0)                            # [6, 5]

        # 静态资源请求（从 events_df 再匹配一次更准）
        jobs_in_c = final_df[mask][["job_id","task_index"]]
        static_c = events_df.merge(jobs_in_c, on=["job_id","task_index"], how="inner")[
            ["priority","cpu_request","memory_request","disk_space_request"]
        ]

        profile.append({
            "cluster": c,
            "任务数": len(seq_c),
            "平均CPU请求": static_c["cpu_request"].mean(),
            "平均内存请求": static_c["memory_request"].mean(),
            "平均磁盘请求": static_c["disk_space_request"].mean(),
            "优先级均值": static_c["priority"].mean(),
            "CPU使用率均值": mean_curve[:,0].mean(),
            "内存使用率均值": mean_curve[:,1].mean(),
            "是否高IO": mean_curve[:,2].mean() > mean_curve[:,2].mean() * 1.5,
        })

    profile_df = pd.DataFrame(profile)
    print("\n=== 8个簇的业务画像 ===")
    print(profile_df.round(4))

    # 3. 画出每个簇的平均时序曲线（最直观！）
    plt.figure(figsize=(15, 10))
    for c in sorted(final_df["cluster"].unique()):
        mask = final_df["cluster"] == c
        mean_curve = orig_sequences[mask].mean(axis=0)
        plt.subplot(2, 4, c+1)
        for i, name in enumerate(feat_names):
            plt.plot(mean_curve[:, i], label=name, linewidth=2)
        plt.title(f"Cluster {c}（{mask.sum()} 个任务）", fontsize=14)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        if c == 0: plt.legend()

    plt.suptitle("8 类 HPC 长任务典型时序行为画像", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"hpc_cluster_profiles_{timestamp}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()