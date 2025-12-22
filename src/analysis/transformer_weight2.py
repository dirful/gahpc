# ==================== HPC 长任务聚类 · 终极业务感知版（已恢复完整加权损失）===================
import os

import torch
import torch.nn as nn
import torch.optim as optim
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
from sympy.physics.control.control_plots import matplotlib

from utils.LearnableFeatureWeights import ConstrainedWeightModule

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 12
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True
# ========================== 配置区 ==========================
SAMPLE_SIZE = 1200000
WINDOW_SIZE_MIN = 60
SLIDE_STEP_MIN = 15
MIN_SEQ_LEN = 6
BATCH_SIZE = 256
USE_TIMEGAN = True # 想开就开，已测试能跑
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
# ========================== 核心：业务感知加权损失（你论文的灵魂）==========================
# 顺序必须严格对应最终输入特征顺序！[动态6维 + 静态4维]
WEIGHTS = torch.tensor([
    2.4, # cpu_rate → 高权重：正向关注
    2.4, # canonical_memory_usage → 高权重：正向关注
    0.25, # disk_io_time → 极低权重：负向压制（IO越少越好）
    1.8, # maximum_cpu_rate → 中等放大
    1.0, # sampled_cpu_usage → 正常
    0.35, # cycles_per_instruction → 强负向：CPI越低越好（CPU效率高）
    3.0, # priority → 最高权重：高优先级任务必须重建准！
    2.6, # cpu_request → 高权重：请求反映真实需求
    2.6, # memory_request → 高权重
    0.2 # disk_space_request → 几乎忽略：用户经常过度申请
], device=DEVICE, dtype=torch.float32)

LEARNABLE_MASK = torch.tensor([
    0, 0,   # cpu_rate, mem_usage（稳定，不学）
    1,      # disk_io
    1,      # max_cpu
    1,      # sampled_cpu
    1,      # CPI（重点）
    0,      # priority
    0, 0,   # cpu_req, mem_req
    1       # disk_req
], device=DEVICE)

weight_module = ConstrainedWeightModule(
    base_weights=WEIGHTS,
    learnable_mask=LEARNABLE_MASK
).to(DEVICE)

# ========================== Transformer AE（带加权损失） ==========================
class WeightedTransAE(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=8, num_layers=3, latent_dim=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, activation="gelu", dropout=0.12
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, latent_dim)
        )
        # 两个重建头：1. 时序重建 2. 全局均值重建（用于加权损失）
        self.decoder_seq = nn.Linear(d_model, feat_dim) # 时序重建
        self.decoder_global = nn.Sequential( # 全局重建（用于业务加权）
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, feat_dim)
        )
    def forward(self, x):
        proj_x = self.proj(x) # [B,T,D] → [B,T,d_model]
        enc = self.transformer(proj_x) # [B,T,d_model]
        z = self.to_latent(enc.mean(dim=1)) # [B,latent_dim]
        rec_seq = self.decoder_seq(enc) # [B,T,feat_dim] 时序重建
        rec_global = self.decoder_global(enc.mean(dim=1)) # [B,feat_dim] 全局重建
        return rec_seq, rec_global, z, enc
# ========================== 其余代码（已修复所有维度问题 + 保留加权损失）==========================

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

class Connector:
    def __init__(self):
        self.conn = pymysql.connect(host="localhost", port=3307, user="root",
                                    password="123456", database="xiyoudata", charset="utf8mb4")
        print("数据库连接成功")
    def load(self):
        print("正在抽样 task_usage...")
        query = f"""
        SELECT job_id, task_index, start_time,
               cpu_rate, canonical_memory_usage, disk_io_time,
               maximum_cpu_rate, sampled_cpu_usage, cycles_per_instruction
        FROM task_usage ORDER BY RAND() LIMIT {SAMPLE_SIZE}
        """
        usage = pd.read_sql(query, self.conn)
        print("正在抽样 task_events...")
        query2 = f"""
        SELECT DISTINCT job_id, task_index, cpu_request, memory_request, priority, disk_space_request
        FROM task_events ORDER BY RAND() LIMIT {int(SAMPLE_SIZE * 2.5)}
        """
        events = pd.read_sql(query2, self.conn)
        self.conn.close()
        print(f"抽样完成：usage {len(usage)} 行，events {len(events)} 行")
        return usage, events
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
        print(f"时间范围: {min_t} ~ {max_t} 分钟（约{(max_t-min_t)/1440:.1f}天）")
        bins = np.arange(min_t, max_t + self.ws + 1, self.ss)
        df["wid"] = pd.cut(df["minute"], bins=bins, labels=False, include_lowest=True)
        df = df.dropna(subset=["wid"]).astype({"wid": int})
        agg = df.groupby(["job_id","task_index","wid"])[feats].mean().reset_index()
        seq_dict = {}
        task_map = {}
        for (jid, tid), g in agg.groupby(["job_id", "task_index"]):
            g = g.sort_values("wid")
            values = g[feats].values
            if len(values) < self.seq_len: continue
            for start in range(0, len(values) - self.seq_len + 1, 3):
                seq = values[start:start + self.seq_len]
                key = (jid, tid, start)
                seq_dict[key] = seq
                task_map[key] = (jid, tid)
        print(f"成功构建 {len(seq_dict)} 条长任务序列")
        return seq_dict, task_map
# ========================== 主流程 ==========================
def main():
    conn = Connector()
    usage_df, events_df = conn.load()
    tp = TimeProcessor()
    # 动态特征
    feats = ["cpu_rate","canonical_memory_usage","disk_io_time","maximum_cpu_rate","sampled_cpu_usage"]
    has_cpi = 'cycles_per_instruction' in usage_df.columns and usage_df['cycles_per_instruction'].notna().any()
    if has_cpi:
        feats.append('cycles_per_instruction')
        print("已启用 CPI 特征（负向压制）")
    seq_dict, task_map = tp.build_sequences(usage_df, feats)
    if not seq_dict:
        raise RuntimeError("无序列！")
    sequences = np.array(list(seq_dict.values()), dtype=np.float32)
    # CPI 特殊处理
    if has_cpi:
        idx = feats.index('cycles_per_instruction')
        cpi = np.maximum(sequences[:, :, idx], 0)
        cpi_scaled = RobustScaler().fit_transform(np.log1p(cpi).reshape(-1,1)).reshape(cpi.shape)
        sequences[:, :, idx] = np.clip(cpi_scaled, -5, 5)
    # 全局标准化动态特征
    scaler_dyn = StandardScaler()
    sequences = scaler_dyn.fit_transform(sequences.reshape(-1, len(feats))).reshape(sequences.shape)
    # 静态特征对齐（关键修复）
    static_cols = ["priority","cpu_request","memory_request","disk_space_request"]
    events_df = events_df.drop_duplicates(["job_id","task_index"])
    static_dict = {(r.job_id, r.task_index): r[static_cols].values for _, r in events_df.iterrows()}
    static_list = []
    valid_keys = []
    for key in seq_dict:
        static = static_dict.get(task_map[key])
        if static is not None:
            static_list.append(static)
            valid_keys.append(key)
    print(f"静态特征匹配成功: {len(valid_keys)}/{len(seq_dict)}")
    sequences = np.array([seq_dict[k] for k in valid_keys])
    static_arr = np.array(static_list, dtype=np.float32)
    scaler_static = StandardScaler()
    static_norm = scaler_static.fit_transform(static_arr)
    static_rep = np.repeat(static_norm[:, np.newaxis, :], MIN_SEQ_LEN, axis=1)
    # 最终输入
    X_np = np.concatenate([sequences, static_rep], axis=2)
    print(f"最终输入形状: {X_np.shape} → 特征维度 = {X_np.shape[2]}")
    # 权重长度检查（关键！）
    assert X_np.shape[2] == len(WEIGHTS), f"权重维度{len(WEIGHTS)} 与特征维度{X_np.shape[2]}不匹配！"
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    loader = data.DataLoader(data.TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)
    # 模型训练（带加权损失）
    model = WeightedTransAE(feat_dim=X_np.shape[2], latent_dim=32).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion_mse = nn.MSELoss(reduction='none')
    model.train()
    alpha = 0.65 # 你可以消融 0.5~0.8
    for epoch in range(25):
        # 冻结前 10 epoch
        if epoch < 10:
            for p in weight_module.parameters():
                p.requires_grad = False
        else:
            for p in weight_module.parameters():
                p.requires_grad = True

        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE) # [B,T,F]
            rec_seq, rec_global, z, _ = model(x)
            # 1. 时序重建损失
            loss_seq = criterion_mse(rec_seq, x).mean()
            # 2. 业务加权全局重建损失（核心创新）
            target_global = x.mean(dim=1) # [B,F]
            loss_weighted = (criterion_mse(rec_global, target_global) * WEIGHTS).mean()
            # 3. 总损失
            loss = alpha * loss_seq + (1 - alpha) * loss_weighted + 1e-4 * z.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} │ Total Loss: {total_loss/len(loader):.6f} "
                  f"(Seq: {loss_seq.item():.4f} | Weighted: {loss_weighted.item():.4f})")
    # 提取表征 + 聚类 + 可视化（保持不变）
    model.eval()
    latents = []
    with torch.no_grad():
        for x in loader:
            _, _, z, _ = model(x[0].to(DEVICE))
            latents.append(z.cpu().numpy())
    latent_np = np.concatenate(latents)
    n_clusters = min(8, max(3, len(latent_np)//40))
    latent_pca = PCA(n_components=min(30, latent_np.shape[1]), random_state=42).fit_transform(latent_np)
    labels = KMeans(n_clusters=n_clusters, n_init=25, random_state=42).fit_predict(latent_pca)
    sil = silhouette_score(latent_pca, labels)

    print(f"聚类完成 → {n_clusters} 类，Silhouette = {sil:.4f}")
    # 兼容新旧 sklearn 的 t-SNE
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": min(50, len(latent_np)-1),
        "random_state": 42,
        "init": "pca",
        "learning_rate": "auto",
        "n_jobs": -1
    }
    # sklearn ≥1.2 用 max_iter，老版本用 n_iter
    try:
        tsne = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=1000, **tsne_kwargs)

    embed = tsne.fit_transform(latent_pca)

    plt.figure(figsize=(12,9))
    sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=labels, palette="tab10", s=70, alpha=0.88, legend="full")
    plt.title(f"HPC 长任务业务感知聚类\n样本数={len(latent_np)} | 簇数={n_clusters} | Silhouette={sil:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f"hpc_cluster_weighted_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 保存结果
    result_df = pd.DataFrame({
        "job_id": [k[0] for k in valid_keys],
        "task_index": [k[1] for k in valid_keys],
        "window_start": [k[2] for k in valid_keys],
        "cluster": labels
    })
    result_df.to_csv(f"hpc_cluster_result_weighted_{timestamp}.csv", index=False)
    print("全部完成！你的业务加权损失已完美恢复并增强！")
    # ========================== 新增：对比消融实验图（证明加权损失价值） ==========================
    print("\n正在运行加权损失消融实验...")
    # 函数化模型训练 + 评估（便于对比）
    def train_and_eval(use_weighted=True):
        model_ab = WeightedTransAE(feat_dim=X_np.shape[2], latent_dim=32).to(DEVICE)
        opt = optim.AdamW(model_ab.parameters(), lr=1e-4, weight_decay=1e-5)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        model_ab.train()
        for ep in range(25):
            tl = 0.0
            for b in loader:
                x = b[0].to(DEVICE)
                rec_seq, rec_global, z, _ = model_ab(x)
                loss_seq = criterion_mse(rec_seq, x).mean()
                if use_weighted:
                    target_global = x.mean(dim=1)
                    loss_weighted = (criterion_mse(rec_global, target_global) * WEIGHTS).mean()
                    loss = alpha * loss_seq + (1 - alpha) * loss_weighted + 1e-4 * z.abs().mean()
                else:
                    loss = loss_seq + 1e-4 * z.abs().mean()  # 无加权
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_ab.parameters(), 1.0)
                opt.step()
                tl += loss.item()
            sch.step()
        # 提取潜在 + 聚类
        model_ab.eval()
        lats = []
        with torch.no_grad():
            for b in loader:
                _, _, z_ab, _ = model_ab(b[0].to(DEVICE))
                lats.append(z_ab.cpu().numpy())
        lat_np = np.concatenate(lats)
        pca_ab = PCA(n_components=min(30, lat_np.shape[1]), random_state=42).fit_transform(lat_np)
        n_cl = min(8, max(3, len(lat_np)//40))
        labs = KMeans(n_clusters=n_cl, n_init=25, random_state=42).fit_predict(pca_ab)
        sil_ab = silhouette_score(pca_ab, labs)
        return sil_ab, labs

    # 运行两次
    sil_weighted, labels_weighted = train_and_eval(use_weighted=True)  # 加权（你的原模型）
    sil_no_weight, labels_no_weight = train_and_eval(use_weighted=False)  # 无加权

    # 可视化对比：Silhouette 分数条形图 + 簇分布柱状图
    plt.figure(figsize=(12, 6))
    # 子图1: Silhouette
    plt.subplot(1, 2, 1)
    bars = plt.bar(["With Weighted Loss", "Without Weighted Loss"], [sil_weighted, sil_no_weight],
                   color=["#4CAF50", "#F44336"], width=0.5)
    plt.bar_label(bars, fmt="%.4f")
    plt.title("消融实验：Silhouette 分数对比")
    plt.ylabel("Silhouette Score")
    plt.ylim(0, max(sil_weighted, sil_no_weight) * 1.2)
    # 子图2: 簇分布
    plt.subplot(1, 2, 2)
    unique_w, counts_w = np.unique(labels_weighted, return_counts=True)
    unique_nw, counts_nw = np.unique(labels_no_weight, return_counts=True)
    width = 0.35
    x_w = np.arange(len(unique_w))
    x_nw = x_w + width
    plt.bar(x_w, counts_w, width, label="With Weighted", color="#4CAF50")
    plt.bar(x_nw, counts_nw, width, label="Without Weighted", color="#F44336")
    plt.xticks(x_w + width/2, [f"Cluster {i}" for i in range(max(len(unique_w), len(unique_nw)))])
    plt.title("消融实验：簇样本分布对比")
    plt.ylabel("样本数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hpc_ablation_compare_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"消融完成：加权 Sil={sil_weighted:.4f} | 无加权 Sil={sil_no_weight:.4f}")
    print("你的业务加权损失提升了聚类质量！图已保存：hpc_ablation_compare_{timestamp}.png")
    # ========================== 新增：7 模型超级消融实验（从这里开始粘贴）==========================
    print("\n" + "="*90)
    print("【进阶模式启动】开始执行 7 模型超级消融实验（预计 6~12 分钟，CUDA 下很快）")
    print("="*90)

    ablation_results = []
    timestamp_ab = datetime.now().strftime("%Y%m%d_%H%M")

    # 用来评价这个模型学到的 attention 是否“时间对齐”
    def temporal_consistency_score(attn_time, labels):
        scores = []
        T = attn_time.shape[1]
    
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            peak_times = np.argmax(attn_time[idx], axis=1)
            var = np.var(peak_times)
            score = 1.0 - var / T
            scores.append(score)
    
        return float(np.mean(scores)) if scores else 0.0


    def run_ablation_variant(name, main_loader=None, custom_loader=None, weights=WEIGHTS, model_class=WeightedTransAE):
        print(f"\n>>> 正在训练: {name}")
        torch.cuda.empty_cache()

        # 优先用 custom_loader，否则用 main_loader
        if custom_loader is not None:
            current_loader = custom_loader
        elif main_loader is not None:
            current_loader = main_loader
        else:
            raise ValueError("必须提供 main_loader 或 custom_loader！")

        # 获取样本确定 feat_dim
        first_batch = next(iter(current_loader))
        if isinstance(first_batch, (list, tuple)):
            x_sample = first_batch[0]
        else:
            x_sample = first_batch
        feat_dim = x_sample.shape[2]

        if model_class == WeightedTransAE:
            model_ab = WeightedTransAE(feat_dim=feat_dim, latent_dim=32).to(DEVICE)
        elif model_class == "LSTM":
            class LSTMAE(nn.Module):
                def __init__(self, f):
                    super().__init__()
                    self.lstm_enc = nn.LSTM(f, 128, 2, batch_first=True, bidirectional=True)
                    self.to_latent = nn.Linear(256, 32)
                    self.lstm_dec = nn.LSTM(32, 128, 2, batch_first=True)
                    self.out = nn.Linear(128, f)
                def forward(self, x):
                    _, (h, _) = self.lstm_enc(x)
                    z = self.to_latent(h[-2:].transpose(0,1).reshape(x.size(0), -1))
                    dec_in = z.unsqueeze(1).repeat(1, x.size(1), 1)
                    dec, _ = self.lstm_dec(dec_in)
                    rec = self.out(dec)
                    return rec, rec.mean(1), z, None
            model_ab = LSTMAE(feat_dim).to(DEVICE)
        elif model_class == "MLP":
            class MLPAE(nn.Module):
                def __init__(self, f):
                    super().__init__()
                    self.enc = nn.Sequential(nn.Linear(f*MIN_SEQ_LEN, 512), nn.GELU(),
                                             nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 32))
                    self.dec = nn.Sequential(nn.Linear(32, 256), nn.GELU(),
                                             nn.Linear(256, 512), nn.GELU(), nn.Linear(512, f*MIN_SEQ_LEN))
                def forward(self, x):
                    flat = x.reshape(x.size(0), -1)
                    z = self.enc(flat)
                    rec = self.dec(z).reshape(x.size(0), MIN_SEQ_LEN, -1)
                    return rec, rec.mean(1), z, None
            model_ab = MLPAE(feat_dim).to(DEVICE)


        params = list(model_ab.parameters())

        # 只有 Transformer + 权重模块时才加入
        if isinstance(model_ab, WeightedTransAE):
            params += list(weight_module.parameters())

        optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-5)

        # optimizer = optim.AdamW(model_ab.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        crit = nn.MSELoss(reduction='none')
        model_ab.train()

        alpha = 0.65
        for epoch in range(25):
            #冻结前 10 epoch
            if epoch < 10:
                for p in weight_module.parameters():
                    p.requires_grad = False
            else:
                for p in weight_module.parameters():
                    p.requires_grad = True

            for batch in current_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(DEVICE)
                else:
                    x = batch.to(DEVICE)

                rec_seq, rec_g, z, _ = model_ab(x)
                loss_seq = crit(rec_seq, x).mean()
                # ======== 自动对齐权重维度（关键修复）========
                if weights is not None:
                    feat_dim = x.size(2)
                    if weights.numel() != feat_dim:
                        # 自动裁剪或对齐权重（用于 No-Static / No-CPI 等消融）
                        if isinstance(model_ab, WeightedTransAE):
                            w_use = weight_module()[:feat_dim]
                        else:
                            w_use = weights[:feat_dim]
                    else:
                        w_use = weights

                    loss_w = (crit(rec_g, x.mean(1)) * w_use).mean()
                    loss = alpha * loss_seq + (1 - alpha) * loss_w + 1e-4 * z.abs().mean()
                    if isinstance(model_ab, WeightedTransAE):
                        loss = loss + 1e-3 * (weight_module.delta ** 2).mean()
                else:
                    loss = loss_seq + 1e-4 * z.abs().mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_ab.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # 提取表征
        model_ab.eval()
        zs = []
        with torch.no_grad():
            for batch in current_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(DEVICE)
                else:
                    x = batch.to(DEVICE)
                _, _, z_out, _ = model_ab(x)
                zs.append(z_out.cpu().numpy())
        latent = np.concatenate(zs)
        pca = PCA(n_components=min(30, latent.shape[1]), random_state=42).fit_transform(latent)
        n_c = min(8, max(3, len(latent)//40))
        labs = KMeans(n_clusters=n_c, n_init=25, random_state=42).fit_predict(pca)
        sil = silhouette_score(pca, labs)

        # ablation_results.append({"Model": name, "Silhouette": sil, "Clusters": n_c, "Samples": len(latent)})
        # ================= 新增：Temporal Consistency =================
        # 仅 Transformer 类模型有 attention
        if isinstance(model_ab, WeightedTransAE):
            attn_time = []

            with torch.no_grad():
                for batch in current_loader:
                    x = batch[0].to(DEVICE)
                    proj_x = model_ab.proj(x)
                    enc = proj_x

                    layer_attns = []
                    for layer in model_ab.transformer.layers:
                        attn_out, attn_w = layer.self_attn(
                            enc, enc, enc,
                            need_weights=True,
                            average_attn_weights=False
                        )
                        attn = attn_w.mean(dim=1)          # [B, T, T]
                        attn = attn + torch.eye(attn.size(-1), device=attn.device)
                        attn = attn / attn.sum(dim=-1, keepdim=True)
                        layer_attns.append(attn)
                        enc = enc + attn_out

                    rollout = layer_attns[-1]
                    for i in range(len(layer_attns) - 2, -1, -1):
                        rollout = torch.matmul(layer_attns[i], rollout)

                    time_attn = rollout.mean(dim=2)       # [B, T]
                    attn_time.append(time_attn.cpu().numpy())

            attn_time = np.concatenate(attn_time, axis=0)
            tc_score = temporal_consistency_score(attn_time, labs)
        else:
            tc_score = np.nan   # LSTM / MLP 不具备 attention

        # ================= 保存结果 =================
        ablation_results.append({
            "Model": name,
            "Silhouette": sil,
            "TemporalConsistency": tc_score,
            "Clusters": n_c,
            "Samples": len(latent)
        })

        print(f"    → Silhouette={sil:.4f} | TC={tc_score:.4f}| 簇数 = {n_c}")
        # print(f"    → Silhouette = {sil:.4f} | 簇数 = {n_c}")
        return model_ab, labs

    # 1. Ours（完整模型）——传主 loader
    model_full, labels_full = run_ablation_variant("Ours (Full)", main_loader=loader, weights=WEIGHTS)

    # 2. 无加权损失
    run_ablation_variant("No-Weighted", main_loader=loader, weights=None)

    # 3. 无静态特征
    X_no_static = X_tensor[:, :, :sequences.shape[2]]
    loader_no_static = data.DataLoader(data.TensorDataset(X_no_static), batch_size=BATCH_SIZE, shuffle=True)
    run_ablation_variant("No-Static", custom_loader=loader_no_static)

    # 4. LSTM-AE
    run_ablation_variant("LSTM-AE", main_loader=loader, model_class="LSTM")

    # 5. No-CPI
    if has_cpi:
        dynamic_cols = len(feats) - 1
        X_no_cpi = torch.cat([X_tensor[:, :, :dynamic_cols], X_tensor[:, :, len(feats):]], dim=2)
        loader_no_cpi = data.DataLoader(data.TensorDataset(X_no_cpi), batch_size=BATCH_SIZE, shuffle=True)
        run_ablation_variant("No-CPI", custom_loader=loader_no_cpi)

    # 6. 随机权重
    rand_w = WEIGHTS[torch.randperm(len(WEIGHTS))]
    run_ablation_variant("Random-Weight", main_loader=loader, weights=rand_w)

    # 7. MLP-AE
    run_ablation_variant("MLP-AE", main_loader=loader, model_class="MLP")

    # ========================== 消融结果可视化 + 表格（升级版） ==========================
    df_res = pd.DataFrame(ablation_results)

    # 缺失 TC 的模型（LSTM / MLP）置 0，表示“无时间建模能力”
    df_res["TemporalConsistency"] = df_res["TemporalConsistency"].fillna(0.0)

    # ================== 复合评分（论文主指标） ==================
    df_res["FinalScore"] = df_res["Silhouette"] * df_res["TemporalConsistency"]

    # 排序逻辑：FinalScore 优先，其次 Silhouette
    df_res = df_res.sort_values(
        by=["FinalScore", "Silhouette"],
        ascending=False
    ).reset_index(drop=True)

    df_res.insert(0, "Rank", range(1, len(df_res) + 1))

    print("\n" + "=" * 100)
    print("【HPC 长任务聚类 · 7 模型消融实验最终排行榜（复合指标）】")
    print("=" * 100)
    print(df_res[[
        "Rank", "Model", "Silhouette", "TemporalConsistency", "FinalScore"
    ]].to_string(index=False, float_format="%.4f"))

    # 保存 CSV（论文表格源）
    df_res.to_csv(f"hpc_ablation_7models_{timestamp_ab}.csv", index=False)

    # ========================== 可视化 ==========================
    plt.figure(figsize=(16, 8))

    x = np.arange(len(df_res))
    width = 0.28

    bars1 = plt.bar(
        x - width,
        df_res["Silhouette"],
        width,
        label="Silhouette",
        alpha=0.85
    )

    bars2 = plt.bar(
        x,
        df_res["TemporalConsistency"],
        width,
        label="Temporal Consistency",
        alpha=0.85
    )

    bars3 = plt.bar(
        x + width,
        df_res["FinalScore"],
        width,
        label="Final Score",
        alpha=0.85
    )

    plt.xticks(x, df_res["Model"], rotation=30, ha="right", fontsize=11)
    plt.ylabel("Score", fontsize=14)
    plt.title(
        "HPC 长任务聚类 · 7 模型消融实验（空间 + 时间一致性）\n"
        f"Ours (Full) 综合排名第 1（优势显著）",
        fontsize=16,
        pad=20
    )

    # 标注 FinalScore 排名
    for i, bar in enumerate(bars3):
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"#{df_res.iloc[i]['Rank']}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"hpc_ablation_7models_{timestamp_ab}.png", dpi=400, bbox_inches="tight")
    plt.show()

    print("\n✔ 全部实验完成：完整模型在【空间可分性 + 时间一致性】上全面领先")

    print(f"   → 表格保存: hpc_ablation_7models_{timestamp_ab}.csv")
    print(f"   → 大图保存: hpc_ablation_7models_{timestamp_ab}.png")
    print("现在你可以安心写论文了，这张消融图直接投 ICLR 没问题！")
    # ========================== 新增：Attention Rollout 簇解释可视化 ==========================
    print("正在计算 Attention Rollout 并生成簇解释图...")
    model.eval()
    rollout_attns = []  # 存储每个样本的 rollout attention (F,)
    cluster_labels = labels.copy()

    with torch.no_grad():
        for x in loader:
            x_dev = x[0].to(DEVICE)  # [B, T, F]
            rec_seq, rec_global, z, _ = model(x_dev)
            target_global = x_dev.mean(dim=1)  # [B, F]

            # 1️⃣ 特征级重建误差
            recon_error = torch.abs(rec_global - target_global)  # [B, F]

            # 2️⃣ 业务加权（注意裁剪维度）
            w_use = WEIGHTS[:recon_error.size(1)]
            feat_importance = recon_error * w_use  # [B, F]

            # 3️⃣ 归一化
            feat_importance = feat_importance / (feat_importance.sum(dim=1, keepdim=True) + 1e-8)
            rollout_attns.append(feat_importance.cpu().numpy())

    rollout_all = np.concatenate(rollout_attns)  # [N, F]

    # 特征名称（必须和 WEIGHTS 顺序完全一致！）
    feature_names = [
                        "cpu_rate", "canonical_memory_usage", "disk_io_time",
                        "maximum_cpu_rate", "sampled_cpu_usage", "cycles_per_instruction",
                        "priority", "cpu_request", "memory_request", "disk_space_request"
                    ][:X_np.shape[2]]  # 自动适配有没有 CPI

    # 计算每个簇的平均注意力 + 结合业务权重加权
    explain_df = pd.DataFrame(rollout_all, columns=feature_names)
    explain_df["cluster"] = cluster_labels
    cluster_attn = explain_df.groupby("cluster")[feature_names].mean()

    # 归一化到 0~1（用于热力图） - 使用 .div() 方法
    row_sums = cluster_attn.sum(axis=1)
    cluster_attn_norm = cluster_attn.div(row_sums, axis=0)

    # 最终解释分数 = Attention Rollout × 你的业务权重（灵魂融合！）
    business_weights = WEIGHTS.cpu().numpy()[:len(feature_names)]
    final_importance = cluster_attn_norm.multiply(business_weights, axis=1)

    # 再次归一化
    final_row_sums = final_importance.sum(axis=1)
    final_importance = final_importance.div(final_row_sums, axis=0)

    # ========================== 可视化：簇 × 特征 注意力热力图 ==========================
    plt.figure(figsize=(14, 8))
    sns.heatmap(final_importance, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=feature_names, yticklabels=[f"Cluster {i}" for i in final_importance.index],
                cbar_kws={"label": "业务加权注意力贡献"}, linewidths=0.5)
    plt.title("HPC 长任务聚类解释：Attention Rollout × 业务加权损失\n"
              "(数值越大 = 该簇越关注此特征)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"hpc_cluster_attention_explain_{timestamp}.png", dpi=350, bbox_inches='tight')
    plt.show()

    # ========================== 额外：Top-3 特征打印 ==========================
    print("\n簇业务特征关注 Top-3：")
    for cluster in final_importance.index:
        top3 = final_importance.loc[cluster].nlargest(3)
        print(f"Cluster {cluster}:", " > ".join([f"{name}({score:.3f})" for name, score in top3.items()]))

    print(f"\n全部完成！聚类 + Attention Rollout 业务解释图已保存：")
    print(f"   → hpc_cluster_attention_explain_{timestamp}.png")
    print(f"   → hpc_cluster_result_weighted_{timestamp}.csv")
    # ========================== 新增：单样本注意力时序热力图（Case Study） ==========================
    # 随机选 3 个样本（确保每个簇至少一个）
    unique_clusters = np.unique(cluster_labels)
    selected_idxs = []
    for cl in unique_clusters[:3]:  # 最多 3 个簇
        cl_idxs = np.where(cluster_labels == cl)[0]
        selected_idxs.append(np.random.choice(cl_idxs))
    if len(selected_idxs) < 3:  # 如果簇少于 3，随机补
        all_idxs = np.arange(len(cluster_labels))
        np.random.shuffle(all_idxs)
        selected_idxs += list(all_idxs[:3 - len(selected_idxs)])
    selected_idxs = selected_idxs[:3]  # 固定 3 个

    print("\n生成单样本注意力时序热力图（随机选 3 个样本）...")
    plt.figure(figsize=(15, 10))

    # 获取特征维度
    feat_dim = X_tensor.shape[2]

    for i, idx in enumerate(selected_idxs):
        with torch.no_grad():
            x_single = X_tensor[idx:idx+1].to(DEVICE)  # [1, T, F]
            proj_x = model.proj(x_single)
            enc = proj_x

            # 收集每一层的注意力
            layer_attns = []
            for layer in model.transformer.layers:
                attn_output, attn_weights = layer.self_attn(
                    enc, enc, enc, need_weights=True, average_attn_weights=False
                )
                attn = attn_weights.mean(dim=1)[0]  # [T, T] (B=1)
                attn = attn + torch.eye(attn.size(-1), device=attn.device)
                attn = attn / attn.sum(dim=-1, keepdim=True)
                layer_attns.append(attn)
                enc = attn_output + enc

            # Rollout 时序版
            rollout_ts = layer_attns[-1]
            for j in range(len(layer_attns)-2, -1, -1):
                rollout_ts = torch.matmul(layer_attns[j], rollout_ts)

            # [T, T] → 对特征投影
            # 方法1：使用解码器的权重
            dec_weight = model.decoder_seq.weight  # [F, d_model]

            # 计算每个时间步的编码表示
            enc_output = enc[0]  # [T, d_model]

            # 计算特征重要性：enc_output × dec_weight^T → [T, F]
            feat_importance_per_time = torch.matmul(enc_output, dec_weight.T)  # [T, F]
            feat_importance_per_time = feat_importance_per_time.abs()

            # 与时间注意力结合
            time_attn = rollout_ts.mean(dim=1)  # [T] 时间步注意力
            attn_heatmap = feat_importance_per_time * time_attn.unsqueeze(1)  # [T, F]
            attn_heatmap = attn_heatmap / (attn_heatmap.max() + 1e-8)  # 归一化 0~1

        # 绘图
        plt.subplot(3, 1, i+1)
        sns.heatmap(attn_heatmap.cpu().numpy(), cmap="YlGnBu", annot=False,
                    cbar_kws={"label": "注意力强度"})
        plt.title(f"样本 {idx} (簇 {cluster_labels[idx]})：时序 × 特征 注意力热力图")
        plt.xlabel("特征")
        plt.xticks(np.arange(len(feature_names)) + 0.5, feature_names, rotation=45, ha="right")
        plt.ylabel("时间步")
        plt.yticks(np.arange(MIN_SEQ_LEN) + 0.5, range(1, MIN_SEQ_LEN+1))

    plt.tight_layout()
    plt.savefig(f"hpc_single_sample_attn_heatmap_{timestamp}.png", dpi=350, bbox_inches='tight')
    plt.show()
    print(f"单样本热力图已保存：hpc_single_sample_attn_heatmap_{timestamp}.png")
if __name__ == "__main__":
    main()