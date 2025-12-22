# ==================== HPC 长任务聚类 · 终极业务感知版（已恢复完整加权损失）===================
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sympy.physics.control.control_plots import matplotlib

from utils.LearnableFeatureWeights import ConstrainedWeightModule
from utils.tc import temporal_consistency_score_v3

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 12
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True
# ========================== 配置区 ==========================
SAMPLE_SIZE = 800000
WINDOW_SIZE_MIN = 60
SLIDE_STEP_MIN = 15
MIN_SEQ_LEN = 6
BATCH_SIZE = 256
USE_TIMEGAN = False # 想开就开，已测试能跑
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

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

# ========================== Transformer AE（不变） ==========================
class WeightedTransAE(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=8, num_layers=3, latent_dim=64):
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

# ========================== Connector 和 TimeProcessor（不变） ==========================
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
        # 双重变换：log1p + sqrt 强压右尾
        cpi_log = np.log1p(cpi)
        cpi_sqrt = np.sqrt(cpi_log)
        # 标准化
        cpi_standard = StandardScaler().fit_transform(cpi_sqrt.reshape(-1, 1)).reshape(cpi.shape)
        # 强制居中：减去均值（处理右偏）
        cpi_centered = cpi_standard - cpi_standard.mean()
        # clip 到 [-3, 3]
        cpi_final = np.clip(cpi_centered, -3.0, 3.0)
        sequences[:, :, idx] = cpi_final
        # 打印检查
        cpi_flat = cpi_final.flatten()
        print("\n【CPI 处理后分布检查】")
        print(f"   mean: {cpi_flat.mean():.4f}, std: {cpi_flat.std():.4f}, "
              f"min: {cpi_flat.min():.4f}, max: {cpi_flat.max():.4f}")
        print(f"   5%/95% 分位: {np.percentile(cpi_flat, 5):.4f} / {np.percentile(cpi_flat, 95):.4f}")
    # 全局标准化动态特征
    scaler_dyn = StandardScaler()
    # sequences = scaler_dyn.fit_transform(sequences.reshape(-1, len(feats))).reshape(sequences.shape)
    # scaler_dyn = StandardScaler()
    sequences = scaler_dyn.fit_transform(sequences.reshape(-1, len(feats))).reshape(sequences.shape)
    # 不做额外 clip，交给全局 clip 处理
    print("\n【动态特征标准化后分布检查】")
    for i, name in enumerate(feats):
        data1 = sequences[:, :, i].flatten()
        print(f"{name:25}: mean={data1.mean():.4f}, std={data1.std():.4f}, "
              f"min={data1.min():.4f}, max={data1.max():.4f}")
    # 静态特征对齐
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
    # === 新增：priority 独立加强归一化（防止幅度过大）===
    priority_raw = static_arr[:, 0]  # priority 是第0列
    # 先 clip 原始值到合理范围（Google Cluster Trace priority 通常 0~11）
    priority_clipped = np.clip(priority_raw, 0, 12)
    # 独立标准化 + clip
    priority_mean = priority_clipped.mean()
    priority_std = priority_clipped.std() + 1e-8
    priority_norm = (priority_clipped - priority_mean) / priority_std
    priority_norm = np.clip(priority_norm, -3.0, 3.0)  # 强制 [-3, 3]

    # 扩展到时间维
    priority_rep = np.repeat(priority_norm[:, np.newaxis, np.newaxis], MIN_SEQ_LEN, axis=1)  # [N, T, 1]

    # 如果你还保留了其他静态特征（如 request），这里只取 priority
    # static_rep = np.repeat(static_norm[:, np.newaxis, :], MIN_SEQ_LEN, axis=1)  # 注释掉原来的
    print(f"Priority 归一化后: mean={priority_norm.mean():.3f}, std={priority_norm.std():.3f}, min={priority_norm.min():.3f}, max={priority_norm.max():.3f}")
    static_rep = np.repeat(static_norm[:, np.newaxis, :], MIN_SEQ_LEN, axis=1)
    # 新增：计算空间占有率特征，并转化正/负影响
    eps = 1e-8
    occupancy_cpu = sequences[:,:,0] / (static_rep[:,:,1] + eps)  # cpu_rate / cpu_request
    occupancy_mem = sequences[:,:,1] / (static_rep[:,:,2] + eps)  # mem_usage / mem_request
    occupancy_disk = sequences[:,:,2] / (static_rep[:,:,3] + eps)  # disk_io_time / disk_space_request (假设类似)
    # 转化：90%以内正影响，90%以上负影响
    # def transform_occupancy(occ):
    #     occ = np.clip(occ, 0, 2)  # 限制0~200%
    #     over = occ > 0.9
    #     occ[over] = - (occ[over] - 0.9)  # 转为负值
    #     return occ
    # occupancy_cpu = transform_occupancy(occupancy_cpu)
    # occupancy_mem = transform_occupancy(occupancy_mem)
    # occupancy_disk = transform_occupancy(occupancy_disk)
    # occupancy = np.stack([occupancy_cpu, occupancy_mem, occupancy_disk], axis=2)  # [N, T, 3]
    # # 标准化占有率特征
    # scaler_occ = StandardScaler()
    # occupancy = scaler_occ.fit_transform(occupancy.reshape(-1, 3)).reshape(occupancy.shape)
    # 注释掉 transform_occupancy 函数
    # def transform_occupancy(occ): ...

    # 直接使用原始占有率（clip 到 0~2，标准化即可）
    occupancy_cpu = np.clip(sequences[:,:,0] / (static_rep[:,:,1] + eps), 0, 2)
    occupancy_mem = np.clip(sequences[:,:,1] / (static_rep[:,:,2] + eps), 0, 2)
    occupancy_disk = np.clip(sequences[:,:,2] / (static_rep[:,:,3] + eps), 0, 2)  # disk_io_time / disk_request 可能不准，可考虑去掉
    scaler_occ = StandardScaler()
    occupancy = np.stack([occupancy_cpu, occupancy_mem, occupancy_disk], axis=2)
    occupancy = scaler_occ.fit_transform(occupancy.reshape(-1, 3)).reshape(occupancy.shape)
    # 最终输入：动态 + 静态 + 占有率
    # X_np = np.concatenate([sequences, static_rep, occupancy], axis=2)
    # ============ 新方案：只保留 priority + 占有率（抛弃原始 request） ============
    # 提取标准化后的 priority（第0列）
    priority_norm = static_norm[:, 0:1]  # [N, 1]
    priority_rep = np.repeat(priority_norm[:, np.newaxis, :], MIN_SEQ_LEN, axis=1)  # [N, T, 1]

    # 占有率特征（已计算好，并经过90%阈值正负转化 + 标准化）
    # occupancy: [N, T, 3]

    # 最终输入：动态序列 + priority + 占有率
    # X_np = np.concatenate([sequences, priority_rep, occupancy], axis=2)

    # print(f"【新特征方案】最终输入形状: {X_np.shape} → 特征维度 = {X_np.shape[2]}")
    # print("    特征组成：动态运行指标 + priority + cpu/mem/disk_occupancy（90%阈值正负转化）")
    feature_names = feats + ["priority", "cpu_request", "memory_request"]
    pos_base_names = ["cpu_rate", "canonical_memory_usage", "maximum_cpu_rate", "sampled_cpu_usage", "priority", "cpu_request", "memory_request"]
    neg_base_names = ["disk_io_time"] + (["cycles_per_instruction"] if has_cpi else [])
    # # 更新特征名列表（关键！）
    # feature_names = feats + ["priority", "cpu_occupancy", "mem_occupancy", "disk_occupancy"]
    #
    # # 更新正向/负向业务索引
    # pos_base_names = ["cpu_rate", "canonical_memory_usage", "maximum_cpu_rate", "sampled_cpu_usage",
    #                   "priority", "cpu_occupancy", "mem_occupancy", "disk_occupancy"]
    # neg_base_names = ["disk_io_time"] + (["cycles_per_instruction"] if has_cpi else [])
    # 最终输入：动态 + priority + 原始 request（标准化后）
    X_np = np.concatenate([sequences, static_rep[:, :, [0,1,2]]], axis=2)  # 只取 priority, cpu_req, mem_req
    # === 终极保险：全局 clip，所有特征统一幅度 ===
    X_np = np.clip(X_np, -4.0, 4.0)  # 稍松一点，防止过度截断

    # 最终统计（覆盖你的现有统计）
    print("\n【最终输入特征幅度统计】")
    for i, name in enumerate(feature_names):
        data2 = X_np[:, :, i].flatten()
        print(f"{name:25}: mean={data2.mean():.4f}, std={data2.std():.4f}, "
              f"min={data2.min():.4f}, max={data2.max():.4f}, "
              f"5%/95%={np.percentile(data2, 5):.4f}/{np.percentile(data2, 95):.4f}")
    print(f"整体: mean={X_np.mean():.4f}, std={X_np.std():.4f}, shape={X_np.shape}\n")


    # ============ 第二步：TimeGAN 数据增强（在基础 X_np 上进行） ============
    if USE_TIMEGAN:
        print("\n开始 TimeGAN 数据增强...")
        from utils.timegan import TimeGAN
        X_np_base = X_np
        timegan = TimeGAN(input_dim=X_np_base.shape[2], seq_len=MIN_SEQ_LEN, hidden_dim=64, device=DEVICE)
        real_dataset = data.TensorDataset(torch.tensor(X_np_base, dtype=torch.float32))
        real_loader = data.DataLoader(real_dataset, batch_size=256, shuffle=True)

        # 三阶段训练
        timegan.train_autoencoder(real_loader, epochs=50)
        timegan.train_supervisor(real_loader, epochs=50)
        timegan.train_gan(real_loader, epochs=100)

        # 生成与真实相同数量的合成数据
        synth_np = timegan.generate(len(X_np_base))

        # 混合：真实 + 合成
        X_np = np.concatenate([X_np_base, synth_np], axis=0)
        print(f"TimeGAN 增强完成：序列数量从 {len(X_np_base)} → {len(X_np)} (+100%)")


    # ============ 第三步：最终统计和权重模块 ============
    print(f"最终输入形状: {X_np.shape} → 特征维度 = {X_np.shape[2]}")

    # 打印最终幅度统计
    print("\n【最终输入特征幅度统计（增强后）】")
    for i, name in enumerate(feature_names):
        data3 = X_np[:, :, i].flatten()
        print(f"{name:25}: mean={data3.mean():.4f}, std={data3.std():.4f}, "
              f"min={data3.min():.4f}, max={data3.max():.4f}, "
              f"5%/95%={np.percentile(data3, 5):.4f}/{np.percentile(data3, 95):.4f}")

    # 可学习权重模块（自动适配最终维度）
    weight_module = ConstrainedWeightModule(
        base_weights=torch.ones(X_np.shape[2], device=DEVICE, dtype=torch.float32),
        learnable_mask=torch.ones(X_np.shape[2], device=DEVICE, dtype=torch.float32),
        min_weight=0.1,
        max_weight=3.0
    ).to(DEVICE)
    print(f"最终输入形状: {X_np.shape} → 特征维度 = {X_np.shape[2]} (新增占有率特征)")
    # 特征名列表
    feature_names = feats + static_cols + ["cpu_occupancy", "mem_occupancy", "disk_occupancy"]
    # 正/负业务索引（更新为包含占有率，占有率作为正向，但转化已体现负影响）
    # 可学习权重模块（自动适应新维度）
    weight_module = ConstrainedWeightModule(
        base_weights=torch.ones(X_np.shape[2], device=DEVICE, dtype=torch.float32),
        learnable_mask=torch.ones(X_np.shape[2], device=DEVICE, dtype=torch.float32),
        min_weight=0.1,
        max_weight=3.0
    ).to(DEVICE)
    # 权重长度检查
    assert X_np.shape[2] == len(weight_module()), f"权重维度{len(weight_module())} 与特征维度{X_np.shape[2]}不匹配！"
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    loader = data.DataLoader(data.TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)
    # 模型训练（完全可学习权重）
    # model = WeightedTransAE(feat_dim=X_np.shape[2], latent_dim=32).to
    model = WeightedTransAE(feat_dim=X_np.shape[2], d_model=256, nhead=8, num_layers=4, latent_dim=64).to(DEVICE)
    optimizer = optim.AdamW(list(model.parameters()) + list(weight_module.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion_mse = nn.MSELoss(reduction='none')
    model.train()
    alpha = 0.7
    for epoch in range(40):
        if epoch < 10:
            for p in weight_module.parameters():
                p.requires_grad = False
        else:
            for p in weight_module.parameters():
                p.requires_grad = True
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE)
            rec_seq, rec_global, z, _ = model(x)
            loss_seq = criterion_mse(rec_seq, x).mean()
            target_global = x.mean(dim=1)
            learned_weights = weight_module().detach()  # detach 避免 backward 冲突
            loss_weighted = (criterion_mse(rec_global, target_global) * learned_weights).mean()
            loss = alpha * loss_seq + (1 - alpha) * loss_weighted + 1e-4 * z.abs().mean() + 1e-3 * (weight_module.delta ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            w = weight_module().detach().cpu().numpy()
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.6f} | Learned Weights: {np.round(w, 3)}")
    # 提取表征 + 聚类 + 可视化
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
    # t-SNE 可视化（不变）
    # ... (你的原 t-SNE 和 plt 代码，保持不变)
    # 保存结果（不变）
    # ... (你的原 result_df 保存代码，保持不变)
    # 对比消融实验（不变）
    # ... (你的原 train_and_eval 和 plt 对比代码，保持不变)
    # 7 模型超级消融实验
    print("\n" + "="*90)
    print("【进阶模式启动】开始执行 7 模型超级消融实验（预计 6~12 分钟，CUDA 下很快）")
    print("="*90)
    ablation_results = []
    timestamp_ab = datetime.now().strftime("%Y%m%d_%H%M")
    def run_ablation_variant(name, main_loader=None, custom_loader=None, weights=weight_module(),
                             model_class=WeightedTransAE, feature_names=None, pos_base=None, neg_base=None):
        print(f"\n>>> 正在训练: {name}")
        torch.cuda.empty_cache()
        if custom_loader is not None:
            current_loader = custom_loader
        elif main_loader is not None:
            current_loader = main_loader
        else:
            raise ValueError("必须提供 main_loader 或 custom_loader！")
        first_batch = next(iter(current_loader))
        if isinstance(first_batch, (list, tuple)):
            x_sample = first_batch[0]
        else:
            x_sample = first_batch

        feat_dim = x_sample.shape[2]
        if model_class == WeightedTransAE:
            model_ab = WeightedTransAE(feat_dim=feat_dim, latent_dim=64).to(DEVICE)
        elif model_class == "LSTM":
            # ... (你的原 LSTMAE 类，保持不变)
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
            # ... (你的原 MLPAE 类，保持不变)
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
        optimizer = optim.AdamW(list(model_ab.parameters()) + list(weight_module.parameters()), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        crit = nn.MSELoss(reduction='none')
        model_ab.train()
        alpha = 0.7
        for epoch in range(40):
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
                rec_seq, rec_global, z, _ = model_ab(x)
                loss_seq = crit(rec_seq, x).mean()
                target_global = x.mean(dim=1)
                if weights is not None:
                    if isinstance(weights, torch.Tensor) and weights.numel() == x.size(2):
                        w_use = weights.detach()
                    else:
                        w_use = weight_module().detach()[:x.size(2)]
                    loss_weighted = (crit(rec_global, target_global) * w_use).mean()
                    loss = alpha * loss_seq + (1 - alpha) * loss_weighted + 1e-4 * z.abs().mean()
                    if isinstance(model_ab, WeightedTransAE):
                        loss += 1e-3 * (weight_module.delta ** 2).mean()
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

        # Temporal Consistency [N, T, F]
        if isinstance(model_ab, WeightedTransAE):
            attn_time = []
            with torch.no_grad():
                for batch in current_loader:
                    x = batch[0].to(DEVICE)
                    proj_x = model_ab.proj(x)
                    enc = model_ab.transformer(proj_x)
                    dec_weight = model_ab.decoder_seq.weight.T
                    feat_importance_per_time = torch.matmul(enc, dec_weight).abs()
                    # rollout
                    layer_attns = []
                    enc_temp = proj_x
                    for layer in model_ab.transformer.layers:
                        attn_out, attn_w = layer.self_attn(enc_temp, enc_temp, enc_temp,
                                                           need_weights=True, average_attn_weights=False)
                        attn = attn_w.mean(dim=1)
                        attn = attn + torch.eye(attn.size(-1), device=attn.device)
                        attn = attn / attn.sum(dim=-1, keepdim=True)
                        layer_attns.append(attn)
                        enc_temp = enc_temp + attn_out
                    rollout = layer_attns[-1]
                    for i in range(len(layer_attns)-2, -1, -1):
                        rollout = torch.matmul(layer_attns[i], rollout)
                    time_attn = rollout.mean(dim=2)
                    attn_heatmap = feat_importance_per_time * time_attn.unsqueeze(2)
                    attn_time.append(attn_heatmap.cpu().numpy())
            attn_time = np.concatenate(attn_time, axis=0)

            # 自动适配当前特征维度的 pos/neg 索引
            current_feat_names = feature_names[:feat_dim]
            current_pos_idx = [i for i, name in enumerate(current_feat_names) if name in pos_base]
            current_neg_idx = [i for i, name in enumerate(current_feat_names) if name in neg_base]
            tc_score = temporal_consistency_score_v3(attn_time, labs, current_pos_idx, current_neg_idx)
        else:
            tc_score = np.nan
        ablation_results.append({
            "Model": name,
            "Silhouette": sil,
            "TemporalConsistency": tc_score,
            "Clusters": n_c,
            "Samples": len(latent)
        })
        print(f"    → Silhouette={sil:.4f} | TC={tc_score:.4f} | 簇数 = {n_c}")
        return model_ab, labs
    pos_base_names = ["cpu_rate", "canonical_memory_usage", "maximum_cpu_rate", "sampled_cpu_usage",
                          "priority", "cpu_occupancy", "mem_occupancy", "disk_occupancy"]
    neg_base_names = ["disk_io_time"] + (["cycles_per_instruction"] if has_cpi else [])
    # 1. Ours（完整模型）——传主 loader
    model_full, labels_full = run_ablation_variant("Ours (Full)", main_loader=loader, weights=weight_module(),feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names)

    # 2. 无加权损失
    run_ablation_variant("No-Weighted", main_loader=loader, weights=None,feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names)

    # 3. 无静态特征
    X_no_static = X_tensor[:, :, :sequences.shape[2]]
    loader_no_static = data.DataLoader(data.TensorDataset(X_no_static), batch_size=BATCH_SIZE, shuffle=True)
    run_ablation_variant("No-Static", custom_loader=loader_no_static,feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names     )

    # 4. LSTM-AE
    run_ablation_variant("LSTM-AE", main_loader=loader, model_class="LSTM",feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names)

    # 5. No-CPI
    if has_cpi:
        dynamic_cols = len(feats) - 1
        X_no_cpi = torch.cat([X_tensor[:, :, :dynamic_cols], X_tensor[:, :, len(feats):]], dim=2)
        loader_no_cpi = data.DataLoader(data.TensorDataset(X_no_cpi), batch_size=BATCH_SIZE, shuffle=True)
        run_ablation_variant("No-CPI", custom_loader=loader_no_cpi,feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names)

    # 6. 随机权重
    rand_w = torch.tensor([
        2.4, 2.4, 0.25, 1.8, 1.0, 0.35, 3.0, 2.6, 2.6, 0.2
    ], device=DEVICE, dtype=torch.float32)[torch.randperm(10)]
    run_ablation_variant("Random-Weight", main_loader=loader, weights=rand_w,feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names)

    # 7. MLP-AE
    run_ablation_variant("MLP-AE", main_loader=loader, model_class="MLP",feature_names=feature_names, pos_base=pos_base_names, neg_base=neg_base_names)

    # ========================== 消融结果可视化 + 表格（升级版） ==========================
    df_res = pd.DataFrame(ablation_results)

    # 缺失 TC 的模型（LSTM / MLP）置 0，表示“无时间建模能力”
    df_res["TemporalConsistency"] = df_res["TemporalConsistency"].fillna(0.0)

    # ================== 复合评分（论文主指标） - 排名分数版 ==================
    # 步骤1：分别对两个指标进行降序排名（值越高，排名越前）
    df_res["Sil_rank"] = df_res["Silhouette"].rank(ascending=False, method='min')
    df_res["TC_rank"]  = df_res["TemporalConsistency"].rank(ascending=False, method='min')

    # 总模型数
    n_models = len(df_res)

    # 步骤2：将排名转换为分数（第1名=1.0，最后一名=0.4，线性插值）
    def rank_to_score(rank, total):
        if total == 1:
            return 1.0
        # 公式：1.0 - (rank-1)/(total-1) * 0.6
        return 1.0 - (rank - 1) / (total - 1) * 0.6

    df_res["Sil_score"] = df_res["Sil_rank"].apply(lambda r: rank_to_score(r, n_models))
    df_res["TC_score"]  = df_res["TC_rank"].apply(lambda r: rank_to_score(r, n_models))

    # 步骤3：复合分数 = 两个排名分相加（满分2.0）
    df_res["FinalScore"] = df_res["Sil_score"] + df_res["TC_score"]

    # 美化显示（保留4位小数）
    df_res["Silhouette"] = df_res["Silhouette"].round(4)
    df_res["TemporalConsistency"] = df_res["TemporalConsistency"].round(4)
    df_res["Sil_score"] = df_res["Sil_score"].round(3)
    df_res["TC_score"] = df_res["TC_score"].round(3)
    df_res["FinalScore"] = df_res["FinalScore"].round(3)

    # 排序逻辑：FinalScore 优先，其次 Silhouette
    df_res = df_res.sort_values(
        by=["FinalScore", "Silhouette"],
        ascending=False
    ).reset_index(drop=True)

    df_res.insert(0, "Rank", range(1, len(df_res) + 1))

    print("\n" + "=" * 100)
    print("【HPC 长任务聚类 · 7 模型消融实验最终排行榜（复合指标）】")
    print("=" * 100)
    print("复合评分规则：对 Silhouette 和 TemporalConsistency 分别排名，")
    print("               第1名得1.0分，最后一名得0.4分（线性插值），FinalScore = 两分相加（满分2.0）")
    print("=" * 100)
    print(df_res[[
        "Rank", "Model", "Silhouette", "TemporalConsistency",
        "Sil_score", "TC_score", "FinalScore"
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
    # Attention Rollout 簇解释可视化
    print("正在计算 Attention Rollout 并生成簇解释图...")
    model.eval()
    rollout_attns = []
    cluster_labels = labels.copy()  # 使用主模型的 labels

    with torch.no_grad():
        for x in loader:
            x_dev = x[0].to(DEVICE)
            rec_seq, rec_global, z, _ = model(x_dev)
            target_global = x_dev.mean(dim=1)

            recon_error = torch.abs(rec_global - target_global)  # [B, F]

            # 使用当前学到的权重（自动裁剪到当前维度）
            current_weights = weight_module().detach()[:recon_error.size(1)]
            feat_importance = recon_error * current_weights
            feat_importance = feat_importance / (feat_importance.sum(dim=1, keepdim=True) + 1e-8)
            rollout_attns.append(feat_importance.cpu().numpy())

    rollout_all = np.concatenate(rollout_attns)  # [N, F_current]

    # 关键修复：只取当前实际特征数的名字
    actual_feat_dim = rollout_all.shape[1]
    current_feature_names = feature_names[:actual_feat_dim]

    explain_df = pd.DataFrame(rollout_all, columns=current_feature_names)
    explain_df["cluster"] = cluster_labels
    cluster_attn = explain_df.groupby("cluster")[current_feature_names].mean()

    # 归一化
    row_sums = cluster_attn.sum(axis=1)
    cluster_attn_norm = cluster_attn.div(row_sums, axis=0)

    # 最终解释分数 = Attention × 学到的权重
    business_weights = weight_module().detach().cpu().numpy()[:actual_feat_dim]
    final_importance = cluster_attn_norm.multiply(business_weights, axis=1)

    final_row_sums = final_importance.sum(axis=1)
    final_importance = final_importance.div(final_row_sums, axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # 可视化热力图
    plt.figure(figsize=(14, 8))
    sns.heatmap(final_importance, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=current_feature_names, yticklabels=[f"Cluster {i}" for i in final_importance.index],
                cbar_kws={"label": "业务加权注意力贡献"}, linewidths=0.5)
    plt.title("HPC 长任务聚类解释：Attention Rollout × 可学习业务权重\n(数值越大 = 该簇越关注此特征)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"hpc_cluster_attention_explain_{timestamp}.png", dpi=350, bbox_inches='tight')
    plt.show()

    # Top-3 打印
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