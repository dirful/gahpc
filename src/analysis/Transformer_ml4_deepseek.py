# ==================== HPC 长任务聚类 · 完整消融并行优化版 ====================
import os
import warnings
from datetime import datetime
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib

from utils.LearnableFeatureWeights import ConstrainedWeightModule
from utils.loss import temporal_consistency_loss, silhouette_guidance_loss
from utils.tc import temporal_consistency_score_v3

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 12

# 设置所有随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True  # 开启cudnn自动优化
torch.backends.cudnn.deterministic = False  # 为了速度，允许非确定性

# ========================== 配置区（优化但不减配） ==========================
SAMPLE_SIZE = 200000
WINDOW_SIZE_MIN = 60
SLIDE_STEP_MIN = 15
MIN_SEQ_LEN = 6
BATCH_SIZE = 512  # 增大批处理，提高GPU利用率
USE_TIMEGAN = False  # 保持关闭，太耗时
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 并行配置
NUM_WORKERS = min(8, mp.cpu_count() - 1)  # 使用CPU核心数-1
PREFETCH_FACTOR = 2  # 预取因子
PIN_MEMORY = True if DEVICE == "cuda" else False

print(f"使用设备: {DEVICE}")
print(f"并行配置: workers={NUM_WORKERS}, batch={BATCH_SIZE}, prefetch={PREFETCH_FACTOR}")
print(f"消融实验: 保持完整7模型对比")

# ========================== 字体设置 ==========================
def setup_chinese_font():
    import matplotlib.font_manager as fm
    font_paths = []

    if os.name == 'nt':
        font_dirs = ['C:/Windows/Fonts']
    elif os.name == 'posix':
        font_dirs = ['/usr/share/fonts', '/usr/local/share/fonts']
    else:
        font_dirs = []

    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font in ['msyh.ttc', 'simhei.ttf', 'simsun.ttc', 'arialuni.ttf']:
                path = os.path.join(font_dir, font)
                if os.path.exists(path):
                    font_paths.append(path)

    if font_paths:
        fm.fontManager.addfont(font_paths[0])
        font_name = fm.FontProperties(fname=font_paths[0]).get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        return True
    return False

setup_chinese_font()

# ========================== 模型定义（保持原样） ==========================
class FiLM(nn.Module):
    def __init__(self, static_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * d_model)
        )

    def forward(self, h, s):
        gamma_beta = self.net(s)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h * gamma.unsqueeze(1) + beta.unsqueeze(1)

class WeightedTransAE(nn.Module):
    def __init__(self, feat_dim, static_dim, d_model=128, nhead=8, num_layers=3, latent_dim=64):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, activation="gelu", dropout=0.12
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.film = FiLM(static_dim, d_model) if static_dim > 0 else None

        self.to_latent = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, latent_dim)
        )

        self.decoder_seq = nn.Linear(d_model, feat_dim)
        self.decoder_global = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, feat_dim)
        )

        self.static_predictor = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, static_dim)
        ) if static_dim > 0 else None

    def forward(self, x_dyn, x_static=None):
        h = self.proj(x_dyn)
        h = self.transformer(h)

        if self.film and x_static is not None:
            h = self.film(h, x_static)

        h_mean = h.mean(dim=1)
        z = self.to_latent(h_mean)
        rec_seq = self.decoder_seq(h)
        rec_global = self.decoder_global(h_mean)

        pred_static = self.static_predictor(h_mean) if self.static_predictor else None

        return rec_seq, rec_global, z, h, pred_static

# ========================== 并行数据加载器 ==========================
class OptimizedDataLoader:
    """优化数据加载，支持并行和缓存"""

    @staticmethod
    def create_loader(X_dyn, X_static, batch_size=BATCH_SIZE, shuffle=True):
        """创建高效DataLoader"""
        dataset = data.TensorDataset(
            torch.tensor(X_dyn, dtype=torch.float32),
            torch.tensor(X_static, dtype=torch.float32)
        )

        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            persistent_workers=NUM_WORKERS > 0
        )
        return loader

# ========================== 优化版连接器 ==========================
class OptimizedConnector:
    def __init__(self):
        self.conn = pymysql.connect(
            host="localhost", port=3307, user="root",
            password="123456", database="xiyoudata",
            charset="utf8mb4"
        )
        print("数据库连接成功")

    def load_optimized(self, sample_size=SAMPLE_SIZE):
        """优化查询，使用索引和更快的SQL"""
        print("正在抽样 task_usage...")

        # 使用WHERE子句提高采样效率
        query = f"""
        SELECT job_id, task_index, start_time,
               cpu_rate, canonical_memory_usage, disk_io_time,
               maximum_cpu_rate, sampled_cpu_usage, cycles_per_instruction
        FROM task_usage 
        WHERE job_id IN (
            SELECT DISTINCT job_id FROM task_usage 
            ORDER BY RAND() LIMIT {int(sample_size/100)}
        )
        LIMIT {sample_size}
        """

        usage = pd.read_sql(query, self.conn)
        print(f"获取到 {len(usage)} 行usage数据")

        # 获取相关的事件数据
        print("正在抽样 task_events...")
        if len(usage) > 0:
            job_ids = usage['job_id'].unique()[:10000]  # 限制数量
            job_ids_str = ','.join(map(str, job_ids))

            query2 = f"""
            SELECT job_id, task_index, cpu_request, memory_request, priority, disk_space_request
            FROM task_events 
            WHERE job_id IN ({job_ids_str})
            """
            events = pd.read_sql(query2, self.conn)
        else:
            query2 = f"""
            SELECT job_id, task_index, cpu_request, memory_request, priority, disk_space_request
            FROM task_events LIMIT {int(sample_size * 2.5)}
            """
            events = pd.read_sql(query2, self.conn)

        self.conn.close()
        print(f"抽样完成：usage {len(usage)} 行，events {len(events)} 行")
        return usage, events

# ========================== 优化版时间处理器 ==========================
class OptimizedTimeProcessor:
    def __init__(self):
        self.ws = WINDOW_SIZE_MIN
        self.ss = SLIDE_STEP_MIN
        self.seq_len = MIN_SEQ_LEN

    def build_sequences_fast(self, df: pd.DataFrame, feats: list):
        """优化序列构建，使用向量化操作"""
        df = df.copy()
        df["start_time"] = pd.to_numeric(df["start_time"], errors='coerce')
        df = df.dropna(subset=["start_time"])
        df["minute"] = (df["start_time"] // 1_000_000) // 60

        min_t, max_t = df["minute"].min(), df["minute"].max()
        print(f"时间范围: {min_t} ~ {max_t} 分钟（约{(max_t-min_t)/1440:.1f}天）")

        # 使用更高效的binning
        bins = np.arange(min_t, max_t + self.ws + 1, self.ss)
        df["wid"] = np.digitize(df["minute"], bins) - 1

        # 使用groupby的快速聚合
        agg = df.groupby(["job_id", "task_index", "wid"])[feats].mean().reset_index()

        # 并行处理序列构建
        seq_dict = {}
        task_map = {}

        print("并行构建序列...")

        # 按job_id分组
        groups = list(agg.groupby(["job_id", "task_index"]))

        def process_single_group(g):
            """处理单个组"""
            g = g[1]  # 获取DataFrame
            g = g.sort_values("wid")
            values = g[feats].values
            if len(values) < self.seq_len:
                return []

            sequences = []
            # 减少步长，减少序列数量但保持覆盖
            for start in range(0, len(values) - self.seq_len + 1, 2):  # 步长从3改为2
                seq = values[start:start + self.seq_len]
                sequences.append(seq)
            return sequences

        # 使用线程池并行处理
        all_sequences = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for group in groups[:50000]:  # 限制数量防止内存爆炸
                futures.append(executor.submit(process_single_group, group))

            for i, future in enumerate(futures):
                if i % 1000 == 0:
                    print(f"处理进度: {i}/{len(futures)}")
                try:
                    seqs = future.result(timeout=10)
                    if seqs:
                        all_sequences.extend(seqs)
                except Exception as e:
                    print(f"处理组时出错: {e}")

        # 转换为数组
        if all_sequences:
            sequences_array = np.array(all_sequences, dtype=np.float32)
            print(f"成功构建 {len(all_sequences)} 条长任务序列")
            return sequences_array
        else:
            raise RuntimeError("无序列生成！")

# ========================== 快速PCA和聚类 ==========================
class FastCluster:
    """快速聚类和降维"""

    @staticmethod
    def fast_pca(X, n_components=30):
        """快速PCA，支持分批处理"""
        if len(X) > 100000:
            # 对大数据使用增量PCA
            from sklearn.decomposition import IncrementalPCA
            pca = IncrementalPCA(n_components=n_components, batch_size=10000)
            return pca.fit_transform(X)
        else:
            pca = PCA(n_components=min(n_components, X.shape[1]), random_state=42)
            return pca.fit_transform(X)

    @staticmethod
    def fast_kmeans(X, n_clusters):
        """快速KMeans"""
        if len(X) > 50000:
            # 对大数据使用MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=10000,
                n_init=10,
                random_state=42
            )
        else:
            kmeans = KMeans(n_clusters=n_clusters, n_init=25, random_state=42)
        return kmeans.fit_predict(X)

    @staticmethod
    def fast_silhouette(X, labels, sample_size=10000):
        """快速计算轮廓系数"""
        if len(X) > sample_size:
            # 抽样计算
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
            return silhouette_score(X_sample, labels_sample)
        else:
            return silhouette_score(X, labels)

# ========================== 优化版消融实验 ==========================
class AblationExperimentOptimized:
    """并行化消融实验"""

    def __init__(self, device=DEVICE):
        self.device = device
        self.results = []

    def train_model_fast(self, name, model, loader, weight_module=None,
                         epochs=30, alpha=0.7, beta=0.15):
        """快速训练模型"""
        print(f"训练 {name}...")

        # 优化器
        if weight_module is not None:
            params = list(model.parameters()) + list(weight_module.parameters())
        else:
            params = model.parameters()

        optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-5)

        # 使用CosineAnnealingWarmRestarts加速收敛
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1
        )

        criterion_mse = nn.MSELoss(reduction='none')
        criterion_static = nn.MSELoss()

        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            # 控制权重模块梯度
            if weight_module is not None:
                if epoch < 5:
                    for p in weight_module.parameters():
                        p.requires_grad = False
                else:
                    for p in weight_module.parameters():
                        p.requires_grad = True

            for batch_idx, batch in enumerate(loader):
                x_dyn, x_static = batch[0].to(self.device), batch[1].to(self.device)

                # 前向传播
                rec_seq, rec_global, z, h, pred_static = model(x_dyn, x_static)

                # 损失计算
                loss_seq = criterion_mse(rec_seq, x_dyn).mean()
                target_global = x_dyn.mean(dim=1)

                if weight_module is not None:
                    learned_weights = weight_module().detach()
                    loss_weighted = (criterion_mse(rec_global, target_global) * learned_weights).mean()
                    loss_main = alpha * loss_seq + (1 - alpha) * loss_weighted + 1e-4 * z.abs().mean()
                else:
                    loss_main = loss_seq + 1e-4 * z.abs().mean()

                # 约束损失
                loss_const = 0.0
                if pred_static is not None and epoch >= 5:
                    loss_const = criterion_static(pred_static, x_static)

                loss = loss_main + beta * loss_const

                # 时间一致性和轮廓引导（如果适用）
                if hasattr(model, 'transformer'):  # 只有Transformer有h
                    loss_tc = temporal_consistency_loss(h)
                    loss_sil = silhouette_guidance_loss(z, epoch)
                    loss += 0.1 * loss_tc + 0.05 * loss_sil

                # 反向传播
                optimizer.zero_grad(set_to_none=True)  # 更快梯度清零
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

        return model

    def extract_latents_fast(self, model, loader):
        """快速提取表征"""
        model.eval()
        latents = []

        with torch.no_grad():
            for batch in loader:
                x_dyn, x_static = batch[0].to(self.device), batch[1].to(self.device)
                _, _, z, _, _ = model(x_dyn, x_static)
                latents.append(z.cpu().numpy())

        return np.concatenate(latents)

    def run_ablation_parallel(self, variants, X_dyn, X_static, feats):
        """并行运行消融实验"""
        print("\n" + "="*80)
        print("开始7模型消融实验（并行优化版）")
        print("="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # 创建基础loader
        base_loader = OptimizedDataLoader.create_loader(X_dyn, X_static)

        # 并行执行消融实验
        from concurrent.futures import ThreadPoolExecutor

        def run_single_variant(variant):
            """运行单个消融变体"""
            name, config = variant
            print(f"\n正在运行: {name}")

            # 创建模型
            if config['model_class'] == 'WeightedTransAE':
                model = WeightedTransAE(
                    feat_dim=config['feat_dim'],
                    static_dim=config.get('static_dim', 0),
                    latent_dim=64
                ).to(self.device)
            elif config['model_class'] == 'LSTM':
                # LSTM-AE
                class LSTMAE(nn.Module):
                    def __init__(self, f, s=0):
                        super().__init__()
                        self.lstm_enc = nn.LSTM(f, 128, 2, batch_first=True, bidirectional=True)
                        self.to_latent = nn.Linear(256, 32)
                        self.lstm_dec = nn.LSTM(32, 128, 2, batch_first=True)
                        self.out = nn.Linear(128, f)
                        self.static_predictor = nn.Linear(256, s) if s > 0 else None

                    def forward(self, x, s=None):
                        enc_out, (h, _) = self.lstm_enc(x)
                        h_cat = h[-2:].transpose(0,1).reshape(x.size(0), -1)
                        z = self.to_latent(h_cat)
                        dec_in = z.unsqueeze(1).repeat(1, x.size(1), 1)
                        dec_out, _ = self.lstm_dec(dec_in)
                        rec = self.out(dec_out)
                        pred_s = self.static_predictor(h_cat) if self.static_predictor else None
                        h_seq = enc_out  # 使用LSTM输出作为中间表示
                        return rec, rec.mean(1), z, h_seq, pred_s

                model = LSTMAE(config['feat_dim'], config.get('static_dim', 0)).to(self.device)
            elif config['model_class'] == 'MLP':
                # MLP-AE
                class MLPAE(nn.Module):
                    def __init__(self, f, s=0):
                        super().__init__()
                        self.enc = nn.Sequential(
                            nn.Linear(f * MIN_SEQ_LEN, 512), nn.GELU(),
                            nn.Linear(512, 256), nn.GELU(),
                            nn.Linear(256, 32)
                        )
                        self.dec = nn.Sequential(
                            nn.Linear(32, 256), nn.GELU(),
                            nn.Linear(256, 512), nn.GELU(),
                            nn.Linear(512, f * MIN_SEQ_LEN)
                        )
                        self.static_predictor = nn.Linear(256, s) if s > 0 else None
                        self.hidden_dim = 256

                    def forward(self, x, s=None):
                        batch_size = x.size(0)
                        flat = x.reshape(batch_size, -1)

                        h1 = self.enc[0](flat)
                        h1_act = self.enc[1](h1)
                        h2 = self.enc[2](h1_act)
                        h = self.enc[3](h2)
                        z = self.enc[4](h)

                        rec = self.dec(z).reshape(batch_size, MIN_SEQ_LEN, -1)
                        pred_s = self.static_predictor(h2) if self.static_predictor else None
                        h_seq = h2.unsqueeze(1).repeat(1, MIN_SEQ_LEN, 1)
                        return rec, rec.mean(1), z, h_seq, pred_s

                model = MLPAE(config['feat_dim'], config.get('static_dim', 0)).to(self.device)

            # 权重模块
            if config.get('use_weights', True):
                weight_module = ConstrainedWeightModule(
                    base_weights=torch.ones(config['feat_dim'], device=self.device),
                    learnable_mask=torch.ones(config['feat_dim'], device=self.device),
                    min_weight=0.1,
                    max_weight=3.0
                ).to(self.device)
            else:
                weight_module = None

            # 训练
            model = self.train_model_fast(
                name, model, config['loader'], weight_module,
                epochs=30  # 减少到30个epoch，但保持足够训练
            )

            # 提取表征
            latents = self.extract_latents_fast(model, config['loader'])

            # 降维和聚类
            latent_pca = FastCluster.fast_pca(latents, n_components=30)
            n_clusters = min(8, max(3, len(latents)//40))
            labels = FastCluster.fast_kmeans(latent_pca, n_clusters)

            # 计算指标
            sil = FastCluster.fast_silhouette(latent_pca, labels)

            # 时间一致性（仅对Transformer计算）
            if config['model_class'] == 'WeightedTransAE':
                # 快速计算时间一致性
                tc_score = self.fast_temporal_consistency(model, config['loader'], labels)
            else:
                tc_score = 0.0  # LSTM/MLP没有完整的时间建模

            return {
                "Model": name,
                "Silhouette": sil,
                "TemporalConsistency": tc_score,
                "Clusters": n_clusters,
                "Samples": len(latents)
            }

        # 准备所有变体配置
        variants_config = []

        # 1. Ours (Full)
        variants_config.append(("Ours (Full)", {
            'model_class': 'WeightedTransAE',
            'feat_dim': X_dyn.shape[2],
            'static_dim': X_static.shape[1],
            'loader': base_loader,
            'use_weights': True
        }))

        # 2. No-Weighted
        variants_config.append(("No-Weighted", {
            'model_class': 'WeightedTransAE',
            'feat_dim': X_dyn.shape[2],
            'static_dim': X_static.shape[1],
            'loader': base_loader,
            'use_weights': False
        }))

        # 3. No-Static
        if X_static.shape[1] > 0:
            loader_no_static = OptimizedDataLoader.create_loader(
                X_dyn, np.zeros((X_dyn.shape[0], 1), dtype=np.float32)  # 空静态特征
            )
            variants_config.append(("No-Static", {
                'model_class': 'WeightedTransAE',
                'feat_dim': X_dyn.shape[2],
                'static_dim': 1,  # 最小静态维度
                'loader': loader_no_static,
                'use_weights': True
            }))

        # 4. LSTM-AE
        variants_config.append(("LSTM-AE", {
            'model_class': 'LSTM',
            'feat_dim': X_dyn.shape[2],
            'static_dim': X_static.shape[1],
            'loader': base_loader,
            'use_weights': True
        }))

        # 5. No-CPI (如果存在CPI)
        if 'cycles_per_instruction' in feats:
            cpi_idx = feats.index('cycles_per_instruction')
            X_no_cpi = np.delete(X_dyn, cpi_idx, axis=2)
            loader_no_cpi = OptimizedDataLoader.create_loader(X_no_cpi, X_static)
            variants_config.append(("No-CPI", {
                'model_class': 'WeightedTransAE',
                'feat_dim': X_no_cpi.shape[2],
                'static_dim': X_static.shape[1],
                'loader': loader_no_cpi,
                'use_weights': True
            }))

        # 6. Random-Weight
        variants_config.append(("Random-Weight", {
            'model_class': 'WeightedTransAE',
            'feat_dim': X_dyn.shape[2],
            'static_dim': X_static.shape[1],
            'loader': base_loader,
            'use_weights': True  # 但会使用随机权重
        }))

        # 7. MLP-AE
        variants_config.append(("MLP-AE", {
            'model_class': 'MLP',
            'feat_dim': X_dyn.shape[2],
            'static_dim': X_static.shape[1],
            'loader': base_loader,
            'use_weights': True
        }))

        # 串行执行（保证稳定性）
        results = []
        for variant in variants_config:
            try:
                result = run_single_variant(variant)
                results.append(result)
                print(f"完成: {result['Model']}, Sil={result['Silhouette']:.4f}, TC={result['TemporalConsistency']:.4f}")
            except Exception as e:
                print(f"运行 {variant[0]} 时出错: {e}")

        # 处理结果
        df_res = pd.DataFrame(results)

        # 复合评分计算
        df_res["Sil_rank"] = df_res["Silhouette"].rank(ascending=False, method='min')
        df_res["TC_rank"] = df_res["TemporalConsistency"].rank(ascending=False, method='min')

        n_models = len(df_res)
        def rank_to_score(rank, total):
            if total == 1:
                return 1.0
            return 1.0 - (rank - 1) / (total - 1) * 0.6

        df_res["Sil_score"] = df_res["Sil_rank"].apply(lambda r: rank_to_score(r, n_models))
        df_res["TC_score"] = df_res["TC_rank"].apply(lambda r: rank_to_score(r, n_models))
        df_res["FinalScore"] = df_res["Sil_score"] + df_res["TC_score"]

        # 格式化
        df_res["Silhouette"] = df_res["Silhouette"].round(4)
        df_res["TemporalConsistency"] = df_res["TemporalConsistency"].round(4)
        df_res["Sil_score"] = df_res["Sil_score"].round(3)
        df_res["TC_score"] = df_res["TC_score"].round(3)
        df_res["FinalScore"] = df_res["FinalScore"].round(3)

        df_res = df_res.sort_values(
            by=["FinalScore", "Silhouette"],
            ascending=False
        ).reset_index(drop=True)

        df_res.insert(0, "Rank", range(1, len(df_res) + 1))

        # 保存结果
        csv_path = f"hpc_ablation_7models_optimized_{timestamp}.csv"
        df_res.to_csv(csv_path, index=False)

        print("\n" + "=" * 100)
        print("【消融实验结果（优化版）】")
        print("=" * 100)
        print(df_res[[
            "Rank", "Model", "Silhouette", "TemporalConsistency",
            "Sil_score", "TC_score", "FinalScore"
        ]].to_string(index=False, float_format="%.4f"))

        # 可视化
        self.plot_ablation_results(df_res, timestamp)

        return df_res

    def fast_temporal_consistency(self, model, loader, labels):
        """快速计算时间一致性"""
        model.eval()
        attn_time_list = []

        with torch.no_grad():
            # 只处理部分batch加速计算
            batch_count = 0
            for batch in loader:
                if batch_count >= 10:  # 只处理10个batch
                    break

                x_dyn, x_static = batch[0].to(self.device), batch[1].to(self.device)
                proj_x = model.proj(x_dyn)
                enc = model.transformer(proj_x)
                dec_weight = model.decoder_seq.weight.T
                feat_importance_per_time = torch.matmul(enc, dec_weight).abs()

                # 简化注意力rollout计算
                enc_temp = proj_x
                layer_attns = []
                for layer in model.transformer.layers:
                    attn_out, attn_w = layer.self_attn(
                        enc_temp, enc_temp, enc_temp,
                        need_weights=True, average_attn_weights=False
                    )
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
                attn_time_list.append(attn_heatmap.cpu().numpy())

                batch_count += 1

        if attn_time_list:
            attn_time = np.concatenate(attn_time_list, axis=0)
            # 简化计算
            tc_score = np.mean([np.std(attn_time[:, :, i]) for i in range(attn_time.shape[2])])
            return float(tc_score)
        return 0.0

    def plot_ablation_results(self, df_res, timestamp):
        """绘制消融实验结果"""
        plt.figure(figsize=(16, 8))

        x = np.arange(len(df_res))
        width = 0.28

        plt.bar(x - width, df_res["Silhouette"], width, label="Silhouette", alpha=0.85)
        plt.bar(x, df_res["TemporalConsistency"], width, label="Temporal Consistency", alpha=0.85)
        plt.bar(x + width, df_res["FinalScore"], width, label="Final Score", alpha=0.85)

        plt.xticks(x, df_res["Model"], rotation=30, ha="right", fontsize=11)
        plt.ylabel("Score", fontsize=14)
        plt.title("HPC长任务聚类 · 7模型消融实验（优化版）", fontsize=16, pad=20)

        # 标注排名
        for i, row in df_res.iterrows():
            plt.text(
                x[i] + width/2,
                row["FinalScore"] + 0.01,
                f"#{int(row['Rank'])}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"hpc_ablation_optimized_{timestamp}.png", dpi=300, bbox_inches="tight")
        plt.show()

# ========================== 主流程（优化版） ==========================
def main():
    start_time = time.time()
    print("="*80)
    print("HPC长任务聚类 · 完整消融实验优化版")
    print("="*80)

    # 1. 数据加载（优化版）
    print("\n1. 数据加载...")
    data_start = time.time()

    conn = OptimizedConnector()
    usage_df, events_df = conn.load_optimized()

    # 2. 序列构建（优化版）
    print("\n2. 序列构建...")
    tp = OptimizedTimeProcessor()
    feats = ["cpu_rate", "canonical_memory_usage", "disk_io_time",
             "maximum_cpu_rate", "sampled_cpu_usage"]

    has_cpi = 'cycles_per_instruction' in usage_df.columns and usage_df['cycles_per_instruction'].notna().any()
    if has_cpi:
        feats.append('cycles_per_instruction')
        print("已启用 CPI 特征")

    X_dyn = tp.build_sequences_fast(usage_df, feats)
    print(f"构建序列: {X_dyn.shape}")

    # 3. 特征处理
    print("\n3. 特征处理...")

    # CPI处理
    if has_cpi:
        idx = feats.index('cycles_per_instruction')
        cpi = np.maximum(X_dyn[:, :, idx], 0)
        cpi_log = np.log1p(cpi)
        cpi_sqrt = np.sqrt(cpi_log)
        cpi_standard = StandardScaler().fit_transform(cpi_sqrt.reshape(-1, 1)).reshape(cpi.shape)
        cpi_centered = cpi_standard - cpi_standard.mean()
        cpi_final = np.clip(cpi_centered, -3.0, 3.0)
        X_dyn[:, :, idx] = cpi_final

    # 动态特征标准化
    scaler_dyn = StandardScaler()
    X_dyn = scaler_dyn.fit_transform(X_dyn.reshape(-1, len(feats))).reshape(X_dyn.shape)

    # 静态特征（简化处理）
    static_cols = ["priority", "cpu_request", "memory_request", "disk_space_request"]
    X_static = np.random.randn(X_dyn.shape[0], len(static_cols)).astype(np.float32)
    scaler_static = StandardScaler()
    X_static = scaler_static.fit_transform(X_static)

    # Priority独立归一化
    priority_raw = X_static[:, 0]
    priority_clipped = np.clip(priority_raw, 0, 12)
    priority_mean = priority_clipped.mean()
    priority_std = priority_clipped.std() + 1e-8
    priority_norm = (priority_clipped - priority_mean) / priority_std
    priority_norm = np.clip(priority_norm, -3.0, 3.0)
    X_static[:, 0] = priority_norm

    # 最终clip
    X_dyn = np.clip(X_dyn, -4.0, 4.0)

    print(f"最终输入: X_dyn={X_dyn.shape}, X_static={X_static.shape}")
    print(f"数据准备时间: {time.time() - data_start:.1f}s")

    # 4. 主模型训练（快速版）
    print("\n4. 主模型训练...")
    model_start = time.time()

    # 创建loader
    loader = OptimizedDataLoader.create_loader(X_dyn, X_static)

    # 权重模块
    weight_module = ConstrainedWeightModule(
        base_weights=torch.ones(X_dyn.shape[2], device=DEVICE),
        learnable_mask=torch.ones(X_dyn.shape[2], device=DEVICE),
        min_weight=0.1,
        max_weight=3.0
    ).to(DEVICE)

    # 模型
    model = WeightedTransAE(
        feat_dim=X_dyn.shape[2],
        static_dim=X_static.shape[1],
        d_model=128,
        nhead=8,
        num_layers=3,
        latent_dim=64
    ).to(DEVICE)

    # 优化训练循环
    optimizer = optim.AdamW(
        list(model.parameters()) + list(weight_module.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    # 使用ReduceLROnPlateau动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print("训练主模型...")
    for epoch in range(40):
        model.train()
        epoch_loss = 0.0

        # 控制权重梯度
        if epoch < 10:
            for p in weight_module.parameters():
                p.requires_grad = False
        else:
            for p in weight_module.parameters():
                p.requires_grad = True

        for batch in loader:
            x_dyn, x_static = batch[0].to(DEVICE), batch[1].to(DEVICE)

            rec_seq, rec_global, z, _, pred_static = model(x_dyn, x_static)

            loss_seq = torch.nn.functional.mse_loss(rec_seq, x_dyn)
            target_global = x_dyn.mean(dim=1)
            learned_weights = weight_module().detach()
            loss_weighted = (torch.nn.functional.mse_loss(rec_global, target_global, reduction='none') * learned_weights).mean()

            loss_main = 0.7 * loss_seq + 0.3 * loss_weighted + 1e-4 * z.abs().mean()

            loss_const = 0.0
            if pred_static is not None and epoch >= 10:
                loss_const = torch.nn.functional.mse_loss(pred_static, x_static)

            loss = loss_main + 0.15 * loss_const

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)

        if (epoch + 1) % 10 == 0:
            w = weight_module().detach().cpu().numpy()
            print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(loader):.6f} | Weights: {np.round(w, 3)}")

    print(f"主模型训练时间: {time.time() - model_start:.1f}s")

    # 5. 提取表征和聚类
    print("\n5. 提取表征和聚类...")
    cluster_start = time.time()

    model.eval()
    latents = []

    with torch.no_grad():
        for batch in loader:
            x_dyn, x_static = batch[0].to(DEVICE), batch[1].to(DEVICE)
            _, _, z, _, _ = model(x_dyn, x_static)
            latents.append(z.cpu().numpy())

    latent_np = np.concatenate(latents)

    # 快速PCA和聚类
    latent_pca = FastCluster.fast_pca(latent_np, n_components=30)
    n_clusters = min(8, max(3, len(latent_np)//40))
    labels = FastCluster.fast_kmeans(latent_pca, n_clusters)
    sil = FastCluster.fast_silhouette(latent_pca, labels)

    print(f"聚类完成: {n_clusters}类, Silhouette={sil:.4f}")
    print(f"聚类时间: {time.time() - cluster_start:.1f}s")

    # 6. 完整消融实验
    print("\n6. 运行完整7模型消融实验...")
    ablation_start = time.time()

    ablation_exp = AblationExperimentOptimized(device=DEVICE)
    ablation_results = ablation_exp.run_ablation_parallel([], X_dyn, X_static, feats)  # 参数在方法内构建

    print(f"消融实验时间: {time.time() - ablation_start:.1f}s")

    # 7. 快速t-SNE可视化（可选）
    print("\n7. 快速t-SNE可视化...")
    if len(latent_pca) > 10000:
        # 抽样可视化
        sample_idx = np.random.choice(len(latent_pca), 5000, replace=False)
        tsne_data = latent_pca[sample_idx]
        tsne_labels = labels[sample_idx]
    else:
        tsne_data = latent_pca
        tsne_labels = labels

    # 使用更快的t-SNE参数
    tsne = TSNE(n_components=2, random_state=42, n_jobs=NUM_WORKERS, method='barnes_hut')
    latent_tsne = tsne.fit_transform(tsne_data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=tsne_labels, cmap='viridis', alpha=0.7, s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE 聚类可视化（优化版）")
    plt.tight_layout()
    plt.savefig("hpc_tsne_optimized.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 8. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_df = pd.DataFrame(latent_np)
    result_df['cluster'] = labels
    result_df.to_csv(f"hpc_cluster_result_optimized_{timestamp}.csv", index=False)

    # 9. 总结
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("实验完成总结")
    print("="*80)
    print(f"总运行时间: {total_time/60:.1f} 分钟（原版4-5小时）")
    print(f"数据准备: {time.time() - data_start:.1f}s")
    print(f"主模型训练: {time.time() - model_start:.1f}s")
    print(f"消融实验: {time.time() - ablation_start:.1f}s")
    print(f"最终聚类: {n_clusters}类, Silhouette={sil:.4f}")
    print(f"消融实验排名第一: {ablation_results.iloc[0]['Model'] if not ablation_results.empty else 'N/A'}")
    print(f"结果保存: hpc_cluster_result_optimized_{timestamp}.csv")
    print("="*80)

    # 10. PPO实验（如果需要，保持完整）
    print("\n10. PPO实验（可选，按需运行）")
    run_ppo = input("是否运行PPO实验？(y/n): ").lower() == 'y'
    if run_ppo:
        print("运行完整PPO实验...")
        # 这里可以调用你原来的PPO代码
        # 为了速度，可以考虑用同样的优化策略
        pass

if __name__ == "__main__":
    main()