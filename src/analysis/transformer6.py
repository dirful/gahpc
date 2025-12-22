import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================
# 全局配置
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_SEQ_LEN = 10
EPOCHS = 25
BATCH_SIZE = 256
LATENT_DIM = 32
N_CLUSTERS = 8

# =========================
# 工具函数
# =========================
def align_weights(weights, feat_dim):
    if weights is None:
        return None
    if weights.numel() > feat_dim:
        return weights[:feat_dim]
    elif weights.numel() < feat_dim:
        pad = torch.ones(feat_dim - weights.numel(), device=weights.device)
        return torch.cat([weights, pad])
    return weights


def temporal_consistency_score(attn_time, labels):
    scores = []
    T = attn_time.shape[1]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        peak_times = np.argmax(attn_time[idx], axis=1)
        var = np.var(peak_times)
        score = 1.0 - var / (T + 1e-6)
        scores.append(score)
    return float(np.mean(scores)) if scores else np.nan


# =========================
# 模型定义
# =========================
class WeightedTransAE(nn.Module):
    def __init__(self, feat_dim, latent_dim=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.to_latent = nn.Linear(128, latent_dim)
        self.decoder = nn.Linear(latent_dim, feat_dim)

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        z = self.to_latent(h.mean(1))
        rec_global = self.decoder(z)
        rec_seq = rec_global.unsqueeze(1).repeat(1, x.size(1), 1)
        return rec_seq, rec_global, z, None


# =========================
# 消融实验主函数
# =========================
def run_ablation_variant(
        name,
        main_loader=None,
        custom_loader=None,
        weights=None,
        model_class=WeightedTransAE,
        n_c=N_CLUSTERS
):
    print(f"\n>>> Training: {name}")
    torch.cuda.empty_cache()

    loader = custom_loader if custom_loader is not None else main_loader
    if loader is None:
        raise ValueError("必须提供 DataLoader")

    # === 确定特征维度 ===
    x0 = next(iter(loader))[0].to(DEVICE)
    feat_dim = x0.shape[2]

    # === 构建模型 ===
    if model_class == WeightedTransAE:
        model = WeightedTransAE(feat_dim, LATENT_DIM).to(DEVICE)
    elif model_class == "LSTM":
        class LSTMAE(nn.Module):
            def __init__(self, f):
                super().__init__()
                self.enc = nn.LSTM(f, 128, 2, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(256, LATENT_DIM)
                self.dec = nn.Linear(LATENT_DIM, f)

            def forward(self, x):
                _, (h, _) = self.enc(x)
                z = self.fc(h[-2:].transpose(0,1).reshape(x.size(0), -1))
                rec_g = self.dec(z)
                rec_s = rec_g.unsqueeze(1).repeat(1, x.size(1), 1)
                return rec_s, rec_g, z, None

        model = LSTMAE(feat_dim).to(DEVICE)

    elif model_class == "MLP":
        class MLPAE(nn.Module):
            def __init__(self, f):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(f * MIN_SEQ_LEN, 512),
                    nn.GELU(),
                    nn.Linear(512, LATENT_DIM)
                )
                self.dec = nn.Linear(LATENT_DIM, f * MIN_SEQ_LEN)

            def forward(self, x):
                flat = x.reshape(x.size(0), -1)
                z = self.enc(flat)
                rec = self.dec(z).reshape(x.size(0), MIN_SEQ_LEN, -1)
                return rec, rec.mean(1), z, None

        model = MLPAE(feat_dim).to(DEVICE)

    # === 训练 ===
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.MSELoss(reduction="none")
    alpha = 0.65

    model.train()
    for _ in range(EPOCHS):
        for (x,) in loader:
            x = x.to(DEVICE)
            rec_s, rec_g, z, _ = model(x)

            loss_seq = crit(rec_s, x).mean()
            w_use = align_weights(weights, x.size(2))

            if w_use is not None:
                loss_w = (crit(rec_g, x.mean(1)) * w_use).mean()
                loss = alpha * loss_seq + (1 - alpha) * loss_w
            else:
                loss = loss_seq

            loss += 1e-4 * z.abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    # === 表征提取 ===
    model.eval()
    zs = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            _, _, z, _ = model(x)
            zs.append(z.cpu().numpy())

    latent = np.concatenate(zs)
    pca = PCA(n_components=min(30, latent.shape[1]), random_state=42)
    emb = pca.fit_transform(latent)
    labs = KMeans(n_clusters=n_c, n_init=25, random_state=42).fit_predict(emb)
    sil = silhouette_score(emb, labs)

    # === Temporal Consistency（仅 Transformer）===
    if isinstance(model, WeightedTransAE):
        attn_time_all = []

        with torch.no_grad():
            for (x,) in loader:
                x = x.to(DEVICE)
                h = model.proj(x)
                attn_roll = None

                for layer in model.transformer.layers:
                    attn_out, attn = layer.self_attn(
                        h, h, h,
                        need_weights=True,
                        average_attn_weights=False
                    )
                    attn = attn.mean(1)
                    attn = attn / attn.sum(dim=-1, keepdim=True)

                    attn_roll = attn if attn_roll is None else torch.matmul(attn, attn_roll)
                    h = h + attn_out

                time_attn = attn_roll.mean(dim=2)
                attn_time_all.append(time_attn.cpu().numpy())

        attn_time_all = np.concatenate(attn_time_all)
        tc = temporal_consistency_score(attn_time_all, labs)
    else:
        tc = np.nan

    print(f"  → Silhouette={sil:.4f} | TC={tc:.4f}")
    return {
        "Model": name,
        "Silhouette": sil,
        "TemporalConsistency": tc,
        "Clusters": n_c,
        "Samples": len(latent)
    }


# =========================
# 主程序入口
# =========================
def main():
    # ===== 示例数据 =====
    N = 40000
    T = MIN_SEQ_LEN
    F = 256

    X = torch.randn(N, T, F)
    loader = data.DataLoader(
        data.TensorDataset(X),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    global WEIGHTS
    WEIGHTS = torch.rand(F).to(DEVICE)

    results = []

    results.append(run_ablation_variant("Ours (Full)", loader, weights=WEIGHTS))
    results.append(run_ablation_variant("No-Weighted", loader, weights=None))
    results.append(run_ablation_variant("LSTM-AE", loader, model_class="LSTM"))
    results.append(run_ablation_variant("MLP-AE", loader, model_class="MLP"))

    df = pd.DataFrame(results).sort_values("Silhouette", ascending=False)
    print("\n==== 消融实验结果 ====")
    print(df)


if __name__ == "__main__":
    main()
