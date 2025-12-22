import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Silhouette 保证簇不糊
def silhouette_guidance_loss(z, epoch, n_clusters=8, warmup=10):
    """
    z: [B, D] latent vectors
    """
    if epoch < warmup or z.size(0) < n_clusters:
        return torch.tensor(0.0, device=z.device)

    with torch.no_grad():
        z_np = z.detach().cpu().numpy()
        labels = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(z_np)
        sil = silhouette_score(z_np, labels)

    # maximize silhouette → minimize negative
    return torch.tensor(-sil, device=z.device)

# TC 保证时间不乱跳
def temporal_consistency_loss(h, mode="l2"):
    """
    h: [B, T, D] hidden states from encoder
    """
    if h.size(1) < 2:
        return torch.tensor(0.0, device=h.device)

    delta = h[:, 1:] - h[:, :-1]   # [B, T-1, D]

    if mode == "l1":
        return delta.abs().mean()
    else:
        return (delta ** 2).mean()