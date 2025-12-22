import optuna
from optuna.trial import TrialState
from sympy.printing.pytorch import torch

from transformer4 import DEVICE


def objective(trial):
    """贝叶斯优化目标函数"""
    # 为每个特征定义搜索空间
    weights = []

    # 动态特征权重（范围：0.1-4.0）
    weights.append(trial.suggest_float('cpu_rate_weight', 0.1, 4.0))
    weights.append(trial.suggest_float('mem_usage_weight', 0.1, 4.0))
    weights.append(trial.suggest_float('disk_io_weight', 0.05, 1.0))  # 负向特征，权重较低
    weights.append(trial.suggest_float('max_cpu_weight', 0.1, 3.0))
    weights.append(trial.suggest_float('sampled_cpu_weight', 0.1, 3.0))

    if has_cpi:
        weights.append(trial.suggest_float('cpi_weight', 0.05, 1.0))

    # 静态特征权重（范围：0.05-2.0，比动态特征范围小）
    weights.append(trial.suggest_float('priority_weight', 0.05, 2.0))
    weights.append(trial.suggest_float('cpu_req_weight', 0.05, 1.5))
    weights.append(trial.suggest_float('mem_req_weight', 0.05, 1.5))
    weights.append(trial.suggest_float('disk_req_weight', 0.01, 0.5))

    # 确保维度正确
    weights_tensor = torch.tensor(weights[:X_np.shape[2]], device=DEVICE, dtype=torch.float32)

    # 快速训练评估
    model = WeightedTransAE(feat_dim=X_np.shape[2], latent_dim=32).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 快速训练（节省时间）
    model.train()
    for epoch in range(10):  # 少量epoch
        for batch in loader:
            x = batch[0].to(DEVICE)
            rec_seq, rec_global, z, _ = model(x)

            target_global = x.mean(dim=1)
            loss_seq = criterion_mse(rec_seq, x).mean()
            loss_weighted = (criterion_mse(rec_global, target_global) * weights_tensor).mean()
            loss = 0.65 * loss_seq + 0.35 * loss_weighted

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 提取表征并聚类
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in loader:
            _, _, z, _ = model(batch[0].to(DEVICE))
            latents.append(z.cpu().numpy())

    latent_np = np.concatenate(latents)

    # PCA降维
    latent_pca = PCA(n_components=min(30, latent_np.shape[1])).fit_transform(latent_np)

    # 尝试多个聚类数，取最佳Silhouette
    best_sil = -1
    for n_clusters in [3, 4, 5, 6, 7, 8]:
        if len(latent_pca) >= n_clusters * 10:  # 确保足够样本
            labels = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit_predict(latent_pca)
            sil = silhouette_score(latent_pca, labels)
            if sil > best_sil:
                best_sil = sil

    # 添加正则化项：鼓励权重集中（避免极端值）
    weight_std = torch.std(weights_tensor).item()
    weight_penalty = 0.1 * weight_std  # 惩罚权重差异过大

    # 最终目标：最大化Silhouette，最小化权重差异
    return best_sil - weight_penalty

# 运行贝叶斯优化
def optimize_weights_bayesian(n_trials=50):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    print("开始贝叶斯优化权重...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 输出最佳结果
    print(f"\n最佳试验 {study.best_trial.number}:")
    print(f"  最佳Silhouette: {study.best_value:.4f}")
    print(f"  最佳权重:")

    best_weights = []
    for param, value in study.best_trial.params.items():
        print(f"    {param}: {value:.3f}")
        best_weights.append(value)

    # 生成权重张量
    best_weights_tensor = torch.tensor(best_weights[:X_np.shape[2]],
                                       device=DEVICE, dtype=torch.float32)

    # 可视化优化历史
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    # 参数重要性
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()

    return best_weights_tensor, study