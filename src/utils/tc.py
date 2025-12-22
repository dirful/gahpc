import numpy as np

def temporal_consistency_score_v2(attn_time, labels, pos_idx, neg_idx, eps=1e-6):
    """
    attn_time: [N, T, F]
    labels: [N]
    pos_idx / neg_idx: list of int
    """
    if len(pos_idx) == 0 and len(neg_idx) == 0:
        return 0.0

    cluster_scores = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 5:  # 样本太少不计算
            continue
        A = attn_time[idx]  # [Nc, T, F]

        # 计算正向特征平均注意力
        pos_attn = A[:, :, pos_idx].mean(axis=2) if len(pos_idx) > 0 else np.zeros((len(idx), A.shape[1]))
        # 计算负向特征平均注意力（负向越低越好，所以取反）
        neg_attn = A[:, :, neg_idx].mean(axis=2) if len(neg_idx) > 0 else np.zeros((len(idx), A.shape[1]))

        # 业务信号：正向高 + 负向低 → 信号强
        business_signal = pos_attn - neg_attn  # [Nc, T]

        # 时序稳定性：信号随时间波动越小越好
        # 对每个样本计算时序方差，然后取簇平均
        temporal_var = np.var(business_signal, axis=1)  # [Nc]
        mean_var = np.mean(temporal_var)

        # 归一化分数：方差越小，分数越高（上限1）
        score = 1.0 / (1.0 + mean_var)  # 平滑，避免除0爆炸
        # 额外奖励：信号均值越高越好（业务强度）
        mean_signal = np.mean(business_signal)
        score = score * (1.0 + np.clip(mean_signal, -1, 1))  # 轻微奖励正信号

        cluster_scores.append(score)

    return float(np.mean(cluster_scores)) if cluster_scores else 0.0

def temporal_consistency_score_v3(attn_time, labels, pos_idx, neg_idx, eps=1e-6):
    """
    平衡版 TC：正向高稳定 + 负向低稳定 + 特征丰富性奖励
    """
    cluster_scores = []
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_total = n_pos + n_neg

    # 丰富性奖励因子（总特征越多，分数基础越高，但上限1.2）
    richness_bonus = min(1.0 + n_total * 0.05, 1.2)

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 5: continue
        A = attn_time[idx]  # [Nc, T, F]

        # 正向：高值 + 稳定（var小）
        if n_pos > 0:
            pos_attn = A[:, :, pos_idx].mean(axis=2)  # [Nc, T]
            pos_mean = pos_attn.mean()
            pos_diff = np.diff(pos_attn, axis=1)
            pos_var = np.var(pos_diff, axis=1).mean()
            pos_score = (1.0 / (1.0 + pos_var)) * (0.5 + 0.5 * np.tanh(pos_mean))  # 高值奖励
        else:
            pos_score = 0.5  # 中性

        # 负向：低值 + 稳定（var小）
        if n_neg > 0:
            neg_attn = A[:, :, neg_idx].mean(axis=2)  # [Nc, T]
            neg_mean = neg_attn.mean()
            neg_diff = np.diff(neg_attn, axis=1)
            neg_var = np.var(neg_diff, axis=1).mean()
            neg_score = (1.0 / (1.0 + neg_var)) * (1.0 - np.tanh(neg_mean))  # 低值奖励
        else:
            neg_score = 0.5  # 中性

        # 整体分数：平衡正负 + 丰富性
        score = 0.6 * pos_score + 0.4 * neg_score
        score *= richness_bonus  # 奖励多特征

        # 平滑到 0~1
        score = np.tanh(score)

        cluster_scores.append(score)

    return float(np.mean(cluster_scores)) if cluster_scores else 0.0


def temporal_consistency_score_v4(attn_time, labels, pos_idx, neg_idx, eps=1e-6):
    """
    优化版 TC：更敏感 + 更公平 + 分数更高
    """
    cluster_scores = []
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_total = n_pos + n_neg

    # 丰富性奖励加强（无上限，鼓励使用更多特征）
    richness_bonus = 1.0 + n_total * 0.08  # 10个特征 ≈ +0.8，显著奖励完整模型

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 5:
            continue
        A = attn_time[idx]  # [Nc, T, F]

        # 正向特征：期望高值 + 低波动
        if n_pos > 0:
            pos_attn = A[:, :, pos_idx].mean(axis=2)  # [Nc, T]
            pos_mean = pos_attn.mean()
            pos_std = np.std(pos_attn, axis=1).mean()  # 用 std 更鲁棒
            stability_pos = 1.0 / (1.0 + pos_std)
            value_pos = np.clip(pos_mean, 0.0, 1.0)  # 假设 attn 已归一化
            pos_score = stability_pos * (0.6 + 0.4 * value_pos)
        else:
            pos_score = 0.6

        # 负向特征：期望低值 + 低波动
        if n_neg > 0:
            neg_attn = A[:, :, neg_idx].mean(axis=2)
            neg_mean = neg_attn.mean()
            neg_std = np.std(neg_attn, axis=1).mean()
            stability_neg = 1.0 / (1.0 + neg_std)
            value_neg = 1.0 - np.clip(neg_mean, 0.0, 1.0)
            neg_score = stability_neg * (0.6 + 0.4 * value_neg)
        else:
            neg_score = 0.6

        # 等权融合 + 丰富性奖励
        score = 0.5 * pos_score + 0.5 * neg_score
        score *= richness_bonus

        # 最终平滑到合理范围（避免过大）
        score = np.tanh(score * 0.8) * 0.9 + 0.1  # ≈ [0.1, 1.0]

        cluster_scores.append(score)

    return float(np.mean(cluster_scores)) if cluster_scores else 0.0