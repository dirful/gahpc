#!/usr/bin/env python3
"""
plots.py - 可视化函数
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

from utils.tools import ensure_outdir

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    seasonal_decompose = None

try:
    import plotly.express as px
except ImportError:
    px = None

def create_violin_plots_cluster_comparison(
        feature_matrix: pd.DataFrame,
        cluster_results: pd.DataFrame,
        emb_df: pd.DataFrame,
        outdir: str
):
    """创建聚类小提琴图"""
    print("Creating cluster violin plots...")
    ensure_outdir(outdir)

    # 合并数据
    merged = emb_df.copy()

    # 确保 emb_df 和 cluster_results 都有 job_id 列
    if 'job_id' in merged.columns:
        merged = merged.set_index('job_id')

    if 'job_id' in cluster_results.columns:
        # 设置 cluster_results 的索引
        cluster_results_indexed = cluster_results.set_index('job_id')

        # 重命名 cluster 列以避免冲突
        if 'cluster' in merged.columns and 'cluster' in cluster_results_indexed.columns:
            cluster_results_indexed = cluster_results_indexed.rename(
                columns={'cluster': 'cluster_from_results'}
            )

        # 合并数据，不指定后缀（已经重命名了列）
        merged = merged.join(cluster_results_indexed, how='left')

        # 如果重命名了列，使用新的列名
        if 'cluster_from_results' in merged.columns:
            merged['cluster'] = merged['cluster_from_results']
            merged = merged.drop(columns=['cluster_from_results'])
    elif 'cluster' in cluster_results.columns and 'cluster' not in merged.columns:
        # 如果 cluster_results 没有 job_id 但有 cluster 列，且 merged 没有 cluster 列
        merged['cluster'] = cluster_results['cluster'].values

    # 提取潜在维度列
    latent_cols = [c for c in merged.columns if str(c).startswith('latent_')]
    if len(latent_cols) == 0:
        print("No latent columns found for violin plots")
        return

    # 确保 cluster 列存在
    if 'cluster' not in merged.columns:
        print("No cluster column found for violin plots")
        return

    # 重塑数据用于绘图
    melt = merged.reset_index().melt(
        id_vars=['job_id', 'cluster'],
        value_vars=latent_cols,
        var_name='latent_dim',
        value_name='value'
    )

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=melt, x='latent_dim', y='value',
                   hue='cluster', split=False, inner='quartile')
    plt.title('Latent Embedding Distribution by Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cluster_violin_embeddings.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

def create_time_series_decomposition(
        window_agg: pd.DataFrame,
        recon_map: Dict,
        outdir: str,
        n_jobs: int = 6
):
    """创建时间序列分解图"""
    if seasonal_decompose is None:
        print("statsmodels not installed; skipping decomposition")
        return

    print("Creating time series decomposition comparisons...")

    # 选择要分析的作业
    job_counts = window_agg['job_id'].value_counts()
    sample_jobs = job_counts.head(n_jobs).index.tolist()

    # 创建子图
    fig, axes = plt.subplots(len(sample_jobs), 4,
                             figsize=(18, 4 * len(sample_jobs)))

    for i, job in enumerate(sample_jobs):
        job_data = window_agg[window_agg['job_id'] == job].sort_values('window_index')
        if job_data.shape[0] < 6:
            continue

        # 获取真实和重建序列
        real_series = job_data['cpu_rate_mean'].values if 'cpu_rate_mean' in job_data.columns else job_data.iloc[:, -1].values
        recon_series = recon_map.get(job)

        if recon_series is None:
            continue

        if recon_series.ndim == 2:
            recon_cpu = recon_series[:, 0]
        else:
            recon_cpu = recon_series

        # 尝试时间序列分解
        period = max(2, min(20, len(real_series) // 2))
        try:
            res_real = seasonal_decompose(real_series, model='additive',
                                          period=period, two_sided=True)
            res_recon = seasonal_decompose(recon_cpu, model='additive',
                                           period=period, two_sided=True)
        except Exception:
            res_real = None
            res_recon = None

        # 绘制原始vs重建
        ax = axes[i, 0]
        ax.plot(real_series, label='real')
        ax.plot(recon_cpu, label='recon', alpha=0.8)
        ax.set_title(f'Job {job} - Original vs Reconstructed')
        ax.legend()

        # 绘制趋势对比
        ax = axes[i, 1]
        if res_real is not None and res_recon is not None:
            ax.plot(res_real.trend, label='real_trend')
            ax.plot(res_recon.trend, label='recon_trend', alpha=0.8)
        else:
            ax.plot(pd.Series(real_series).rolling(3, min_periods=1).mean(),
                    label='real_trend')
            ax.plot(pd.Series(recon_cpu).rolling(3, min_periods=1).mean(),
                    label='recon_trend')
        ax.set_title('Trend Comparison')
        ax.legend()

        # 绘制残差
        ax = axes[i, 2]
        if res_real is not None and res_recon is not None:
            rr = res_real.resid
            rrr = res_recon.resid
        else:
            rr = real_series - pd.Series(real_series).rolling(3, min_periods=1).mean()
            rrr = recon_cpu - pd.Series(recon_cpu).rolling(3, min_periods=1).mean()
        ax.plot(rr, label='real_resid')
        ax.plot(rrr, label='recon_resid', alpha=0.8)
        ax.set_title('Residuals')
        ax.legend()

        # 绘制残差差异
        ax = axes[i, 3]
        diff = (rr - rrr)
        ax.plot(diff, label='resid_diff')
        ax.set_title('Residual Difference (phase change)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ts_decomp_recon_compare.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

def create_temporal_pattern_analysis(
        window_agg: pd.DataFrame,
        emb_time_series: Dict,
        outdir: str
):
    """创建时间模式分析图"""
    print("Creating temporal pattern analysis using embeddings...")

    # 聚合时间模式
    lat_dims = None
    aggregated = {}
    counts = {}

    for job, arr in emb_time_series.items():
        T = arr.shape[0]
        lat_dims = arr.shape[1]
        for t in range(T):
            aggregated.setdefault(t, np.zeros(lat_dims))
            counts.setdefault(t, 0)
            aggregated[t] += arr[t]
            counts[t] += 1

    # 创建数据框
    rows = []
    for t in sorted(aggregated.keys()):
        mean_vec = aggregated[t] / max(1, counts[t])
        row = {'window_index': t}
        for d in range(lat_dims):
            row[f'latent_{d}'] = mean_vec[d]
        rows.append(row)

    df = pd.DataFrame(rows)
    ks = [c for c in df.columns if c.startswith('latent_')][:3]

    # 创建子图
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for i, k in enumerate(ks):
        # 时间模式图
        ax = axes[i, 0]
        ax.plot(df['window_index'], df[k])
        ax.set_title(f'Temporal pattern of {k}')
        ax.grid(True, alpha=0.3)

        # 核密度估计图
        ax = axes[i, 1]
        sns.kdeplot(df[k], fill=True, ax=ax)
        ax.set_title(f'KDE of {k}')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'temporal_pattern_embeddings.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

def create_interactive_umap(emb_df: pd.DataFrame, outdir: str):
    """创建交互式UMAP图（如果plotly可用）"""
    if px is None:
        print('Plotly not available; skipping interactive UMAP')
        return

    try:
        fig = px.scatter(emb_df, x='umap1', y='umap2',
                         color='cluster', hover_data=['job_id'])
        fig.write_html(os.path.join(outdir, 'interactive_umap_clusters.html'))
    except Exception as e:
        print('Plotly interactive UMAP failed:', e)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_business_weights(feature_names, weights, save_path="feature_importance.png"):
    """
    绘制业务感知权重柱状图
    :param feature_names: 特征名称列表 (feats)
    :param weights: 学习到的权重数组 (numpy array)
    """
    # 设置学术风格
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文（如果需要）
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 自动对特征进行分类（业务感知颜色映射）
    colors = []
    for name in feature_names:
        name = name.lower()
        if 'cpu' in name:
            colors.append('#3498db')  # 蓝色 - 计算资源
        elif 'mem' in name:
            colors.append('#2ecc71')  # 绿色 - 内存资源
        elif 'disk' in name or 'io' in name:
            colors.append('#e67e22')  # 橙色 - 存储资源
        elif 'cycles' in name or 'cpi' in name:
            colors.append('#e74c3c')  # 红色 - 指令效率
        else:
            colors.append('#95a5a6')  # 灰色 - 其他

    # 2. 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(feature_names))

    # 绘制水平柱状图
    # 确保 weights 是 CPU 上的 Numpy 数组
    import torch
    if torch.is_tensor(weights):
        weights = weights.detach().cpu().numpy()

    bars = ax.barh(y_pos, weights, color=colors, edgecolor='black', alpha=0.8)

    # 添加基准线 (Weight = 1.0)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='初始权重基准 (1.0)')

    # 3. 细节美化
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=12)
    ax.invert_yaxis()  # 让排名靠前的在上面
    ax.set_xlabel('学习到的特征权重 (Importance Score)', fontsize=14)
    ax.set_title('基于可学习权重模块的任务特征重要性校准', fontsize=16, pad=20)

    # 在柱状图末尾添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', fontsize=11, fontweight='bold')

    # 添加图例说明业务含义
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498db', lw=4, label='计算密集型特征'),
        Line2D([0], [0], color='#2ecc71', lw=4, label='内存负载特征'),
        Line2D([0], [0], color='#e67e22', lw=4, label='存储/IO特征'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='微架构效率特征'),
        Line2D([0], [0], color='red', lw=1.5, ls='--', label='初始基准线')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)

    plt.tight_layout()
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.show()

# 使用示例（在你训练完 weight_module 后调用）
# final_weights = weight_module().detach().cpu().numpy()
# plot_business_weights(feats, final_weights)