#!/usr/bin/env python3
"""
clustering.py - 聚类分析
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
    import hdbscan
except ImportError:
    hdbscan = None

def run_hdbscan(emb2d: np.ndarray, min_cluster_size: int = 20) -> np.ndarray:
    """运行HDBSCAN聚类"""
    if hdbscan is None:
        raise RuntimeError("hdbscan is required (pip install hdbscan)")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(emb2d)
    return labels

def create_cluster_visualizations(
        emb_df: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        outdir: str
):
    """创建聚类可视化"""
    # UMAP散点图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_df['umap1'], emb_df['umap2'],
                          c=emb_df['cluster'], cmap='tab10', s=20)
    plt.colorbar(scatter)
    plt.title('UMAP Embedding colored by cluster')
    plt.savefig(os.path.join(outdir, 'umap_clusters.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 聚类数量分布
    plt.figure(figsize=(6, 4))
    vc = pd.Series(emb_df['cluster']).value_counts().sort_index()
    plt.bar(range(len(vc)), vc.values, color=plt.cm.tab10(range(len(vc))))
    plt.title('Job Distribution Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.savefig(os.path.join(outdir, 'cluster_counts.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # CPU使用率按聚类分布
    if 'job_id' in emb_df.columns and 'cpu_mean' in feature_matrix.columns:
        fm = feature_matrix.copy()
        fm['job_id'] = fm.index
        fm = fm.reset_index(drop=True).merge(
            emb_df[['job_id', 'cluster']], on='job_id', how='left'
        )
        cpu_by_cluster = fm.groupby('cluster')['cpu_mean'].mean()
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(cpu_by_cluster)), cpu_by_cluster.values,
                color=plt.cm.tab10(range(len(cpu_by_cluster))))
        plt.title('Average CPU Usage by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('CPU mean')
        plt.savefig(os.path.join(outdir, 'cluster_cpu_mean.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()