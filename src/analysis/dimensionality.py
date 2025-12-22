#!/usr/bin/env python3
"""
dimensionality.py - 降维分析
"""
import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    umap = None

def run_umap(embeddings: np.ndarray, n_components: int = 2,
             random_state: int = 42) -> np.ndarray:
    """运行UMAP降维"""
    if umap is None:
        raise RuntimeError("UMAP is required (pip install umap-learn)")

    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(embeddings)

def perform_pca_analysis(
        feature_matrix: pd.DataFrame,
        outdir: str,
        n_components: int = 6
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """执行PCA分析"""
    print("Performing PCA analysis...")

    X = feature_matrix.select_dtypes(include=[np.number]).fillna(0).values
    if X.shape[0] < 2:
        print("Not enough rows for PCA")
        return None, None, None

    # 标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    comps = pca.fit_transform(Xs)

    # 绘制解释方差图
    explained = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA Scree (cumulative)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, 'pca_scree.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 保存载荷和主成分
    loadings = pd.DataFrame(pca.components_,
                            columns=feature_matrix.select_dtypes(include=[np.number]).columns)
    loadings.to_csv(os.path.join(outdir, 'pca_loadings.csv'))

    comps_df = pd.DataFrame(comps, index=feature_matrix.index,
                            columns=[f'PC{i+1}' for i in range(comps.shape[1])])
    comps_df.to_csv(os.path.join(outdir, 'pca_components.csv'))

    return pca, comps_df, loadings