#!/usr/bin/env python3
"""
main.py - 主入口文件
"""
import os

import numpy as np
import pandas as pd
import torch
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 导入自定义模块
from utils.tools import ensure_outdir, set_seed
from config.config import parse_args, get_features_list, setup_device
from data.database import (
    load_task_usage_200k,
    prepare_window_agg_from_task_usage,
    load_window_agg
)
from data.preprocessor import (
    build_per_job_sequences,
    normalize_sequences,
    create_feature_matrix,
    sample_sequences
)
from model.autoencoders import LSTMAutoencoder, GRUAutoencoder
from training.trainer import (
    train_autoencoder,
    extract_embeddings,
    create_reconstruction_map
)
from analysis.dimensionality import perform_pca_analysis, run_umap
from analysis.clustering import run_hdbscan, create_cluster_visualizations
from analysis.anomaly import add_anomaly_columns
from view.plots import (
    create_violin_plots_cluster_comparison,
    create_time_series_decomposition,
    create_temporal_pattern_analysis,
    create_interactive_umap
)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    ensure_outdir(args.outdir)
    set_seed(42)

    # 设置设备
    device = setup_device(args.device)

    # 1. 加载或构建窗口聚合数据
    if args.window_agg:
        window_agg = load_window_agg(args.window_agg)
    else:
        df_task = load_task_usage_200k(args.db_uri)
        window_agg = prepare_window_agg_from_task_usage(
            df_task,
            window_seconds=args.window_seconds,
            time_unit=args.time_unit
        )
        # 保存生成的数据
        temp_path = os.path.join(args.outdir, 'window_agg_generated.parquet')
        window_agg.to_parquet(temp_path, index=False)
        print(f"Saved generated window_agg to {temp_path}")

    # 2. 构建序列
    features = get_features_list(args.features)
    print('Using features for sequences:', features)

    sequences, job_ids = build_per_job_sequences(
        window_agg,
        features,
        max_len=args.max_seq_len
    )

    # 3. 采样（如果需要）
    sequences, job_ids = sample_sequences(
        sequences,
        job_ids,
        args.sample_jobs
    )
    print(f'Total jobs loaded for AE: {len(sequences)}')

    # 4. 标准化序列
    arr3d, scaler, mask = normalize_sequences(sequences)

    # 5. 创建特征矩阵
    feature_matrix = create_feature_matrix(sequences, job_ids)

    # 6. 创建和训练模型
    input_dim = arr3d.shape[2]
    if args.ae_type == 'lstm':
        model = LSTMAutoencoder(input_dim=input_dim)
    else:
        model = GRUAutoencoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        )

    print('Training autoencoder...')
    model = train_autoencoder(
        model,
        arr3d,
        mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )

    # 7. 提取嵌入
    print('Extracting embeddings and reconstructions...')
    embeddings, reconstructions = extract_embeddings(
        model,
        arr3d,
        device=device,
        batch_size=args.batch_size
    )

    # 8. 创建重建映射
    recon_map = create_reconstruction_map(reconstructions, sequences, job_ids)

    # 9. PCA分析
    pca_model, pca_components, pca_loadings = perform_pca_analysis(
        feature_matrix,
        args.outdir,
        n_components=6
    )

    # 10. 创建嵌入数据框
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f'latent_{i}' for i in range(embeddings.shape[1])]
    )
    emb_df['job_id'] = job_ids

    # 11. 异常检测
    emb_df = add_anomaly_columns(emb_df, embeddings)

    # 12. UMAP降维
    try:
        umap2d = run_umap(embeddings, n_components=args.umap_components)
    except Exception as e:
        print(f"UMAP failed: {e}; falling back to PCA for 2D")
        pca2 = PCA(n_components=2)
        umap2d = pca2.fit_transform(embeddings)

    emb_df['umap1'] = umap2d[:, 0]
    emb_df['umap2'] = umap2d[:, 1]

    # 13. 聚类
    try:
        labels = run_hdbscan(umap2d, min_cluster_size=args.min_cluster_size)
    except Exception as e:
        print(f"HDBSCAN failed: {e}; falling back to KMeans")
        km = KMeans(n_clusters=min(args.n_clusters, len(embeddings)), random_state=42)
        labels = km.fit_predict(embeddings)

    emb_df['cluster'] = labels
    emb_df.to_csv(os.path.join(args.outdir, 'embeddings_clusters.csv'), index=False)

    # 14. 聚类可视化
    create_cluster_visualizations(emb_df, feature_matrix, args.outdir)

    # 15. 创建聚类结果
    cluster_results = pd.DataFrame({'job_id': job_ids, 'cluster': labels})
    cluster_stats = None

    if 'job_id' in feature_matrix.columns:
        fm = feature_matrix.copy()
        fm['job_id'] = fm.index
        fm = fm.reset_index(drop=True).merge(
            emb_df[['job_id', 'cluster']], on='job_id', how='left'
        )
        cluster_stats = fm.groupby('cluster').agg(['mean', 'std', 'count']).round(4)
        cluster_stats.to_csv(os.path.join(args.outdir, 'cluster_statistics.csv'))

    # 16. 其他可视化
    if emb_df is not None and cluster_results is not None:
        create_violin_plots_cluster_comparison(
            feature_matrix,
            cluster_results,
            emb_df,
            args.outdir
        )

    create_time_series_decomposition(
        window_agg,
        recon_map,
        args.outdir,
        n_jobs=8
    )

    # 17. 时间模式分析
    emb_time_series = {}
    for i, jid in enumerate(job_ids[:500]):
        seq = arr3d[i]
        with torch.no_grad():
            x = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(device)
            _, latent = model(x)
            latent_ts = np.tile(
                latent.cpu().numpy()[0][np.newaxis, :],
                (seq.shape[0], 1)
            )
            emb_time_series[jid] = latent_ts

    create_temporal_pattern_analysis(window_agg, emb_time_series, args.outdir)

    # 18. 交互式可视化
    create_interactive_umap(emb_df, args.outdir)

    # 19. 保存最终结果
    emb_df.to_csv(os.path.join(args.outdir, 'embeddings_final.csv'), index=False)

    print('Pipeline complete. Outputs saved to', args.outdir)

if __name__ == '__main__':
    main()