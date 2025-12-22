"""
修复版主程序
"""
import argparse
import pickle
import json
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseConnector
from feature_engine import FeatureEngine
from param_optimizer import ParamOptimizer
from clustering import MultiViewClustering
from visualization import VisualizationSystem
from config import EXPERIMENT_CONFIG, CLUSTERING_CONFIG, VISUALIZATION_CONFIG

def main_fixed():
    """修复版主函数"""
    print("="*80)
    print("HPC工作负载分析与分类系统 - 修复版")
    print("="*80)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"hpc_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 创建子目录
    for subdir in ['raw_data', 'features', 'clustering', 'visualizations', 'reports']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 步骤1: 数据加载
    print("\n" + "="*60)
    print("步骤1: 数据加载")
    print("="*60)

    db = DatabaseConnector()
    if not db.connect():
        print("❌ 数据库连接失败!")
        return

    try:
        # 加载数据
        print("加载任务时间序列数据...")
        df = db.load_task_time_series(num_tasks=100, min_samples=10)

        if df is None or len(df) == 0:
            print("尝试加载高频任务...")
            df = db.load_high_frequency_tasks(num_tasks=50, frequency_threshold=20)

        if df is None or len(df) == 0:
            print("❌ 数据加载失败!")
            return

        print(f"✅ 数据加载成功!")
        print(f"  总数据点: {len(df)}")
        print(f"  任务数量: {df.groupby(['job_id', 'task_index']).ngroups}")

        # 保存原始数据
        raw_path = os.path.join(output_dir, 'raw_data', 'task_data.csv')
        df.to_csv(raw_path, index=False)
        print(f"  原始数据已保存: {raw_path}")

    finally:
        db.disconnect()

    # 步骤2: 特征工程
    print("\n" + "="*60)
    print("步骤2: 特征工程")
    print("="*60)

    feature_engine = FeatureEngine()

    print("提取特征...")
    features_df = feature_engine.extract_multi_view_features(df, max_tasks=200)

    if features_df is None or len(features_df) == 0:
        print("❌ 特征提取失败!")
        return

    # 保存特征
    features_path = os.path.join(output_dir, 'features', 'task_features.csv')
    features_df.to_csv(features_path, index=False)

    print(f"✅ 特征提取完成!")
    print(f"  特征维度: {features_df.shape}")
    print(f"  特征已保存: {features_path}")

    # 步骤3: 聚类分析
    print("\n" + "="*60)
    print("步骤3: 聚类分析")
    print("="*60)

    clustering = MultiViewClustering({'n_clusters': 5})

    # 准备特征
    features = clustering.prepare_features(features_df)
    features_scaled = clustering.standardize_features()

    # 降维
    print("执行降维...")
    embeddings = clustering.dimensionality_reduction(method='umap')

    # 聚类
    print("执行聚类...")
    labels = clustering.perform_clustering(method='kmeans')

    # 分析聚类
    print("分析聚类结果...")
    cluster_stats_df, cluster_profiles = clustering.analyze_clusters(features_df)

    if cluster_stats_df is None:
        print("❌ 聚类分析失败!")
        return

    # 保存聚类结果
    clustering_dir = os.path.join(output_dir, 'clustering')

    # 带标签的特征数据
    features_with_labels = features_df.copy()
    features_with_labels['cluster'] = labels[:len(features_df)]
    features_with_labels_path = os.path.join(clustering_dir, 'features_with_clusters.csv')
    features_with_labels.to_csv(features_with_labels_path, index=False)

    # 聚类统计
    cluster_stats_path = os.path.join(clustering_dir, 'cluster_statistics.csv')
    cluster_stats_df.to_csv(cluster_stats_path, index=False)

    print(f"✅ 聚类分析完成!")
    print(f"  聚类数量: {len(cluster_stats_df)}")

    # 显示聚类分布
    print(f"\n📊 聚类分布:")
    for _, row in cluster_stats_df.iterrows():
        cluster_id = row['cluster_id']
        size = row['size']
        percentage = row['percentage']
        print(f"  聚类 {cluster_id}: {size} 个任务 ({percentage:.1f}%)")

    # 步骤4: 可视化
    print("\n" + "="*60)
    print("步骤4: 可视化")
    print("="*60)

    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    visualization = VisualizationSystem(output_dir=viz_dir)

    try:
        print("创建可视化图表...")

        # 1. 聚类散点图
        print("  创建聚类散点图...")
        visualization.create_cluster_scatter(embeddings, labels)

        # 2. 小提琴图
        print("  创建特征分布小提琴图...")
        visualization.create_violin_plots(features_with_labels)

        # 3. 热力图
        print("  创建聚类热力图...")
        visualization.create_heatmap(features_with_labels)

        # 4. 时间序列图
        print("  创建时间序列叠加图...")
        # 准备任务ID列表
        job_task_ids = []
        for _, row in features_df.iterrows():
            if 'job_id' in row and 'task_index' in row:
                job_task_ids.append((row['job_id'], row['task_index']))

        if job_task_ids:
            visualization.create_time_series_overlay(df, labels, job_task_ids, n_samples=2)

        # 5. 仪表板
        print("  创建聚类仪表板...")
        visualization.create_dashboard(cluster_stats_df, cluster_profiles)

        # 保存所有图形
        visualization.save_all_figures()

        print(f"✅ 可视化完成!")

    except Exception as e:
        print(f"⚠️ 可视化错误: {e}")
        import traceback
        traceback.print_exc()

    # 步骤5: 生成报告
    print("\n" + "="*60)
    print("步骤5: 生成报告")
    print("="*60)

    reports_dir = os.path.join(output_dir, 'reports')
    report_path = os.path.join(reports_dir, 'analysis_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("HPC工作负载分析与分类报告\n")
        f.write("="*60 + "\n\n")

        f.write(f"分析时间: {timestamp}\n")
        f.write(f"输出目录: {output_dir}\n\n")

        f.write("1. 数据概况\n")
        f.write("-"*40 + "\n")
        f.write(f"   总数据点: {len(df)}\n")
        f.write(f"   任务数量: {features_df.shape[0]}\n")
        f.write(f"   特征数量: {features_df.shape[1]}\n\n")

        f.write("2. 聚类结果\n")
        f.write("-"*40 + "\n")
        for _, row in cluster_stats_df.iterrows():
            cluster_id = row['cluster_id']
            size = row['size']
            percentage = row['percentage']
            f.write(f"   聚类 {cluster_id}: {size} 个任务 ({percentage:.1f}%)\n")
        f.write("\n")

        f.write("3. 工作负载类型\n")
        f.write("-"*40 + "\n")
        for cluster_id, profile in cluster_profiles.items():
            f.write(f"\n   类型 {cluster_id}:\n")
            f.write(f"     主导资源: {profile.get('dominant_resource', 'N/A')}\n")
            f.write(f"     行为模式: {profile.get('behavior_type', 'N/A')}\n")
            f.write(f"     波动特性: {profile.get('volatility_level', 'N/A')}\n")
        f.write("\n")

        f.write("4. 生成文件\n")
        f.write("-"*40 + "\n")

        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")

            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

    print(f"✅ 报告生成完成!")
    print(f"  报告文件: {report_path}")

    # 保存最终结果
    final_results = {
        'timestamp': timestamp,
        'output_dir': output_dir,
        'features_df_shape': features_df.shape,
        'cluster_stats': cluster_stats_df.to_dict('records'),
        'cluster_profiles': cluster_profiles
    }

    results_path = os.path.join(output_dir, 'final_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"✅ 最终结果已保存: {results_path}")

    print("\n" + "="*80)
    print("✅ 分析完成!")
    print(f"所有结果保存在: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main_fixed()