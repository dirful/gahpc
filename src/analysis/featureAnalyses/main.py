"""
完整的HPC工作负载分析主程序
"""
import argparse
import pickle
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np



from database import DatabaseConnector
from feature_engine import FeatureEngine
from param_optimizer import ParamOptimizer
from clustering import MultiViewClustering
from visualization import VisualizationSystem
from config import EXPERIMENT_CONFIG, CLUSTERING_CONFIG, VISUALIZATION_CONFIG

class HPCWorkloadAnalyzer:
    def __init__(self, config=None):
        self.config = config or EXPERIMENT_CONFIG
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"hpc_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = {
            'config': self.config,
            'timestamp': self.timestamp,
            'output_dir': self.output_dir
        }

    def setup_directories(self):
        """创建输出目录结构"""
        subdirs = ['raw_data', 'features', 'clustering', 'visualizations', 'reports']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        return self.output_dir

    def run_data_loading(self, sample_size=None):
        """步骤1: 数据加载"""
        print("\n" + "="*60)
        print("步骤1: 数据加载")
        print("="*60)

        db = DatabaseConnector()
        if not db.connect():
            print("❌ 数据库连接失败!")
            return None

        try:
            # 使用配置的样本大小
            sample_size = sample_size or self.config['sample_size']

            print(f"加载数据 (样本大小: {sample_size})...")

            # 方法1: 尝试加载高频任务
            df = db.load_high_frequency_tasks(
                num_tasks=min(50, sample_size // 100),
                frequency_threshold=20
            )

            if df is None or len(df) == 0:
                print("高频任务加载失败，尝试普通任务...")
                df = db.load_task_time_series(
                    num_tasks=min(100, sample_size // 50),
                    min_samples=10
                )

            if df is None or len(df) == 0:
                print("❌ 数据加载失败!")
                return None

            # 保存原始数据
            raw_data_path = os.path.join(self.output_dir, 'raw_data', 'task_data.csv')
            df.to_csv(raw_data_path, index=False)

            print(f"✅ 数据加载成功!")
            print(f"  总数据点: {len(df):,}")
            print(f"  唯一任务数: {df.groupby(['job_id', 'task_index']).ngroups}")
            print(f"  原始数据已保存: {raw_data_path}")

            self.results['raw_data'] = df
            self.results['raw_data_path'] = raw_data_path

            return df

        finally:
            db.disconnect()

    def run_feature_engineering(self, df, optimize_params=True):
        """步骤2: 特征工程与参数优化"""
        print("\n" + "="*60)
        print("步骤2: 特征工程与参数优化")
        print("="*60)

        if df is None or len(df) == 0:
            print("❌ 输入数据为空!")
            return None

        feature_engine = FeatureEngine()

        # 参数优化
        if optimize_params and self.config.get('optimize_params', True):
            print("执行参数优化...")

            # 使用少量数据进行快速优化
            print("使用少量数据进行参数优化...")
            optimizer = ParamOptimizer(feature_engine)

            # 为了速度，使用更少的任务进行优化
            optimization_df = df.copy()
            if len(df) > 50000:
                # 采样用于优化
                optimization_df = df.sample(50000, random_state=42)

            best_params, best_score, opt_features = optimizer.grid_search(
                optimization_df,
                param_grid={
                    'cpu_weight': [0.3, 0.4, 0.5],
                    'mem_weight': [0.2, 0.3, 0.4],
                    'io_weight': [0.1, 0.2, 0.3],
                    'diff_weight': [0.05, 0.1, 0.15],
                    'volatility_weight': [0.1, 0.2, 0.3]
                }
            )

            if best_params:
                feature_engine.set_parameters(best_params)
                print(f"✅ 参数优化完成!")
                print(f"  最佳得分: {best_score:.4f}")
                print(f"  最佳参数: {best_params}")

                self.results['optimized_params'] = best_params
                self.results['optimization_score'] = best_score
                self.results['optimization_history'] = optimizer.history

                # 保存优化历史
                opt_history_path = os.path.join(self.output_dir, 'features', 'optimization_history.csv')
                pd.DataFrame(optimizer.history).to_csv(opt_history_path, index=False)

            else:
                print("⚠️ 参数优化失败，使用默认参数")
        else:
            print("跳过参数优化，使用默认参数")

        # 提取特征（使用所有数据）
        print("\n提取多视图特征...")
        features_df = feature_engine.extract_multi_view_features(
            df,
            max_tasks=self.config.get('max_tasks', 2000)
        )

        if features_df is None or len(features_df) == 0:
            print("❌ 特征提取失败!")
            return None

        # 保存特征
        features_path = os.path.join(self.output_dir, 'features', 'task_features.csv')
        features_df.to_csv(features_path, index=False)

        # 特征统计
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        feature_stats = features_df[numeric_cols].describe()
        stats_path = os.path.join(self.output_dir, 'features', 'feature_statistics.csv')
        feature_stats.to_csv(stats_path)

        print(f"✅ 特征工程完成!")
        print(f"  特征维度: {features_df.shape}")
        print(f"  特征已保存: {features_path}")
        print(f"  统计信息: {stats_path}")

        self.results['features_df'] = features_df
        self.results['feature_engine'] = feature_engine

        return features_df

    def run_clustering(self, features_df, n_clusters=None):
        """步骤3: 聚类分析"""
        print("\n" + "="*60)
        print("步骤3: 聚类分析")
        print("="*60)

        if features_df is None or len(features_df) == 0:
            print("❌ 特征数据为空!")
            return None

        # 使用配置的聚类数量
        n_clusters = n_clusters or self.config.get('n_clusters', CLUSTERING_CONFIG['n_clusters'])
        clustering_config = CLUSTERING_CONFIG.copy()
        clustering_config['n_clusters'] = n_clusters

        clustering = MultiViewClustering(clustering_config)

        print(f"执行 {n_clusters} 类聚类分析...")

        # 准备特征
        features = clustering.prepare_features(features_df)

        # 标准化
        features_scaled = clustering.standardize_features()

        # 降维
        embeddings = clustering.dimensionality_reduction(
            method=clustering_config.get('dim_reduction_method', 'umap')
        )

        # 执行聚类
        labels = clustering.perform_clustering(
            method=clustering_config.get('clustering_method', 'kmeans')
        )

        # 分析聚类结果
        cluster_stats_df, cluster_profiles = clustering.analyze_clusters(features_df)

        if cluster_stats_df is None:
            print("❌ 聚类分析失败!")
            return None

        # 保存聚类结果
        clustering_dir = os.path.join(self.output_dir, 'clustering')

        # 带标签的特征数据
        features_with_labels = features_df.copy()
        features_with_labels['cluster'] = labels[:len(features_df)]
        features_with_labels_path = os.path.join(clustering_dir, 'features_with_clusters.csv')
        features_with_labels.to_csv(features_with_labels_path, index=False)

        # 聚类统计
        cluster_stats_path = os.path.join(clustering_dir, 'cluster_statistics.csv')
        cluster_stats_df.to_csv(cluster_stats_path, index=False)

        # 聚类配置文件
        cluster_profiles_path = os.path.join(clustering_dir, 'cluster_profiles.json')
        with open(cluster_profiles_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            def convert_for_json(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            json.dump(convert_for_json(cluster_profiles), f, indent=2, ensure_ascii=False)

        print(f"✅ 聚类分析完成!")
        print(f"  聚类数量: {n_clusters}")
        print(f"  带标签数据: {features_with_labels_path}")
        print(f"  聚类统计: {cluster_stats_path}")
        print(f"  聚类配置: {cluster_profiles_path}")

        # 显示聚类概况
        print(f"\n📊 聚类概况:")
        for _, row in cluster_stats_df.iterrows():
            cluster_id = row['cluster_id']
            size = row['size']
            percentage = row['percentage']
            print(f"  聚类 {cluster_id}: {size} 个任务 ({percentage:.1f}%)")

        self.results.update({
            'clustering': clustering,
            'embeddings': embeddings,
            'labels': labels,
            'cluster_stats_df': cluster_stats_df,
            'cluster_profiles': cluster_profiles,
            'features_with_labels': features_with_labels
        })

        return clustering, embeddings, labels

    def run_visualization(self, df, features_df, clustering, embeddings, labels):
        """步骤4: 可视化"""
        print("\n" + "="*60)
        print("步骤4: 可视化")
        print("="*60)

        if clustering is None:
            print("❌ 聚类结果为空，跳过可视化")
            return

        # 创建可视化输出目录
        viz_output_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_output_dir, exist_ok=True)

        # 创建可视化系统
        viz = VisualizationSystem(
            config=VISUALIZATION_CONFIG,
            output_dir=viz_output_dir
        )

        print("创建可视化图表...")

        try:
            # 1. 聚类散点图
            print("  创建聚类散点图...")
            viz.create_cluster_scatter(
                embeddings,
                labels,
                title="HPC Workload Clustering"
            )

            # 2. 特征分布小提琴图
            print("  创建特征分布小提琴图...")
            features_with_labels = features_df.copy()
            features_with_labels['cluster'] = labels[:len(features_df)]
            viz.create_violin_plots(features_with_labels, cluster_col='cluster')

            # 3. 聚类热力图
            print("  创建聚类热力图...")
            viz.create_heatmap(features_with_labels, cluster_col='cluster')

            # 4. 质心图（如果有）
            if hasattr(clustering, 'cluster_centers') and clustering.cluster_centers is not None:
                print("  创建质心图...")
                viz.create_centroid_plot(
                    clustering.cluster_centers,
                    clustering.feature_names
                )

            # 5. 时间序列叠加图
            print("  创建时间序列叠加图...")
            # 准备任务ID列表
            job_task_ids = []
            if 'job_id' in features_df.columns and 'task_index' in features_df.columns:
                # 直接从特征DataFrame获取任务ID
                for _, row in features_df.iterrows():
                    job_task_ids.append((row['job_id'], row['task_index']))
            else:
                # 从原始数据获取
                task_groups = df.groupby(['job_id', 'task_index'])
                task_keys = list(task_groups.groups.keys())

                # 确保任务ID与特征顺序对应
                for i in range(min(len(features_df), len(task_keys))):
                    job_id, task_index = task_keys[i]
                    job_task_ids.append((job_id, task_index))

            if job_task_ids:
                viz.create_time_series_overlay(
                    df, labels, job_task_ids, n_samples=3
                )
            else:
                print("  ⚠️ 无法创建时间序列图：缺少任务ID信息")

            # 6. 聚类仪表板
            print("  创建聚类仪表板...")
            if 'cluster_stats_df' in self.results and 'cluster_profiles' in self.results:
                viz.create_dashboard(
                    self.results['cluster_stats_df'],
                    self.results['cluster_profiles']
                )
            else:
                print("  ⚠️ 无法创建仪表板：缺少聚类统计信息")

            # 保存所有图形
            viz.save_all_figures()

            print(f"✅ 可视化完成!")
            print(f"  所有图表已保存到: {viz_output_dir}")

            self.results['visualization'] = viz

        except Exception as e:
            print(f"⚠️ 可视化创建过程中出错: {e}")
            import traceback
            traceback.print_exc()

    def generate_report(self):
        """步骤5: 生成分析报告"""
        print("\n" + "="*60)
        print("步骤5: 生成分析报告")
        print("="*60)

        reports_dir = os.path.join(self.output_dir, 'reports')

        # 生成文本报告
        report_path = os.path.join(reports_dir, 'analysis_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("HPC工作负载分析与分类报告\n")
            f.write("="*60 + "\n\n")

            f.write(f"分析时间: {self.timestamp}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")

            # 1. 数据概况
            f.write("1. 数据概况\n")
            f.write("-"*40 + "\n")
            if 'raw_data' in self.results:
                df = self.results['raw_data']
                f.write(f"   总数据点: {len(df):,}\n")
                f.write(f"   唯一任务数: {df.groupby(['job_id', 'task_index']).ngroups}\n")
                f.write(f"   时间范围: {df['start_time'].min()} - {df['start_time'].max()}\n\n")

            # 2. 特征工程
            f.write("2. 特征工程\n")
            f.write("-"*40 + "\n")
            if 'features_df' in self.results:
                features_df = self.results['features_df']
                f.write(f"   特征数量: {features_df.shape[1]}\n")
                f.write(f"   任务数量: {features_df.shape[0]}\n")

                if 'optimized_params' in self.results:
                    params = self.results['optimized_params']
                    f.write(f"   优化参数: {params}\n")
                    f.write(f"   优化得分: {self.results.get('optimization_score', 'N/A')}\n")
                f.write("\n")

            # 3. 聚类结果
            f.write("3. 聚类结果\n")
            f.write("-"*40 + "\n")
            if 'cluster_stats_df' in self.results:
                cluster_stats_df = self.results['cluster_stats_df']
                f.write(f"   聚类数量: {len(cluster_stats_df)}\n\n")

                f.write("   聚类分布:\n")
                for _, row in cluster_stats_df.iterrows():
                    cluster_id = row['cluster_id']
                    size = row['size']
                    percentage = row['percentage']
                    f.write(f"     聚类 {cluster_id}: {size} 个任务 ({percentage:.1f}%)\n")
                f.write("\n")

            # 4. 工作负载类型描述
            f.write("4. 工作负载类型描述\n")
            f.write("-"*40 + "\n")
            if 'cluster_profiles' in self.results:
                cluster_profiles = self.results['cluster_profiles']

                for cluster_id, profile in cluster_profiles.items():
                    f.write(f"\n   聚类 {cluster_id}:\n")
                    f.write(f"     资源强度: {profile.get('resource_intensity', 'N/A')}\n")
                    f.write(f"     主导资源: {profile.get('dominant_resource', 'N/A')}\n")
                    f.write(f"     行为模式: {profile.get('behavior_type', 'N/A')}\n")
                    f.write(f"     波动特性: {profile.get('volatility_level', 'N/A')}\n")

                    # 调度建议
                    f.write(f"     调度建议: ")
                    resource = profile.get('dominant_resource', '')
                    intensity = profile.get('resource_intensity', '')
                    volatility = profile.get('volatility_level', '')

                    if intensity == 'High' and resource:
                        if resource == 'CPU':
                            f.write("分配高CPU节点，考虑CPU亲和性\n")
                        elif resource == 'Memory':
                            f.write("保证足够内存，避免swap\n")
                        elif resource == 'IO':
                            f.write("使用高速存储，优化IO调度\n")
                        else:
                            f.write("根据主导资源进行专项优化\n")
                    elif volatility == 'High':
                        f.write("预留缓冲资源，使用弹性调度策略\n")
                    else:
                        f.write("标准调度策略，资源按需分配\n")
                f.write("\n")

            # 5. 应用建议
            f.write("5. 应用建议\n")
            f.write("-"*40 + "\n")
            f.write("   • 调度优化: 基于工作负载类型实现差异化调度\n")
            f.write("   • 资源分配: 为不同类别设置资源保障和限制\n")
            f.write("   • 容量规划: 识别集群资源瓶颈，优化资源配置\n")
            f.write("   • 性能预测: 建立基于类型的性能预测模型\n")
            f.write("   • 仿真研究: 使用聚类结果作为GAN/TimeGAN输入\n")
            f.write("   • 资源竞争分析: 分析不同类型任务间的资源竞争模式\n")
            f.write("   • 调度策略优化: 为不同类型任务制定最优调度策略\n")
            f.write("\n")

            # 6. 文件清单
            f.write("6. 生成文件清单\n")
            f.write("-"*40 + "\n")

            def list_files(dir_path, indent=4):
                for root, dirs, files in os.walk(dir_path):
                    level = root.replace(dir_path, '').count(os.sep)
                    indent_str = ' ' * indent * level
                    f.write(f"{indent_str}{os.path.basename(root)}/\n")

                    subindent = ' ' * indent * (level + 1)
                    for file in files:
                        f.write(f"{subindent}{file}\n")

            f.write(f"\n{self.output_dir}/\n")
            list_files(self.output_dir, indent=4)

        print(f"✅ 报告生成完成!")
        print(f"  报告文件: {report_path}")

        # 在控制台也显示报告摘要
        print("\n" + "="*60)
        print("报告摘要")
        print("="*60)

        if 'cluster_profiles' in self.results:
            print("\n发现的工作负载类型:")
            for cluster_id, profile in self.results['cluster_profiles'].items():
                print(f"\n  类型 {cluster_id}:")
                print(f"    特征: {profile.get('resource_intensity', '')}-{profile.get('dominant_resource', '')}")
                print(f"    行为: {profile.get('behavior_type', '')}, {profile.get('volatility_level', '')}波动")

        print(f"\n详细报告请查看: {report_path}")
        print(f"所有结果保存在: {self.output_dir}")

    def save_final_results(self):
        """保存最终结果"""
        print("\n保存最终结果...")

        # 保存pickle文件
        pickle_path = os.path.join(self.output_dir, 'final_results.pkl')
        with open(pickle_path, 'wb') as f:
            # 移除可能无法pickle的对象
            save_results = self.results.copy()
            if 'visualization' in save_results:
                del save_results['visualization']
            if 'raw_data' in save_results and isinstance(save_results['raw_data'], pd.DataFrame):
                # 只保存数据路径，不保存整个DataFrame
                save_results['raw_data'] = None

            pickle.dump(save_results, f)

        print(f"✅ 最终结果已保存: {pickle_path}")

    def run_complete_analysis(self, sample_size=None, n_clusters=None, optimize_params=None):
        """运行完整分析流程"""
        print("="*80)
        print("HPC工作负载分析与分类系统")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # 创建目录结构
        self.setup_directories()

        # 步骤1: 数据加载
        df = self.run_data_loading(sample_size)
        if df is None:
            return None

        # 步骤2: 特征工程
        optimize = optimize_params if optimize_params is not None else self.config.get('optimize_params', True)
        features_df = self.run_feature_engineering(df, optimize_params=optimize)
        if features_df is None:
            return None

        # 步骤3: 聚类分析
        n_clusters = n_clusters or self.config.get('n_clusters', 5)
        clustering_result = self.run_clustering(features_df, n_clusters=n_clusters)
        if clustering_result is None:
            return None

        clustering, embeddings, labels = clustering_result

        # 步骤4: 可视化
        self.run_visualization(df, features_df, clustering, embeddings, labels)

        # 步骤5: 生成报告
        self.generate_report()

        # 保存最终结果
        self.save_final_results()

        print("\n" + "="*80)
        print("✅ 分析完成!")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"所有结果保存在: {self.output_dir}")
        print("="*80)

        return self.results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HPC工作负载分析与分类系统')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='样本大小（默认: 使用配置文件设置）')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='最大任务数（默认: 使用配置文件设置）')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='聚类数量（默认: 5）')
    parser.add_argument('--optimize', action='store_true', default=None,
                        help='启用参数优化')
    parser.add_argument('--no_optimize', dest='optimize', action='store_false',
                        help='禁用参数优化')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认: 自动生成时间戳目录）')

    args = parser.parse_args()

    # 更新配置
    config = EXPERIMENT_CONFIG.copy()

    if args.sample_size is not None:
        config['sample_size'] = args.sample_size

    if args.max_tasks is not None:
        config['max_tasks'] = args.max_tasks

    if args.optimize is not None:
        config['optimize_params'] = args.optimize

    # 创建分析器
    analyzer = HPCWorkloadAnalyzer(config)

    # 运行完整分析
    results = analyzer.run_complete_analysis(
        sample_size=args.sample_size,
        n_clusters=args.n_clusters,
        optimize_params=args.optimize
    )

    return results

if __name__ == "__main__":
    # 运行完整分析
    main()