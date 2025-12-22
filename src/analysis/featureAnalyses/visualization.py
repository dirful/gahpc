"""
修复版可视化模块
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import VISUALIZATION_CONFIG

class VisualizationSystem:
    def __init__(self, config=None, output_dir=None):
        """
        初始化可视化系统

        Args:
            config: 配置字典
            output_dir: 输出目录
        """
        self.config = config or VISUALIZATION_CONFIG

        # 设置输出目录
        if output_dir is not None:
            self.output_dir = output_dir
        elif 'output_dir' in self.config:
            self.output_dir = self.config['output_dir']
        else:
            self.output_dir = 'visualizations'

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        self.figures = {}

    def create_cluster_scatter(self, embeddings, labels, title="Cluster Visualization", save_name="cluster_scatter"):
        """创建聚类散点图"""
        # 确保标签是字符串类型
        labels_str = labels.astype(str)

        fig = px.scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            color=labels_str,
            title=title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'color': 'Cluster'},
            opacity=0.7
        )

        self.figures['cluster_scatter'] = fig
        self.save_figure(fig, save_name)

        return fig

    def create_violin_plots(self, features_df, cluster_col='cluster', save_name="violin_plots"):
        """创建小提琴图"""
        # 选择要显示的特征
        key_features = [
            'cpu_rate_mean', 'canonical_memory_usage_mean', 'disk_io_time_mean',
            'cpu_rate_std', 'weighted_resource', 'weighted_volatility'
        ]

        available_features = [f for f in key_features if f in features_df.columns]

        if not available_features:
            print("⚠️ 没有找到可用的特征创建小提琴图")
            return None

        # 限制特征数量
        available_features = available_features[:6]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=available_features,
            vertical_spacing=0.1
        )

        for i, feature in enumerate(available_features):
            row = i // 3 + 1
            col = i % 3 + 1

            # 获取唯一的聚类ID
            cluster_ids = sorted([c for c in features_df[cluster_col].unique() if c != -1])

            # 为每个聚类创建数据
            for cluster_id in cluster_ids:
                cluster_data = features_df[features_df[cluster_col] == cluster_id]
                feature_values = cluster_data[feature].dropna()

                if len(feature_values) > 0:
                    fig.add_trace(
                        go.Violin(
                            y=feature_values,
                            name=f'Cluster {cluster_id}',
                            box_visible=True,
                            meanline_visible=True,
                            points=False,
                            side='positive' if cluster_id % 2 == 0 else 'negative'
                        ),
                        row=row, col=col
                    )

        fig.update_layout(
            title="Resource Usage Distribution by Cluster",
            height=600,
            showlegend=True
        )

        self.figures['violin_plots'] = fig
        self.save_figure(fig, save_name)

        return fig

    def create_heatmap(self, features_df, cluster_col='cluster', save_name="cluster_heatmap"):
        """创建特征热力图"""
        # 获取唯一的聚类ID（排除噪声点-1）
        cluster_ids = sorted([c for c in features_df[cluster_col].unique() if c != -1])

        if not cluster_ids:
            print("⚠️ 没有有效的聚类ID")
            return None

        # 计算每个聚类的特征均值
        cluster_means = []
        feature_names = []

        # 选择数值型特征
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if cluster_col in numeric_cols:
            numeric_cols.remove(cluster_col)

        # 选择最重要的特征（方差最大的）
        if len(numeric_cols) > 10:
            feature_vars = []
            for col in numeric_cols:
                feature_vars.append(features_df[col].var())

            # 取方差最大的前10个特征
            important_idx = np.argsort(feature_vars)[-10:][::-1]
            selected_features = [numeric_cols[i] for i in important_idx]
        else:
            selected_features = numeric_cols

        for cluster_id in cluster_ids:
            cluster_data = features_df[features_df[cluster_col] == cluster_id]
            means = cluster_data[selected_features].mean().values
            cluster_means.append(means)

        if not cluster_means:
            print("⚠️ 没有计算到聚类均值")
            return None

        cluster_means = np.array(cluster_means)

        fig = go.Figure(data=go.Heatmap(
            z=cluster_means.T,
            x=[f'Cluster {i}' for i in cluster_ids],
            y=selected_features,
            colorscale='Viridis',
            colorbar=dict(title="平均值")
        ))

        fig.update_layout(
            title="Cluster Feature Heatmap",
            xaxis_title="Cluster",
            yaxis_title="Feature",
            height=500
        )

        self.figures['heatmap'] = fig
        self.save_figure(fig, save_name)

        return fig

    def create_centroid_plot(self, cluster_centers, feature_names=None, save_name="centroid_plot"):
        """创建质心图"""
        if cluster_centers is None:
            print("⚠️ 没有聚类中心数据")
            return None

        n_clusters, n_features = cluster_centers.shape

        # 选择最重要的特征
        if n_features > 10:
            centroid_var = np.var(cluster_centers, axis=0)
            important_idx = np.argsort(centroid_var)[-10:][::-1]
            cluster_centers = cluster_centers[:, important_idx]

            if feature_names is not None:
                feature_names = [feature_names[i] for i in important_idx]

        fig = go.Figure()

        for i in range(n_clusters):
            x_values = feature_names if feature_names is not None else list(range(cluster_centers.shape[1]))
            fig.add_trace(go.Scatter(
                x=x_values,
                y=cluster_centers[i],
                name=f'Cluster {i}',
                mode='lines+markers'
            ))

        fig.update_layout(
            title="Cluster Centroids",
            xaxis_title="Feature",
            yaxis_title="Value",
            xaxis_tickangle=45,
            height=500
        )

        self.figures['centroids'] = fig
        self.save_figure(fig, save_name)

        return fig

    def create_time_series_overlay(self, original_df, cluster_labels,
                                   job_task_ids, n_samples=3, save_prefix="timeseries_cluster"):
        """创建时间序列叠加图"""
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # 跳过噪声点
                continue

            # 获取该聚类的任务索引
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            # 随机选择几个任务
            sample_indices = np.random.choice(
                cluster_indices,
                min(n_samples, len(cluster_indices)),
                replace=False
            )

            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(f"CPU Usage - Cluster {cluster_id}",
                                f"Memory Usage - Cluster {cluster_id}",
                                f"IO Time - Cluster {cluster_id}"),
                shared_xaxes=True,
                vertical_spacing=0.05
            )

            all_cpu = []
            all_mem = []
            all_io = []

            for idx in sample_indices:
                if idx >= len(job_task_ids):
                    continue

                job_id, task_index = job_task_ids[idx]

                # 获取任务数据
                task_data = original_df[
                    (original_df['job_id'] == job_id) &
                    (original_df['task_index'] == task_index)
                    ].sort_values('start_time')

                if len(task_data) > 0:
                    # 截取前100个时间点
                    n_points = min(100, len(task_data))

                    # CPU
                    cpu_series = task_data['cpu_rate'].values[:n_points]
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(n_points)),
                            y=cpu_series,
                            mode='lines',
                            opacity=0.3,
                            showlegend=False,
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
                    all_cpu.append(cpu_series)

                    # Memory
                    if 'canonical_memory_usage' in task_data.columns:
                        mem_series = task_data['canonical_memory_usage'].values[:n_points]
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(n_points)),
                                y=mem_series,
                                mode='lines',
                                opacity=0.3,
                                showlegend=False,
                                line=dict(width=1)
                            ),
                            row=2, col=1
                        )
                        all_mem.append(mem_series)

                    # IO
                    if 'disk_io_time' in task_data.columns:
                        io_series = task_data['disk_io_time'].values[:n_points]
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(n_points)),
                                y=io_series,
                                mode='lines',
                                opacity=0.3,
                                showlegend=False,
                                line=dict(width=1)
                            ),
                            row=3, col=1
                        )
                        all_io.append(io_series)

            # 添加平均线
            if all_cpu:
                # 对齐长度
                min_len = min(len(series) for series in all_cpu)
                if min_len > 0:
                    cpu_aligned = [series[:min_len] for series in all_cpu]
                    mean_cpu = np.nanmean(cpu_aligned, axis=0)

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(min_len)),
                            y=mean_cpu,
                            mode='lines',
                            name='Mean',
                            line=dict(color='black', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )

            if all_mem:
                min_len = min(len(series) for series in all_mem)
                if min_len > 0:
                    mem_aligned = [series[:min_len] for series in all_mem]
                    mean_mem = np.nanmean(mem_aligned, axis=0)

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(min_len)),
                            y=mean_mem,
                            mode='lines',
                            name='Mean',
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )

            if all_io:
                min_len = min(len(series) for series in all_io)
                if min_len > 0:
                    io_aligned = [series[:min_len] for series in all_io]
                    mean_io = np.nanmean(io_aligned, axis=0)

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(min_len)),
                            y=mean_io,
                            mode='lines',
                            name='Mean',
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=False
                        ),
                        row=3, col=1
                    )

            fig.update_layout(
                title=f"Time Series Patterns - Cluster {cluster_id}",
                height=600,
                showlegend=True
            )

            fig.update_xaxes(title_text="Time Step", row=3, col=1)

            self.figures[f'timeseries_cluster_{cluster_id}'] = fig
            self.save_figure(fig, f'{save_prefix}_{cluster_id}')

    def create_dashboard(self, cluster_stats_df, cluster_profiles, save_name="clustering_dashboard"):
        """创建仪表板"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Cluster Distribution',
                'Dominant Resources',
                'Behavior Types',
                'CPU Usage by Cluster',
                'Memory Usage by Cluster',
                'Volatility Distribution'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'box'}, {'type': 'box'}, {'type': 'box'}]
            ],
            vertical_spacing=0.15
        )

        # 1. 聚类分布饼图
        sizes = cluster_stats_df['size'].values
        labels = [f'Cluster {i}' for i in range(len(sizes))]

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=sizes,
                hole=0.3,
                textinfo='label+percent'
            ),
            row=1, col=1
        )

        # 2. 主导资源
        resource_counts = {}
        for cluster_id, profile in cluster_profiles.items():
            resource = profile.get('dominant_resource', 'Balanced')
            resource_counts[resource] = resource_counts.get(resource, 0) + 1

        if resource_counts:
            fig.add_trace(
                go.Bar(
                    x=list(resource_counts.keys()),
                    y=list(resource_counts.values()),
                    marker_color='lightblue'
                ),
                row=1, col=2
            )

        # 3. 行为类型
        behavior_counts = {}
        for cluster_id, profile in cluster_profiles.items():
            behavior = profile.get('behavior_type', 'Unknown')
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        if behavior_counts:
            fig.add_trace(
                go.Bar(
                    x=list(behavior_counts.keys()),
                    y=list(behavior_counts.values()),
                    marker_color='lightcoral'
                ),
                row=1, col=3
            )

        fig.update_layout(
            title="HPC Workload Clustering Dashboard",
            height=700,
            showlegend=False
        )

        self.figures['dashboard'] = fig
        self.save_figure(fig, save_name)

        return fig

    def save_figure(self, fig, filename):
        """保存图形"""
        if self.config.get('save_plots', True):
            filepath = os.path.join(self.output_dir, f"{filename}.html")
            try:
                fig.write_html(filepath)
                print(f"  ✓ 图形已保存: {filepath}")
            except Exception as e:
                print(f"  ✗ 图形保存失败: {e}")

    def save_all_figures(self):
        """保存所有图形"""
        print(f"保存所有可视化图形到: {self.output_dir}")
        for name, fig in self.figures.items():
            if fig is not None:
                self.save_figure(fig, name)