"""
聚类分析模块
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import umap
from config import CLUSTERING_CONFIG

class MultiViewClustering:
    def __init__(self, config=None):
        self.config = config or CLUSTERING_CONFIG
        self.n_clusters = int(self.config['n_clusters'])  # 确保是int类型
        self.cluster_labels = None
        self.cluster_centers = None
        self.embeddings = None

    def prepare_features(self, features_df):
        """
        准备聚类特征
        """
        # 分离特征列
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # 移除ID列
        id_cols = ['job_id', 'task_index']
        for col in id_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)

        self.feature_names = numeric_cols
        self.features = features_df[numeric_cols].values

        print(f"准备聚类特征: {self.features.shape[0]} 个样本, {self.features.shape[1]} 个特征")

        return self.features

    def standardize_features(self):
        """标准化特征"""
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.features)
        return self.features_scaled

    def dimensionality_reduction(self, method=None, n_components=None):
        """降维"""
        method = method or self.config.get('dim_reduction_method', 'umap')
        n_components = n_components or self.config.get('n_components', 2)

        if method == 'pca':
            pca = PCA(n_components=n_components)
            self.embeddings = pca.fit_transform(self.features_scaled)
            self.explained_variance = pca.explained_variance_ratio_
            print(f"PCA降维完成，解释方差: {self.explained_variance.sum():.3f}")

        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            self.embeddings = reducer.fit_transform(self.features_scaled)
            print(f"UMAP降维完成，维度: {self.embeddings.shape}")

        else:
            # 不降维
            self.embeddings = self.features_scaled

        return self.embeddings

    def perform_clustering(self, method=None):
        """执行聚类"""
        method = method or self.config.get('clustering_method', 'kmeans')

        print(f"使用 {method} 进行聚类...")

        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            self.cluster_labels = clusterer.fit_predict(self.embeddings)
            self.cluster_centers = clusterer.cluster_centers_

        elif method == 'gmm':
            gmm = GaussianMixture(
                n_components=self.n_clusters,
                random_state=42,
                covariance_type='tied'
            )
            gmm.fit(self.embeddings)
            self.cluster_labels = gmm.predict(self.embeddings)

        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            self.cluster_labels = dbscan.fit_predict(self.embeddings)
            # 重新标记聚类
            unique_labels = np.unique(self.cluster_labels)
            self.n_clusters = len(unique_labels[unique_labels != -1])

        elif method == 'hierarchical':
            hierarchical = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            self.cluster_labels = hierarchical.fit_predict(self.embeddings)

        # 打印聚类分布
        self._print_cluster_distribution()

        # 计算聚类质量指标
        self._evaluate_clustering()

        return self.cluster_labels

    def _print_cluster_distribution(self):
        """打印聚类分布"""
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\n聚类分布:")
        for label, count in zip(unique, counts):
            if label == -1:
                print(f"  噪声点: {count} ({count/len(self.cluster_labels)*100:.1f}%)")
            else:
                print(f"  聚类 {label}: {count} ({count/len(self.cluster_labels)*100:.1f}%)")

    def _evaluate_clustering(self):
        """评估聚类质量"""
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        if len(np.unique(self.cluster_labels)) > 1 and np.any(self.cluster_labels != -1):
            # 过滤噪声点
            mask = self.cluster_labels != -1
            if np.sum(mask) > 1:
                sil_score = silhouette_score(self.embeddings[mask], self.cluster_labels[mask])
                db_score = davies_bouldin_score(self.embeddings[mask], self.cluster_labels[mask])

                print(f"\n聚类质量指标:")
                print(f"  轮廓系数: {sil_score:.3f}")
                print(f"  Davies-Bouldin指数: {db_score:.3f}")

                self.silhouette_score = sil_score
                self.davies_bouldin_score = db_score

    def analyze_clusters(self, features_df):
        """
        分析聚类特征
        """
        if self.cluster_labels is None:
            print("请先执行聚类!")
            return None

        features_df = features_df.copy()
        features_df['cluster'] = self.cluster_labels

        # 计算每个聚类的统计信息
        cluster_stats = []

        for cluster_id in range(self.n_clusters):
            if cluster_id == -1:  # 跳过噪声点
                continue

            cluster_data = features_df[features_df['cluster'] == cluster_id]

            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features_df) * 100
            }

            # 关键特征的平均值
            key_features = ['cpu_rate_mean', 'canonical_memory_usage_mean',
                            'disk_io_time_mean', 'weighted_resource', 'weighted_volatility']

            for feature in key_features:
                if feature in cluster_data.columns:
                    stats[f'{feature}_mean'] = cluster_data[feature].mean()
                    stats[f'{feature}_std'] = cluster_data[feature].std()

            cluster_stats.append(stats)

        cluster_stats_df = pd.DataFrame(cluster_stats)

        # 分析聚类特征
        self.cluster_profiles = self._create_cluster_profiles(features_df)

        return cluster_stats_df, self.cluster_profiles

    def _create_cluster_profiles(self, features_df):
        """
        创建聚类特征描述
        """
        profiles = {}

        for cluster_id in range(self.n_clusters):
            if cluster_id == -1:
                continue

            cluster_data = features_df[features_df['cluster'] == cluster_id]

            profile = {
                'dominant_resource': None,
                'behavior_type': None,
                'volatility_level': None,
                'resource_intensity': None
            }

            # 判断主导资源
            cpu_mean = cluster_data['cpu_rate_mean'].mean() if 'cpu_rate_mean' in cluster_data.columns else 0
            mem_mean = cluster_data['canonical_memory_usage_mean'].mean() if 'canonical_memory_usage_mean' in cluster_data.columns else 0
            io_mean = cluster_data['disk_io_time_mean'].mean() if 'disk_io_time_mean' in cluster_data.columns else 0

            resource_values = {'CPU': cpu_mean, 'Memory': mem_mean, 'IO': io_mean}
            dominant_resource = max(resource_values, key=resource_values.get)

            # 资源强度
            max_value = max(resource_values.values())
            if max_value > 0:
                intensity = max_value / (sum(resource_values.values()) + 1e-10)
                if intensity > 0.7:
                    profile['resource_intensity'] = 'High'
                    profile['dominant_resource'] = dominant_resource
                elif intensity > 0.4:
                    profile['resource_intensity'] = 'Medium'
                else:
                    profile['resource_intensity'] = 'Balanced'

            # 判断行为类型
            weighted_diff = cluster_data['weighted_diff'].mean() if 'weighted_diff' in cluster_data.columns else 0
            weighted_vol = cluster_data['weighted_volatility'].mean() if 'weighted_volatility' in cluster_data.columns else 0

            if weighted_vol > 0.3:
                profile['behavior_type'] = 'Volatile'
            elif weighted_diff > 0.1:
                profile['behavior_type'] = 'Increasing'
            elif weighted_diff < -0.1:
                profile['behavior_type'] = 'Decreasing'
            else:
                profile['behavior_type'] = 'Stable'

            # 波动水平
            if weighted_vol < 0.1:
                profile['volatility_level'] = 'Low'
            elif weighted_vol < 0.3:
                profile['volatility_level'] = 'Medium'
            else:
                profile['volatility_level'] = 'High'

            profiles[cluster_id] = profile

        return profiles