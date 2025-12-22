"""
参数优化模块
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from feature_engine import FeatureEngine

class ParamOptimizer:
    def __init__(self, feature_engine):
        self.feature_engine = feature_engine
        self.best_params = None
        self.history = []

    def _convert_params(self, params):
        """转换参数为Python原生类型"""
        if isinstance(params, dict):
            return {k: float(v) if isinstance(v, np.floating) else v
                    for k, v in params.items()}
        return params

    def objective_function(self, params, X, y=None):
        """目标函数：最大化聚类质量"""
        # 转换参数
        params = self._convert_params(params)

        # 更新特征引擎参数
        self.feature_engine.alpha_params.update(params)

    def evaluate_clustering(self, features, params):
        """
        评估聚类质量
        """
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 执行聚类
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        # 计算评估指标
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(features_scaled, labels)
            ch_score = calinski_harabasz_score(features_scaled, labels)
        else:
            sil_score = -1
            ch_score = -1

        # 组合得分
        score = 0.6 * sil_score + 0.4 * (ch_score / 1000)

        return score, sil_score, ch_score

    def grid_search(self, df, param_grid=None):
        """
        网格搜索参数优化
        """
        if param_grid is None:
            param_grid = {
                'cpu_weight': [0.2, 0.4, 0.6],
                'mem_weight': [0.1, 0.3, 0.5],
                'io_weight': [0.1, 0.2, 0.3],
                'diff_weight': [0.05, 0.1, 0.15],
                'volatility_weight': [0.1, 0.2, 0.3]
            }

        best_score = -np.inf
        best_params = {}
        best_features = None

        print("开始网格搜索参数优化...")

        # 生成参数组合
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_combinations = np.prod([len(v) for v in param_values])
        print(f"总参数组合数: {total_combinations}")

        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))

            # 设置参数
            self.feature_engine.set_parameters(params)

            # 提取特征
            features_df = self.feature_engine.extract_multi_view_features(df, max_tasks=500)

            if len(features_df) < 10:
                continue

            # 选择数值型特征
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'job_id' in numeric_cols:
                numeric_cols.remove('job_id')
            if 'task_index' in numeric_cols:
                numeric_cols.remove('task_index')

            features = features_df[numeric_cols].values

            # 评估聚类
            score, sil_score, ch_score = self.evaluate_clustering(features, params)

            # 记录历史
            self.history.append({
                'params': params.copy(),
                'score': score,
                'silhouette': sil_score,
                'calinski_harabasz': ch_score
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_features = features_df

                print(f"\n[{i+1}/{total_combinations}] 新最佳参数:")
                print(f"  得分: {score:.4f}")
                print(f"  轮廓系数: {sil_score:.4f}")
                print(f"  Calinski-Harabasz: {ch_score:.2f}")
                print(f"  参数: {params}")

        self.best_params = best_params
        print(f"\n优化完成！最佳得分: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        return best_params, best_score, best_features

    def random_search(self, df, n_iter=20):
        """
        随机搜索参数优化
        """
        best_score = -np.inf
        best_params = {}

        print(f"开始随机搜索优化 ({n_iter} 次迭代)...")

        for i in range(n_iter):
            # 随机生成参数
            params = {
                'cpu_weight': np.random.uniform(0.1, 0.8),
                'mem_weight': np.random.uniform(0.1, 0.8),
                'io_weight': np.random.uniform(0.05, 0.4),
                'diff_weight': np.random.uniform(0.01, 0.3),
                'volatility_weight': np.random.uniform(0.05, 0.4)
            }

            # 归一化权重
            total = sum(params.values())
            if total > 0:
                for key in params:
                    params[key] = params[key] / total

            # 设置参数
            self.feature_engine.set_parameters(params)

            # 提取特征
            features_df = self.feature_engine.extract_multi_view_features(df, max_tasks=300)

            if len(features_df) < 10:
                continue

            # 选择数值型特征
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            features = features_df[numeric_cols].values

            # 评估聚类
            score, sil_score, ch_score = self.evaluate_clustering(features, params)

            # 记录历史
            self.history.append({
                'params': params.copy(),
                'score': score,
                'silhouette': sil_score,
                'calinski_harabasz': ch_score
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()

                print(f"\n[{i+1}/{n_iter}] 新最佳参数:")
                print(f"  得分: {score:.4f}")
                print(f"  参数: {params}")

        self.best_params = best_params
        return best_params, best_score