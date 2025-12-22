"""
配置文件
"""

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'database': 'xiyoudata',
    'user': 'root',
    'password': '123456'
}

# 特征工程配置
FEATURE_CONFIG = {
    'window_sizes': [5, 10, 20],
    'default_params': {
        'cpu_weight': 0.4,
        'mem_weight': 0.3,
        'io_weight': 0.2,
        'diff_weight': 0.1,
        'volatility_weight': 0.2
    }
}

# 聚类配置
CLUSTERING_CONFIG = {
    'n_clusters': 5,
    'clustering_method': 'kmeans',  # 'kmeans', 'gmm', 'dbscan', 'hierarchical'
    'dim_reduction_method': 'umap',  # 'umap', 'pca', None
    'n_components': 2
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'save_plots': True,
    'output_dir': 'visualizations',
    'plot_format': 'html',
    'figure_size': (1200, 800),
    'theme': 'plotly_white'
}

# 实验配置
EXPERIMENT_CONFIG = {
    'sample_size': 50000,
    'max_tasks': 1000,
    'optimize_params': True,
    'random_seed': 42,
    'min_samples_per_task': 10,
    'output_base_dir': 'hpc_analysis_results'
}