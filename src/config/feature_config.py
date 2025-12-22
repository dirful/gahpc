# config/feature_config.py
class FeatureConfig:
    """特征配置管理"""

    @staticmethod
    def get_feature_config(data_mode='job_stats'):
        """获取特征配置"""
        configs = {
            'job_stats': {
                'feature_list': [
                    'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                    'mem_mean', 'mem_std', 'mem_max',
                    'machines_count', 'task_count',
                    'duration_sec', 'cpu_intensity', 'mem_intensity',
                    'task_density', 'cpu_cv', 'mem_cv'
                ],
                'seq_len': 1,
                'description': 'Job-level统计特征'
            },
            'time_series': {
                'feature_list': None,  # 动态确定
                'seq_len': 24,
                'description': '时间序列特征'
            },
            'task': {
                'feature_list': ['cpu', 'mem', 'disk_io', 'duration'],
                'seq_len': 1,
                'description': 'Task-level原始特征'
            }
        }

        return configs.get(data_mode, configs['job_stats'])

    @staticmethod
    def get_llm_stats_mapping(data_mode='job_stats'):
        """获取LLM统计信息映射"""
        mappings = {
            'job_stats': {
                'cpu': 'cpu_mean',
                'mem': 'mem_mean',
                'duration': 'duration_sec'
            },
            'time_series': {
                'cpu': 'cpu_avg',
                'mem': 'mem_avg',
                'duration': 'duration'
            },
            'task': {
                'cpu': 'cpu',
                'mem': 'mem',
                'disk_io': 'disk_io',
                'duration': 'duration'
            }
        }

        return mappings.get(data_mode, mappings['job_stats'])