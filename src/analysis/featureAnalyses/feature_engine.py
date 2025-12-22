"""
修复版特征工程模块
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngine:
    def __init__(self, config=None):
        if config is None:
            self.window_sizes = [5, 10, 20]
            self.alpha_params = {
                'cpu_weight': 0.4,
                'mem_weight': 0.3,
                'io_weight': 0.2,
                'diff_weight': 0.1,
                'volatility_weight': 0.2
            }
        else:
            self.window_sizes = config.get('window_sizes', [5, 10, 20])
            self.alpha_params = config.get('default_params', {
                'cpu_weight': 0.4,
                'mem_weight': 0.3,
                'io_weight': 0.2,
                'diff_weight': 0.1,
                'volatility_weight': 0.2
            })

    def set_parameters(self, params):
        """设置可调参数"""
        self.alpha_params.update(params)

    def extract_task_features(self, task_data):
        """
        提取单个任务的特征

        Args:
            task_data: 单个任务的数据DataFrame
        """
        features = {}

        if task_data.empty or len(task_data) < 3:
            print(f"任务数据不足: {len(task_data)} 个点")
            return features

        # 确保按时间排序
        task_data = task_data.sort_values('start_time')

        try:
            # 基本统计特征
            for metric in ['cpu_rate', 'canonical_memory_usage', 'disk_io_time']:
                if metric in task_data.columns:
                    series = task_data[metric].dropna().values

                    if len(series) > 0:
                        # 基本统计
                        features[f'{metric}_mean'] = np.mean(series)
                        features[f'{metric}_std'] = np.std(series) if len(series) > 1 else 0
                        features[f'{metric}_min'] = np.min(series)
                        features[f'{metric}_max'] = np.max(series)
                        features[f'{metric}_median'] = np.median(series)

                        # 分位数
                        features[f'{metric}_q25'] = np.percentile(series, 25)
                        features[f'{metric}_q75'] = np.percentile(series, 75)
                        features[f'{metric}_iqr'] = features[f'{metric}_q75'] - features[f'{metric}_q25']

                        # 高阶统计（需要足够的数据点）
                        if len(series) > 2:
                            try:
                                features[f'{metric}_skew'] = pd.Series(series).skew()
                                features[f'{metric}_kurt'] = pd.Series(series).kurtosis()
                            except:
                                features[f'{metric}_skew'] = 0
                                features[f'{metric}_kurt'] = 0

                    else:
                        # 如果系列为空，设置默认值
                        features[f'{metric}_mean'] = 0
                        features[f'{metric}_std'] = 0
                        features[f'{metric}_min'] = 0
                        features[f'{metric}_max'] = 0
                        features[f'{metric}_median'] = 0
                        features[f'{metric}_q25'] = 0
                        features[f'{metric}_q75'] = 0
                        features[f'{metric}_iqr'] = 0
                        features[f'{metric}_skew'] = 0
                        features[f'{metric}_kurt'] = 0

            # 增量特征（需要至少2个点）
            if 'cpu_rate' in task_data.columns:
                cpu_series = task_data['cpu_rate'].dropna().values
                if len(cpu_series) > 1:
                    cpu_diff = np.diff(cpu_series)
                    features['cpu_diff_mean'] = np.mean(cpu_diff)
                    features['cpu_diff_std'] = np.std(cpu_diff) if len(cpu_diff) > 1 else 0
                    features['cpu_diff_max'] = np.max(np.abs(cpu_diff)) if len(cpu_diff) > 0 else 0
                else:
                    features['cpu_diff_mean'] = 0
                    features['cpu_diff_std'] = 0
                    features['cpu_diff_max'] = 0

            # 波动率特征
            if 'cpu_rate' in task_data.columns:
                cpu_series = task_data['cpu_rate'].dropna().values
                for window in self.window_sizes:
                    if len(cpu_series) > window:
                        volatility = []
                        for i in range(window, len(cpu_series)):
                            window_data = cpu_series[i-window:i]
                            if np.mean(window_data) > 0:
                                vol = np.std(window_data) / np.mean(window_data)
                            else:
                                vol = 0
                            volatility.append(vol)

                        if volatility:
                            features[f'cpu_volatility_{window}_mean'] = np.mean(volatility)
                            features[f'cpu_volatility_{window}_max'] = np.max(volatility)
                        else:
                            features[f'cpu_volatility_{window}_mean'] = 0
                            features[f'cpu_volatility_{window}_max'] = 0
                    else:
                        features[f'cpu_volatility_{window}_mean'] = 0
                        features[f'cpu_volatility_{window}_max'] = 0

            # 突发性特征
            if 'cpu_rate' in task_data.columns:
                cpu_series = task_data['cpu_rate'].dropna().values
                if len(cpu_series) > 10:
                    threshold = np.percentile(cpu_series, 75)
                    bursts = cpu_series > threshold
                    features['burst_ratio'] = np.mean(bursts)
                    features['burst_duration_mean'] = self._mean_burst_duration(bursts)
                else:
                    features['burst_ratio'] = 0
                    features['burst_duration_mean'] = 0

            # 相位特征（周期性检测）
            if 'cpu_rate' in task_data.columns:
                cpu_series = task_data['cpu_rate'].dropna().values
                if len(cpu_series) > 20:
                    autocorr = self._autocorrelation(cpu_series, max_lag=min(10, len(cpu_series)-1))
                    features['autocorr_lag1'] = autocorr[1] if len(autocorr) > 1 else 0
                    features['autocorr_max'] = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
                else:
                    features['autocorr_lag1'] = 0
                    features['autocorr_max'] = 0

            # 资源比率特征
            cpu_mean = features.get('cpu_rate_mean', 0)
            mem_mean = features.get('canonical_memory_usage_mean', 0)
            io_mean = features.get('disk_io_time_mean', 0)

            if mem_mean > 0:
                features['cpu_mem_ratio'] = cpu_mean / mem_mean
            else:
                features['cpu_mem_ratio'] = 0

            if io_mean > 0:
                features['cpu_io_ratio'] = cpu_mean / io_mean
                features['mem_io_ratio'] = mem_mean / io_mean
            else:
                features['cpu_io_ratio'] = 0
                features['mem_io_ratio'] = 0

            # 持续时间特征
            if 'duration' in task_data.columns:
                features['duration'] = task_data['duration'].iloc[0] if len(task_data) > 0 else 0
            features['num_samples'] = len(task_data)

        except Exception as e:
            print(f"特征提取错误: {e}")
            # 返回空特征字典
            return {}

        return features

    def _mean_burst_duration(self, burst_mask):
        """计算平均突发持续时间"""
        if not np.any(burst_mask):
            return 0

        # 找到突发的开始和结束
        burst_changes = np.diff(np.concatenate(([0], burst_mask.astype(int), [0])))
        burst_starts = np.where(burst_changes == 1)[0]
        burst_ends = np.where(burst_changes == -1)[0]

        if len(burst_starts) > 0 and len(burst_ends) > 0:
            durations = burst_ends - burst_starts
            return np.mean(durations) if len(durations) > 0 else 0

        return 0

    def _autocorrelation(self, series, max_lag=10):
        """计算自相关"""
        n = len(series)
        if n <= max_lag:
            return np.zeros(max_lag + 1)

        autocorr = []
        series_mean = np.mean(series)
        series_var = np.var(series)

        if series_var == 0:
            return np.zeros(max_lag + 1)

        for lag in range(max_lag + 1):
            if lag < n:
                numerator = np.sum((series[:n-lag] - series_mean) * (series[lag:] - series_mean))
                denominator = (n - lag) * series_var
                autocorr.append(numerator / denominator)
            else:
                autocorr.append(0)

        return np.array(autocorr)

    def create_parametric_features(self, features_dict):
        """
        创建参数化特征（可调公式层）
        """
        parametric = {}

        # 提取基础特征
        cpu_mean = features_dict.get('cpu_rate_mean', 0)
        mem_mean = features_dict.get('canonical_memory_usage_mean', 0)
        io_mean = features_dict.get('disk_io_time_mean', 0)

        cpu_std = features_dict.get('cpu_rate_std', 0)
        mem_std = features_dict.get('canonical_memory_usage_std', 0)
        io_std = features_dict.get('disk_io_time_std', 0)

        cpu_diff_mean = features_dict.get('cpu_diff_mean', 0)

        # 可调公式特征
        # 1. 加权资源特征
        parametric['weighted_resource'] = (
                self.alpha_params['cpu_weight'] * cpu_mean +
                self.alpha_params['mem_weight'] * mem_mean +
                self.alpha_params['io_weight'] * io_mean
        )

        # 2. 加权波动特征
        parametric['weighted_volatility'] = (
                self.alpha_params['volatility_weight'] * (cpu_std + mem_std + io_std)
        )

        # 3. 增量特征
        parametric['weighted_diff'] = self.alpha_params['diff_weight'] * cpu_diff_mean

        # 4. 突发特征
        burst_ratio = features_dict.get('burst_ratio', 0)
        burst_duration = features_dict.get('burst_duration_mean', 0)
        parametric['burst_score'] = burst_ratio * burst_duration

        # 5. 资源均衡度
        resource_values = [cpu_mean, mem_mean, io_mean]
        non_zero = [v for v in resource_values if v > 0]
        if len(non_zero) > 1:
            parametric['resource_balance'] = np.std(non_zero) / (np.mean(non_zero) + 1e-10)
        else:
            parametric['resource_balance'] = 0

        # 6. 资源主导度
        if resource_values:
            max_val = max(resource_values)
            sum_val = sum(resource_values)
            if sum_val > 0:
                parametric['resource_dominance'] = max_val / sum_val
            else:
                parametric['resource_dominance'] = 0

        return parametric

    def extract_multi_view_features(self, df, max_tasks=1000):
        """
        提取多视图特征

        Args:
            df: 原始数据DataFrame
            max_tasks: 最大任务数限制
        """
        print("开始提取多视图特征...")

        if df is None or df.empty:
            print("❌ 输入数据为空!")
            return pd.DataFrame()

        # 确保必要的列存在
        required_columns = ['job_id', 'task_index', 'cpu_rate', 'canonical_memory_usage']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ 缺少必要的列: {missing_columns}")
            return pd.DataFrame()

        # 按job_id和task_index分组
        print(f"数据分组...")
        task_groups = df.groupby(['job_id', 'task_index'])
        num_tasks = len(task_groups)
        print(f"发现 {num_tasks} 个任务")

        # 获取任务列表（限制数量）
        task_list = list(task_groups.groups.keys())
        if len(task_list) > max_tasks:
            print(f"限制任务数为 {max_tasks} (原 {len(task_list)})")
            task_list = task_list[:max_tasks]

        all_features = []
        skipped_tasks = 0

        for i, (job_id, task_index) in enumerate(task_list):
            try:
                # 获取任务数据
                task_data = task_groups.get_group((job_id, task_index))

                # 检查数据是否足够
                if len(task_data) < 5:  # 至少需要5个数据点
                    skipped_tasks += 1
                    continue

                # 提取基础特征
                basic_features = self.extract_task_features(task_data)

                # 如果特征提取失败，跳过
                if not basic_features:
                    skipped_tasks += 1
                    continue

                # 提取参数化特征
                parametric_features = self.create_parametric_features(basic_features)

                # 合并特征
                task_features = {
                    'job_id': job_id,
                    'task_index': task_index,
                    **basic_features,
                    **parametric_features
                }

                all_features.append(task_features)

                if (i + 1) % 100 == 0:
                    print(f"已处理 {i+1}/{len(task_list)} 个任务")

            except Exception as e:
                print(f"处理任务 {job_id}-{task_index} 时出错: {e}")
                skipped_tasks += 1
                continue

        if not all_features:
            print("❌ 没有成功提取任何特征!")
            return pd.DataFrame()

        # 创建DataFrame
        features_df = pd.DataFrame(all_features)

        print(f"特征提取完成:")
        print(f"  成功处理: {len(features_df)} 个任务")
        print(f"  跳过任务: {skipped_tasks} 个")
        print(f"  特征数量: {len(features_df.columns)}")

        # 处理缺失值
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
                print(f"  填充缺失值: {col} -> {median_val:.4f}")

        # 显示前几个特征
        if len(features_df) > 0:
            print(f"\n前3个任务的特征示例:")
            print(features_df.head(3).T)

        return features_df