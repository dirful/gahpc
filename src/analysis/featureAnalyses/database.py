"""
修复版数据库模块 - 处理数据类型问题
"""
import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
from config import DB_CONFIG

class DatabaseConnector:
    def __init__(self, config=None):
        self.config = config or DB_CONFIG
        self.connection = None

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                print(f"成功连接到数据库 {self.config['database']}")
                return True
        except Error as e:
            print(f"数据库连接错误: {e}")
            return False

    def disconnect(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("数据库连接已关闭")

    def _convert_to_native_types(self, params):
        """将numpy类型转换为Python原生类型"""
        if params is None:
            return None

        if isinstance(params, (list, tuple)):
            return [self._convert_to_native_types(p) for p in params]
        elif isinstance(params, dict):
            return {k: self._convert_to_native_types(v) for k, v in params.items()}
        elif isinstance(params, np.integer):
            return int(params)
        elif isinstance(params, np.floating):
            return float(params)
        elif isinstance(params, np.ndarray):
            return params.tolist()
        else:
            return params

    def execute_query(self, query, params=None):
        """执行查询并处理数据类型"""
        try:
            # 转换参数类型
            native_params = self._convert_to_native_types(params)

            # 使用pandas的read_sql，它会自动处理类型转换
            df = pd.read_sql(query, self.connection, params=native_params)
            return df
        except Exception as e:
            print(f"查询执行错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_task_time_series(self, num_tasks=100, min_samples=5):
        """
        加载任务的完整时间序列数据

        Args:
            num_tasks: 要加载的任务数量
            min_samples: 每个任务最少需要的数据点数
        """
        print(f"加载 {num_tasks} 个任务的完整时间序列数据...")

        try:
            # 步骤1: 先找到有足够数据点的任务
            print("查找有足够数据点的任务...")

            # 查询每个任务的数据点数量
            task_count_query = """
                               SELECT
                                   job_id, task_index, COUNT(*) as num_samples,
                                   MIN(start_time) as first_time,
                                   MAX(end_time) as last_time
                               FROM task_usage
                               WHERE cpu_rate IS NOT NULL
                                 AND canonical_memory_usage IS NOT NULL
                               GROUP BY job_id, task_index
                               HAVING COUNT(*) >= %s
                               ORDER BY num_samples DESC
                                   LIMIT %s \
                               """

            # 使用execute_query来处理数据类型
            task_counts = self.execute_query(task_count_query,
                                             params=[min_samples, num_tasks * 2])

            if task_counts is None or len(task_counts) == 0:
                print("❌ 没有找到有足够数据点的任务")
                return None

            print(f"找到 {len(task_counts)} 个有足够数据点的任务")

            # 步骤2: 批量加载任务数据（更高效的方式）
            print("批量加载任务数据...")

            # 收集所有符合条件的任务ID
            task_ids = []
            for _, row in task_counts.iterrows():
                # 转换为Python原生类型
                job_id = int(row['job_id'])
                task_index = int(row['task_index'])
                task_ids.append((job_id, task_index))

            # 构建IN查询（更高效）
            if not task_ids:
                print("❌ 没有有效的任务ID")
                return None

            # 限制任务数量
            task_ids = task_ids[:num_tasks]

            # 构建WHERE条件
            conditions = []
            params = []
            for job_id, task_index in task_ids:
                conditions.append("(job_id = %s AND task_index = %s)")
                params.extend([job_id, task_index])

            where_clause = " OR ".join(conditions)

            # 一次性加载所有任务数据
            batch_query = f"""
            SELECT 
                start_time, end_time, job_id, task_index, machine_id,
                cpu_rate, canonical_memory_usage, disk_io_time,
                maximum_cpu_rate, maximum_memory_usage,
                local_disk_space_usage, total_page_cache,
                cycles_per_instruction
            FROM task_usage
            WHERE ({where_clause})
            ORDER BY job_id, task_index, start_time
            """

            df = self.execute_query(batch_query, params=params)

            if df is None or len(df) == 0:
                print("❌ 批量加载数据失败")
                return None

            print(f"批量加载完成，共 {len(df)} 条记录")

            # 检查数据质量
            task_groups = df.groupby(['job_id', 'task_index'])
            valid_tasks = []

            for (job_id, task_index), group in task_groups:
                if len(group) >= min_samples:
                    # 确保数据按时间排序
                    group = group.sort_values('start_time').copy()

                    # 计算衍生特征
                    group['duration'] = group['end_time'] - group['start_time']
                    group['time_from_start'] = group['start_time'] - group['start_time'].min()

                    valid_tasks.append(group)

            if not valid_tasks:
                print("❌ 没有找到有足够数据点的有效任务")
                return None

            # 合并所有有效任务数据
            combined_df = pd.concat(valid_tasks, ignore_index=True)

            print(f"\n✅ 成功加载 {len(valid_tasks)} 个任务的完整时间序列")
            print(f"总数据点: {len(combined_df)}")
            print(f"每个任务平均数据点: {len(combined_df) / len(valid_tasks):.1f}")

            # 显示统计信息
            task_stats = []
            for (job_id, task_index), group in combined_df.groupby(['job_id', 'task_index']):
                task_stats.append({
                    'job_id': job_id,
                    'task_index': task_index,
                    'num_samples': len(group),
                    'time_span': group['end_time'].max() - group['start_time'].min(),
                    'cpu_mean': group['cpu_rate'].mean(),
                    'cpu_std': group['cpu_rate'].std(),
                    'mem_mean': group['canonical_memory_usage'].mean(),
                    'mem_std': group['canonical_memory_usage'].std()
                })

            stats_df = pd.DataFrame(task_stats)

            print(f"\n📊 任务统计:")
            print(f"  任务数量: {len(stats_df)}")
            print(f"  最小序列长度: {stats_df['num_samples'].min()}")
            print(f"  最大序列长度: {stats_df['num_samples'].max()}")
            print(f"  平均序列长度: {stats_df['num_samples'].mean():.1f}")
            print(f"  最小时间跨度: {stats_df['time_span'].min()}")
            print(f"  最大时间跨度: {stats_df['time_span'].max()}")
            print(f"  平均CPU使用率: {stats_df['cpu_mean'].mean():.4f}")
            print(f"  平均内存使用: {stats_df['mem_mean'].mean():.1f}")

            return combined_df

        except Exception as e:
            print(f"数据加载错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_high_frequency_tasks(self, num_tasks=50, frequency_threshold=30):
        """
        加载高频任务（有大量数据点的任务）
        """
        print(f"加载高频任务数据（至少 {frequency_threshold} 个数据点）...")

        try:
            # 查询高频任务
            query = """
                    SELECT
                        job_id, task_index, COUNT(*) as num_samples,
                        AVG(cpu_rate) as avg_cpu,
                        AVG(canonical_memory_usage) as avg_mem
                    FROM task_usage
                    WHERE cpu_rate IS NOT NULL
                      AND canonical_memory_usage IS NOT NULL
                    GROUP BY job_id, task_index
                    HAVING COUNT(*) >= %s
                    ORDER BY num_samples DESC
                        LIMIT %s \
                    """

            tasks = self.execute_query(query,
                                       params=[frequency_threshold, num_tasks])

            if tasks is None or len(tasks) == 0:
                print(f"❌ 没有找到至少有 {frequency_threshold} 个数据点的任务")
                return None

            print(f"找到 {len(tasks)} 个高频任务")

            # 批量加载这些任务的数据
            task_ids = []
            for _, row in tasks.iterrows():
                task_ids.append((int(row['job_id']), int(row['task_index'])))

            # 构建查询
            conditions = []
            params = []
            for job_id, task_index in task_ids:
                conditions.append("(job_id = %s AND task_index = %s)")
                params.extend([job_id, task_index])

            where_clause = " OR ".join(conditions)

            batch_query = f"""
            SELECT 
                start_time, end_time, job_id, task_index, machine_id,
                cpu_rate, canonical_memory_usage, disk_io_time,
                maximum_cpu_rate, maximum_memory_usage,
                local_disk_space_usage, total_page_cache,
                cycles_per_instruction
            FROM task_usage
            WHERE ({where_clause})
            ORDER BY job_id, task_index, start_time
            """

            df = self.execute_query(batch_query, params=params)

            if df is None or len(df) == 0:
                return None

            # 处理数据
            df = df.sort_values(['job_id', 'task_index', 'start_time']).copy()
            df['duration'] = df['end_time'] - df['start_time']

            print(f"✅ 加载了 {len(df)} 条高频任务记录")

            return df

        except Exception as e:
            print(f"高频任务加载错误: {e}")
            return None