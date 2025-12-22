"""
快速测试脚本
"""
import pandas as pd
import numpy as np
from database import DatabaseConnector

def quick_test():
    """快速测试数据库连接和数据加载"""
    print("快速测试...")

    # 1. 测试连接
    db = DatabaseConnector()
    if not db.connect():
        print("连接失败")
        return

    try:
        # 2. 测试一个简单查询
        print("\n测试简单查询...")
        test_query = "SELECT COUNT(*) as count FROM task_usage"
        result = db.execute_query(test_query)

        if result is not None:
            print(f"task_usage表记录数: {result['count'].iloc[0]}")

        # 3. 测试数据类型转换
        print("\n测试数据类型转换...")
        test_params = [np.int64(100), np.float64(10.5), 200, "test"]
        print(f"原始参数: {test_params}")
        print(f"转换后: {db._convert_to_native_types(test_params)}")

        # 4. 加载少量数据测试
        print("\n加载少量数据测试...")
        small_query = """
                      SELECT job_id, task_index, start_time, cpu_rate
                      FROM task_usage
                      WHERE cpu_rate IS NOT NULL
                      ORDER BY job_id, task_index, start_time
                          LIMIT 50 \
                      """

        small_df = db.execute_query(small_query)

        if small_df is not None:
            print(f"加载了 {len(small_df)} 条记录")
            print("\n前10条记录:")
            print(small_df.head(10))

            # 检查数据类型
            print("\n数据类型:")
            print(small_df.dtypes)

        # 5. 测试批量查询
        print("\n测试批量查询少量任务...")
        batch_query = """
                      SELECT
                          start_time, end_time, job_id, task_index,
                          cpu_rate, canonical_memory_usage
                      FROM task_usage
                      WHERE job_id IN (3418309, 3418314)
                      ORDER BY job_id, task_index, start_time
                          LIMIT 100 \
                      """

        batch_df = db.execute_query(batch_query)

        if batch_df is not None and len(batch_df) > 0:
            print(f"批量加载了 {len(batch_df)} 条记录")

            # 分组查看
            groups = batch_df.groupby(['job_id', 'task_index'])
            print(f"找到 {groups.ngroups} 个任务组")

            for (job_id, task_index), group in groups:
                print(f"\n任务 {job_id}-{task_index}: {len(group)} 个点")
                if len(group) > 1:
                    print(f"  时间范围: {group['start_time'].min()} - {group['start_time'].max()}")
                    print(f"  CPU范围: {group['cpu_rate'].min():.4f} - {group['cpu_rate'].max():.4f}")

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.disconnect()

    print("\n测试完成!")

if __name__ == "__main__":
    quick_test()