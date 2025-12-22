import pandas as pd
import pymysql

SAMPLE_SIZE = 200000

class Connector:
    def __init__(self):
        self.conn = pymysql.connect(host="localhost", port=3307, user="root",
                                    password="123456", database="xiyoudata", charset="utf8mb4")
        print("数据库连接成功")

    def load(self):
        print("正在抽样 task_usage...")
        query = f"""
        SELECT job_id, task_index, start_time,
               cpu_rate, canonical_memory_usage, disk_io_time,
               maximum_cpu_rate, sampled_cpu_usage, cycles_per_instruction
        FROM task_usage ORDER BY RAND() LIMIT {SAMPLE_SIZE}
        """
        usage = pd.read_sql(query, self.conn)
        print("正在抽样 task_events...")
        query2 = f"""
        SELECT DISTINCT job_id, task_index, cpu_request, memory_request, priority, disk_space_request
        FROM task_events ORDER BY RAND() LIMIT {int(SAMPLE_SIZE * 2.5)}
        """
        events = pd.read_sql(query2, self.conn)
        self.conn.close()
        print(f"抽样完成：usage {len(usage)} 行，events {len(events)} 行")
        return usage, events