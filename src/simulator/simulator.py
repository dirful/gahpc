# src/simulator/simulator.py
import numpy as np
from log.logger import get_logger

logger = get_logger(__name__)

class HPCSimulator:
    def __init__(self, cfg):
        self.cfg = cfg

        # 集群配置
        self.num_nodes = cfg.num_nodes
        self.max_jobs_per_node = cfg.max_jobs_per_node

        # 节点资源容量
        self.node_cpu_capacity = getattr(cfg, 'node_cpu_capacity', 1.0)
        self.node_mem_capacity = getattr(cfg, 'node_mem_capacity', 1.0)

        # 先初始化所有属性
        self.nodes = []
        self.job_queue = []  # 先初始化job_queue
        self.current_time = 0
        self.total_jobs_completed = 0
        self.total_reward = 0
        self.history = []

        # 然后调用reset
        self.reset()

        logger.info(f"HPC Simulator initialized: {self.num_nodes} nodes")

    def reset(self, clear_jobs=False):
        """重置环境状态

        Args:
            clear_jobs: 是否清空作业队列（默认False，保持作业）
        """
        # 初始化节点状态
        self.nodes = []
        for i in range(self.num_nodes):
            self.nodes.append({
                'id': i,
                'cpu_used': 0.0,
                'mem_used': 0.0,
                'jobs': [],
                'available': True
            })

        # 是否清空作业队列
        if clear_jobs:
            self.job_queue = []
            logger.info("Simulator reset (清空作业)")
        else:
            # 确保job_queue已初始化
            if not hasattr(self, 'job_queue'):
                self.job_queue = []
            logger.info(f"Simulator reset (保留 {len(self.job_queue)} 个作业)")

        # 重置时间
        self.current_time = 0

        # 重置指标
        self.total_jobs_completed = 0
        self.total_reward = 0

        # 初始状态
        state = self._get_state()

        return state

    def _get_state(self):
        """获取当前状态表示"""
        state = []

        # 节点资源使用情况
        for node in self.nodes:
            state.append(node['cpu_used'] / self.node_cpu_capacity)
            state.append(node['mem_used'] / self.node_mem_capacity)

        # 队列信息
        queue_length = len(self.job_queue)
        state.append(queue_length / 10.0)  # 归一化

        # 如果有作业在队列中，添加第一个作业的信息
        if self.job_queue:
            job = self.job_queue[0]
            state.extend([
                job.get('cpu', 0.5),
                job.get('mem', 0.3),
                job.get('duration', 300) / 3600.0  # 归一化为小时
            ])
        else:
            state.extend([0.0, 0.0, 0.0])

        # 确保状态维度固定
        while len(state) < 30:  # 固定为30维
            state.append(0.0)

        return np.array(state[:30], dtype=np.float32)

    def add_jobs(self, jobs):
        """添加作业到队列"""
        if not isinstance(jobs, list):
            jobs = [jobs]

        for job in jobs:
            # 验证作业
            job = self._validate_job(job)
            self.job_queue.append(job)

        logger.info(f"Added {len(jobs)} jobs to queue, total: {len(self.job_queue)}")

    def _validate_job(self, job):
        """验证作业参数"""
        validated = {}

        # CPU: 确保在合理范围内
        validated['cpu'] = np.clip(
            job.get('cpu', 0.5),
            self.cfg.min_cpu,
            self.cfg.max_cpu
        )

        # 内存: 确保在合理范围内
        validated['mem'] = np.clip(
            job.get('mem', 0.3),
            self.cfg.min_mem,
            self.cfg.max_mem
        )

        # 磁盘IO
        validated['disk_io'] = np.clip(
            job.get('disk_io', 0.2),
            self.cfg.min_disk_io if hasattr(self.cfg, 'min_disk_io') else 0.0,
            self.cfg.max_disk_io if hasattr(self.cfg, 'max_disk_io') else 10.0
        )

        # 持续时间
        validated['duration'] = np.clip(
            job.get('duration', 300),
            self.cfg.min_duration,
            self.cfg.max_duration
        )

        # 作业ID
        validated['id'] = job.get('id', len(self.job_queue) + 1)

        return validated

    def step(self, action):
        """执行一步动作 - 增强版本"""
        # 检查是否有作业
        if not self.job_queue:
            logger.warning("No jobs in queue, skipping step")
            next_state = self._get_state()
            reward = -0.1  # 空闲惩罚
            done = self._check_done()
            info = {'status': 'idle', 'message': '队列为空'}
            return next_state, reward, done, info

        # 解析动作
        try:
            node_id, cpu_allocation, mem_allocation = self._parse_action(action)
        except Exception as e:
            logger.error(f"动作解析失败: {e}")
            next_state = self._get_state()
            reward = -1.0
            done = False
            info = {'status': 'action_error', 'error': str(e)}
            return next_state, reward, done, info

        # 检查节点ID有效性
        if node_id < 0 or node_id >= len(self.nodes):
            logger.warning(f"无效的节点ID: {node_id}, 可用节点: 0-{len(self.nodes)-1}")
            # 选择第一个可用节点作为后备
            node_id = 0

        # 获取作业和节点
        job = self.job_queue[0]  # 总是处理队列中的第一个作业
        node = self.nodes[node_id]

        # 计算需要的资源
        cpu_needed = job['cpu'] * cpu_allocation
        mem_needed = job['mem'] * mem_allocation

        # 检查资源可用性
        cpu_available = self.node_cpu_capacity - node['cpu_used']
        mem_available = self.node_mem_capacity - node['mem_used']

        allocation_successful = False
        if cpu_needed <= cpu_available and mem_needed <= mem_available:
            # 成功分配
            node['cpu_used'] += cpu_needed
            node['mem_used'] += mem_needed
            node['jobs'].append({
                'job_id': job['id'],
                'cpu': cpu_needed,
                'mem': mem_needed,
                'duration': job['duration'],
                'start_time': self.current_time
            })

            # 从队列中移除作业
            completed_job = self.job_queue.pop(0)

            # 计算奖励
            reward = self._calculate_reward(completed_job, node_id, cpu_allocation, mem_allocation)

            allocation_successful = True
            info = {
                'status': 'success',
                'job_id': job['id'],
                'node_id': node_id,
                'cpu_allocated': cpu_needed,
                'mem_allocated': mem_needed,
                'reward': reward,
                'queue_remaining': len(self.job_queue)
            }

            logger.info(f"作业 {job['id']} 成功分配到节点 {node_id}, 奖励: {reward:.2f}")

        else:
            # 资源不足
            reward = -0.5
            info = {
                'status': 'resource_insufficient',
                'job_id': job['id'],
                'node_id': node_id,
                'cpu_needed': cpu_needed,
                'cpu_available': cpu_available,
                'mem_needed': mem_needed,
                'mem_available': mem_available,
                'suggestion': '尝试其他节点或减少资源分配'
            }

            logger.debug(f"节点 {node_id} 资源不足，作业 {job['id']} 保留在队列中")

        # 更新时间
        self.current_time += 1
        self.total_reward += reward

        if allocation_successful:
            self.total_jobs_completed += 1

        # 获取下一个状态
        next_state = self._get_state()

        # 检查是否结束
        done = self._check_done()

        return next_state, reward, done, info

    def _parse_action(self, action):
        """解析动作"""
        if isinstance(action, np.ndarray):
            if len(action) >= 3:
                # 连续动作
                node_id = int((action[0] + 1) * (self.num_nodes - 1) / 2)  # 映射到节点ID
                cpu_allocation = (action[1] + 1) / 2  # 映射到[0, 1]
                mem_allocation = (action[2] + 1) / 2  # 映射到[0, 1]
            else:
                # 标量动作
                node_id = int((action + 1) * (self.num_nodes - 1) / 2)
                cpu_allocation = 1.0
                mem_allocation = 1.0
        else:
            # 整数动作（离散）
            node_id = action % self.num_nodes
            cpu_allocation = 1.0
            mem_allocation = 1.0

        # 确保在有效范围内
        node_id = max(0, min(node_id, self.num_nodes - 1))
        cpu_allocation = max(0.0, min(cpu_allocation, 1.0))
        mem_allocation = max(0.0, min(mem_allocation, 1.0))

        return node_id, cpu_allocation, mem_allocation

    def _calculate_reward(self, job, node_id, cpu_allocation, mem_allocation):
        """计算增强的奖励"""
        reward = 0.0

        # 基础奖励：成功调度
        reward += 2.0

        # 资源利用效率奖励
        utilization_efficiency = (cpu_allocation + mem_allocation) / 2.0
        reward += utilization_efficiency

        # 节点负载均衡惩罚
        node = self.nodes[node_id]
        cpu_util = node['cpu_used'] / self.node_cpu_capacity
        mem_util = node['mem_used'] / self.node_mem_capacity
        avg_util = (cpu_util + mem_util) / 2.0

        if avg_util > 0.9:
            reward -= 1.0  # 节点过载惩罚
        elif avg_util < 0.3:
            reward -= 0.5  # 节点利用不足惩罚

        # 作业特性奖励
        if job['duration'] < 100:  # 短作业奖励
            reward += 0.5
        elif job['duration'] > 1000:  # 长作业惩罚
            reward -= 0.3

        if job['cpu'] > 0.8:  # CPU密集型作业
            reward += 0.3
        if job['mem'] > 0.8:  # 内存密集型作业
            reward += 0.3

        # 队列长度惩罚（鼓励快速处理）
        queue_len = len(self.job_queue)
        if queue_len > 5:
            reward -= 0.1 * (queue_len - 5)

        # 确保奖励在合理范围内
        reward = max(-5.0, min(10.0, reward))

        return reward

    def _check_done(self):
        """检查是否结束"""
        # 检查队列是否为空且所有节点都空闲
        queue_empty = len(self.job_queue) == 0
        all_nodes_idle = all(node['cpu_used'] < 0.1 and node['mem_used'] < 0.1
                             for node in self.nodes)

        # 或者达到最大时间步
        max_steps = getattr(self.cfg, 'max_steps', 100)
        time_limit = self.current_time >= max_steps

        return (queue_empty and all_nodes_idle) or time_limit

    def export_state(self):
        """导出详细的环境状态用于LLM"""
        state = {
            'current_time': self.current_time,
            'num_nodes': self.num_nodes,
            'nodes': [],
            'queue_length': len(self.job_queue),
            'jobs_completed': self.total_jobs_completed,
            'total_reward': self.total_reward,
            'system_load': self._calculate_system_load()
        }

        # 详细的节点信息
        for i, node in enumerate(self.nodes):
            cpu_util = node['cpu_used'] / self.node_cpu_capacity
            mem_util = node['mem_used'] / self.node_mem_capacity

            state['nodes'].append({
                'node_id': i,
                'cpu_used': node['cpu_used'],
                'mem_used': node['mem_used'],
                'num_jobs': len(node['jobs']),
                'cpu_utilization': cpu_util,
                'mem_utilization': mem_util,
                'status': 'overloaded' if cpu_util > 0.8 or mem_util > 0.8 else
                'underutilized' if cpu_util < 0.2 and mem_util < 0.2 else
                'normal'
            })

        # 队列中的作业信息
        queue_info = []
        for i, job in enumerate(self.job_queue[:5]):  # 最多显示5个
            job_type = []
            if job.get('cpu', 0) > 0.7:
                job_type.append('CPU密集型')
            if job.get('mem', 0) > 0.7:
                job_type.append('内存密集型')
            if job.get('duration', 0) < 60:
                job_type.append('短作业')
            elif job.get('duration', 0) > 600:
                job_type.append('长作业')

            queue_info.append({
                'position': i + 1,
                'cpu': job.get('cpu', 0),
                'mem': job.get('mem', 0),
                'duration': job.get('duration', 0),
                'type': '、'.join(job_type) if job_type else '普通作业'
            })

        state['queue_jobs'] = queue_info

        # 系统负载分析
        total_cpu_used = sum(node['cpu_used'] for node in self.nodes)
        total_mem_used = sum(node['mem_used'] for node in self.nodes)
        total_capacity = self.num_nodes * (self.node_cpu_capacity + self.node_mem_capacity)
        total_used = total_cpu_used + total_mem_used

        state['system_load'] = {
            'avg_cpu_utilization': total_cpu_used / (self.num_nodes * self.node_cpu_capacity),
            'avg_mem_utilization': total_mem_used / (self.num_nodes * self.node_mem_capacity),
            'total_utilization': total_used / total_capacity,
            'recommendation': self._get_load_recommendation(total_used / total_capacity)
        }

        return state

    def _calculate_system_load(self):
        """计算系统负载"""
        total_cpu = sum(node['cpu_used'] for node in self.nodes)
        total_mem = sum(node['mem_used'] for node in self.nodes)

        max_cpu = self.num_nodes * self.node_cpu_capacity
        max_mem = self.num_nodes * self.node_mem_capacity

        return {
            'cpu_load': total_cpu / max_cpu if max_cpu > 0 else 0,
            'mem_load': total_mem / max_mem if max_mem > 0 else 0
        }

    def _get_load_recommendation(self, utilization):
        """根据利用率提供推荐"""
        if utilization < 0.3:
            return "集群负载较低，可以接受大型作业"
        elif utilization < 0.7:
            return "集群负载适中，适合各种类型的作业"
        else:
            return "集群负载较高，建议提交小型或中型的作业"

    def get_metrics(self):
        """获取性能指标"""
        total_cpu_used = sum(node['cpu_used'] for node in self.nodes)
        total_mem_used = sum(node['mem_used'] for node in self.nodes)

        metrics = {
            'total_jobs_completed': self.total_jobs_completed,
            'total_reward': self.total_reward,
            'average_node_utilization': {
                'cpu': total_cpu_used / (self.num_nodes * self.node_cpu_capacity),
                'mem': total_mem_used / (self.num_nodes * self.node_mem_capacity)
            },
            'queue_length': len(self.job_queue),
            'simulation_time': self.current_time
        }

        return metrics

    def render(self):
        """渲染当前状态（文本形式）"""
        print(f"\n=== HPC Simulator Status (Time: {self.current_time}) ===")
        print(f"Jobs in queue: {len(self.job_queue)}")
        print(f"Jobs completed: {self.total_jobs_completed}")
        print(f"Total reward: {self.total_reward:.2f}")

        print("\nNode Status:")
        for i, node in enumerate(self.nodes):
            cpu_pct = node['cpu_used'] / self.node_cpu_capacity * 100
            mem_pct = node['mem_used'] / self.node_mem_capacity * 100
            print(f"  Node {i}: CPU {cpu_pct:.1f}%, Mem {mem_pct:.1f}%, Jobs: {len(node['jobs'])}")

        if self.job_queue:
            print("\nNext job in queue:")
            job = self.job_queue[0]
            print(f"  CPU: {job.get('cpu', 0.0):.2f}, Mem: {job.get('mem', 0.0):.2f}, "
                  f"Duration: {job.get('duration', 0.0):.1f}s")

        print("=" * 50)

    def test_action_parsing(self):
        """测试动作解析"""
        test_actions = [
            np.array([0.5, 0.3, -0.2, 0.1]),  # 4维动作
            np.array([-0.8, 0.2]),  # 2维动作
            np.array([0.0]),  # 1维动作
            2,  # 整数动作
            [0.5, 0.3, -0.2]  # 列表动作
        ]

        for i, action in enumerate(test_actions):
            try:
                node_id, cpu, mem = self._parse_action(action)
                print(f"Test {i+1}: action={action}, node_id={node_id}, cpu={cpu:.2f}, mem={mem:.2f}")
            except Exception as e:
                print(f"Test {i+1} failed: {e}")