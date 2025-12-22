# src/llm/enhanced_llm_job_gen.py

import json
import re
import numpy as np
import pandas as pd
import time
import math
import random
from log.logger import get_logger
from .llm_job_gen import LLMJobGenerator

logger = get_logger(__name__)

class EnhancedLLMJobGenerator(LLMJobGenerator):
    """增强的LLM作业生成器，专门生成job-level特征"""

    def __init__(self, cfg, llm_client, gan_trainer=None):
        super().__init__(cfg, llm_client, gan_trainer)

        # 定义必需的job-level特征
        self.required_features = [
            'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
            'mem_mean', 'mem_std', 'mem_max',
            'machines_count', 'task_count',
            'start_time', 'end_time',  # 基础特征
            'duration_sec', 'cpu_intensity', 'mem_intensity',
            'task_density', 'cpu_cv', 'mem_cv'
        ]

        # 调试标志
        self.debug = getattr(cfg, 'llm_debug', False)

    def generate_job_level(self, stats, num=None):
        """生成与DBLoader匹配的job-level特征"""
        num = num or getattr(self.cfg, 'llm_num_generate', 30)

        logger.info(f"使用LLM生成{num}个job-level特征...")

        # 1. 使用LLM生成基础特征
        prompt = self._build_job_level_prompt(stats, num)
        if self.debug:
            logger.debug(f"完整Prompt:\n{prompt}")

        try:
            response = self.llm.ask(prompt)
            if self.debug:
                logger.debug(f"LLM完整响应:\n{response}")

            # 2. 解析LLM输出
            jobs = self._parse_job_level_response_debug(response, num)

            # 3. 如果没有成功解析或数量不足，使用后备方法
            if not jobs or len(jobs) < num:
                logger.warning(f"LLM解析失败或数量不足({len(jobs) if jobs else 0}/{num})，使用后备方法")
                logger.debug(f"响应内容: {response[:500]}")
                jobs = self._fallback_generate_job_level(stats, num)

            # 4. 确保每个作业都有所有必需特征
            jobs = self._ensure_complete_features(jobs, num)

            # 5. 转换为DataFrame
            df = pd.DataFrame(jobs)

            # 6. 确保特征类型正确
            df = self._ensure_feature_types(df)

            logger.info(f"成功生成 {len(df)} 个job-level特征")
            if self.debug and len(df) > 0:
                logger.debug(f"第一个生成的特征: {df.iloc[0].to_dict()}")

            return df

        except Exception as e:
            logger.error(f"LLM生成过程失败: {e}", exc_info=True)
            # 直接使用后备方法
            logger.info("使用纯后备方法生成数据")
            jobs = self._fallback_generate_job_level(stats, num)
            df = pd.DataFrame(jobs)
            df = self._ensure_feature_types(df)
            return df

    def _build_job_level_prompt(self, stats, num):
        """构建生成job-level特征的prompt"""

        stats_summary = self._format_stats_for_prompt(stats)

        prompt = f"""你是一个Google集群数据分析专家。请生成 {num} 个作业的统计特征，每个作业包含以下17个字段：

必需字段及其描述：
1. cpu_mean: 平均CPU使用率 (0.0-1.0)，通常0.1-0.4
2. cpu_std: CPU使用率标准差 (0.0-0.3)，反映波动性
3. cpu_max: 最大CPU使用率 (0.0-1.0)，必须 ≥ cpu_mean
4. cpu_95th: CPU 95分位数 (0.0-1.0)，在cpu_mean和cpu_max之间
5. mem_mean: 平均内存使用 (MB, 100-50000)，通常1000-10000
6. mem_std: 内存标准差 (MB, 50-10000)
7. mem_max: 最大内存使用 (MB, ≥ mem_mean)
8. machines_count: 使用的机器数 (1-50)，通常1-10
9. task_count: 任务总数 (1-1000)，通常10-100
10. start_time: 开始时间 (微秒，大整数，如1680000000000)
11. end_time: 结束时间 (微秒，> start_time)
12. duration_sec: 持续时间 (秒) = (end_time - start_time)/1000
13. cpu_intensity: CPU强度 = cpu_mean * duration_sec
14. mem_intensity: 内存强度 = mem_mean * duration_sec
15. task_density: 任务密度 = task_count / duration_sec
16. cpu_cv: CPU变异系数 = cpu_std / (cpu_mean + ε)
17. mem_cv: 内存变异系数 = mem_std / (mem_mean + ε)

注意逻辑一致性：
1. cpu_max ≥ cpu_mean ≥ 0
2. cpu_95th 应该在 cpu_mean 和 cpu_max 之间
3. mem_max ≥ mem_mean ≥ 0
4. end_time > start_time
5. duration_sec > 0
6. 变异系数(cpu_cv, mem_cv)通常在0.1-10之间

作业类型分布建议：
- 30% CPU密集型: cpu_mean > 0.6, mem_mean < 2000
- 30% 内存密集型: mem_mean > 8000, cpu_mean < 0.4
- 20% 混合型: cpu_mean 0.4-0.6, mem_mean 2000-8000
- 20% 轻量型: cpu_mean < 0.3, mem_mean < 1000

历史统计参考：
{stats_summary}

请返回严格的JSON数组，每个元素是一个包含上述17个字段的对象。
不要包含任何额外的解释或文本。

示例（只显示前2个字段）：
[
  {{
    "cpu_mean": 0.75,
    "cpu_std": 0.12,
    "cpu_max": 0.95,
    "cpu_95th": 0.88,
    "mem_mean": 1500.0,
    "mem_std": 300.0,
    "mem_max": 2200.0,
    "machines_count": 3,
    "task_count": 25,
    "start_time": 1680000000000,
    "end_time": 1680001800000,
    "duration_sec": 1800.0,
    "cpu_intensity": 1350.0,
    "mem_intensity": 2700000.0,
    "task_density": 0.0139,
    "cpu_cv": 0.16,
    "mem_cv": 0.2
  }}
]
"""
        return prompt

    def _format_stats_for_prompt(self, stats):
        """格式化统计信息用于prompt"""
        if not stats:
            return "No historical stats"
        out = []
        for k, v in stats.items():
            if isinstance(v, dict):
                out.append(f"- {k}: mean={v.get('mean',0):.2f}, p50={v.get('p50',0):.2f}, p90={v.get('p90',0):.2f}, count={v.get('count',0)}")
            else:
                out.append(f"- {k}: {v}")
        return "\n".join(out)

    def _parse_job_level_response_debug(self, response, expected_num):
        """带调试信息的解析函数"""
        logger.debug(f"开始解析LLM响应，响应长度: {len(response)}")

        # 首先尝试清理响应
        cleaned_response = response.strip()

        # 尝试1: 查找JSON数组
        json_patterns = [
            r'\[\s*\{.*\}\s*\]',  # 标准JSON数组
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # 多个JSON对象
        ]

        for pattern in json_patterns:
            try:
                matches = re.findall(pattern, cleaned_response, re.DOTALL)
                if matches:
                    logger.debug(f"找到 {len(matches)} 个匹配模式: {pattern}")
                    for i, match in enumerate(matches[:2]):
                        logger.debug(f"匹配 {i}: {match[:200]}...")

                    # 尝试解析第一个匹配
                    json_str = matches[0]
                    # 清理JSON
                    json_str = self._clean_json_string(json_str)

                    logger.debug(f"清理后的JSON: {json_str[:500]}...")

                    jobs = json.loads(json_str)
                    if isinstance(jobs, list):
                        logger.info(f"成功解析 {len(jobs)} 个job-level特征")
                        return jobs
                    elif isinstance(jobs, dict):
                        # 如果是单个对象，包装成列表
                        logger.info(f"解析到单个对象，包装成列表")
                        return [jobs]
            except Exception as e:
                logger.debug(f"JSON模式 {pattern} 解析失败: {str(e)[:100]}")
                continue

        # 尝试2: 逐行解析
        try:
            logger.debug("尝试逐行解析...")
            lines = cleaned_response.split('\n')
            jobs = []

            for line_num, line in enumerate(lines):
                line = line.strip()
                # 跳过空行和明显不是JSON的行
                if not line or line.startswith('#') or line.startswith('//') or line.startswith('/*'):
                    continue

                # 尝试找到JSON对象
                if '{' in line and '}' in line:
                    # 提取可能的JSON部分
                    start = line.find('{')
                    end = line.rfind('}') + 1
                    if end > start:
                        json_part = line[start:end]
                        try:
                            json_part = self._clean_json_string(json_part)
                            job = json.loads(json_part)
                            if isinstance(job, dict):
                                # 检查是否有足够的关键字段
                                required_keys = ['cpu_mean', 'mem_mean', 'task_count']
                                if any(key in job for key in required_keys):
                                    jobs.append(job)
                                    logger.debug(f"第{line_num}行解析成功")
                        except Exception as e:
                            logger.debug(f"第{line_num}行解析失败: {str(e)[:50]}")

            if jobs:
                logger.info(f"逐行解析获得 {len(jobs)} 个特征")
                return jobs[:expected_num]
        except Exception as e:
            logger.debug(f"逐行解析完全失败: {e}")

        # 尝试3: 提取所有数字模式
        try:
            logger.debug("尝试数字提取模式...")
            # 查找所有数字模式
            numbers = re.findall(r'\b\d+\.?\d*\b', cleaned_response)
            logger.debug(f"找到 {len(numbers)} 个数字")

            if len(numbers) >= expected_num * 10:  # 至少有足够多的数字
                jobs = []
                # 假设每10个数字构成一个作业
                step = 10
                for i in range(0, min(len(numbers), expected_num * step), step):
                    chunk = numbers[i:i+step]
                    if len(chunk) >= 5:  # 至少有5个数字
                        try:
                            job = {
                                'cpu_mean': float(chunk[0]) % 1.0 if float(chunk[0]) > 0 else 0.3,
                                'cpu_std': float(chunk[1]) % 0.5 if float(chunk[1]) > 0 else 0.1,
                                'cpu_max': min(1.0, float(chunk[2]) % 1.0) if float(chunk[2]) > 0 else 0.5,
                                'cpu_95th': min(1.0, float(chunk[3]) % 1.0) if float(chunk[3]) > 0 else 0.4,
                                'mem_mean': float(chunk[4]) % 50000 if float(chunk[4]) > 0 else 2000,
                            }
                            jobs.append(job)
                        except:
                            continue

                if jobs:
                    logger.info(f"数字提取获得 {len(jobs)} 个特征")
                    return jobs[:expected_num]
        except Exception as e:
            logger.debug(f"数字提取失败: {e}")

        return None

    def _clean_json_string(self, json_str):
        """清理JSON字符串"""
        # 替换单引号为双引号
        json_str = re.sub(r"'", '"', json_str)

        # 移除尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # 修复没有引号的键（如果可能）
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # 尝试修复 "key: value" 为 "key": value
            line = re.sub(r'(\w+)\s*:', r'"\1":', line)
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _fallback_generate_job_level(self, stats, num):
        """后备方法生成job-level特征"""
        logger.info("使用统计驱动的后备方法生成job-level特征")

        jobs = []
        base_time = int(time.time() * 1000000)

        # 从统计中获取参数
        cpu_mean_stats = stats.get('cpu_mean', {'mean': 0.3, 'std': 0.15})
        mem_mean_stats = stats.get('mem_mean', {'mean': 2000, 'std': 1500})

        cpu_mean = cpu_mean_stats.get('mean', 0.3) if isinstance(cpu_mean_stats, dict) else 0.3
        mem_mean_val = mem_mean_stats.get('mean', 2000) if isinstance(mem_mean_stats, dict) else 2000

        for i in range(num):
            # 决定作业类型
            if i < num * 0.3:  # CPU密集型
                cpu_mean_val = np.random.uniform(0.6, 0.9)
                mem_mean_val = np.random.uniform(500, 2000)
                task_count = np.random.randint(10, 50)
            elif i < num * 0.6:  # 内存密集型
                cpu_mean_val = np.random.uniform(0.1, 0.4)
                mem_mean_val = np.random.uniform(8000, 20000)
                task_count = np.random.randint(5, 20)
            elif i < num * 0.8:  # 混合型
                cpu_mean_val = np.random.uniform(0.4, 0.6)
                mem_mean_val = np.random.uniform(2000, 8000)
                task_count = np.random.randint(15, 40)
            else:  # 轻量型
                cpu_mean_val = np.random.uniform(0.05, 0.3)
                mem_mean_val = np.random.uniform(100, 1000)
                task_count = np.random.randint(1, 10)

            # 生成特征
            cpu_std = np.random.uniform(0.05, 0.2)
            cpu_max = min(1.0, cpu_mean_val + np.random.exponential(0.2))
            cpu_95th = cpu_mean_val + cpu_std * 1.645

            mem_std = mem_mean_val * np.random.uniform(0.1, 0.3)
            mem_max = mem_mean_val + mem_std * np.random.uniform(1.5, 3)

            machines_count = np.random.randint(1, 11)

            # 时间特征
            duration_sec = np.random.exponential(1800)  # 平均30分钟
            duration_sec = max(60, min(86400, duration_sec))  # 1分钟到1天

            start_time = base_time - int(duration_sec * 1000 * np.random.uniform(1, 10))
            end_time = start_time + int(duration_sec * 1000)

            # 计算派生特征
            cpu_intensity = cpu_mean_val * duration_sec
            mem_intensity = mem_mean_val * duration_sec
            task_density = task_count / duration_sec
            cpu_cv = cpu_std / (cpu_mean_val + 1e-8)
            mem_cv = mem_std / (mem_mean_val + 1e-8)

            job = {
                'cpu_mean': float(cpu_mean_val),
                'cpu_std': float(cpu_std),
                'cpu_max': float(cpu_max),
                'cpu_95th': float(cpu_95th),
                'mem_mean': float(mem_mean_val),
                'mem_std': float(mem_std),
                'mem_max': float(mem_max),
                'machines_count': int(machines_count),
                'task_count': int(task_count),
                'start_time': int(start_time),
                'end_time': int(end_time),
                'duration_sec': float(duration_sec),
                'cpu_intensity': float(cpu_intensity),
                'mem_intensity': float(mem_intensity),
                'task_density': float(task_density),
                'cpu_cv': float(cpu_cv),
                'mem_cv': float(mem_cv)
            }

            jobs.append(job)

        return jobs

    def _ensure_complete_features(self, jobs, expected_num):
        """确保每个作业都有所有必需特征"""
        if not jobs:
            return self._fallback_generate_job_level({}, expected_num)

        complete_jobs = []

        for job in jobs:
            complete_job = {}

            # 确保每个必需特征都存在
            for feat in self.required_features:
                if feat in job:
                    complete_job[feat] = job[feat]
                else:
                    # 提供默认值
                    if feat == 'cpu_mean':
                        complete_job[feat] = np.random.uniform(0.1, 0.8)
                    elif feat == 'cpu_std':
                        complete_job[feat] = np.random.uniform(0.05, 0.2)
                    elif feat == 'cpu_max':
                        complete_job[feat] = min(1.0, complete_job.get('cpu_mean', 0.3) + np.random.uniform(0.1, 0.3))
                    elif feat == 'cpu_95th':
                        complete_job[feat] = complete_job.get('cpu_mean', 0.3) + complete_job.get('cpu_std', 0.1) * 1.645
                    elif feat == 'mem_mean':
                        complete_job[feat] = np.random.uniform(500, 15000)
                    elif feat == 'mem_std':
                        complete_job[feat] = complete_job.get('mem_mean', 2000) * np.random.uniform(0.1, 0.3)
                    elif feat == 'mem_max':
                        complete_job[feat] = complete_job.get('mem_mean', 2000) + complete_job.get('mem_std', 500)
                    elif feat == 'machines_count':
                        complete_job[feat] = np.random.randint(1, 11)
                    elif feat == 'task_count':
                        complete_job[feat] = np.random.randint(1, 101)
                    elif feat == 'start_time':
                        complete_job[feat] = int(time.time() * 1000000) - np.random.randint(0, 86400000)
                    elif feat == 'end_time':
                        duration = np.random.exponential(1800) * 1000
                        complete_job[feat] = complete_job.get('start_time', int(time.time()*1000000)) + int(duration)
                    elif feat == 'duration_sec':
                        if 'start_time' in complete_job and 'end_time' in complete_job:
                            complete_job[feat] = (complete_job['end_time'] - complete_job['start_time']) / 1000.0
                        else:
                            complete_job[feat] = np.random.exponential(1800)
                    elif feat == 'cpu_intensity':
                        complete_job[feat] = complete_job.get('cpu_mean', 0.3) * complete_job.get('duration_sec', 1800)
                    elif feat == 'mem_intensity':
                        complete_job[feat] = complete_job.get('mem_mean', 2000) * complete_job.get('duration_sec', 1800)
                    elif feat == 'task_density':
                        complete_job[feat] = complete_job.get('task_count', 10) / (complete_job.get('duration_sec', 1800) + 1e-6)
                    elif feat == 'cpu_cv':
                        complete_job[feat] = complete_job.get('cpu_std', 0.1) / (complete_job.get('cpu_mean', 0.3) + 1e-8)
                    elif feat == 'mem_cv':
                        complete_job[feat] = complete_job.get('mem_std', 500) / (complete_job.get('mem_mean', 2000) + 1e-8)

            complete_jobs.append(complete_job)

        # 确保数量正确
        if len(complete_jobs) < expected_num:
            additional = self._fallback_generate_job_level({}, expected_num - len(complete_jobs))
            complete_jobs.extend(additional)

        return complete_jobs[:expected_num]

    def _ensure_feature_types(self, df):
        """确保特征类型正确"""
        for col in self.required_features:
            if col not in df.columns:
                continue

            try:
                if 'cpu' in col or col in ['cpu_cv', 'mem_cv']:
                    # CPU相关特征在0-1之间
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0.3).clip(0, 1)
                elif 'mem' in col or col in ['mem_mean', 'mem_std', 'mem_max']:
                    # 内存特征（MB）
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(2000).clip(0, 100000)
                elif col in ['task_count', 'machines_count']:
                    # 计数特征
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(10).clip(1, 1000).astype(int)
                elif col in ['duration_sec', 'cpu_intensity', 'mem_intensity', 'task_density']:
                    # 连续数值特征
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(1800).clip(0, 1000000)
                elif col in ['start_time', 'end_time']:
                    # 时间戳
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(int(time.time() * 1000000))
            except Exception as e:
                logger.warning(f"处理列 {col} 时出错: {e}")
                # 设置默认值
                if 'cpu' in col:
                    df[col] = 0.3
                elif 'mem' in col:
                    df[col] = 2000.0
                elif col in ['task_count', 'machines_count']:
                    df[col] = 10
                else:
                    df[col] = 0.0

        return df

    # 重写父类方法，确保调用正确的格式
    def generate_job(self, stats, gan_samples=None, env_state=None, num=None):
        """重写父类方法，生成job-level特征"""
        return self.generate_job_level(stats, num)