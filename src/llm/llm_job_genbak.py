# src/llm/llm_job_gen.py
import json
import math  # 添加 math 模块导入
import re
from log.logger import get_logger
from utils.tools import validate_and_fix_jobs

logger = get_logger(__name__)

class LLMJobGenerator:
    def __init__(self, cfg, llm_client, gan_trainer=None):
        self.cfg = cfg
        self.llm = llm_client
        self.gan = gan_trainer

    # 在llm_job_gen.py中修改_build_prompt方法
    def _build_prompt(self, stats, gan_samples, env_state, num=30):
        """构建更有效的提示"""

        # 提取关键的统计信息用于提示
        stats_summary = self._format_stats_for_prompt(stats)

        # 提取环境状态的关键信息
        env_summary = self._format_env_for_prompt(env_state)

        # 提取GAN样本的特征
        gan_summary = self._format_gan_for_prompt(gan_samples)

        prompt = f"""你是一个HPC集群调度专家。请基于以下信息生成{num}个不同的作业。
    
    ## 历史作业统计信息
    {stats_summary}
    
    ## 当前集群状态
    {env_summary}
    
    ## GAN生成的参考样本
    {gan_summary}
    
    ## 生成要求
    1. 生成 EXACTLY {num} 个不同的作业
    2. 每个作业必须有4个属性：cpu, mem, disk_io, duration
    3. 数值范围：
       - cpu: 0.0到1.0之间的浮点数（CPU使用率）
       - mem: 0.0到1.0之间的浮点数（内存使用率）
       - disk_io: 0.0到10.0之间的浮点数（磁盘IO）
       - duration: 1到3600之间的整数（运行时间，秒）
    
    4. 多样性要求：
       - 至少有一个CPU密集型作业（cpu > 0.7）
       - 至少有一个内存密集型作业（mem > 0.7）
       - 至少有一个短作业（duration < 60）
       - 至少有一个长作业（duration > 600）
    
    5. 考虑当前集群负载：
       - 如果集群负载高（节点利用率 > 70%），生成较小的作业
       - 如果集群负载低（节点利用率 < 30%），生成较大的作业
    
    6. 输出格式：必须是有效的JSON数组，例如：
    [
      {{"cpu": 0.85, "mem": 0.3, "disk_io": 2.5, "duration": 45}},
      {{"cpu": 0.25, "mem": 0.9, "disk_io": 1.2, "duration": 720}},
      {{"cpu": 0.5, "mem": 0.5, "disk_io": 0.8, "duration": 300}}
    ]
    
    请确保每个作业都不同，并且符合上述要求。
    """
        return prompt

    def _format_stats_for_prompt(self, stats):
        """格式化统计信息用于提示"""
        if not stats:
            return "无历史统计信息可用"

        summary = []
        for key, values in stats.items():
            if isinstance(values, dict):
                # 只显示关键统计量
                summary.append(f"- {key}: 均值={values.get('mean', 0):.2f}, "
                               f"标准差={values.get('std', 0):.2f}, "
                               f"范围=[{values.get('min', 0):.2f}, {values.get('max', 0):.2f}]")
            else:
                summary.append(f"- {key}: {values}")

        return "\n".join(summary)

    def _format_env_for_prompt(self, env_state):
        """格式化环境状态用于提示"""
        if not env_state:
            return "集群状态未知"

        try:
            # 提取关键信息
            nodes_info = env_state.get('nodes', [])
            total_nodes = len(nodes_info)

            # 计算平均利用率
            cpu_utils = [n.get('cpu_utilization', 0) for n in nodes_info if 'cpu_utilization' in n]
            mem_utils = [n.get('mem_utilization', 0) for n in nodes_info if 'mem_utilization' in n]

            avg_cpu_util = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0
            avg_mem_util = sum(mem_utils) / len(mem_utils) if mem_utils else 0

            queue_length = env_state.get('queue_length', 0)
            completed_jobs = env_state.get('jobs_completed', 0)

            return (f"- 集群节点数: {total_nodes}\n"
                    f"- 平均CPU利用率: {avg_cpu_util:.1%}\n"
                    f"- 平均内存利用率: {avg_mem_util:.1%}\n"
                    f"- 当前队列长度: {queue_length}\n"
                    f"- 已完成作业数: {completed_jobs}")
        except Exception as e:
            return f"集群状态解析失败: {str(e)}"

    def _format_gan_for_prompt(self, gan_samples):
        """格式化GAN样本用于提示"""
        if not gan_samples:
            return "无GAN参考样本"

        try:
            # 只显示前3个样本
            display_samples = gan_samples[:3]
            samples_str = []
            for i, sample in enumerate(display_samples):
                samples_str.append(f"  样本{i+1}: cpu={sample.get('cpu', 0):.2f}, "
                                   f"mem={sample.get('mem', 0):.2f}, "
                                   f"duration={sample.get('duration', 0):.0f}s")

            return "GAN生成的参考样本:\n" + "\n".join(samples_str)
        except Exception as e:
            return f"GAN样本解析失败: {str(e)}"

    def _safe_parse_response(self, out):
        """安全解析 LLM 响应，处理可能的 Python 表达式"""
        try:
            # 尝试直接解析 JSON
            start = out.find('[')
            end = out.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found")

            json_text = out[start:end]

            # 清理 JSON 文本中的 Python 表达式
            cleaned_text = self._clean_python_expressions(json_text)

            # 解析 JSON
            jobs = json.loads(cleaned_text)

            # 进一步处理：评估任何合法的表达式
            jobs = self._evaluate_expressions(jobs)

            return jobs

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            logger.debug(f"原始响应: {out}")
            # 尝试更激进的清理
            cleaned = self._extract_and_clean_json(out)
            if cleaned:
                try:
                    return json.loads(cleaned)
                except:
                    pass
            return None
        except Exception as e:
            logger.error(f"解析失败: {e}")
            return None

    def _clean_python_expressions(self, text):
        """清理文本中的 Python 表达式"""
        # 移除常见的 Python 表达式
        patterns = [
            (r'math\.\w+\([^)]*\)', self._eval_math_expression),  # math.sqrt(0.5)
            (r'np\.random\.\w+\([^)]*\)', self._eval_numpy_expression),  # np.random.uniform(0,1)
            (r'random\.\w+\([^)]*\)', self._eval_random_expression),  # random.random()
            (r'\b\d+\.?\d*\s*[\+\-\*/]\s*\d+\.?\d*\b', self._eval_arithmetic),  # 0.5 * 2
        ]

        for pattern, eval_func in patterns:
            # 找到所有匹配
            matches = list(re.finditer(pattern, text))
            for match in reversed(matches):  # 反向替换以避免位置偏移
                expr = match.group(0)
                try:
                    result = eval_func(expr)
                    text = text[:match.start()] + str(result) + text[match.end():]
                except Exception:
                    # 如果无法评估，替换为默认值
                    text = text[:match.start()] + "0.5" + text[match.end():]

        return text

    def _eval_math_expression(self, expr):
        """安全地评估 math 表达式"""
        # 限制可用的 math 函数
        safe_dict = {
            'math': math,
            'sqrt': math.sqrt,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }

        # 移除 math. 前缀，直接评估
        if expr.startswith('math.'):
            # 尝试直接计算常见的 math 函数
            if 'sqrt(' in expr:
                val = float(re.search(r'sqrt\(([^)]+)\)', expr).group(1))
                return math.sqrt(val)
            elif 'exp(' in expr:
                val = float(re.search(r'exp\(([^)]+)\)', expr).group(1))
                return math.exp(val)
            elif 'log(' in expr:
                val = float(re.search(r'log\(([^)]+)\)', expr).group(1))
                return math.log(val)

        # 回退方案：使用 eval 但限制命名空间
        try:
            return eval(expr, {"__builtins__": None}, safe_dict)
        except:
            return 0.5  # 默认值

    def _eval_numpy_expression(self, expr):
        """安全地评估 numpy 表达式"""
        # 如果不需要 numpy，返回随机值
        import random
        if 'uniform' in expr:
            # 解析 np.random.uniform(a, b)
            match = re.search(r'uniform\(([^,]+),\s*([^)]+)\)', expr)
            if match:
                low = float(match.group(1))
                high = float(match.group(2))
                return random.uniform(low, high)
        elif 'random' in expr and '()' in expr:
            return random.random()
        elif 'normal' in expr:
            match = re.search(r'normal\(([^,]+),\s*([^)]+)\)', expr)
            if match:
                mean = float(match.group(1))
                std = float(match.group(2))
                return random.gauss(mean, std)

        return random.random()

    def _eval_random_expression(self, expr):
        """安全地评估 random 表达式"""
        import random
        if expr == 'random.random()' or expr == 'random()':
            return random.random()
        elif 'uniform' in expr:
            match = re.search(r'uniform\(([^,]+),\s*([^)]+)\)', expr)
            if match:
                low = float(match.group(1))
                high = float(match.group(2))
                return random.uniform(low, high)
        return random.random()

    def _eval_arithmetic(self, expr):
        """安全地评估算术表达式"""
        try:
            # 使用 ast 安全评估
            import ast
            # 限制操作符
            node = ast.parse(expr, mode='eval')
            if not all(isinstance(n, (ast.Num, ast.BinOp, ast.UnaryOp)) for n in ast.walk(node)):
                raise ValueError("Unsafe expression")

            # 创建安全的环境
            safe_dict = {'__builtins__': None}
            code = compile(node, '<string>', 'eval')
            result = eval(code, safe_dict, {})
            return float(result)
        except:
            # 如果失败，尝试直接计算
            try:
                return float(eval(expr, {"__builtins__": None}, {}))
            except:
                return 0.5

    def _extract_and_clean_json(self, text):
        """从文本中提取和清理 JSON"""
        # 尝试提取 JSON 数组
        pattern = r'\[\s*\{[^}]+\}\s*(?:,\s*\{[^}]+\}\s*)*\]'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # 使用第一个匹配
            json_str = matches[0]

            # 清理：移除注释、尾随逗号等
            json_str = re.sub(r'//.*', '', json_str)  # 移除单行注释
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # 移除多行注释
            json_str = re.sub(r',\s*}', '}', json_str)  # 移除尾随逗号
            json_str = re.sub(r',\s*]', ']', json_str)  # 移除尾随逗号

            return json_str

        return None

    # 在llm_job_gen.py中修改_evaluate_expressions方法
    def _evaluate_expressions(self, jobs):
        """递归处理 job 数据中的表达式 - 修复math未定义"""
        try:
            # 确保math模块已导入
            import math

            if isinstance(jobs, list):
                return [self._evaluate_expressions(job) for job in jobs]
            elif isinstance(jobs, dict):
                result = {}
                for key, value in jobs.items():
                    if isinstance(value, (int, float)):
                        result[key] = float(value)
                    elif isinstance(value, str):
                        # 尝试评估字符串表达式
                        try:
                            # 检查是否包含数学表达式
                            if any(op in value for op in ['math.', 'sqrt', 'exp', 'log']):
                                # 安全地评估数学表达式
                                safe_dict = {'math': math}
                                # 移除危险函数
                                safe_dict['__builtins__'] = None
                                result[key] = float(eval(value, safe_dict, {}))
                            elif any(op in value for op in ['+', '-', '*', '/', '(', ')']):
                                # 安全评估算术表达式
                                result[key] = self._eval_arithmetic(value)
                            else:
                                result[key] = float(value)
                        except Exception as e:
                            logger.warning(f"表达式评估失败 {key}={value}: {e}")
                            result[key] = 0.0
                    else:
                        result[key] = float(value) if value else 0.0
                return result
            else:
                return float(jobs) if jobs else 0.0
        except Exception as e:
            logger.error(f"表达式评估失败: {e}")
            return jobs  # 返回原始值

    def generate_job(self, stats, gan_samples=None, env_state=None, num=None):
        num = num or self.cfg.llm_num_generate

        # 构建提示
        prompt = self._build_prompt(stats, gan_samples, env_state, num)

        logger.info(f"LLM提示长度: {len(prompt)}")
        logger.debug(f"LLM提示内容:\n{prompt[:500]}...")

        # 获取LLM响应
        out = self.llm.ask(prompt)
        logger.debug(f"LLM原始响应:\n{out[:1000]}...")

        # 尝试解析响应
        jobs = self._safe_parse_response(out)

        # 如果解析失败或作业数量不对，尝试多种方法
        if not jobs or len(jobs) != num:
            logger.warning(f"LLM解析失败或数量不对，期望{num}个，得到{len(jobs) if jobs else 0}个")
            jobs = self._fallback_generation(stats, gan_samples, env_state, num, out)

        # 确保作业多样性
        jobs = self._ensure_diversity(jobs, num)

        # 验证和修复
        try:
            jobs = validate_and_fix_jobs(jobs, self.cfg, stats)
        except Exception as e:
            logger.error(f"作业验证失败: {e}")
            jobs = self._create_default_jobs(num)

        logger.info(f"最终生成 {len(jobs)} 个作业")
        for i, job in enumerate(jobs[:3]):  # 只显示前3个
            logger.info(f"  作业{i+1}: cpu={job.get('cpu', 0):.2f}, "
                        f"mem={job.get('mem', 0):.2f}, "
                        f"duration={job.get('duration', 0):.0f}s")

        return jobs

    def _fallback_generation(self, stats, gan_samples, env_state, num, llm_output):
        """后备生成方法"""
        jobs = []

        # 方法1: 尝试从LLM输出中提取数字
        try:
            import re
            # 查找所有数字
            numbers = re.findall(r'\b\d+\.?\d*\b', llm_output)
            if len(numbers) >= num * 4:  # 每个作业需要4个数字
                for i in range(num):
                    if i * 4 + 3 < len(numbers):
                        job = {
                            'cpu': float(numbers[i*4]) % 1.0,
                            'mem': float(numbers[i*4+1]) % 1.0,
                            'disk_io': float(numbers[i*4+2]) % 10.0,
                            'duration': float(numbers[i*4+3]) % 3600
                        }
                        jobs.append(job)
                if jobs:
                    logger.info(f"从LLM输出中提取了 {len(jobs)} 个作业")
                    return jobs
        except Exception as e:
            logger.warning(f"从LLM输出提取失败: {e}")

        # 方法2: 基于统计信息生成
        try:
            jobs = self._generate_from_stats(stats, num)
            if jobs:
                logger.info(f"基于统计信息生成了 {len(jobs)} 个作业")
                return jobs
        except Exception as e:
            logger.warning(f"基于统计信息生成失败: {e}")

        # 方法3: 使用GAN样本
        if gan_samples and len(gan_samples) >= num:
            # 从GAN样本中选择
            import random
            jobs = random.sample(gan_samples, min(num, len(gan_samples)))
            logger.info(f"从GAN样本中选择了 {len(jobs)} 个作业")
            return jobs

        # 方法4: 创建默认但不同的作业
        return self._create_diverse_default_jobs(num)

    def _generate_from_stats(self, stats, num):
        """基于统计信息生成作业"""
        jobs = []

        # 从统计信息中提取关键值
        cpu_mean = stats.get('cpu', {}).get('mean', 0.5) if isinstance(stats.get('cpu'), dict) else 0.5
        cpu_std = stats.get('cpu', {}).get('std', 0.2) if isinstance(stats.get('cpu'), dict) else 0.2
        mem_mean = stats.get('mem', {}).get('mean', 0.3) if isinstance(stats.get('mem'), dict) else 0.3
        mem_std = stats.get('mem', {}).get('std', 0.15) if isinstance(stats.get('mem'), dict) else 0.15
        dur_mean = stats.get('duration', {}).get('mean', 300) if isinstance(stats.get('duration'), dict) else 300
        dur_std = stats.get('duration', {}).get('std', 200) if isinstance(stats.get('duration'), dict) else 200

        import random
        import numpy as np

        for i in range(num):
            # 基于统计分布生成
            job = {
                'cpu': np.clip(np.random.normal(cpu_mean, cpu_std), 0.0, 1.0),
                'mem': np.clip(np.random.normal(mem_mean, mem_std), 0.0, 1.0),
                'disk_io': random.uniform(0.0, 5.0),
                'duration': max(1, int(np.random.normal(dur_mean, dur_std)))
            }

            # 确保多样性
            if i == 0:  # CPU密集型
                job['cpu'] = random.uniform(0.7, 1.0)
            elif i == 1:  # 内存密集型
                job['mem'] = random.uniform(0.7, 1.0)
            elif i == 2:  # 短作业
                job['duration'] = random.randint(1, 60)
            elif i == 3:  # 长作业
                job['duration'] = random.randint(600, 3600)

            jobs.append(job)

        return jobs

    def _ensure_diversity(self, jobs, target_num):
        """确保生成的作业具有足够的多样性"""
        if len(jobs) < 2:
            return jobs

        # 计算作业之间的相似度
        diverse_jobs = []

        for job in jobs:
            if not diverse_jobs:
                diverse_jobs.append(job)
                continue

            # 检查与现有作业的相似度
            is_similar = False
            for existing_job in diverse_jobs:
                similarity = self._calculate_job_similarity(job, existing_job)
                if similarity > 0.7:  # 相似度阈值
                    is_similar = True
                    break

            if not is_similar or len(diverse_jobs) < target_num // 2:
                diverse_jobs.append(job)

        # 如果多样性不足，生成一些新的作业
        if len(diverse_jobs) < target_num:
            needed = target_num - len(diverse_jobs)
            new_jobs = self._create_diverse_default_jobs(needed)
            diverse_jobs.extend(new_jobs)

        return diverse_jobs[:target_num]

    def _calculate_job_similarity(self, job1, job2):
        """计算两个作业的相似度"""
        # 基于关键特征的欧几里得距离
        features1 = [
            job1.get('cpu', 0),
            job1.get('mem', 0),
            job1.get('duration', 0) / 3600  # 归一化到小时
        ]

        features2 = [
            job2.get('cpu', 0),
            job2.get('mem', 0),
            job2.get('duration', 0) / 3600
        ]

        # 计算相似度（距离越小，相似度越高）
        distance = sum((a - b) ** 2 for a, b in zip(features1, features2)) ** 0.5

        # 将距离转换为相似度 (0-1)
        max_distance = 1.5  # 最大可能距离
        similarity = 1.0 - min(distance / max_distance, 1.0)

        return similarity

    def _create_diverse_default_jobs(self, num_jobs):
        """创建多样化的默认作业"""
        import random
        import numpy as np

        jobs = []

        # 定义不同的作业模板
        templates = [
            # 小型计算密集型
            {'cpu_mean': 0.7, 'cpu_std': 0.1, 'mem_mean': 0.2, 'mem_std': 0.05, 'duration_mean': 300, 'duration_std': 100},
            # 小型内存密集型
            {'cpu_mean': 0.2, 'cpu_std': 0.05, 'mem_mean': 0.7, 'mem_std': 0.1, 'duration_mean': 600, 'duration_std': 200},
            # 中型平衡型
            {'cpu_mean': 0.5, 'cpu_std': 0.1, 'mem_mean': 0.5, 'mem_std': 0.1, 'duration_mean': 1800, 'duration_std': 600},
            # 大型长期运行
            {'cpu_mean': 0.8, 'cpu_std': 0.1, 'mem_mean': 0.8, 'mem_std': 0.1, 'duration_mean': 7200, 'duration_std': 1800},
            # 短时突发型
            {'cpu_mean': 0.9, 'cpu_std': 0.05, 'mem_mean': 0.3, 'mem_std': 0.05, 'duration_mean': 60, 'duration_std': 30},
        ]

        # 确保每种模板至少有一个作业
        num_templates = len(templates)
        jobs_per_template = max(1, num_jobs // num_templates)

        job_id = 0
        for i in range(num_templates):
            template = templates[i]
            for j in range(jobs_per_template):
                if len(jobs) >= num_jobs:
                    break

                job = {
                    'job_id': f'default_job_{job_id}',
                    'cpu': np.random.normal(template['cpu_mean'], template['cpu_std']),
                    'mem': np.random.normal(template['mem_mean'], template['mem_std']),
                    'duration': np.random.normal(template['duration_mean'], template['duration_std']),
                    'priority': random.randint(1, 5),
                    'submission_time': job_id * 5,
                    'source': 'default_generator'
                }

                # 确保值在合理范围内
                job['cpu'] = max(0.1, min(1.0, job['cpu']))
                job['mem'] = max(0.1, min(1.0, job['mem']))
                job['duration'] = max(10, job['duration'])

                jobs.append(job)
                job_id += 1

        # 如果还需要更多作业，使用随机模板
        while len(jobs) < num_jobs:
            template = random.choice(templates)
            job = {
                'job_id': f'default_job_{job_id}',
                'cpu': np.random.normal(template['cpu_mean'], template['cpu_std']),
                'mem': np.random.normal(template['mem_mean'], template['mem_std']),
                'duration': np.random.normal(template['duration_mean'], template['duration_std']),
                'priority': random.randint(1, 5),
                'submission_time': job_id * 5,
                'source': 'default_generator'
            }

            # 确保值在合理范围内
            job['cpu'] = max(0.1, min(1.0, job['cpu']))
            job['mem'] = max(0.1, min(1.0, job['mem']))
            job['duration'] = max(10, job['duration'])

            jobs.append(job)
            job_id += 1

        return jobs

    def _create_default_jobs(self, num):
        """简单的创建默认作业（用于验证失败时）"""
        import random

        jobs = []
        for i in range(num):
            jobs.append({
                'cpu': random.uniform(0.2, 0.8),
                'mem': random.uniform(0.2, 0.8),
                'duration': random.randint(60, 1800),
                'disk_io': random.uniform(0.5, 3.0)
            })
        return jobs