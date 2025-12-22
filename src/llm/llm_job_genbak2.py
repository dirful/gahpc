# src/llm/llm_job_gen.py
import json
import re
import math
import random
import numpy as np
from log.logger import get_logger
from utils.tools import validate_and_fix_jobs  # keep your existing validator if present

logger = get_logger(__name__)

class LLMJobGenerator:
    def __init__(self, cfg, llm_client, gan_trainer=None):
        self.cfg = cfg
        self.llm = llm_client
        self.gan = gan_trainer

    def _build_prompt(self, stats, gan_samples, env_state, num=30):
        stats_summary = self._format_stats_for_prompt(stats)
        env_summary = self._format_env_for_prompt(env_state)
        gan_summary = self._format_gan_for_prompt(gan_samples)

        # 强约束：要求 submission_time 字段以产生重叠（用于竞争）
        prompt = f"""You are an HPC cluster job generator. Produce EXACTLY {num} distinct jobs as a JSON array.
Each job must have fields: cpu (0.0-1.0), mem (0.0-1.0), disk_io (0.0-10.0), duration (1-3600 integer), submission_time (int seconds).
Constraints:
 - At least 1 CPU-heavy job (cpu > 0.7)
 - At least 1 MEM-heavy job (mem > 0.7)
 - At least 1 short job (duration < 60)
 - At least 1 long job (duration > 600)
Competition requirement (to model resource contention):
 - Include at least 3 groups where two or more jobs have submission_time within the same 0-120 second window and the sum(cpu) > 0.9 OR sum(mem) > 0.9.

Cluster stats:
{stats_summary}

Cluster state:
{env_summary}

GAN reference (top samples):
{gan_summary}

Output: ONLY a JSON array, e.g.:
[
  {{"cpu":0.85,"mem":0.3,"disk_io":2.5,"duration":45,"submission_time":10}},
  ...
]
Return exactly {num} jobs and nothing else.
"""
        return prompt

    def _format_stats_for_prompt(self, stats):
        if not stats:
            return "No historical stats"
        out = []
        for k, v in stats.items():
            if isinstance(v, dict):
                out.append(f"- {k}: mean={v.get('mean',0):.2f}, p50={v.get('p50',0):.2f}, p90={v.get('p90',0):.2f}, count={v.get('count',0)}")
            else:
                out.append(f"- {k}: {v}")
        return "\n".join(out)

    def _format_env_for_prompt(self, env_state):
        if not env_state:
            return "Cluster state unknown"
        try:
            nodes = env_state.get('nodes', [])
            cpu_utils = [n.get('cpu_utilization', 0) for n in nodes if 'cpu_utilization' in n]
            mem_utils = [n.get('mem_utilization', 0) for n in nodes if 'mem_utilization' in n]
            avg_cpu = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0
            avg_mem = sum(mem_utils) / len(mem_utils) if mem_utils else 0
            return f"- nodes={len(nodes)}, avg_cpu={avg_cpu:.1%}, avg_mem={avg_mem:.1%}, queue_len={env_state.get('queue_length',0)}"
        except Exception as e:
            return f"Cluster parse failed: {e}"

    def _format_gan_for_prompt(self, gan_samples):
        if not gan_samples:
            return "No GAN samples"
        try:
            s = []
            for i, sample in enumerate(gan_samples[:3]):
                s.append(f" sample{i+1}: cpu={sample.get('cpu',0):.2f}, mem={sample.get('mem',0):.2f}, dur={sample.get('duration',0)}")
            return "\n".join(s)
        except Exception as e:
            return f"GAN parse failed: {e}"

    def _safe_parse_response(self, out):
        """尽力从 LLM 文本中提取 JSON array，并做清理"""
        try:
            # 提取第一个 JSON array
            start = out.find('[')
            end = out.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found")
            json_text = out[start:end]

            # 允许常见非严格 JSON 表达：替换单引号、移除尾逗号
            cleaned = json_text
            cleaned = re.sub(r"'", '"', cleaned)
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            # 移除注释
            cleaned = re.sub(r"//.*", "", cleaned)
            cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

            # 评估并清理表达式（如 math.sqrt(4)）
            cleaned = self._clean_python_expressions(cleaned)

            jobs = json.loads(cleaned)
            jobs = self._evaluate_expressions(jobs)
            return jobs
        except Exception as e:
            logger.warning(f"Safe parse failed: {e}")
            # fallback aggressive extraction
            extracted = self._extract_and_clean_json(out)
            if extracted:
                try:
                    return json.loads(extracted)
                except Exception as e2:
                    logger.error(f"Fallback parse failed: {e2}")
                    return None
            return None

    def _clean_python_expressions(self, text):
        # 简单替换 math.*(...) & numeric arithmetic expressions via regex -> evaluated
        patterns = [
            (r'math\.\w+\([^)]*\)', self._eval_math_expression),
            (r'np\.random\.\w+\([^)]*\)', self._eval_numpy_expression),
            (r'random\.\w+\([^)]*\)', self._eval_random_expression),
            (r'\b\d+\.?\d*\s*[\+\-\*/]\s*\d+\.?\d*\b', self._eval_arithmetic),
        ]
        for pattern, func in patterns:
            for m in re.finditer(pattern, text):
                try:
                    val = func(m.group(0))
                    text = text[:m.start()] + str(val) + text[m.end():]
                except Exception:
                    text = text[:m.start()] + "0.5" + text[m.end():]
        return text

    def _eval_math_expression(self, expr):
        try:
            # allow simple calls like math.sqrt(4)
            expr = expr.replace('math.', '')
            return float(eval(expr, {"__builtins__": None}, {"sqrt": math.sqrt, "exp": math.exp, "log": math.log}))
        except Exception:
            return 0.5

    def _eval_numpy_expression(self, expr):
        # basic handling of uniform/normal
        if 'uniform' in expr:
            nums = re.findall(r'[\d\.]+', expr)
            if len(nums) >= 2:
                return random.uniform(float(nums[0]), float(nums[1]))
        if 'normal' in expr:
            nums = re.findall(r'[\d\.]+', expr)
            if len(nums) >= 2:
                return random.gauss(float(nums[0]), float(nums[1]))
        return random.random()

    def _eval_random_expression(self, expr):
        if 'random' in expr:
            return random.random()
        return 0.5

    def _eval_arithmetic(self, expr):
        try:
            # use ast for safety
            import ast
            node = ast.parse(expr, mode='eval')
            return float(eval(compile(node, '<string>', 'eval'), {"__builtins__":None}, {}))
        except Exception:
            try:
                return float(eval(expr, {"__builtins__": None}, {}))
            except:
                return 0.5

    def _extract_and_clean_json(self, text):
        pattern = r'\[\s*\{[^}]+\}\s*(?:,\s*\{[^}]+\}\s*)*\]'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            json_str = matches[0]
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
            json_str = re.sub(r"'", '"', json_str)
            return json_str
        return None

    def _evaluate_expressions(self, jobs):
        """递归将字符串表达式转换为数值，并 clamp 到合法范围"""
        def _process(obj):
            if isinstance(obj, list):
                return [_process(x) for x in obj]
            if isinstance(obj, dict):
                res = {}
                for k, v in obj.items():
                    if isinstance(v, (int, float)):
                        res[k] = float(v)
                    elif isinstance(v, str):
                        # try numeric
                        try:
                            res[k] = float(v)
                            continue
                        except:
                            pass
                        # try arithmetic eval
                        try:
                            res[k] = float(self._eval_arithmetic(v))
                            continue
                        except:
                            res[k] = 0.0
                    else:
                        res[k] = float(v) if v else 0.0
                return res
            else:
                return float(jobs) if jobs else 0.0
        try:
            out = _process(jobs)
            return out
        except Exception as e:
            logger.error(f"Evaluate expressions failed: {e}")
            return jobs

    def _clamp_and_fill(self, job):
        """保障字段并 clamp 范围"""
        job = dict(job) if isinstance(job, dict) else {}
        cpu = float(job.get('cpu', 0.0))
        mem = float(job.get('mem', 0.0))
        disk_io = float(job.get('disk_io', 0.0))
        duration = int(job.get('duration', 1))
        submission_time = int(job.get('submission_time', 0))

        cpu = max(0.0, min(1.0, cpu))
        mem = max(0.0, min(1.0, mem))
        disk_io = max(0.0, min(10.0, disk_io))
        duration = max(1, min(3600, duration))
        submission_time = max(0, submission_time)

        return {'cpu': cpu, 'mem': mem, 'disk_io': disk_io, 'duration': duration, 'submission_time': submission_time}

    def _generate_competition_scenarios(self, stats, k=6):
        """基于统计信息生成确保竞争的样本（用于补充/替换）"""
        scn = []
        for group in range(2):  # create two contention groups
            base = random.randint(0, 120)
            group_size = k // 2
            for i in range(group_size):
                cpu = float(np.clip(np.random.normal(0.6 + 0.1*group, 0.15), 0.0, 1.0))
                mem = float(np.clip(np.random.normal(0.5 + 0.1*group, 0.12), 0.0, 1.0))
                dur = int(max(1, np.random.normal(stats.get('duration',{}).get('mean',300), 100)))
                scn.append({'cpu': cpu, 'mem': mem, 'disk_io': random.uniform(0,5), 'duration': dur, 'submission_time': base + random.randint(0,60)})
        return scn

    def _check_competition(self, jobs, window=120, cpu_thresh=0.9, mem_thresh=0.9, min_groups=3):
        """检测是否存在足够的竞争组（两个或以上作业在 window 内且 sum(cpu) > cpu_thresh 或 sum(mem) > mem_thresh）"""
        if not jobs:
            return False
        jobs_sorted = sorted(jobs, key=lambda j: j['submission_time'])
        n = len(jobs_sorted)
        groups = 0
        for i in range(n):
            base = jobs_sorted[i]['submission_time']
            window_jobs = [j for j in jobs_sorted if base <= j['submission_time'] <= base + window]
            if len(window_jobs) >= 2:
                sum_cpu = sum([j['cpu'] for j in window_jobs])
                sum_mem = sum([j['mem'] for j in window_jobs])
                if sum_cpu > cpu_thresh or sum_mem > mem_thresh:
                    groups += 1
            if groups >= min_groups:
                return True
        return False

    def _ensure_competition(self, jobs, stats, needed_groups=3):
        """如果 LLM 生成不满足竞争要求，混入一些对抗竞争样本"""
        if self._check_competition(jobs, min_groups=needed_groups):
            return jobs
        scn = self._generate_competition_scenarios(stats, k=6)
        # replace last few jobs with scenarios
        result = jobs[:max(0, len(jobs)-len(scn))] + scn
        # if still not enough, append more
        if not self._check_competition(result, min_groups=needed_groups):
            result.extend(self._generate_competition_scenarios(stats, k=6))
        # clamp all jobs
        return [self._clamp_and_fill(j) for j in result][:len(jobs)]

    def generate_job(self, stats, gan_samples=None, env_state=None, num=None):
        num = num or getattr(self.cfg, 'llm_num_generate', 30)

        prompt = self._build_prompt(stats, gan_samples, env_state, num)
        logger.info(f"LLM prompt length: {len(prompt)}")
        logger.debug(f"LLM prompt (head): {prompt[:800]}")

        out = self.llm.ask(prompt)
        logger.debug(f"LLM raw output head: {out[:1000]}")

        jobs = self._safe_parse_response(out)
        if not jobs or not isinstance(jobs, list) or len(jobs) != num:
            logger.warning(f"LLM parse failed or count mismatch: expected {num}, got {len(jobs) if jobs else 0}")
            jobs = self._fallback_generation(stats, gan_samples, env_state, num, out)

        # normalize/validate/clamp
        jobs = [self._clamp_and_fill(j) for j in jobs]
        # ensure diversity & competition if configured
        if getattr(self.cfg, 'require_competition_in_generation', True):
            jobs = self._ensure_competition(jobs, stats, needed_groups=getattr(self.cfg, 'competition_groups', 3))

        # final validation via project util
        try:
            jobs = validate_and_fix_jobs(jobs, self.cfg, stats)
        except Exception as e:
            logger.error(f"validate_and_fix_jobs failed: {e}")
            jobs = self._create_diverse_default_jobs(num)

        logger.info(f"Generated {len(jobs)} jobs")
        for i, j in enumerate(jobs[:3]):
            logger.info(f" Job{i+1}: cpu={j['cpu']:.2f}, mem={j['mem']:.2f}, dur={j['duration']:.0f}, sub={j['submission_time']}")

        return jobs

    def _fallback_generation(self, stats, gan_samples, env_state, num, llm_output):
        """多策略后备生成：抽数字、基于统计生成、GAN样本、对抗样本"""
        jobs = []

        # Extract numbers heuristic
        try:
            nums = re.findall(r'\b\d+\.?\d*\b', llm_output)
            if len(nums) >= num*4:
                for i in range(num):
                    a = float(nums[i*4]) % 1.0
                    b = float(nums[i*4+1]) % 1.0
                    c = float(nums[i*4+2]) % 10.0
                    d = int(float(nums[i*4+3]) % 3600)
                    st = random.randint(0, 300)
                    jobs.append({'cpu':a, 'mem':b, 'disk_io':c, 'duration':d or 1, 'submission_time':st})
                if jobs:
                    logger.info(f"Extracted {len(jobs)} jobs from LLM numbers")
                    return jobs
        except Exception as e:
            logger.warning(f"Numeric extraction failed: {e}")

        # From stats
        try:
            jobs = self._generate_from_stats(stats, num)
            if jobs:
                logger.info("Generated from stats fallback")
                return jobs
        except Exception as e:
            logger.warning(f"Generate from stats failed: {e}")

        # Use GAN samples if available
        if gan_samples and len(gan_samples) >= num:
            sel = random.sample(gan_samples, num)
            jobs = []
            for s in sel:
                jobs.append({'cpu': float(s.get('cpu',0)), 'mem': float(s.get('mem',0)), 'disk_io': float(s.get('disk_io',0)),
                             'duration': int(s.get('duration',1)), 'submission_time': random.randint(0,300)})
            logger.info("Used GAN samples as fallback")
            return jobs

        # Finally create defaults
        return self._create_diverse_default_jobs(num)

    def _generate_from_stats(self, stats, num):
        jobs = []
        cpu_mean = stats.get('cpu', {}).get('mean', 0.5) if isinstance(stats.get('cpu'), dict) else 0.5
        cpu_std = stats.get('cpu', {}).get('std', 0.2) if isinstance(stats.get('cpu'), dict) else 0.2
        mem_mean = stats.get('mem', {}).get('mean', 0.3) if isinstance(stats.get('mem'), dict) else 0.3
        mem_std = stats.get('mem', {}).get('std', 0.15) if isinstance(stats.get('mem'), dict) else 0.15
        dur_mean = stats.get('duration', {}).get('mean', 300) if isinstance(stats.get('duration'), dict) else 300
        dur_std = stats.get('duration', {}).get('std', 200) if isinstance(stats.get('duration'), dict) else 200

        for i in range(num):
            job = {
                'cpu': float(np.clip(np.random.normal(cpu_mean, cpu_std), 0.0, 1.0)),
                'mem': float(np.clip(np.random.normal(mem_mean, mem_std), 0.0, 1.0)),
                'disk_io': float(random.uniform(0.0, 5.0)),
                'duration': int(max(1, np.random.normal(dur_mean, dur_std))),
                'submission_time': random.randint(0, 600)
            }
            if i == 0:
                job['cpu'] = random.uniform(0.7, 1.0)
            elif i == 1:
                job['mem'] = random.uniform(0.7, 1.0)
            elif i == 2:
                job['duration'] = random.randint(1, 60)
            elif i == 3:
                job['duration'] = random.randint(600, 3600)
            jobs.append(job)
        return jobs

    def _create_diverse_default_jobs(self, num):
        jobs = []
        templates = [
            {'cpu_mean':0.7,'cpu_std':0.1,'mem_mean':0.2,'mem_std':0.05,'duration_mean':300,'duration_std':100},
            {'cpu_mean':0.2,'cpu_std':0.05,'mem_mean':0.7,'mem_std':0.1,'duration_mean':600,'duration_std':200},
            {'cpu_mean':0.5,'cpu_std':0.1,'mem_mean':0.5,'mem_std':0.1,'duration_mean':1800,'duration_std':600},
            {'cpu_mean':0.8,'cpu_std':0.1,'mem_mean':0.8,'mem_std':0.1,'duration_mean':7200,'duration_std':1800},
            {'cpu_mean':0.9,'cpu_std':0.05,'mem_mean':0.3,'mem_std':0.05,'duration_mean':60,'duration_std':30},
        ]
        job_id = 0
        while len(jobs) < num:
            t = random.choice(templates)
            cpu = float(np.clip(np.random.normal(t['cpu_mean'], t['cpu_std']), 0.1, 1.0))
            mem = float(np.clip(np.random.normal(t['mem_mean'], t['mem_std']), 0.1, 1.0))
            dur = int(max(10, np.random.normal(t['duration_mean'], t['duration_std'])))
            jobs.append({'job_id': f'default_{job_id}', 'cpu': cpu, 'mem': mem, 'duration': dur,
                         'disk_io': random.uniform(0.1, 3.0), 'submission_time': job_id * 5, 'source':'default'})
            job_id += 1
        return jobs
