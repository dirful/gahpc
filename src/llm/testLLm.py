# diagnose_llm.py
import sys
sys.path.append('src')

from log.logger import get_logger
from config.config import Config
from llm.llm_client import LLMClient
from llm.enhanced_llm_job_gen import EnhancedLLMJobGenerator
import json

logger = get_logger(__name__)

def test_llm_response():
    """测试LLM响应"""
    cfg = Config()

    # 启用调试
    cfg.llm_debug = True

    # 创建LLM客户端
    llm_client = LLMClient(
        model=cfg.llm_model,
        host=cfg.ollama_host
    )

    # 创建生成器
    llm_gen = EnhancedLLMJobGenerator(cfg, llm_client)

    # 测试prompt
    stats = {
        'cpu_mean': {'mean': 0.3, 'std': 0.15, 'min': 0.0, 'max': 1.0, 'p50': 0.25, 'p90': 0.6},
        'mem_mean': {'mean': 2000.0, 'std': 1500.0, 'min': 10.0, 'max': 50000.0, 'p50': 1500.0, 'p90': 4000.0},
        'duration_sec': {'mean': 1800.0, 'std': 1200.0, 'min': 1.0, 'max': 86400.0, 'p50': 1500.0, 'p90': 3600.0}
    }

    # 1. 查看生成的prompt
    prompt = llm_gen._build_job_level_prompt(stats, 5)  # 只请求5个进行测试
    print("=" * 80)
    print("生成的Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    # 2. 测试LLM响应
    print("\n正在请求LLM...")
    response = llm_client.ask(prompt)

    print("=" * 80)
    print("LLM原始响应:")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # 3. 测试解析
    print("\n测试解析...")
    jobs = llm_gen._parse_job_level_response_debug(response, 5)

    if jobs:
        print(f"成功解析 {len(jobs)} 个作业:")
        for i, job in enumerate(jobs[:3]):
            print(f"\n作业 {i+1}:")
            for key, value in job.items():
                print(f"  {key}: {value}")
    else:
        print("解析失败")

        # 测试后备生成
        print("\n测试后备生成...")
        backup_jobs = llm_gen._fallback_generate_job_level(stats, 3)
        print(f"后备生成 {len(backup_jobs)} 个作业")
        print(json.dumps(backup_jobs[0], indent=2))

if __name__ == "__main__":
    test_llm_response()