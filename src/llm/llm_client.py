# src/llm/llm_client.py
from ollama import Client
from log.logger import get_logger

logger = get_logger(__name__)

class LLMClient:
    def __init__(self, model="yxchia/qwen2.5-1.5b-instruct:Q8_0", host="http://localhost:11434", **kwargs):
        self.model = model
        self.client = Client(host=host)

        # 保存可选参数
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 500)
        self.top_p = kwargs.get('top_p', 0.9)
        self.top_k = kwargs.get('top_k', 40)

        logger.info(f"LLMClient initialized: model={model}, host={host}")

    def ask(self, prompt: str, **kwargs) -> str:
        """向LLM发送请求"""
        logger.info("[LLM] prompt length=%d", len(prompt))

        # 合并参数
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        try:
            # 使用 ollama 的 chat 方法，传入可选参数
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": self.top_p,
                    "top_k": self.top_k
                },
                stream=False
            )

            # Ollama response shape: {"message": {"content": "..."}}
            content = resp.get("message", {}).get("content", "")

            if isinstance(content, dict):
                # sometimes Ollama returns structured content: join if list
                content = content.get("text", "") if "text" in content else str(content)

            logger.debug(f"LLM response length: {len(content)}")
            return content or ""

        except Exception as e:
            logger.error(f"LLM请求失败: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, prompt):
        """模拟LLM响应（用于调试）"""
        logger.warning("使用模拟LLM响应")

        # 根据prompt返回不同的模拟响应
        if "job" in prompt.lower() or "cpu" in prompt.lower():
            return '''[
              {"cpu": 0.75, "mem": 0.5, "disk_io": 0.3, "duration": 300},
              {"cpu": 0.25, "mem": 0.8, "disk_io": 0.1, "duration": 600},
              {"cpu": 0.5, "mem": 0.3, "disk_io": 0.2, "duration": 450}
            ]'''
        else:
            return "这是模拟的LLM响应"

    def generate(self, prompt, **kwargs):
        """兼容旧接口"""
        return self.ask(prompt, **kwargs)