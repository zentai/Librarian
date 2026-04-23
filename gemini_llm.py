import os
from google import genai
from dotenv import load_dotenv


# ---------- 1. Gemini LLM 适配器（兼容 llm(prompt) -> str）----------
class GeminiLLM:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        自动从环境变量 GEMINI_API_KEY 读取密钥。
        如果没有设置，尝试加载 .env 文件。
        """
        load_dotenv()  # 加载 .env 中的变量
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 GEMINI_API_KEY，请设置环境变量或在 .env 文件中提供"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, prompt: str) -> str:
        """调用 Gemini 并返回文本响应"""
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text
