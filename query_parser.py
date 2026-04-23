# test_gemini_parser.py
import json
from typing import Dict

# 新版 Gemini SDK
from gemini_llm import GeminiLLM
from retrieval_planner import RetrievalPlanner


# ---------- 2. QueryParser（你的原有逻辑，稍加增强 JSON 提取）----------
class QueryParser:
    def __init__(self, llm):
        self.llm = llm

    def parse(self, query: str) -> Dict:
        prompt = f"""
You are a query parser.

Task:
Convert user query into structured intent and sub-queries.

Rules:
- DO NOT answer the question
- Only extract intent and search directions
- Keep output STRICT JSON
- sub_queries <= 5

Query:
{query}

Output format:
{{
  "intent": "...",
  "sub_queries": ["...", "..."]
}}
"""
        response = self.llm(prompt)
        parsed = self._safe_parse(response)
        return self._validate(parsed)

    def _safe_parse(self, text: str) -> Dict:
        """鲁棒解析：自动提取 JSON 块"""
        import json

        # 如果模型返回了 markdown 代码块，提取其中的 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text)
        except Exception as e:
            print(f"Parsing JSON e: {e}")
            # 降级：把整个响应当成 sub_queries 的一部分
            return {"intent": "unknown", "sub_queries": [text[:100]]}

    def _validate(self, data: Dict) -> Dict:
        intent = data.get("intent", "unknown")
        sub_queries = data.get("sub_queries", [])
        if not isinstance(sub_queries, list):
            sub_queries = [str(sub_queries)]
        sub_queries = sub_queries[:5]  # 限制长度
        return {"intent": intent, "sub_queries": sub_queries}
