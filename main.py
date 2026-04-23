# 新版 Gemini SDK
import json
from gemini_llm import GeminiLLM
from query_parser import QueryParser
from retrieval_planner import RetrievalPlanner


# ---------- 3. main 函数：交互式测试 ----------
def main():
    print("🔍 初始化 Gemini 客户端...")
    try:
        # 你可以更换模型，例如 "gemini-2.0-flash" 或 "gemini-3-flash-preview"
        llm = GeminiLLM(model_name="gemini-3.1-flash-lite-preview")
        parser = QueryParser(llm)
        planner = RetrievalPlanner(llm)

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print("✅ 就绪！输入查询（输入 exit 退出）")
    while True:
        query = input("\n📝 请输入查询: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        try:
            parsed = parser.parse(query)
            plans = planner.plan(parsed)

            print("\n[parse]")
            # print(json.dumps(parsed, indent=2, ensure_ascii=False))
            print(parsed)

            print("\n[plans]")
            print(json.dumps(plans, indent=2, ensure_ascii=False))
            # print(plans)

        except Exception as e:
            print(f"❌ 解析出错: {e}")


if __name__ == "__main__":
    main()
