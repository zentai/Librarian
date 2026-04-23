# ---------- 3. RetrievalPlanner ----------
from typing import List
from typing import Dict
import json
from gemini_llm import GeminiLLM


class RetrievalPlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, parsed_query: Dict) -> Dict:
        print(f"[PLAN] candidates, {parsed_query}")
        candidates = self._generate_candidates(parsed_query)
        print(f"[PLAN] scored")
        scored = self._score_plans(candidates, parsed_query)
        print(f"[PLAN] selected")
        selected = self._select_top_k(scored, k=3)

        return {
            "plans": scored,
            "selected_plans": selected,
        }

    # --- LM: generate candidates ---
    def _generate_candidates(self, parsed_query: Dict) -> List[Dict]:
        prompt = f"""
You are a retrieval planner.

Your ONLY job:
Generate structured retrieval plans that can be executed on a database.

You are NOT allowed to:
- analyze
- explain
- summarize
- answer the question
- introduce unrelated entities, companies, products, or industries
- generate abstract or research-style queries
- invent tables, datasets, or schema names outside the allowed list

OUTPUT REQUIREMENTS (MANDATORY):

1. Generate EXACTLY 3 plans
2. Each plan MUST have a DIFFERENT "type"
3. Allowed types ONLY:
   - "events"
   - "aggregate"
   - "similar"

4. plan_id MUST be:
   - "P1", "P2", "P3"

5. Each plan MUST use ONLY the allowed targets below

6. Each plan MUST contain a STRUCTURED query spec
   Do NOT output natural-language search queries

7. reason MUST be short
   - no more than 8 words
   - describe what data is retrieved
   - do NOT explain why it matters

DOMAIN LOCK (CRITICAL):

1. All plans MUST stay inside the domain of the input query
2. All entity names MUST come from:
   - the input query
   - the parsed intent
   - the parsed sub_queries
   - or directly implied close variants
3. Do NOT introduce unrelated entities
4. If the input is about Gemini, do NOT use NVIDIA, OpenAI, Tesla, etc.
5. If the input is about OpenAI, do NOT use Gemini, Claude, NVIDIA, etc.
6. If you are unsure, stay generic inside the same domain

ALLOWED TARGETS ONLY:

- "release_events"
- "model_pricing"
- "api_limits"
- "model_specs"
- "usage_policies"

TYPE GUIDELINES:

1. events
- target should usually be "release_events"
- operation should be "select"
- use for announcements, release notes, official updates, launch events

2. aggregate
- target should usually be one of:
  - "model_pricing"
  - "api_limits"
  - "model_specs"
- operation should be "select" or "aggregate"
- use for numeric values, limits, prices, specs, quotas

3. similar
- target should usually be one of:
  - "release_events"
  - "usage_policies"
  - "model_specs"
- operation should be "similar"
- use for related cases inside the SAME domain

QUERY SPEC FORMAT:

Each plan MUST follow this structure exactly:

{{
  "plan_id": "P1",
  "type": "events",
  "target": "release_events",
  "filters": {{
    "entity": "...",
    "keywords": ["...", "..."],
    "year": 2024
  }},
  "fields": ["...", "..."],
  "operation": "select",
  "limit": 5,
  "reason": "..."
}}

FIELD RULES:

1. filters MUST be concrete
- use specific entity names
- use specific keywords
- use year or numeric values if available

2. fields MUST be concrete column-like names
- examples:
  - "model"
  - "price_input"
  - "price_output"
  - "context_window"
  - "rate_limit"
  - "release_date"
  - "policy_name"

3. operation MUST be one of:
- "select"
- "aggregate"
- "similar"

4. limit MUST be a small integer
- usually 5 or 10

SEMANTIC RELEVANCE RULES:

1. Plans MUST reflect the parsed intent directly
2. Plans MUST cover the most important evidence directions from sub_queries
3. Prefer first-party factual evidence over indirect comparison
4. Prefer raw facts before synthesis

GOOD OUTPUT PATTERN:

- events: retrieve official release/update records
- aggregate: retrieve pricing/spec/limit numbers
- similar: retrieve related cases in the same topic family

BAD OUTPUT PATTERN:

- unrelated companies
- unrelated industries
- finance metrics when the query is about API pricing policy
- hardware architecture when the query is about model usage terms

OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):

[
  {{
    "plan_id": "P1",
    "type": "events",
    "target": "release_events",
    "filters": {{
      "entity": "...",
      "keywords": ["...", "..."]
    }},
    "fields": ["...", "..."],
    "operation": "select",
    "limit": 5,
    "reason": "..."
  }},
  {{
    "plan_id": "P2",
    "type": "aggregate",
    "target": "model_pricing",
    "filters": {{
      "entity": "...",
      "keywords": ["...", "..."]
    }},
    "fields": ["...", "..."],
    "operation": "select",
    "limit": 10,
    "reason": "..."
  }},
  {{
    "plan_id": "P3",
    "type": "similar",
    "target": "usage_policies",
    "filters": {{
      "entity": "...",
      "keywords": ["...", "..."]
    }},
    "fields": ["...", "..."],
    "operation": "similar",
    "limit": 5,
    "reason": "..."
  }}
]

INPUT:
{{parsed_query}}
"""
        response = self.llm(prompt)
        print(f"[PLAN] LLM: {response}")
        return self._safe_parse_list(response)

    # --- Code: scoring ---
    def _score_plans(self, plans: List[Dict], parsed_query: Dict) -> List[Dict]:
        scored = []

        intent = parsed_query.get("intent", "")
        sub_queries = parsed_query.get("sub_queries", [])

        for plan in plans:
            score = 0.0
            query_text = plan.get("query", "").lower()

            # (1) intent alignment
            if intent and intent.lower() in query_text:
                score += 0.3

            # (2) keyword coverage
            for sq in sub_queries:
                if sq.lower() in query_text:
                    score += 0.1

            # (3) diversity boost
            if plan.get("type") == "aggregate":
                score += 0.2
            elif plan.get("type") == "similar":
                score += 0.15

            # (4) simplicity
            if len(query_text) < 60:
                score += 0.1

            plan["score"] = round(score, 3)
            scored.append(plan)

        return sorted(scored, key=lambda x: x["score"], reverse=True)

    # --- Code: selection ---
    def _select_top_k(self, plans: List[Dict], k: int = 3) -> List[Dict]:
        return plans[:k]

    # --- util ---
    def _safe_parse_list(self, text: str) -> List[Dict]:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass

        return [
            {
                "plan_id": "fallback",
                "type": "events",
                "query": text[:100],
                "reason": "fallback",
            }
        ]
