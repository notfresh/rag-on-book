#!/usr/bin/env python3
"""
RAG on Book — ask.py：和书对话

流程: 问题 → embedding → 向量检索 top-k 块 → LLM 生成回答（带引用）
      → 输出回答 + 引用列表（章节定位 + 原文文件, 可回到书里细看）

用法:
  python3 ask.py "DDIA 里事务隔离级别有哪些?"        # 问答（默认 top-5）
  python3 ask.py "什么是 LSM 树" -k 8                # 调整召回块数
  python3 ask.py "LSM 树和 B+ 树的区别" --no-llm     # 只看检索结果（调试）
  python3 ask.py "一致性" --book ddia                # 指定书（默认全部）

LLM: DeepSeek API（key 从 ~/.hermes/.env 或环境变量 DEEPSEEK_API_KEY 读取）
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

from index import DB_PATH, COLLECTION, ENCODE_THREADS, get_client

# ---------- LLM 配置 ----------
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
MAX_ANSWER_TOKENS = 1200


def load_deepseek_key():
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    dotenv = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(dotenv):
        with open(dotenv) as f:
            for line in f:
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip('"\'')
    return None


def embed_query(model, query):
    """bge 检索时 query 也要加指令前缀（模型约定, 提升中文检索效果）。"""
    return model.encode([query], normalize_embeddings=True)[0]


def search(client, model, query, book, top_k):
    from pymilvus import MilvusClient
    qv = embed_query(model, query)
    filter_expr = f"book == '{book}'" if book else None
    res = client.search(
        COLLECTION, data=[qv.tolist()], limit=top_k,
        output_fields=["book", "part", "chapter", "chapter_title", "section",
                       "subsection", "heading_path", "source", "content"],
        filter=filter_expr,
    )
    hits = []
    for h in res[0]:
        hits.append({
            "score": round(h["distance"] * 100),
            **h["entity"],
        })
    return hits


def build_prompt(query, hits):
    """检索片段带编号拼进 prompt; 引用编号 [N] 与片段一一对应。"""
    parts = []
    for i, h in enumerate(hits, 1):
        loc = h["heading_path"]
        parts.append(f"[{i}] 出处: {loc}\n{h['content']}")
    context = "\n\n".join(parts)
    return f"""你是一位精通《设计数据密集型应用》(DDIA) 的读书助手。
请严格根据下面给出的书中片段回答问题。规则:
1. 只依据片段内容回答; 片段没有的信息, 明确说"书中片段未涉及"。
2. 每个关键论断后标注引用编号 [N]（对应片段编号）。
3. 回答用中文, 简洁、结构清晰, 不超过 500 字。

---- 书中片段 ----
{context}

---- 问题 ----
{query}
"""


def ask_llm(prompt):
    key = load_deepseek_key()
    if not key:
        print("Error: 未找到 DEEPSEEK_API_KEY（~/.hermes/.env 或环境变量）")
        sys.exit(1)
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是严谨的读书助手, 严格依据给定片段回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": MAX_ANSWER_TOKENS,
    }
    req = urllib.request.Request(
        BASE_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    answer = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    print(f"  [llm] {MODEL} {time.time()-t0:.1f}s, tokens={usage.get('total_tokens')}")
    return answer


def render(query, hits, answer=None):
    print("=" * 60)
    print(f"问: {query}")
    if answer:
        print("-" * 60)
        print(answer)
    print("-" * 60)
    print("引用（可回到书里细看）:")
    for i, h in enumerate(hits, 1):
        score = h["score"]
        print(f"  [{i}] {h['heading_path']}  (相关度 {score}%)")
        print(f"      {h['source']}")
        snippet = h["content"].replace("\n", " ")[:80]
        print(f"      …{snippet}…")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="和书对话")
    ap.add_argument("query", help="你的问题")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="召回块数（默认 5）")
    ap.add_argument("--book", default=None, help="指定书（默认全部）")
    ap.add_argument("--no-llm", action="store_true", help="只看检索结果, 不调 LLM")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(ENCODE_THREADS)
    from sentence_transformers import SentenceTransformer
    print(f"加载模型 bge-small-zh-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    client = get_client()
    if not client.has_collection(COLLECTION):
        print("Error: 向量库不存在, 先跑 python3 index.py --full")
        sys.exit(1)
    client.load_collection(COLLECTION)

    hits = search(client, model, args.query, args.book, args.top_k)
    if not hits:
        print("没有命中。")
        return

    if args.no_llm:
        render(args.query, hits)
        return

    prompt = build_prompt(args.query, hits)
    answer = ask_llm(prompt)
    render(args.query, hits, answer)


if __name__ == "__main__":
    main()
