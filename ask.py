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

LLM: 由本文件顶部的 LLM 配置段决定（默认 MiniMax-M3，key 从 ~/.hermes/.env 或环境变量读）。
     设 RAGBOOK_LLM_PROVIDER=deepseek 可切回 DeepSeek（需账户已充值）。
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

from index import COLLECTION, ENCODE_THREADS, get_client
import books as books_mod

# ===== LLM 配置：本项目唯一的 provider/模型/端点配置点 =====
# 换 provider 只改这一段（env RAGBOOK_LLM_PROVIDER），web.py 等调用方一律走
# ask.MODEL / ask.BASE_URL / ask.load_llm_key()，不允许在别处硬编端点。
# 无兜底链：provider 报错就直接报错，不自动切换到另一个 provider。
LLM_PROVIDER = os.environ.get("RAGBOOK_LLM_PROVIDER", "minimax-cn")   # minimax-cn | deepseek
if LLM_PROVIDER == "deepseek":
    BASE_URL = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"
    _KEY_ENV = "DEEPSEEK_API_KEY"
    REASONING_SPLIT = False
else:
    BASE_URL = "https://api.minimaxi.com/v1/chat/completions"
    MODEL = "MiniMax-M3"          # 模型 ID 大小写敏感
    _KEY_ENV = "MINIMAX_CN_API_KEY"
    REASONING_SPLIT = True        # M3 默认把思考内联进 content，reasoning_split 分离到 reasoning_content
MAX_ANSWER_TOKENS = 4000   # M3 的 reasoning 与正文共享这个预算：太小会让思考吃光额度、
                            # 正文(content)字段整个消失（finish_reason=length，无 content 键）。
                            # 4000 是实测能覆盖「整章总结」这类最长 prompt 且不被截断的值。


def load_llm_key():
    """读当前 provider 的 API key（先环境变量，后 ~/.hermes/.env）。

    DeepSeek 充值后，设 RAGBOOK_LLM_PROVIDER=deepseek 即可切回（无需改代码）。
    """
    env_key = os.environ.get(_KEY_ENV)
    if env_key:
        return env_key
    dotenv = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(dotenv):
        with open(dotenv) as f:
            for line in f:
                if line.startswith(_KEY_ENV + "="):
                    return line.strip().split("=", 1)[1].strip().strip('"\'')
    return None


def strip_think(text: str) -> str:
    """剥掉 M3 可能内联进 content 的 <think>...</think> 思考块（只留最终答案）。"""
    if not text:
        return text
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = re.sub(r"<think>.*", "", out, flags=re.S)   # 截断产生的未闭合块
    return out.strip() or text.strip()


def post_llm(body: dict, timeout: int = 120) -> dict:
    """打一次 LLM 请求（无兜底链：provider 报错就抛，不切到别的 provider）。

    web.py 等调用方一律走这里，避免在别处重复写端点/key 逻辑。
    """
    key = load_llm_key()
    if not key:
        raise RuntimeError(f"未找到 {_KEY_ENV}（~/.hermes/.env 或环境变量）")
    req = urllib.request.Request(
        BASE_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _message_content(data: dict) -> str | None:
    """取 choices[0].message.content；M3 额度被思考吃光时该键会整个不存在（不是空串）。"""
    choices = data.get("choices") or []
    if not choices:
        return None
    return (choices[0].get("message") or {}).get("content")


def embed_query(model, query):
    """bge 检索时 query 也要加指令前缀（模型约定, 提升中文检索效果）。"""
    return model.encode([query], normalize_embeddings=True)[0]


def search(client, model, query, book, top_k):
    """向量检索。client 必须是**这本书自己**的向量库客户端。

    因为一书一库，库里只有这一本书的块，filter 不再需要 book 条件
    （保留 book 字段只是冗余，不影响检索）。
    """
    qv = embed_query(model, query)
    res = client.search(
        COLLECTION, data=[qv.tolist()], limit=top_k,
        output_fields=["book", "part", "chapter", "chapter_title", "section",
                       "subsection", "heading_path", "source", "content"],
    )
    hits = []
    for h in res[0]:
        hits.append({
            "score": round(h["distance"] * 100),
            **h["entity"],
        })
    return hits


# ---------- 书上下文（多书支持，2026-11-18） ----------

def book_context(book_id: str | None) -> tuple[str, str]:
    """取书的显示名 + 作者。取不到就用泛称。

    返回 (display, author)。display 形如 "《设计数据密集型应用》"。
    """
    if not book_id:
        return "这本书", ""
    try:
        m = books_mod.get_book(book_id)
    except books_mod.BookError:
        return "这本书", ""
    title = m.get("title") or book_id
    author = m.get("author") or ""
    return f"《{title}》", author


def build_prompt(query, hits, book_id: str | None = None):
    """检索片段带编号拼进 prompt; 引用编号 [N] 与片段一一对应。

    多书支持（2026-11-18）：书名/作者从 manifest 取，不再硬编码 DDIA。
    """
    display, author = book_context(book_id)
    parts = []
    for i, h in enumerate(hits, 1):
        loc = h["heading_path"]
        parts.append(f"[{i}] 出处: {loc}\n{h['content']}")
    context = "\n\n".join(parts)
    who = f"{display}（{author}）" if author else display
    return f"""你是一位精通{who}的读书助手。
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
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是严谨的读书助手, 严格依据给定片段回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": MAX_ANSWER_TOKENS,
    }
    if REASONING_SPLIT:
        body["reasoning_split"] = True
    t0 = time.time()
    data = post_llm(body, timeout=180)
    msg = (data.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content")
    if content is None:
        # M3 的 reasoning 与正文共享 max_tokens：额度被思考吃光时 finish_reason=length，
        # 且响应里根本没有 content 键（不是空串）。这里报出真实原因，别让它变成 KeyError。
        reason = data.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
        raise RuntimeError(
            f"LLM 未返回正文 content（finish_reason="
            f"{(data.get('choices') or [{}])[0].get('finish_reason')}，"
            f"reasoning_tokens={reason}，max_tokens={MAX_ANSWER_TOKENS}）"
        )
    answer = strip_think(content.strip())
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
    ap.add_argument("--book", default=None, help="指定书 id（必填，一书一库）")
    ap.add_argument("--no-llm", action="store_true", help="只看检索结果, 不调 LLM")
    args = ap.parse_args()

    if not args.book:
        print("Error: 一书一库，必须指定 --book（可用书: "
              + ", ".join(books_mod.list_book_ids()) + "）")
        sys.exit(1)

    import torch
    torch.set_num_threads(ENCODE_THREADS)
    from sentence_transformers import SentenceTransformer
    print(f"加载模型 bge-small-zh-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    try:
        client = get_client(args.book)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not client.has_collection(COLLECTION):
        print(f"Error: 书 {args.book} 没有向量库, 先跑 "
              f"python3 index.py --book {args.book}")
        sys.exit(1)
    client.load_collection(COLLECTION)

    hits = search(client, model, args.query, args.book, args.top_k)
    if not hits:
        print("没有命中。")
        return

    if args.no_llm:
        render(args.query, hits)
        return

    prompt = build_prompt(args.query, hits, args.book)
    answer = ask_llm(prompt)
    render(args.query, hits, answer)


if __name__ == "__main__":
    main()
