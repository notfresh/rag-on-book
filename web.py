#!/usr/bin/env python3
"""
RAG on Book Web 端 — Flask 后端（V3：多书 + 新界面）
监听 127.0.0.1:82，由任务101 反代 /t/ragbook/ 过来。

复用 ask.py 的核心：search() / build_prompt() / ask_llm()
进程级缓存：bge 模型只加载一次。

V2：SQLite 持久化会话 + 会话管理 + 意图分类 + 元信息过滤
V3（2026-11-18）：多书支持
  - 书清单来自 books.py（manifest 是单一来源），不再硬编码 DDIA
  - 元信息提问策略按书走：chapter_book 走"第N章"，note_set 走"主题名"
  - 章节表从 outline.json 自动读，不再手写 CHAPTER_SECTIONS 字典
  - 前端重写：书选择器 + 目录侧栏 + 右侧原文抽屉 + 新配色
"""

import json
import os
import sqlite3
import sys
import time
import uuid
from threading import Lock

# 让 ask.py 能被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string

import ask
import books as books_mod
from index import ENCODE_THREADS

app = Flask(__name__)

# ---------- SQLite 持久化 ----------
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "sessions.db")


def _db():
    """每次调用拿一个新连接（SQLite + 多线程 Flask）。"""
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # 并发读不阻塞写
    return conn


def init_db():
    with _db() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新对话',
                book TEXT,                    -- 会话绑定的书（多书支持，2026-11-18）
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,         -- 'user' | 'assistant'
                content TEXT NOT NULL,
                hits TEXT,                  -- JSON
                llm_ms INTEGER,
                ts REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_session_updated ON sessions(updated_at DESC);
        """)


def _ensure_book_column():
    """老库补 book 列（ALTER 是元数据操作，不重写数据）。"""
    with _db() as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(sessions)")]
        if "book" not in cols:
            c.execute("ALTER TABLE sessions ADD COLUMN book TEXT")


def get_or_create_session(sid: str | None, book: str | None = None) -> str:
    """有 sid 就拿，没有就新建。返回 sid。"""
    if sid:
        with _db() as c:
            row = c.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
            if row:
                return sid
    # 新建
    new_sid = str(uuid.uuid4())
    now = time.time()
    with _db() as c:
        c.execute("INSERT INTO sessions(id, title, book, created_at, updated_at) VALUES(?,?,?,?,?)",
                  (new_sid, "新对话", book, now, now))
    return new_sid


def add_message(sid: str, role: str, content: str, hits: list | None = None, llm_ms: int = 0):
    with _db() as c:
        c.execute(
            "INSERT INTO messages(session_id, role, content, hits, llm_ms, ts) VALUES(?,?,?,?,?,?)",
            (sid, role, content, json.dumps(hits or [], ensure_ascii=False), llm_ms, time.time()),
        )
        c.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid))


def get_messages(sid: str) -> list[dict]:
    with _db() as c:
        rows = c.execute(
            "SELECT role, content, hits, llm_ms, ts FROM messages WHERE session_id=? ORDER BY id",
            (sid,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "role": r["role"],
            "content": r["content"],
            "hits": json.loads(r["hits"] or "[]"),
            "llm_ms": r["llm_ms"],
            "ts": r["ts"],
        })
    return out


def list_sessions(limit: int = 50) -> list[dict]:
    """按 updated_at 倒序。"""
    with _db() as c:
        rows = c.execute(
            """SELECT s.id, s.title, s.book, s.created_at, s.updated_at,
                      (SELECT content FROM messages WHERE session_id=s.id AND role='user' ORDER BY id LIMIT 1) AS first_q,
                      (SELECT COUNT(*) FROM messages WHERE session_id=s.id) AS msg_count
               FROM sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [{
        "id": r["id"],
        "title": r["title"],
        "book": r["book"] or "",
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "first_q": r["first_q"] or "",
        "msg_count": r["msg_count"],
    } for r in rows]


def rename_session(sid: str, title: str):
    title = (title or "").strip()[:80] or "新对话"
    with _db() as c:
        c.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, time.time(), sid))


def delete_session(sid: str):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))


# 初始化表（启动时）
init_db()
_ensure_book_column()


# ---------- 全局模型客户端（进程复用） ----------
_model = None
_clients: dict[str, object] = {}   # book_id → MilvusClient（每本书一个独立库）
_lock = Lock()


def ensure_loaded():
    """懒加载：第一次请求时才加载模型。

    向量库客户端按书开 —— 每本书一个独立的 Milvus Lite 文件，
    开着的客户端缓存在 _clients 里复用（Milvus Lite 是文件锁，重复打开会失败）。
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                torch.set_num_threads(ENCODE_THREADS)
                from sentence_transformers import SentenceTransformer
                print("[web] 加载 bge-small-zh-v1.5 ...", flush=True)
                _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
                print("[web] 模型就绪", flush=True)


def get_book_client(book_id: str):
    """取（并缓存）某本书的向量库客户端。"""
    if book_id in _clients:
        return _clients[book_id]
    with _lock:
        if book_id in _clients:
            return _clients[book_id]
        from index import get_client, COLLECTION
        if not books_mod.has_vector_store(book_id):
            raise RuntimeError(
                f"书 {book_id} 没有向量库, 先跑 python3 index.py --book {book_id}")
        client = get_client(book_id)
        if not client.has_collection(COLLECTION):
            raise RuntimeError(
                f"书 {book_id} 的向量库里没有 collection, 先跑 "
                f"python3 index.py --book {book_id}")
        client.load_collection(COLLECTION)
        _clients[book_id] = client
        print(f"[web] 向量库就绪: {book_id}", flush=True)
        return client


# ---------- 意图分类（V2 多轮对话管理） ----------
# 路径：规则初筛 -> LLM 兜底分类 -> 路由（chitchat / followup / question）

# 规则初筛：纯问候词 / 寒暄（不区分大小写、不去标点）
_CHITCHAT_PATTERNS = [
    "你好", "您好", "你好啊", "你好呀", "hi", "HI", "Hi",
    "hello", "Hello", "HELLO", "嗨", "哈喽", "哈啰",
    "在吗", "在么", "在?", "在？",
    "你是谁", "你叫什么", "你叫啥",
    "你是AI吗", "你是机器人吗", "你是人吗",
    "thanks", "thank you", "thx", "谢了", "谢谢", "感谢",
    "ok", "OK", "好", "好的", "嗯", "嗯嗯",
    "早上好", "中午好", "下午好", "晚上好", "晚安",
]


def rule_classify(query: str) -> str | None:
    """规则初筛。返回 'chitchat' 或 None（None 表示交给 LLM 兜底）。"""
    q = query.strip().strip("？?！!。，, ").lower()
    if not q:
        return "chitchat"  # 空 query 走闲聊（前端几乎不会发空，但兜底）
    if len(q) <= 12 and any(p.lower() == q or p.lower() in q for p in _CHITCHAT_PATTERNS):
        return "chitchat"
    return None


def llm_classify(query: str, history_tail: list[dict] | None = None) -> str:
    """LLM 兜底分类。三选一：chitchat / followup / question。

    history_tail: 最近 1-2 轮问答，用于判定 followup（用代词/省略可能指上文）。
    返回值必在三选一中（任何异常都回退到 'question'，保证主流程不挂）。
    """
    history_ctx = ""
    if history_tail:
        history_ctx = "\n\n最近对话:\n" + "\n\n".join(
            f"[{m['role']}] {m['content'][:200]}" for m in history_tail
        ) + "\n\n如果用户最新消息明显在追问上文（用代词/省略/接续），判 followup。"
    prompt = f"""你是意图路由器。根据用户最新消息判断意图：

chitchat: 闲聊/打招呼/问你是谁/与书无关的对话
followup: 追问（用了代词、省略、明显在接续上文），需要历史才能理解
question: 新提问（独立问题，可不依赖上下文回答）

只回一个词：chitchat / followup / question{history_ctx}

用户消息：{query}"""
    try:
        # 用 ask_llm 但关掉 search 调用，直接走 LLM：借用 build_prompt 模式 + 单轮
        body = {
            "model": ask.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 8,
            "reasoning_split": True,
            "thinking": {"type": "disabled"},
        }
        data = ask.post_llm(body, timeout=20)
        ans = (ask._message_content(data) or "").strip().lower()
        for label in ("chitchat", "followup", "question"):
            if label in ans:
                return label
        return "question"
    except Exception:
        return "question"


def classify(query: str, history_tail: list[dict] | None = None) -> tuple[str, str]:
    """主入口。返回 (intent, source) —— source 是 'rule' 或 'llm'，便于调试。"""
    by_rule = rule_classify(query)
    if by_rule:
        return by_rule, "rule"
    return llm_classify(query, history_tail), "llm"


# ---------- 闲聊路径 ----------
def chitchat_reply(query: str, history_tail: list[dict] | None = None, book_id: str | None = None) -> str:
    """闲聊直接对话，不检索。"""
    display, author = ask.book_context(book_id)
    who = f"{display}（{author}）" if author else display
    msgs = [{
        "role": "system",
        "content": f"""你是{who}的读书搭子。用户当前没有具体问题，只是闲聊或打招呼。
请用轻松友好的口吻简短回应（不超过 80 字），不主动背诵书的目录、不重复介绍自己。
可以适度幽默，但不要油嘴滑舌。""",
    }]
    if history_tail:
        # 闲聊也带历史（聊天气氛连贯）
        for m in history_tail[-6:]:
            msgs.append({"role": m["role"], "content": m["content"][:300]})
    msgs.append({"role": "user", "content": query})
    try:
        body = {
            "model": ask.MODEL,
            "messages": msgs,
            "temperature": 0.7,
            "max_tokens": 200,
            "reasoning_split": True,
            "thinking": {"type": "disabled"},
        }
        data = ask.post_llm(body, timeout=30)
        content = ask._message_content(data)
        if content is None:
            return "（闲聊回应失败：LLM 未返回正文 content）"
        return ask.strip_think(content.strip())
    except Exception as e:
        return f"（闲聊回应失败：{e}）"


# ---------- Layer 1：元信息过滤（精确召回，不调 embedding） ----------
# 两种策略，由书的 manifest 决定：
#   chapter_book → "第N章" / "第N章某节"
#   note_set     → 主题名（书名或分组名）
import re


_outline_cache: dict[str, dict] = {}


def get_outline(book_id: str) -> dict:
    """读某本书的目录树（parse.py 生成的 outline.json）。"""
    if book_id in _outline_cache:
        return _outline_cache[book_id]
    path = os.path.join(books_mod.BOOKS_DIR, book_id, "outline.json")
    if not os.path.exists(path):
        return {"kind": "topic", "items": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {"kind": "topic", "items": []}
    _outline_cache[book_id] = data
    return data


def detect_meta_query(query: str, book: dict) -> tuple[str | None, dict]:
    """检测 query 是不是元信息提问。返回 (milvus_filter_expr, meta_info)。

    两个都为空表示不是元信息提问，走 bge 向量检索。
    """
    meta_cfg = book.get("meta") or {}
    meta_kind = meta_cfg.get("kind")
    book_id = book["id"]

    if meta_kind == books_mod.META_CHAPTER:
        return _detect_chapter_query(query, book)
    if meta_kind == books_mod.META_TOPIC:
        return _detect_topic_query(query, book)
    return None, {}


def _detect_chapter_query(query: str, book: dict) -> tuple[str | None, dict]:
    """章节式："第N章" / "第N章某节"。章节表从 outline.json 读（parse.py 自动生成）。

    注意（2026-11-18）：outline.items 里除了正文章，还有"前置/后记/致谢/导读"
    这些附加条目（OCR 版面书的附加区域）。判断章号范围时**只能数正文章**，
    否则 max_chapter 会把附加条目也算进去（32 而不是 23），越界提示就错了。
    """
    outline = get_outline(book["id"])
    items = outline.get("items") or []
    # 没有 kind 字段的（旧书 DDIA）都是正文章
    chapters = [it for it in items if (it.get("kind") or "chapter") == "chapter"]
    by_key = {it["key"]: it for it in chapters}

    ch_num = books_mod.detect_chapter_num(query)
    if not ch_num:
        return None, {}
    if ch_num not in by_key:
        # 边界：用户问的章号超出范围 → 上层直接提示，不调 LLM
        return None, {"out_of_range": True, "max_chapter": len(chapters), "chapter": ch_num}

    item = by_key[ch_num]
    ch_title = item.get("title", "")

    # 是否点名了节（把"第N章"和"是什么/讲什么"等噪声词去掉，剩下的去匹配小节）
    noise = re.sub(r'第\s*\d{1,3}\s*章', '', query)
    noise = re.sub(r'(是什么|讲什么|讲讲|介绍|总结|主要内容|大概|简介|讲的什么|说的是|内容是)', '', noise)
    noise = noise.strip(" ？?，,。.：: ")
    matched = books_mod.fuzzy_match_section(noise, item.get("sections") or []) if noise else None

    if matched:
        filter_expr = f"book == '{book['id']}' && chapter == '{ch_num}' && section == '{matched}'"
        return filter_expr, {
            "chapter": ch_num, "chapter_title": ch_title, "section": matched,
            "skip_embedding": True, "sort_by": "pk", "book": book["id"],
        }
    filter_expr = f"book == '{book['id']}' && chapter == '{ch_num}'"
    return filter_expr, {
        "chapter": ch_num, "chapter_title": ch_title, "section": None,
        "skip_embedding": True, "sort_by": "pk", "book": book["id"],
    }


def _detect_topic_query(query: str, book: dict) -> tuple[str | None, dict]:
    """主题式：认出主题名（书名或分组名）→ 精确召回该主题的全部块。

    note_set 没有"第N章"，主题名就是一级定位。
    """
    outline = get_outline(book["id"])
    topic = books_mod.match_topic(query, book, outline)
    if not topic:
        return None, {}

    outline = get_outline(book["id"])
    # 主题可能是分组名（"算法"）或具体文件名（"快速排序"）
    group_names = [it["key"] for it in outline.get("items") or []]
    if topic in group_names:
        # 分组级：召回该组下所有文件 —— 用 part 字段过滤
        filter_expr = f"book == '{book['id']}' && part == '{topic}'"
        return filter_expr, {
            "topic": topic, "scope": "group", "skip_embedding": True,
            "sort_by": "pk", "book": book["id"],
        }

    # 文件级：chapter_title 存的是主题名（文件名去掉分组前缀）
    filter_expr = f"book == '{book['id']}' && chapter_title == '{topic}'"
    return filter_expr, {
        "topic": topic, "scope": "file", "skip_embedding": True,
        "sort_by": "pk", "book": book["id"],
    }


def meta_search(client, filter_expr: str, meta_info: dict) -> list[dict]:
    """元信息精确召回：用 filter 查 milvus，按 pk 升序（章节内顺序）。"""
    from index import COLLECTION
    res = client.query(
        COLLECTION,
        filter=filter_expr,
        output_fields=["book", "part", "chapter", "chapter_title", "section",
                       "subsection", "heading_path", "source", "content"],
        limit=400,  # 一章/一个主题最多也就几百块
        sort="pk",  # milvus query 支持 sort
    )
    return res


# ---------- 提问/追问路径：历史拼进 prompt ----------
def build_prompt_with_history(query: str, hits: list, history_tail: list[dict],
                              book_id: str | None = None) -> str:
    """在 ask.build_prompt 基础上，把最近 2 轮问答拼进 prompt，让 LLM 理解代词/省略。"""
    display, author = ask.book_context(book_id)
    who = f"{display}（{author}）" if author else display
    parts = []
    for i, h in enumerate(hits, 1):
        loc = h["heading_path"]
        parts.append(f"[{i}] 出处: {loc}\n{h['content']}")
    context = "\n\n".join(parts)

    history_ctx = ""
    if history_tail:
        history_ctx = "\n---- 最近对话（用于理解代词/省略） ----\n"
        history_ctx += "\n".join(
            f"[{m['role']}] {m['content'][:300]}" for m in history_tail[-4:]  # 最近 2 轮 = 4 条
        ) + "\n"

    return f"""你是一位精通{who}的读书助手。
请严格根据下面给出的书中片段回答问题。规则:
1. 只依据片段内容回答; 片段没有的信息, 明确说"书中片段未涉及"。
2. 每个关键论断后标注引用编号 [N]（对应片段编号）。
3. 回答用中文, 简洁、结构清晰, 不超过 500 字。
4. 如果用户当前问题依赖上文（用了代词、省略），结合"最近对话"理解意图。{history_ctx}
---- 书中片段 ----
{context}

---- 当前问题 ----
{query}"""


# ---------- 路由 ----------
def _parse_json_body() -> dict:
    """宽容的 JSON body 解析，替代 request.get_json(silent=True)。

    - 接受裸换行（\x0a）等控制字符：textarea 内容里用户按回车会产生裸 \n，
      默认 strict JSON parser 会拒绝、Flask get_json(silent=True) 返 None，
      进而 `data.get("query") or ""` 拿到 ""，触发 "query 不能为空"。
    - 解析失败（含空 body / 非 JSON）返空 dict，行为与 silent=True 一致。
    """
    raw = request.get_data(cache=False, as_text=True)
    if not raw:
        return {}
    try:
        return json.loads(raw, strict=False) or {}
    except (ValueError, TypeError):
        return {}


def _resolve_book(book_id: str | None) -> dict | None:
    """取指定书；取不到或没指定 → 第一本已建索引的书。"""
    try:
        if book_id:
            return books_mod.get_book(book_id)
        candidates = [b for b in books_mod.list_books() if b.get("chunk_count")]
        return candidates[0] if candidates else None
    except books_mod.BookError:
        return None


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/books")
def api_books():
    """书清单 + 目录树（前端渲染书选择器与目录侧栏）。"""
    out = []
    for b in books_mod.list_books():
        outline = get_outline(b["id"])
        out.append({
            "id": b["id"],
            "title": b["title"],
            "subtitle": b.get("subtitle", ""),
            "author": b.get("author", ""),
            "kind": b["kind"],
            "meta_kind": b.get("meta", {}).get("kind", ""),
            # 空状态的推荐问题：优先读 book.json 的 meta.suggestions，
            # 没配就让前端按目录树自己推导 —— 不再按 meta_kind 硬编 DDIA 那四条（换书就不对了）。
            "suggestions": b.get("meta", {}).get("suggestions", []),
            "chunk_count": b.get("chunk_count", 0),
            "outline": outline,
        })
    return jsonify({"books": out})


@app.route("/api/outline/<book_id>")
def api_outline(book_id):
    """单本书的目录树。"""
    try:
        books_mod.get_book(book_id)
    except books_mod.BookError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"book_id": book_id, "outline": get_outline(book_id)})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    # 宽容 JSON 解析：textarea 内容里裸换行符（\x0a）会被默认 strict parser 拒绝，
    # 触发 "query 不能为空"。strict=False 接受裸控制字符。
    data = _parse_json_body()
    query = (data.get("query") or "").strip()
    sid_in = data.get("session_id")
    top_k = int(data.get("top_k") or 5)
    book_id = data.get("book") or None
    no_llm = bool(data.get("no_llm"))

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    book = _resolve_book(book_id)
    if book is None:
        return jsonify({"error": "没有可用的书（先跑 parse.py + index.py）"}), 503
    book_id = book["id"]

    sid = get_or_create_session(sid_in, book=book_id)

    # 拉最近 4 条历史（用于意图分类 + prompt 拼历史）
    history_tail = get_messages(sid)[-4:] if sid else []

    # 0. 元信息提问（"第N章"/主题名）→ 优先精确召回，跳过 bge + 闲聊分类
    meta_filter, meta_info = detect_meta_query(query, book)
    if meta_info.get("out_of_range"):
        # 章节号超出范围：明确告诉用户，避免 LLM 误判
        max_ch = meta_info.get("max_chapter", 0)
        ch_num_oob = meta_info.get("chapter", "?")
        display, _ = ask.book_context(book_id)
        msg = f"{display} 共 {max_ch} 章，没有第{ch_num_oob}章。"
        add_message(sid, "user", query)
        add_message(sid, "assistant", msg, hits=[], llm_ms=0)
        return jsonify({"session_id": sid, "answer": msg, "hits": [],
                        "search_ms": 0, "llm_ms": 0, "book": book_id,
                        "intent": "meta_oob", "intent_src": "rule"})
    if meta_filter:
        try:
            ensure_loaded()
            book_client = get_book_client(book_id)
        except Exception as e:
            return jsonify({"error": f"模型加载失败: {e}"}), 500
        add_message(sid, "user", query)
        t0 = time.time()
        chunks = meta_search(book_client, meta_filter, meta_info)
        search_ms = int((time.time() - t0) * 1000)
        if not chunks:
            msg = f"未找到相关内容"
            if meta_info.get("chapter"):
                msg = f"未找到第{meta_info['chapter']}章的内容"
                if meta_info.get("section"):
                    msg += f"「{meta_info['section']}」节"
            elif meta_info.get("topic"):
                msg = f"未找到「{meta_info['topic']}」相关内容"
            add_message(sid, "assistant", msg, hits=[], llm_ms=0)
            return jsonify({"session_id": sid, "answer": msg, "hits": [],
                            "search_ms": search_ms, "llm_ms": 0, "book": book_id,
                            "intent": "meta", "intent_src": "rule"})
        # 用元信息专用 prompt（不限制"严格按片段"）
        hits_clean = [_hit_to_clean(c, score=100) for c in chunks]  # meta 召回不打分
        prompt = build_meta_prompt(query, chunks, meta_info, book_id)
        t0 = time.time()
        try:
            answer = ask.ask_llm(prompt)
        except Exception as e:
            return jsonify({"error": f"LLM 调用失败: {e}", "hits": hits_clean,
                            "session_id": sid, "intent": "meta", "intent_src": "rule"}), 502
        llm_ms = int((time.time() - t0) * 1000)
        add_message(sid, "assistant", answer, hits=hits_clean, llm_ms=llm_ms)
        # meta 路径也要自动给会话起标题
        auto_title_if_new(sid, query)
        return jsonify({
            "session_id": sid,
            "answer": answer,
            "hits": hits_clean,
            "search_ms": search_ms,
            "llm_ms": llm_ms,
            "book": book_id,
            "intent": "meta",
            "intent_src": "rule",
            "meta": meta_info,
        })

    # 1. 意图分类（规则初筛 → LLM 兜底）
    intent, intent_src = classify(query, history_tail=history_tail)

    # 2. 闲聊路径：不检索，直接对话
    if intent == "chitchat":
        t0 = time.time()
        try:
            answer = chitchat_reply(query, history_tail=history_tail, book_id=book_id)
        except Exception as e:
            return jsonify({"error": f"闲聊 LLM 失败: {e}", "session_id": sid}), 502
        llm_ms = int((time.time() - t0) * 1000)

        add_message(sid, "user", query)
        add_message(sid, "assistant", answer, hits=[], llm_ms=llm_ms)

        return jsonify({
            "session_id": sid,
            "answer": answer,
            "hits": [],
            "search_ms": 0,
            "llm_ms": llm_ms,
            "book": book_id,
            "intent": intent,
            "intent_src": intent_src,
        })

    # 3. 提问/追问路径：检索 + LLM
    try:
        ensure_loaded()
        book_client = get_book_client(book_id)
    except Exception as e:
        return jsonify({"error": f"模型加载失败: {e}"}), 500

    t0 = time.time()
    hits = ask.search(book_client, _model, query, book=book_id, top_k=top_k)
    search_ms = int((time.time() - t0) * 1000)

    add_message(sid, "user", query)

    if not hits:
        add_message(sid, "assistant", "（未命中任何片段）", hits=[], llm_ms=0)
        return jsonify({"session_id": sid, "answer": "（未命中任何片段）", "hits": [],
                        "search_ms": search_ms, "llm_ms": 0, "book": book_id,
                        "intent": intent, "intent_src": intent_src})

    if no_llm:
        answer = None
        llm_ms = 0
    else:
        t0 = time.time()
        # 追问路径：把历史拼进 prompt；提问路径也拼（无害，prompt 明确说"如不依赖可忽略"）
        prompt = build_prompt_with_history(query, hits, history_tail, book_id)
        try:
            answer = ask.ask_llm(prompt)
        except Exception as e:
            return jsonify({"error": f"LLM 调用失败: {e}", "hits": _hits_clean(hits),
                            "session_id": sid, "intent": intent, "intent_src": intent_src}), 502
        llm_ms = int((time.time() - t0) * 1000)

    hits_clean = _hits_clean(hits)
    add_message(sid, "assistant", answer, hits=hits_clean, llm_ms=llm_ms)

    # 首问自动给会话起个标题（取问题前 30 字）
    auto_title_if_new(sid, query)

    return jsonify({
        "session_id": sid,
        "answer": answer,
        "hits": hits_clean,
        "search_ms": search_ms,
        "llm_ms": llm_ms,
        "book": book_id,
        "intent": intent,
        "intent_src": intent_src,
    })


def auto_title_if_new(sid: str, query: str):
    """如果会话标题还是默认"新对话"，用 query 前 30 字设为标题。"""
    with _db() as c:
        row = c.execute("SELECT title FROM sessions WHERE id=?", (sid,)).fetchone()
        if row and row["title"] == "新对话":
            auto_title = (query[:30] + ("…" if len(query) > 30 else "")).strip()
            c.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (auto_title, sid, time.time()))


def _hits_clean(hits):
    out = []
    for h in hits:
        out.append({
            "score": h["score"],
            "book": h.get("book", ""),
            "heading_path": h["heading_path"],
            "source": h["source"],
            "snippet": h["content"][:240].replace("\n", " "),
        })
    return out


def _hit_to_clean(c: dict, score: int = 100) -> dict:
    """milvus query() 返回的实体转 citation dict（meta 召回默认 score=100）。"""
    return {
        "score": score,
        "book": c.get("book", ""),
        "heading_path": c.get("heading_path", ""),
        "source": c.get("source", ""),
        "snippet": c.get("content", "")[:240].replace("\n", " "),
    }


def build_meta_prompt(query: str, chunks: list[dict], meta_info: dict, book_id: str | None = None) -> str:
    """元信息提问专用 prompt：拿到指定章/节/主题的全文，让 LLM 自由总结/介绍。"""
    display, author = ask.book_context(book_id)
    who = f"{display}（{author}）" if author else display

    if meta_info.get("chapter"):
        scope = f"第{meta_info['chapter']}章「{meta_info.get('chapter_title','')}」"
        if meta_info.get("section"):
            scope += f"「{meta_info['section']}」节"
    elif meta_info.get("topic"):
        scope = f"「{meta_info['topic']}」主题"
    else:
        scope = "相关内容"

    parts = []
    for i, c in enumerate(chunks, 1):
        loc = c.get("heading_path", "")
        parts.append(f"[{i}] {loc}\n{c.get('content','')}")
    context = "\n\n".join(parts)

    return f"""你是{who}的读书助手。用户询问的是 {scope}。

下面是该范围按书内顺序排列的全部内容（可能较长）。请根据用户的具体问题：
1. 如果用户问"是什么/讲什么/主要内容"：用结构化方式总结的核心主题、关键概念、典型应用场景。
2. 如果用户问某个具体概念：定位到相关片段，直接回答并标注引用 [N]。
3. 回答用中文，分点呈现时使用 markdown，每条 200-400 字。

---- {scope} 全文 ----
{context}

---- 用户问题 ----
{query}"""


@app.route("/api/sessions", methods=["GET"])
def api_sessions_list():
    """会话列表（按 updated_at 倒序）。"""
    return jsonify({"sessions": list_sessions()})


@app.route("/api/sessions", methods=["POST"])
def api_sessions_create():
    """新建空会话（不立即提问）。"""
    data = _parse_json_body()
    sid = get_or_create_session(None, book=data.get("book") or None)
    return jsonify({"session_id": sid})


@app.route("/api/sessions/<sid>/rename", methods=["POST"])
def api_sessions_rename(sid):
    data = _parse_json_body()
    rename_session(sid, data.get("title", ""))
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def api_sessions_delete(sid):
    delete_session(sid)
    return jsonify({"ok": True})


@app.route("/api/session/<sid>")
def api_session(sid):
    """单个会话：消息 + 标题 + 绑定的书。

    title 必须返回 —— 前端切换/刷新会话时靠它显示会话名，
    漏了会导致"重命名后刷新就没了"。
    """
    with _db() as c:
        row = c.execute("SELECT title, book FROM sessions WHERE id=?", (sid,)).fetchone()
    return jsonify({
        "session_id": sid,
        "messages": get_messages(sid),
        "title": (row["title"] if row else "") or "新对话",
        "book": (row["book"] if row else "") or "",
    })


@app.route("/api/chapter/<book_id>/<path:rel>")
def api_source(book_id: str, rel: str):
    """按需返回某本书里某个源文件的 markdown（前端 marked 渲染）。

    用于"点击引用 → 展开原文"功能。路径只允许读 books/<book_id>/ 之内。
    """
    try:
        books_mod.get_book(book_id)
    except books_mod.BookError as e:
        return jsonify({"error": str(e)}), 404
    # rel 形如 "content/zh/ch3.md" 或 "notes/算法 - 排序.md"
    rel = rel.split("#")[0]                     # 去掉锚点
    base = os.path.realpath(os.path.join(books_mod.BOOKS_DIR, book_id))
    target = os.path.realpath(os.path.join(base, rel))
    if not target.startswith(base + os.sep) or not os.path.exists(target):
        return jsonify({"error": f"文件不存在: {rel}"}), 404
    with open(target, "r", encoding="utf-8") as f:
        md_content = f.read()
    # 剥掉 YAML frontmatter —— 否则渲染出来是一堆 "title: 事务" 的元数据行，不是正文
    md_content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", md_content, count=1, flags=re.S)
    return jsonify({"book": book_id, "source": rel, "markdown": md_content})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "model_loaded": _model is not None, "db": DB_PATH,
                    "books": [b["id"] for b in books_mod.list_books()]})


# ---------- 前端 HTML ----------
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>RAG on Book · 读书问答</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<style>
  :root {
    --bg:#0b0d10; --surface:#12151a; --surface-2:#171b22; --elevated:#1c212a;
    --line:#242a33; --line-soft:#1e242c;
    --fg:#e6e9ee; --fg-dim:#9aa4b2; --fg-faint:#6b7686;
    --accent:#d9a441; --accent-soft:rgba(217,164,65,.12); --accent-line:rgba(217,164,65,.3);
    --user:#1c2430; --user-line:#2b3644;
    --ok:#5fb98b; --radius:10px; --radius-sm:7px;
    --sidebar-w:268px; --toc-w:230px;
    --font:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  * { box-sizing:border-box; }
  html,body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--fg); font-family:var(--font);
         font-size:14px; line-height:1.6; overflow:hidden; }

  /* ---------- 布局骨架 ---------- */
  /* 三列固定：sidebar | toc | main。用 grid-column 显式指定，
     这样即使 toc 被 display:none 拿掉，main 仍在第 3 列，不会串行换行。
     （初版用 auto-placement + .with-toc 切列数，toc 一 display:none
       main 就掉到第二行 —— 主区塌到左上角） */
  .app { display:grid; grid-template-columns:var(--sidebar-w) var(--toc-w) 1fr; height:100vh; }
  .app:not(.with-toc) { grid-template-columns:var(--sidebar-w) 0 1fr; }
  .app:not(.with-toc) .toc { display:none; }
  .sidebar { grid-column:1; grid-row:1; }
  .toc { grid-column:2; grid-row:1; }
  .main { grid-column:3; grid-row:1; }

  /* ---------- 左侧：会话栏 ---------- */
  .sidebar { background:var(--surface); border-right:1px solid var(--line);
             display:flex; flex-direction:column; min-height:0; }
  .brand { padding:16px 18px 12px; display:flex; align-items:center; gap:9px; }
  .brand .mark { width:26px; height:26px; border-radius:7px; display:grid; place-items:center;
                 background:linear-gradient(135deg,var(--accent),#b8791f); color:#1a1206;
                 font-weight:700; font-size:13px; letter-spacing:-.5px; }
  .brand .name { font-weight:600; font-size:14.5px; letter-spacing:.2px; }
  .brand .ver { font-size:11px; color:var(--fg-faint); }

  .newbtn { margin:4px 14px 12px; padding:8px 12px; background:transparent; color:var(--fg);
            border:1px solid var(--line); border-radius:var(--radius-sm); cursor:pointer;
            font-family:inherit; font-size:13px; display:flex; align-items:center; justify-content:center;
            gap:7px; transition:border-color .15s, background .15s; }
  .newbtn:hover { border-color:var(--accent-line); background:var(--accent-soft); color:var(--accent); }
  .newbtn .plus { font-size:15px; line-height:1; }

  .sess-scroll { flex:1; overflow-y:auto; padding-bottom:12px; }
  .sess-item { padding:9px 14px 9px 16px; cursor:pointer; display:flex; align-items:flex-start; gap:8px;
               border-left:2px solid transparent; transition:background .12s; }
  .sess-item:hover { background:var(--surface-2); }
  .sess-item.active { background:var(--accent-soft); border-left-color:var(--accent); }
  .sess-item .title { flex:1; min-width:0; font-size:13px; overflow:hidden;
                      text-overflow:ellipsis; white-space:nowrap; }
  .sess-item.active .title { color:var(--accent); }
  .sess-item .meta { font-size:11px; color:var(--fg-faint); margin-top:2px;
                     overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .sess-item .actions { display:none; gap:2px; }
  .sess-item:hover .actions { display:flex; }
  .sess-item .actions button { background:transparent; border:0; color:var(--fg-faint);
                               cursor:pointer; padding:2px 4px; font-size:13px; border-radius:4px; }
  .sess-item .actions button:hover { color:var(--fg); background:var(--elevated); }
  .empty-list { padding:20px 16px; color:var(--fg-faint); font-size:12px; text-align:center; line-height:1.8; }

  .side-foot { padding:10px 14px; border-top:1px solid var(--line-soft); font-size:11px;
               color:var(--fg-faint); display:flex; justify-content:space-between; align-items:center; }

  /* ---------- 中间：目录栏 ---------- */
  .toc { background:var(--bg); border-right:1px solid var(--line); overflow-y:auto; padding:14px 0 24px;
         display:flex; flex-direction:column; min-height:0; }
  .toc-head { padding:0 16px 10px; }
  .toc-head .t { font-size:11px; text-transform:uppercase; letter-spacing:.9px; color:var(--fg-faint); }
  .toc-item { padding:6px 16px 6px 18px; font-size:12.5px; color:var(--fg-dim); cursor:pointer;
              border-left:2px solid transparent; display:flex; justify-content:space-between; gap:8px; }
  .toc-item:hover { color:var(--fg); background:var(--surface); }
  .toc-item.sub { padding-left:30px; font-size:12px; color:var(--fg-faint); }
  .toc-item.sub:hover { color:var(--fg-dim); }
  .toc-item .n { color:var(--fg-faint); font-family:var(--mono); font-size:11px; flex-shrink:0; }
  .toc-group { padding:12px 16px 4px; font-size:10.5px; text-transform:uppercase;
               letter-spacing:.9px; color:var(--fg-faint); }

  /* ---------- 右侧：主区 ---------- */
  .main { display:flex; flex-direction:column; min-width:0; min-height:0; background:var(--bg); }
  .topbar { padding:11px 20px; border-bottom:1px solid var(--line); display:flex;
            align-items:center; gap:12px; background:rgba(11,13,16,.85); backdrop-filter:blur(12px); }
  .book-select { display:flex; align-items:center; gap:8px; }
  .book-select select { background:var(--surface-2); color:var(--fg); border:1px solid var(--line);
                        border-radius:var(--radius-sm); padding:5px 9px; font-family:inherit; font-size:13px;
                        cursor:pointer; outline:none; max-width:280px; }
  .book-select select:focus { border-color:var(--accent-line); }
  .book-meta { font-size:11.5px; color:var(--fg-faint); }
  .topbar .spacer { flex:1; }
  .sess-title { color:var(--fg-dim); font-size:13px; max-width:320px; overflow:hidden;
                text-overflow:ellipsis; white-space:nowrap; cursor:text; padding:3px 8px;
                border-radius:5px; border:1px solid transparent; }
  .sess-title:hover { border-color:var(--line); }
  .sess-title[contenteditable=true] { background:var(--surface-2); border-color:var(--accent-line);
                                      outline:none; color:var(--fg); }
  .toc-toggle { display:none; background:transparent; border:1px solid var(--line); color:var(--fg-dim);
                border-radius:var(--radius-sm); padding:5px 9px; cursor:pointer; font-size:13px; }

  /* ---------- 消息流 ---------- */
  .chat { flex:1; overflow-y:auto; padding:24px 20px 8px; }
  .thread { max-width:820px; margin:0 auto; }
  .msg { margin-bottom:22px; display:flex; gap:12px; }
  .msg .avatar { width:26px; height:26px; border-radius:7px; flex-shrink:0; display:grid; place-items:center;
                 font-size:11px; font-weight:600; margin-top:1px; }
  .msg.user .avatar { background:var(--user); color:var(--fg-dim); border:1px solid var(--user-line); }
  .msg.assistant .avatar { background:linear-gradient(135deg,var(--accent),#b8791f); color:#1a1206; }
  .msg .body { flex:1; min-width:0; }
  .msg.user .body { background:var(--user); border:1px solid var(--user-line); border-radius:var(--radius);
                    padding:9px 13px; white-space:pre-wrap; display:inline-block; max-width:100%; }
  .msg.assistant .body { font-size:14.2px; }
  .msg .body h1, .msg .body h2, .msg .body h3 { margin:14px 0 7px; line-height:1.35; font-weight:600; }
  .msg .body h1 { font-size:18px; } .msg .body h2 { font-size:16px; } .msg .body h3 { font-size:14.5px; }
  .msg .body p { margin:7px 0; }
  .msg .body ul, .msg .body ol { margin:7px 0; padding-left:20px; }
  .msg .body li { margin:3px 0; }
  .msg .body code { background:var(--surface-2); padding:1.5px 5px; border-radius:4px;
                    font-family:var(--mono); font-size:12.5px; color:#e8c789; }
  .msg .body pre { background:var(--surface); border:1px solid var(--line-soft); padding:11px 13px;
                   border-radius:var(--radius-sm); overflow-x:auto; }
  .msg .body pre code { background:transparent; padding:0; color:var(--fg); }
  .msg .body blockquote { border-left:2px solid var(--accent-line); padding:2px 0 2px 12px;
                          margin:9px 0; color:var(--fg-dim); }
  .msg .body a { color:var(--accent); }
  .msg .body a.ref { text-decoration:none; padding:1px 4px; border-radius:4px; font-weight:500;
                     background:var(--accent-soft); color:var(--accent); cursor:pointer; font-size:12.5px; }
  .msg .body a.ref:hover { background:var(--accent-line); }

  /* ---------- 引用区 ---------- */
  .hits { margin-top:14px; border-top:1px solid var(--line-soft); padding-top:10px; }
  .hits-head { color:var(--fg-faint); font-size:11px; margin-bottom:6px;
               display:flex; align-items:center; gap:6px; }
  .hit-group { margin-bottom:5px; }
  .hit-group > summary { cursor:pointer; padding:6px 10px; background:var(--surface);
                         border:1px solid var(--line-soft); border-radius:var(--radius-sm);
                         list-style:none; display:flex; justify-content:space-between;
                         align-items:center; gap:10px; font-size:12.5px; }
  .hit-group > summary::-webkit-details-marker { display:none; }
  .hit-group > summary::before { content:'▸'; color:var(--fg-faint); font-size:10px;
                                 transition:transform .15s; display:inline-block; margin-right:2px; }
  .hit-group[open] > summary::before { transform:rotate(90deg); }
  .hit-group .sec-name { color:var(--fg-dim); font-weight:500; }
  .hit-group .sec-cnt { color:var(--fg-faint); font-size:11px; font-family:var(--mono); flex-shrink:0; }
  .hit { margin:4px 0 0; padding:9px 11px; background:var(--surface); border:1px solid var(--line-soft);
         border-radius:var(--radius-sm); scroll-margin-top:20px; transition:background 1.4s ease-out; }
  .hit.highlight { background:var(--accent-soft); border-color:var(--accent-line); }
  .hit .path { color:var(--fg-dim); font-size:12.5px; display:flex; flex-wrap:wrap; align-items:center; gap:7px; }
  .hit .path .idx { color:var(--accent); font-family:var(--mono); font-size:11px; }
  .hit .path .score { color:var(--fg-faint); font-family:var(--mono); font-size:11px; }
  .hit .src-btn { color:var(--fg-faint); text-decoration:none; font-size:11.5px; padding:1px 6px;
                  border:1px solid var(--line); border-radius:4px; cursor:pointer; }
  .hit .src-btn:hover { color:var(--accent); border-color:var(--accent-line); }
  .hit .snip { color:var(--fg-dim); margin-top:5px; line-height:1.65; font-size:13px; }
  .hit .snip p { margin:3px 0; }

  .meta { color:var(--fg-faint); font-size:11px; margin-top:9px; font-family:var(--mono); }

  /* ---------- 原文抽屉 ---------- */
  .drawer-backdrop { position:fixed; inset:0; background:rgba(0,0,0,.5); opacity:0; pointer-events:none;
                     transition:opacity .22s; z-index:40; }
  .drawer-backdrop.open { opacity:1; pointer-events:auto; }
  .drawer { position:fixed; top:0; right:0; height:100vh; width:min(720px,92vw);
            background:var(--surface); border-left:1px solid var(--line); z-index:41;
            transform:translateX(100%); transition:transform .24s cubic-bezier(.4,0,.2,1);
            display:flex; flex-direction:column; }
  .drawer.open { transform:translateX(0); }
  .drawer-head { padding:13px 18px; border-bottom:1px solid var(--line); display:flex;
                 align-items:center; gap:10px; flex-shrink:0; }
  .drawer-head .t { font-size:13.5px; font-weight:600; flex:1; min-width:0; overflow:hidden;
                    text-overflow:ellipsis; white-space:nowrap; }
  .drawer-head .close { background:transparent; border:0; color:var(--fg-faint); cursor:pointer;
                        font-size:17px; padding:2px 6px; border-radius:5px; }
  .drawer-head .close:hover { color:var(--fg); background:var(--surface-2); }
  .drawer-body { flex:1; overflow-y:auto; padding:18px 22px 40px; line-height:1.75; font-size:14px; }
  .drawer-body h1 { font-size:20px; margin:16px 0 9px; padding-bottom:7px; border-bottom:1px solid var(--line); }
  .drawer-body h2 { font-size:16px; color:var(--accent); margin:16px 0 7px; }
  .drawer-body h3 { font-size:14px; color:var(--fg-dim); margin:13px 0 6px; }
  .drawer-body p { margin:7px 0; }
  .drawer-body ul, .drawer-body ol { margin:7px 0; padding-left:22px; }
  .drawer-body code { background:var(--surface-2); padding:1.5px 5px; border-radius:4px;
                      font-family:var(--mono); font-size:12.5px; color:#e8c789; }
  .drawer-body pre { background:var(--bg); border:1px solid var(--line-soft); padding:11px 13px;
                     border-radius:var(--radius-sm); overflow-x:auto; }
  .drawer-body pre code { background:transparent; padding:0; color:var(--fg); }
  .drawer-body blockquote { border-left:2px solid var(--accent-line); padding-left:12px;
                            margin:9px 0; color:var(--fg-dim); }
  .drawer-body img { max-width:100%; border-radius:6px; }
  .drawer-body .highlight-anchor { background:var(--accent-soft); border-radius:4px; padding:1px 3px; }

  /* ---------- 输入区 ---------- */
  .composer { border-top:1px solid var(--line); padding:14px 20px 16px; background:var(--surface); }
  .input-wrap { max-width:820px; margin:0 auto; position:relative; }
  .input-row { display:flex; gap:9px; align-items:flex-end; }
  textarea { flex:1; background:var(--bg); color:var(--fg); border:1px solid var(--line);
            border-radius:var(--radius); padding:10px 13px; resize:none; font-family:inherit;
            font-size:14px; line-height:1.55; min-height:44px; max-height:180px; outline:none;
            transition:border-color .15s; }
  textarea:focus { border-color:var(--accent-line); }
  textarea::placeholder { color:var(--fg-faint); }
  .send { background:var(--accent); color:#1a1206; border:0; padding:0 18px; height:44px;
          border-radius:var(--radius); cursor:pointer; font-family:inherit; font-size:13.5px;
          font-weight:600; flex-shrink:0; transition:opacity .15s; }
  .send:disabled { opacity:.45; cursor:wait; }
  .hint { max-width:820px; margin:7px auto 0; font-size:11px; color:var(--fg-faint);
          display:flex; justify-content:space-between; gap:10px; }

  /* ---------- 空状态 ---------- */
  .empty { max-width:640px; margin:8vh auto 0; text-align:center; }
  .empty .eyebrow { font-size:11px; text-transform:uppercase; letter-spacing:1.4px;
                    color:var(--fg-faint); margin-bottom:14px; }
  .empty h2 { font-size:24px; font-weight:600; margin:0 0 8px; letter-spacing:-.3px; }
  .empty p { color:var(--fg-dim); font-size:13.5px; margin:0 0 26px; }
  .suggest { display:grid; gap:8px; text-align:left; }
  .suggest button { background:var(--surface); border:1px solid var(--line-soft); color:var(--fg-dim);
                    padding:11px 14px; border-radius:var(--radius-sm); cursor:pointer;
                    font-family:inherit; font-size:13px; text-align:left; transition:all .15s;
                    display:flex; align-items:center; gap:10px; }
  .suggest button:hover { border-color:var(--accent-line); color:var(--fg); background:var(--surface-2); }
  .suggest button .arrow { color:var(--fg-faint); font-size:12px; }

  /* ---------- 滚动条 ---------- */
  ::-webkit-scrollbar { width:9px; height:9px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:#2a313b; border-radius:5px; border:2px solid var(--bg); }
  ::-webkit-scrollbar-thumb:hover { background:#39424f; }

  /* ---------- 响应式 ---------- */
  @media (max-width:1180px) {
    /* 三个选择器都写上：基础规则里的 .app:not(.with-toc) 特指 0,2,0，
       单写 .app（0,1,0）压不住它 —— 列数没切成 1 列，主区就被塞进 grid-column:1 的 0px 列里。*/
    .app, .app.with-toc, .app:not(.with-toc) { grid-template-columns:var(--sidebar-w) 0 1fr; }
    .app .toc { display:none; }
    .app .toc.open { display:flex; position:fixed; top:0; bottom:0; left:var(--sidebar-w);
                     /* 必须高于 .scrim(34)，否则遮罩盖在目录上，手指滑动全被遮罩吃掉、一碰就关。
                        与 .sidebar(35) 持平。*/ 
                     width:var(--toc-w); z-index:35; box-shadow:8px 0 30px rgba(0,0,0,.4); }
    .toc-toggle { display:block; }
  }
  @media (max-width:820px) {
    :root { --sidebar-w:0px; --toc-w:0px; }
    /* 同 1180 断点：必须同时压住 .app:not(.with-toc)，否则列数切不成 1 列、主区塌成 0 宽。*/
    .app, .app.with-toc, .app:not(.with-toc) { grid-template-columns:1fr; }
    .app .sidebar { grid-column:1; }
    .app .toc { grid-column:1; }
    .app .main { grid-column:1; }
    .sidebar { position:fixed; top:0; bottom:0; left:0; width:270px; z-index:35;
               transform:translateX(-100%); transition:transform .22s; }
    .sidebar.open { transform:translateX(0); box-shadow:8px 0 30px rgba(0,0,0,.5); }
    .menu-btn { display:block !important; }
    .app .toc.open { left:0; width:min(280px,85vw); }
    .chat { padding:16px 14px 8px; }
    .composer { padding:11px 14px 13px; }
    .drawer { width:100vw; }
    .topbar { padding:10px 14px; gap:8px; }
    /* 手机宽度一行放不下 5 个元素：去掉 book-meta（空状态里已写「作者 · 全部内容已建索引」），
       书选择器吃满剩余宽度，其余按钮不换行、不收缩 —— 否则「新建对话」会被截成「新…」、
       「目录」会折成两行。*/
    .book-meta { display:none; }
    .book-select { flex:1 1 auto; min-width:0; }
    .book-select select { width:100%; max-width:none; }
    .menu-btn, .toc-toggle { flex-shrink:0; white-space:nowrap; }
    .sess-title { max-width:96px; flex-shrink:0; white-space:nowrap; }
  }
  .menu-btn { display:none; background:transparent; border:1px solid var(--line); color:var(--fg-dim);
              border-radius:var(--radius-sm); padding:5px 9px; cursor:pointer; font-size:13px; }
  .scrim { position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:34; display:none; }
  .scrim.open { display:block; }
</style>
</head>
<body>

<div class="app" id="app">
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="mark">R</div>
      <div>
        <div class="name">RAG on Book</div>
        <div class="ver">读书问答 · 本地检索</div>
      </div>
    </div>
    <button class="newbtn" onclick="newChat()"><span class="plus">＋</span> 新对话</button>
    <div class="sess-scroll"><div id="sessList"></div></div>
    <div class="side-foot">
      <span id="bookCount"></span>
      <span id="vecInfo"></span>
    </div>
  </aside>

  <nav class="toc" id="toc">
    <div class="toc-head"><div class="t" id="tocTitle">目录</div></div>
    <div id="tocList"></div>
  </nav>

  <main class="main">
    <div class="topbar">
      <button class="menu-btn" onclick="toggleSidebar()">☰</button>
      <div class="book-select">
        <select id="bookSel" onchange="onBookChange(this.value)"></select>
      </div>
      <span class="book-meta" id="bookMeta"></span>
      <button class="toc-toggle" onclick="toggleToc()">目录</button>
      <span class="spacer"></span>
      <span class="sess-title" id="sessTitle" title="点击重命名">新对话</span>
    </div>
    <div class="chat" id="msgs"></div>
    <div class="composer">
      <div class="input-wrap">
        <div class="input-row">
          <textarea id="q" placeholder="问点什么…（Enter 发送，Shift+Enter 换行）" autofocus></textarea>
          <button class="send" id="btn" onclick="send()">发送</button>
        </div>
        <div class="hint">
          <span id="hintLeft"></span>
          <span>Enter 发送 · Shift+Enter 换行</span>
        </div>
      </div>
    </div>
  </main>
</div>

<div class="scrim" id="scrim" onclick="closeMobile()"></div>
<div class="drawer-backdrop" id="drawerBg" onclick="closeDrawer()"></div>
<aside class="drawer" id="drawer">
  <div class="drawer-head">
    <span class="t" id="drawerTitle">原文</span>
    <button class="close" onclick="closeDrawer()">✕</button>
  </div>
  <div class="drawer-body" id="drawerBody"></div>
</aside>

<script>
if (window.marked) {
  marked.setOptions({ breaks: true, gfm: true, headerIds: true, headerPrefix: 'h-', mangle: false });
}
function md(text) {
  if (!text) return '';
  if (!window.marked || !window.DOMPurify) return escape(text);
  const html = marked.parse(String(text));
  return DOMPurify.sanitize(html, { ADD_ATTR: ['target', 'id'] });
}
function linkifyRefs(html) {
  return html.replace(/\[(\d{1,2})\]/g, (m, n) => `<a href="#ref-${n}" class="ref" data-ref="${n}">[${n}]</a>`);
}
function escape(s) {
  return String(s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

// ===== 全局状态 =====
let SID = localStorage.getItem('ragbook_sid') || null;
let BOOKS = [];
let BOOK = localStorage.getItem('ragbook_book') || null;
let OUTLINE = {kind:'topic', items:[]};
const $list = document.getElementById('sessList');
const $msgs = document.getElementById('msgs');
const $q = document.getElementById('q');
const $btn = document.getElementById('btn');
const $title = document.getElementById('sessTitle');
// API 基址：直连（/）和反代（/t/ragbook/）都能跑 —— 相对路径，不要硬编码前缀
const BASE = location.pathname.endsWith('/') ? location.pathname : location.pathname.replace(/[^/]*$/, '');
const $bookSel = document.getElementById('bookSel');
const $bookMeta = document.getElementById('bookMeta');
const $tocList = document.getElementById('tocList');
const $tocTitle = document.getElementById('tocTitle');
const $drawer = document.getElementById('drawer');
const $drawerBody = document.getElementById('drawerBody');
const $drawerTitle = document.getElementById('drawerTitle');
const $drawerBg = document.getElementById('drawerBg');

function currentBook() { return BOOKS.find(b => b.id === BOOK) || null; }

// ===== 书清单 =====
async function loadBooks() {
  const r = await fetch(BASE+'api/books');
  const d = await r.json();
  BOOKS = d.books || [];
  if (!BOOKS.length) {
    $bookSel.innerHTML = '<option>暂无书目</option>';
    return;
  }
  if (!BOOK || !BOOKS.some(b => b.id === BOOK)) {
    const first = BOOKS.find(b => b.chunk_count) || BOOKS[0];
    BOOK = first.id;
    localStorage.setItem('ragbook_book', BOOK);
  }
  $bookSel.innerHTML = BOOKS.map(b =>
    `<option value="${b.id}"${b.id===BOOK?' selected':''}>${escape(b.title)}</option>`).join('');
  document.getElementById('bookCount').textContent = BOOKS.length + ' 本书';
  applyBook();
}

function applyBook() {
  const b = currentBook();
  if (!b) return;
  OUTLINE = b.outline || {kind:'topic', items:[]};
  $bookMeta.textContent = [b.author, b.chunk_count ? (b.chunk_count + ' 块') : '未建索引'].filter(Boolean).join(' · ');
  document.getElementById('vecInfo').textContent = '';
  localStorage.setItem('ragbook_book', BOOK);
  renderOutline();
  renderEmpty();
  // 占位文案也不按书硬编章号/术语：chapter 类统一给通用示例，避免黄仁勋那本提示「第8章·弱隔离级别」
  $q.placeholder = b.meta_kind === 'chapter'
    ? '问点什么…（如「第8章讲什么」「某一节讲了什么」）'
    : '问点什么…（如「快排怎么写」「TCP 三次握手为什么是三次」）';
}

function onBookChange(v) {
  BOOK = v;
  applyBook();
  // 切书 = 新话题，清空当前会话视图（后端会话保留）
  SID = null;
  localStorage.removeItem('ragbook_sid');
  loadSessions();
  $msgs.innerHTML = '';
  $title.textContent = '新对话';
  renderEmpty();
}

function renderOutline() {
  const items = OUTLINE.items || [];
  const app = document.getElementById('app');
  if (items.length) app.classList.add('with-toc'); else app.classList.remove('with-toc');
  if (OUTLINE.kind === 'chapter') {
    $tocTitle.textContent = '章节';
    // it.kind 区分条目类型（parse.py 生成）：
    //   chapter = 正文章（key 是章号）| region = 前置/后记/致谢 | guide = 导读文章
    // 三类的提问文案不同：章用"第N章 标题"，其余直接用标题。
    const askText = it => it.kind === 'chapter'
      ? `第${it.key}章 ${it.title}`
      : (it.title || '');
    $tocList.innerHTML = items.map(it => {
      const subs = (it.sections || []).map(s =>
        `<div class="toc-item sub" onclick="askAbout('${escape(askText(it)).replace(/'/g, "\\'")} ${escape(s).replace(/'/g, "\\'")}')">${escape(s)}</div>`).join('');
      const label = it.kind === 'chapter'
        ? `${escape(it.key)} · ${escape(it.title)}`
        : escape(it.title);
      return `<div class="toc-item" onclick="askAbout('${escape(askText(it)).replace(/'/g, "\\'")}')">
                <span>${label}</span></div>${subs}`;
    }).join('');
  } else {
    $tocTitle.textContent = '主题';
    // files 元素是 {name, topic} 对象（parse.py 生成的），取 name 展示、topic 提问
    const fname = f => (f && typeof f === 'object') ? (f.name || '') : String(f || '');
    const ftopic = f => (f && typeof f === 'object') ? (f.topic || f.name || '') : String(f || '');
    $tocList.innerHTML = items.map(it => {
      const files = (it.files || []).map(f =>
        `<div class="toc-item sub" onclick="askAbout('${escape(ftopic(f)).replace(/'/g, "\\'")}')">${escape(fname(f))}</div>`).join('');
      return `<div class="toc-group">${escape(it.title)}</div>
              <div class="toc-item" onclick="askAbout('${escape(it.title).replace(/'/g, "\\'")}')"><span>${escape(it.title)}</span>
                <span class="n">${(it.files||[]).length}</span></div>${files}`;
    }).join('');
  }
}

function askAbout(text) {
  $q.value = text;
  $q.focus();
  send();
}

// 空状态的推荐问题：book.json 的 meta.suggestions > 按目录树推导 > 通用兜底。
// 以前按 meta_kind 硬编第四章那四条，任何 chapter 类的书都套用 DDIA 的问题。
function suggestionsFor(b) {
  if (b.suggestions && b.suggestions.length) return b.suggestions.slice(0, 4);
  const items = (b.outline && b.outline.items) || [];
  if (b.meta_kind === 'chapter') {
    const chs = items.filter(it => /^\d+$/.test(it.key || ''));
    const picks = [chs[0], chs[Math.floor(chs.length / 2)], chs[chs.length - 1]].filter(Boolean);
    const out = picks.map(it => `第${it.key}章讲了什么？`);
    const withSec = chs.find(it => (it.sections || []).length > 1);
    if (withSec) out.push(`「${withSec.sections[1]}」讲了什么？`);
    if (out.length) return out.slice(0, 4);
  }
  return ['快速排序怎么实现？', 'TCP 三次握手为什么是三次？', 'B+ 树索引的原理？', 'MySQL 索引为什么用 B+ 树？'];
}

function renderEmpty() {
  const b = currentBook();
  if (!b) { $msgs.innerHTML = '<div class="empty"><h2>还没有可用的书</h2><p>先跑 parse.py + index.py 建索引</p></div>'; return; }
  const sug = suggestionsFor(b);
  $msgs.innerHTML = `<div class="empty">
    <div class="eyebrow">${escape(b.subtitle || b.title)}</div>
    <h2>${escape(b.title)}</h2>
    <p>${escape(b.author || '')}${b.author ? ' · ' : ''}全部内容已建索引，问点具体的</p>
    <div class="suggest">${sug.map(s =>
      `<button onclick="askAbout('${escape(s).replace(/'/g, "\\'")}')"><span>${escape(s)}</span><span class="arrow">→</span></button>`).join('')}</div>
  </div>`;
}

// ===== 会话 =====
$title.addEventListener('click', () => {
  const old = $title.textContent;
  $title.contentEditable = 'true';
  $title.focus();
  const r = document.createRange(); r.selectNodeContents($title);
  const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
  const finish = () => {
    $title.contentEditable = 'false';
    $title.removeEventListener('blur', finish);
    $title.removeEventListener('keydown', onKey);
    const newTitle = ($title.textContent || '').trim() || old;
    $title.textContent = newTitle;
    if (SID && newTitle !== old) {
      fetch(BASE+'api/sessions/' + SID + '/rename', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({title: newTitle})
      }).then(() => loadSessions());
    }
  };
  const onKey = e => { if (e.key === 'Enter') { e.preventDefault(); $title.blur(); }
                      if (e.key === 'Escape') { $title.textContent = old; $title.blur(); } };
  $title.addEventListener('blur', finish);
  $title.addEventListener('keydown', onKey);
});

$q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$q.addEventListener('input', () => {
  $q.style.height = 'auto';
  $q.style.height = Math.min($q.scrollHeight, 180) + 'px';
});

async function loadSessions() {
  const r = await fetch(BASE+'api/sessions');
  const d = await r.json();
  $list.innerHTML = '';
  if (!d.sessions.length) {
    $list.innerHTML = '<div class="empty-list">还没有对话<br>点「＋ 新对话」开始</div>';
    return;
  }
  d.sessions.forEach(s => {
    const div = document.createElement('div');
    div.className = 'sess-item' + (s.id === SID ? ' active' : '');
    div.dataset.sid = s.id;
    const preview = s.first_q ? s.first_q.slice(0, 30) : '空对话';
    const ago = relTime(s.updated_at);
    const bookTag = s.book ? (BOOKS.find(b=>b.id===s.book)?.title || s.book) : '';
    div.innerHTML = `
      <div class="title">
        <div>${escape(s.title)}</div>
        <div class="meta">${escape(bookTag ? bookTag + ' · ' : '')}${escape(preview)} · ${ago}</div>
      </div>
      <div class="actions">
        <button onclick="renameInline(event,'${s.id}')" title="重命名">✎</button>
        <button onclick="delSession(event,'${s.id}')" title="删除">🗑</button>
      </div>`;
    div.onclick = e => {
      if (e.target.closest('.actions')) return;
      switchSession(s.id);
    };
    $list.appendChild(div);
  });
}

function relTime(ts) {
  const d = (Date.now()/1000 - ts);
  if (d < 60) return '刚刚';
  if (d < 3600) return Math.floor(d/60) + '分钟前';
  if (d < 86400) return Math.floor(d/3600) + '小时前';
  if (d < 86400*7) return Math.floor(d/86400) + '天前';
  return new Date(ts*1000).toLocaleDateString('zh-CN', {month:'numeric', day:'numeric'});
}

async function loadSession(sid) {
  const r = await fetch(BASE+'api/session/' + sid);
  const d = await r.json();
  $msgs.innerHTML = '';
  if (!d.messages.length) {
    renderEmpty();
  } else {
    d.messages.forEach(m => {
      if (m.role === 'user') appendMsg('user', m.content);
      else appendMsg('assistant', m.content, m.hits, m.llm_ms ? `LLM ${m.llm_ms}ms` : '');
    });
  }
  $msgs.scrollTop = $msgs.scrollHeight;
}

async function switchSession(sid) {
  SID = sid;
  localStorage.setItem('ragbook_sid', SID);
  document.querySelectorAll('.sess-item').forEach(el => el.classList.toggle('active', el.dataset.sid === sid));
  const r = await fetch(BASE+'api/session/' + sid);
  const d = await r.json();
  $title.textContent = d.title || '新对话';
  if (d.book && d.book !== BOOK) { BOOK = d.book; $bookSel.value = BOOK; applyBook(); }
  await loadSession(sid);
}

async function newChat() {
  const r = await fetch(BASE+'api/sessions', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({book: BOOK})
  });
  const d = await r.json();
  SID = d.session_id;
  localStorage.setItem('ragbook_sid', SID);
  await loadSessions();
  switchSession(SID);
  $q.focus();
}

async function renameInline(ev, sid) {
  ev.stopPropagation();
  const item = ev.target.closest('.sess-item');
  const titleEl = item.querySelector('.title > div');
  const old = titleEl.textContent;
  const neu = prompt('新标题：', old);
  if (neu === null || neu.trim() === '' || neu === old) return;
  await fetch(BASE+'api/sessions/' + sid + '/rename', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({title: neu.trim().slice(0,80)})
  });
  if (sid === SID) $title.textContent = neu.trim().slice(0,80);
  await loadSessions();
}

async function delSession(ev, sid) {
  ev.stopPropagation();
  if (!confirm('删除这个会话？')) return;
  await fetch(BASE+'api/sessions/' + sid, {method:'DELETE'});
  if (sid === SID) {
    SID = null;
    localStorage.removeItem('ragbook_sid');
    $msgs.innerHTML = '';
    $title.textContent = '新对话';
    renderEmpty();
  }
  await loadSessions();
}

async function send() {
  const q = $q.value.trim();
  if (!q) return;
  if (!SID) {
    const r = await fetch(BASE+'api/sessions', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({book: BOOK})
    });
    SID = (await r.json()).session_id;
  }
  $btn.disabled = true;
  $q.value = '';
  $q.style.height = 'auto';
  removeEmpty();
  appendMsg('user', q);
  const placeholder = appendMsg('assistant', '正在检索 + 生成回答…');

  const ctrl = new AbortController();
  const guard = setTimeout(() => ctrl.abort(), 120000);
  try {
    const r = await fetch(BASE+'api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query:q, session_id:SID, top_k:5, book:BOOK }),
      signal: ctrl.signal
    });
    const raw = await r.text();
    let d = null;
    if (raw && (r.headers.get('content-type') || '').includes('application/json')) {
      try { d = JSON.parse(raw); } catch { d = null; }
    }
    placeholder.remove();
    if (!d) {
      appendMsg('assistant', `⚠️ 服务响应异常（HTTP ${r.status}）。若回答已生成会自动显示，否则请稍后重试。`);
      try { await loadSession(SID); } catch {}
    } else if (d.error) {
      appendMsg('assistant', '⚠️ ' + d.error, d.hits || []);
    } else {
      const intentTag = d.intent ? ` · ${d.intent}[${d.intent_src}]` : '';
      appendMsg('assistant', d.answer || '（无回答）', d.hits || [],
                `检索 ${d.search_ms}ms · LLM ${d.llm_ms}ms${intentTag}`);
      SID = d.session_id;
      localStorage.setItem('ragbook_sid', SID);
      await loadSessions();
      document.querySelectorAll('.sess-item').forEach(el => el.classList.toggle('active', el.dataset.sid === SID));
      const item = document.querySelector(`.sess-item[data-sid="${SID}"]`);
      if (item) $title.textContent = item.querySelector('.title > div').textContent;
    }
  } catch (e) {
    placeholder.remove();
    if (e.name === 'AbortError') {
      appendMsg('assistant', '⏳ 生成超时被中断。回答可能仍在生成：请稍等几秒刷新查看，或重新发送问题。');
    } else {
      appendMsg('assistant', '⚠️ 网络错误：' + e.message + '。请检查服务是否在线后重试。');
    }
  } finally {
    clearTimeout(guard);
  }
  $btn.disabled = false;
  $q.focus();
  $msgs.scrollTop = $msgs.scrollHeight;
}

function appendMsg(role, text, hits, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const bodyHtml = role === 'assistant' ? linkifyRefs(md(text || '')) : escape(text || '');
  const hitsHtml = hits && hits.length ? renderHits(hits) : '';
  const metaHtml = meta ? `<div class="meta">${meta}</div>` : '';
  div.innerHTML = `<div class="avatar">${role==='user'?'我':'书'}</div>
                   <div class="body">${bodyHtml}${hitsHtml}${metaHtml}</div>`;
  const body = div.querySelector('.body');
  body.addEventListener('click', e => {
    const a = e.target.closest('a.ref');
    if (a) { e.preventDefault(); jumpToRef(a.dataset.ref, div); return; }
    const src = e.target.closest('.src-btn');
    if (src) { e.preventDefault(); openSource(src.dataset.book, src.dataset.source, src.dataset.anchor, div); }
  });
  $msgs.appendChild(div);
  $msgs.scrollTop = $msgs.scrollHeight;
  return div;
}

function jumpToRef(n, msgDiv) {
  const target = msgDiv.querySelector(`.hit#ref-${n}`);
  if (!target) return;
  let p = target.parentElement;
  while (p && p !== msgDiv) {
    if (p.tagName === 'DETAILS' && !p.open) p.open = true;
    p = p.parentElement;
  }
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.add('highlight');
  setTimeout(() => target.classList.remove('highlight'), 1500);
}

// ===== 原文抽屉 =====
const sourceCache = {};
async function openSource(book, source, anchor, msgDiv) {
  if (!book || !source) return;
  const key = book + '|' + source;
  $drawerTitle.textContent = source;
  $drawerBody.innerHTML = '<div style="color:var(--fg-faint)">加载中…</div>';
  $drawer.classList.add('open');
  $drawerBg.classList.add('open');
  try {
    let html = sourceCache[key];
    if (!html) {
      const url = BASE+'api/chapter/' + encodeURIComponent(book) + '/' +
                  source.split('#')[0].split('/').map(encodeURIComponent).join('/');
      const r = await fetch(url);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      html = md(d.markdown);
      sourceCache[key] = html;
    }
    $drawerBody.innerHTML = html;
    if (anchor) {
      const target = $drawerBody.querySelector('#' + CSS.escape(anchor));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        target.classList.add('highlight-anchor');
        setTimeout(() => target.classList.remove('highlight-anchor'), 1800);
      }
    }
  } catch (e) {
    $drawerBody.innerHTML = '<div style="color:#e08585">加载失败：' + escape(e.message) + '</div>';
  }
}
function closeDrawer() {
  $drawer.classList.remove('open');
  $drawerBg.classList.remove('open');
}

function renderHits(hits) {
  const groups = {};
  hits.forEach((h, i) => {
    const parts = (h.heading_path || '').split(' > ');
    const section = parts.length >= 2 ? parts.slice(0,2).join(' › ') : (parts[0] || '其他');
    if (!groups[section]) groups[section] = { items: [], indices: [] };
    groups[section].items.push(h);
    groups[section].indices.push(i + 1);
  });
  const sections = Object.keys(groups);
  let html = `<div class="hits"><div class="hits-head">📎 引用 ${hits.length} 条 · ${sections.length} 节</div>`;
  sections.forEach(sec => {
    const g = groups[sec];
    const idxRange = g.indices.length === 1 ? `[${g.indices[0]}]` : `[${g.indices[0]}–${g.indices[g.indices.length-1]}]`;
    html += `<details class="hit-group"><summary><span class="sec-name">${escape(sec)}</span><span class="sec-cnt">${idxRange} · ${g.items.length}条</span></summary>`;
    g.items.forEach((h, k) => {
      const origIdx = g.indices[k];
      const anchor = (h.source || '').split('#')[1] || '';
      const srcBtn = h.source
        ? `<span class="src-btn" data-book="${escape(h.book || BOOK)}" data-source="${escape(h.source)}" data-anchor="${escape(anchor)}">查看原文</span>`
        : '';
      const snipMd = md('…' + (h.snippet || '') + '…');
      html += `<div class="hit" id="ref-${origIdx}">
        <div class="path"><span class="idx">[${origIdx}]</span>${escape(h.heading_path)}
          <span class="score">${h.score}%</span>${srcBtn}</div>
        <div class="snip">${snipMd}</div>
      </div>`;
    });
    html += `</details>`;
  });
  html += `</div>`;
  return html;
}

function removeEmpty() {
  const e = $msgs.querySelector('.empty'); if (e) e.remove();
}

// ===== 移动端 =====
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('scrim').classList.toggle('open');
}
function toggleToc() {
  document.getElementById('toc').classList.toggle('open');
  document.getElementById('scrim').classList.toggle('open');
}
function closeMobile() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('toc').classList.remove('open');
  document.getElementById('scrim').classList.remove('open');
}

(async () => {
  await loadBooks();
  await loadSessions();
  if (SID) {
    const item = document.querySelector(`.sess-item[data-sid="${SID}"]`);
    if (item) await switchSession(SID);
    else { SID = null; localStorage.removeItem('ragbook_sid'); renderEmpty(); }
  } else {
    renderEmpty();
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 仅本机访问，由任务101 反代 /t/ragbook/ 过来
    app.run(host="127.0.0.1", port=82, debug=False, threaded=True)
