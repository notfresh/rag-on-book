#!/usr/bin/env python3
"""
RAG-on-Book Web 端 — Flask 后端（V2：会话管理 + SQLite 持久化）
监听 127.0.0.1:82，由任务101 反代 /t/ragbook/ 过来。

复用 ask.py 的核心：search() / build_prompt() / ask_llm()
进程级缓存：bge 模型只加载一次。

V2 新增：
- SQLite 持久化会话（/root/projects/rag-on-book/data/sessions.db）
- 会话列表/重命名/删除/创建
- 消息持久化（每条问答含 hits 都落库）
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


def get_or_create_session(sid: str | None) -> str:
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
        c.execute("INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?,?,?,?)",
                  (new_sid, "新对话", now, now))
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
            """SELECT s.id, s.title, s.created_at, s.updated_at,
                      (SELECT content FROM messages WHERE session_id=s.id AND role='user' ORDER BY id LIMIT 1) AS first_q,
                      (SELECT COUNT(*) FROM messages WHERE session_id=s.id) AS msg_count
               FROM sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [{
        "id": r["id"],
        "title": r["title"],
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


# ---------- 全局模型客户端（进程复用） ----------
_model = None
_milvus = None
_lock = Lock()


def ensure_loaded():
    """懒加载：第一次请求时才加载模型 + milvus client。"""
    global _model, _milvus
    if _model is not None and _milvus is not None:
        return
    with _lock:
        if _model is not None and _milvus is not None:
            return
        import torch
        torch.set_num_threads(ENCODE_THREADS)
        from sentence_transformers import SentenceTransformer
        from index import get_client, COLLECTION
        print("[web] 加载 bge-small-zh-v1.5 ...", flush=True)
        _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        client = get_client()
        if not client.has_collection(COLLECTION):
            raise RuntimeError("向量库不存在, 先跑 python3 index.py --full")
        client.load_collection(COLLECTION)
        _milvus = client
        print("[web] 模型 + milvus 就绪", flush=True)


# ---------- 意图分类（V2 多轮对话管理） ----------
# 路径：规则初筛 -> LLM 兜底分类 -> 路由（chitchat / followup / question）

# 规则初筛：纯问候词 / 寒暄（不区分大小写、不去标点）
_CHITCHAT_PATTERNS = [
    "你好", "您好", "你好啊", "你好呀", "你好呀", "hi", "HI", "Hi",
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
        history_ctx = "\n\n最近对话:\n" + "\n".join(
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
        }
        import urllib.request
        req = urllib.request.Request(
            ask.BASE_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ask.load_deepseek_key()}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            ans = data["choices"][0]["message"]["content"].strip().lower()
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
CHITCHAT_SYSTEM = """你是《设计数据密集型应用》(DDIA) 的读书搭子。用户当前没有具体问题，只是闲聊或打招呼。
请用轻松友好的口吻简短回应（不超过 80 字），不主动背诵书的目录、不重复介绍自己。
可以适度幽默，但不要油嘴滑舌。"""


# ---------- Layer 1: 元信息过滤（"第N章" 精确召回） ----------
# parse.py 已经把 chapter 存成数字字符串 "1"-"14"，section 存成中文标题
import re

# DDIA 章节白名单（14 章 + 各自 section）
META_BOOK = "ddia"  # 现在只支持 DDIA

CHAPTER_SECTIONS = {
    "1": ["章引言", "分析型与事务型系统", "云服务与自托管", "分布式与单节点系统", "数据系统、法律与社会"],
    "2": ["章引言", "描述性能", "可靠性与容错", "可维护性", "可伸缩性", "案例研究：社交网络首页时间线"],
    "3": ["章引言", "关系模型与文档模型", "图数据模型", "数据框、矩阵与数组", "事件溯源与 CQRS"],
    "4": ["章引言", "OLTP 系统的存储与索引", "分析型数据存储", "多维索引与全文索引"],
    "5": ["章引言", "编码数据的格式", "数据流的模式"],
    "6": ["章引言", "单主复制", "复制延迟的问题", "多主复制", "无主复制"],
    "7": ["章引言", "键值数据的分片", "分片与二级索引", "请求路由", "分片的利与弊"],
    "8": ["章引言", "事务到底是什么？", "弱隔离级别", "可串行化", "分布式事务"],
    "9": ["章引言", "故障与部分失效", "不可靠的网络", "不可靠的时钟", "知识、真相和谎言"],
    "10": ["章引言", "线性一致性", "共识", "ID 生成器和逻辑时钟"],
    "11": ["章引言", "批处理模型", "批处理用例", "分布式系统中的批处理", "使用 Unix 工具的批处理"],
    "12": ["章引言", "传递事件流", "数据库与流", "流处理"],
    "13": ["章引言", "数据集成", "追求正确性", "分拆数据库"],
    "14": ["章引言", "隐私与追踪", "预测分析"],
}
# chapter 数字 → 中文标题（用于回答里给 LLM 看上下文）
CHAPTER_TITLES = {
    "1": "数据系统架构中的权衡", "2": "定义非功能性需求", "3": "数据模型与查询语言",
    "4": "存储与检索", "5": "编码与演化", "6": "复制", "7": "分片", "8": "事务",
    "9": "分布式系统的麻烦", "10": "一致性与共识", "11": "批处理", "12": "流处理",
    "13": "流式系统的哲学", "14": "做正确的事情",
}


def _fuzzy_match_section(section_hint: str, chapter: str) -> str | None:
    """在指定章的 section 白名单里模糊匹配用户说的关键词。返回标准 section 名。"""
    candidates = CHAPTER_SECTIONS.get(chapter, [])
    hint = section_hint.strip()
    if not hint:
        return None
    # 精确相等
    if hint in candidates:
        return hint
    # 包含关系
    for c in candidates:
        if hint in c or c in hint:
            return c
    # 首字 + 关键字（如"事务"匹配"事务到底是什么？" / "弱隔离级别"）
    # 取 hint 中 2 字以上的中文片段去匹配
    for length in (4, 3, 2):
        for i in range(len(hint) - length + 1):
            sub = hint[i:i+length]
            if not re.match(r'^[\u4e00-\u9fa5]+$', sub):
                continue
            for c in candidates:
                if sub in c:
                    return c
    return None


def detect_meta_query(query: str) -> tuple[str | None, dict]:
    """检测 query 是不是元信息提问（"第N章"/"chN"/"第N章第X节"）。

    返回 (milvus_filter_expr, meta_info)。两者都为 None 表示不是元信息提问，走 bge 向量检索。
    """
    q = query.strip()
    # 1. 章节号识别（阿拉伯数字 + 中文数字）
    ch_num = None
    # 阿拉伯数字：第1章 / 第 1 章
    m = re.search(r'第\s*(\d{1,2})\s*章', q)
    if m:
        ch_num = m.group(1)
    if not ch_num:
        # 中文数字：第一章 / 第一章是什么
        m = re.search(r'第([一二三四五六七八九十]+)\s*章', q)
        if m:
            cn = m.group(1)
            cn_map = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10"}
            # 支持组合：十三=10+3，二十五=20+5，二十三=20+3，二十=20
            if cn in cn_map:
                ch_num = cn_map[cn]
            elif len(cn) == 2 and cn[0] == "十":
                ch_num = "1" + cn_map.get(cn[1], "")
            elif len(cn) == 2 and cn[1] == "十":
                ch_num = str(int(cn_map.get(cn[0],"0")) * 10)
            elif len(cn) == 3 and cn[1] == "十":
                ch_num = str(int(cn_map.get(cn[0],"0")) * 10 + int(cn_map.get(cn[2],"0")))
            else:
                ch_num = None
    if not ch_num:
        m = re.search(r'\bch\s*(\d{1,2})\b', q, re.IGNORECASE)
        if m:
            ch_num = m.group(1)
    if not ch_num:
        return None, {}
    if ch_num not in CHAPTER_SECTIONS:
        # 边界：用户问的章号超出 DDIA 范围（1-14）。返 out_of_range 让上层直接提示。
        return None, {"out_of_range": True, "max_chapter": len(CHAPTER_SECTIONS), "chapter": ch_num}

    # 2. 章节白名单里的章节：chapter == "N"
    # 不调 bge，纯精确召回（用 query 中"章号之后的部分"做关键词，让 LLM 提炼）
    ch_title = CHAPTER_TITLES[ch_num]

    # 3. 是否点名了节（query 含"章引言"/"可串行化"/"事务到底是什么"等关键词）
    section_hint = ""
    # 把"第N章"和"是什么/讲什么/总结"等噪声词去掉，剩下的去匹配 section
    noise = re.sub(r'第\s*\d{1,2}\s*章', '', q)
    noise = re.sub(r'(是什么|讲什么|讲讲|介绍|总结|主要内容|大概|简介|讲的什么|说的是|内容是)', '', noise)
    noise = noise.strip(" ？?，,。.：: ")
    section_hint = noise

    matched_section = None
    if section_hint:
        matched_section = _fuzzy_match_section(section_hint, ch_num)

    if matched_section:
        # 精确到节：用 section 过滤，再按章节内 chunk 顺序（pk）排序
        filter_expr = f"chapter == '{ch_num}' && section == '{matched_section}'"
        return filter_expr, {
            "chapter": ch_num,
            "chapter_title": ch_title,
            "section": matched_section,
            "skip_embedding": True,
            "sort_by": "pk",  # 该节内按 markdown 顺序
        }
    else:
        # 精确到章：召回整章所有块
        filter_expr = f"chapter == '{ch_num}'"
        return filter_expr, {
            "chapter": ch_num,
            "chapter_title": ch_title,
            "section": None,
            "skip_embedding": True,
            "sort_by": "pk",
        }


def meta_search(client, filter_expr: str, meta_info: dict) -> list[dict]:
    """元信息精确召回：用 filter 查 milvus，按 pk 升序（章节内顺序）。"""
    from index import COLLECTION
    res = client.query(
        COLLECTION,
        filter=filter_expr,
        output_fields=["book", "part", "chapter", "chapter_title", "section",
                       "subsection", "heading_path", "source", "content"],
        limit=200,  # 一章最多也就几十块
        sort="pk",  # milvus query 支持 sort
    )
    return res


def chitchat_reply(query: str, history_tail: list[dict] | None = None) -> str:
    """闲聊直接对话，不检索。"""
    msgs = [{"role": "system", "content": CHITCHAT_SYSTEM}]
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
        }
        import urllib.request
        req = urllib.request.Request(
            ask.BASE_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ask.load_deepseek_key()}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"（闲聊回应失败：{e}）"


# ---------- 提问/追问路径：历史拼进 prompt ----------
def build_prompt_with_history(query: str, hits: list, history_tail: list[dict]) -> str:
    """在 ask.build_prompt 基础上，把最近 2 轮问答拼进 prompt，让 LLM 理解代词/省略。"""
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

    return f"""你是一位精通《设计数据密集型应用》(DDIA) 的读书助手。
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
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    sid_in = data.get("session_id")
    top_k = int(data.get("top_k") or 5)
    book = data.get("book") or None
    no_llm = bool(data.get("no_llm"))

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    sid = get_or_create_session(sid_in)

    # 拉最近 4 条历史（用于意图分类 + prompt 拼历史）
    history_tail = get_messages(sid)[-4:] if sid else []

    # 0. 元信息提问（"第N章"/"第N章某节"）→ 优先精确召回，跳过 bge + 闲聊分类
    meta_filter, meta_info = detect_meta_query(query)
    if meta_info.get("out_of_range"):
        # 章节号超出范围：明确告诉用户，避免 LLM 误判
        max_ch = meta_info.get("max_chapter", 14)
        ch_num_oob = meta_info.get("chapter", "?")
        msg = f"DDIA 共 {max_ch} 章，没有第{ch_num_oob}章。"
        sid_record = get_or_create_session(sid_in)
        add_message(sid_record, "user", query)
        add_message(sid_record, "assistant", msg, hits=[], llm_ms=0)
        return jsonify({"session_id": sid_record, "answer": msg, "hits": [],
                        "search_ms": 0, "llm_ms": 0,
                        "intent": "meta_oob", "intent_src": "rule"})
    if meta_filter:
        try:
            ensure_loaded()
        except Exception as e:
            return jsonify({"error": f"模型加载失败: {e}"}), 500
        add_message(sid, "user", query)
        t0 = time.time()
        chunks = meta_search(_milvus, meta_filter, meta_info)
        search_ms = int((time.time() - t0) * 1000)
        if not chunks:
            msg = f"未找到第{meta_info.get('chapter','')}章的内容"
            if meta_info.get('section'):
                msg += f"「{meta_info['section']}」节"
            add_message(sid, "assistant", msg, hits=[], llm_ms=0)
            return jsonify({"session_id": sid, "answer": msg, "hits": [],
                            "search_ms": search_ms, "llm_ms": 0,
                            "intent": "meta", "intent_src": "rule"})
        # 用元信息专用 prompt（不限制"严格按片段"）
        hits_clean = [_hit_to_clean(c, score=100) for c in chunks]  # meta 召回不打分
        prompt = build_meta_prompt(query, chunks, meta_info)
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
            answer = chitchat_reply(query, history_tail=history_tail)
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
            "intent": intent,
            "intent_src": intent_src,
        })

    # 3. 提问/追问路径：检索 + LLM
    try:
        ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"模型加载失败: {e}"}), 500

    t0 = time.time()
    hits = ask.search(_milvus, _model, query, book=book, top_k=top_k)
    search_ms = int((time.time() - t0) * 1000)

    add_message(sid, "user", query)

    if not hits:
        add_message(sid, "assistant", "（未命中任何片段）", hits=[], llm_ms=0)
        return jsonify({"session_id": sid, "answer": "（未命中任何片段）", "hits": [],
                        "search_ms": search_ms, "llm_ms": 0,
                        "intent": intent, "intent_src": intent_src})

    if no_llm:
        answer = None
        llm_ms = 0
    else:
        t0 = time.time()
        # 追问路径：把历史拼进 prompt；提问路径也拼（无害，prompt 明确说"如不依赖可忽略"）
        prompt = build_prompt_with_history(query, hits, history_tail)
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
        "intent": intent,
        "intent_src": intent_src,
    })


def auto_title_if_new(sid: str, query: str):
    """如果会话标题还是默认"新对话"，用 query 前 30 字设为标题。"""
    with _db() as c:
        row = c.execute("SELECT title FROM sessions WHERE id=?", (sid,)).fetchone()
        if row and row["title"] == "新对话":
            auto_title = (query[:30] + ("…" if len(query) > 30 else "")).strip()
            c.execute("UPDATE sessions SET title=? WHERE id=?", (auto_title, sid))


def _hits_clean(hits):
    out = []
    for h in hits:
        out.append({
            "score": h["score"],
            "heading_path": h["heading_path"],
            "source": h["source"],
            "snippet": h["content"][:240].replace("\n", " "),
        })
    return out


def _hit_to_clean(c: dict, score: int = 100) -> dict:
    """milvus query() 返回的实体转 citation dict（meta 召回默认 score=100）。"""
    return {
        "score": score,
        "heading_path": c.get("heading_path", ""),
        "source": c.get("source", ""),
        "snippet": c.get("content", "")[:240].replace("\n", " "),
    }


def build_meta_prompt(query: str, chunks: list[dict], meta_info: dict) -> str:
    """元信息提问专用 prompt：拿到指定章/节的全文，让 LLM 自由总结/介绍。"""
    ch = meta_info.get("chapter")
    ch_title = meta_info.get("chapter_title", "")
    section = meta_info.get("section")

    scope = f"第{ch}章「{ch_title}」"
    if section:
        scope += f"「{section}」节"

    parts = []
    for i, c in enumerate(chunks, 1):
        loc = c.get("heading_path", "")
        parts.append(f"[{i}] {loc}\n{c.get('content','')}")
    context = "\n\n".join(parts)

    return f"""你是《设计数据密集型应用》(DDIA) 的读书助手。用户询问的是书的 {scope}。

下面是该章/节按书内顺序排列的全部内容（可能较长）。请根据用户的具体问题：
1. 如果用户问"是什么/讲什么/主要内容"：用结构化方式总结该章/节的核心主题、关键概念、典型应用场景。
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
    sid = get_or_create_session(None)
    return jsonify({"session_id": sid})


@app.route("/api/sessions/<sid>/rename", methods=["POST"])
def api_sessions_rename(sid):
    data = request.get_json(silent=True) or {}
    rename_session(sid, data.get("title", ""))
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def api_sessions_delete(sid):
    delete_session(sid)
    return jsonify({"ok": True})


@app.route("/api/session/<sid>")
def api_session(sid):
    return jsonify({"session_id": sid, "messages": get_messages(sid)})


@app.route("/api/chapter/<int:chapter>")
def api_chapter(chapter: int):
    """按需返回 DDIA 指定章完整 markdown（前端 marked 渲染）。

    用于"点击引用 → 展开整章"功能，避免启动时预加载所有章。
    """
    if chapter < 1 or chapter > 14:
        return jsonify({"error": f"DDIA 共 14 章，没有第{chapter}章"}), 404
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "books", "ddia", "content", "zh", f"ch{chapter}.md")
    if not os.path.exists(md_path):
        return jsonify({"error": f"章节文件不存在: ch{chapter}.md"}), 404
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    # 算每节标题的锚 id（前端跳锚用）
    return jsonify({"chapter": chapter, "markdown": md_content})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "model_loaded": _model is not None, "db": DB_PATH})


# ---------- 前端 HTML ----------
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RAG on Book · DDIA 问答</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<style>
  :root { --bg:#0f1419; --panel:#1a2028; --line:#2a3441; --fg:#d4dae3; --muted:#7a8694; --accent:#5b9dff; --user:#1e3a5f; --asst:#1f2c3a; --sidebar:#131922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; height:100vh; display:flex; }
  aside { width:240px; background:var(--sidebar); border-right:1px solid var(--line); display:flex; flex-direction:column; flex-shrink:0; }
  aside .brand { padding:14px 16px; border-bottom:1px solid var(--line); font-weight:600; }
  aside .newbtn { margin:10px 12px; padding:8px; background:var(--accent); color:#fff; border:none; border-radius:6px; cursor:pointer; font-weight:600; }
  aside .newbtn:hover { background:#4a8ae8; }
  aside .list { flex:1; overflow-y:auto; }
  .sess-item { padding:10px 14px; border-bottom:1px solid var(--line); cursor:pointer; display:flex; align-items:center; gap:8px; }
  .sess-item:hover { background:#1a2330; }
  .sess-item.active { background:#1e3a5f; }
  .sess-item .title { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:13px; }
  .sess-item .title small { color:var(--muted); font-size:11px; display:block; margin-top:2px; }
  .sess-item .actions { display:none; gap:4px; }
  .sess-item:hover .actions, .sess-item.active .actions { display:flex; }
  .sess-item .actions button { background:transparent; border:0; color:var(--muted); cursor:pointer; padding:2px 4px; font-size:14px; }
  .sess-item .actions button:hover { color:var(--fg); }
  aside .empty-list { padding:24px 16px; color:var(--muted); font-size:12px; text-align:center; }

  main { flex:1; display:flex; flex-direction:column; min-width:0; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:var(--muted); font-size:12px; }
  header .spacer { flex:1; }
  header .sess-title { color:var(--accent); font-size:13px; max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  header .sess-title[contenteditable=true] { background:#0f1419; padding:2px 8px; border-radius:4px; outline:1px solid var(--accent); }

  .chat { flex:1; overflow-y:auto; padding:20px; }
  .msg { max-width:780px; margin:0 auto 16px; display:flex; gap:10px; align-items:flex-start; }
  .msg .role { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0; }
  .msg.user .role { background:var(--user); }
  .msg.assistant .role { background:var(--asst); }
  .msg .bubble { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 12px; line-height:1.6; word-wrap:break-word; width:fit-content; max-width:600px; min-height:0; align-self:flex-start; }
  .msg.user .bubble { background:var(--user); border-color:#2a4a75; white-space:pre-wrap; padding:6px 12px; align-self:flex-end; }
  .msg .bubble h1, .msg .bubble h2, .msg .bubble h3 { margin:8px 0 6px; line-height:1.3; }
  .msg .bubble h1 { font-size:18px; }
  .msg .bubble h2 { font-size:16px; }
  .msg .bubble h3 { font-size:14px; }
  .msg .bubble p { margin:4px 0; }
  .msg .bubble ul, .msg .bubble ol { margin:4px 0; padding-left:22px; }
  .msg .bubble li { margin:2px 0; }
  .msg .bubble code { background:#0f1419; padding:1px 5px; border-radius:3px; font-size:12px; }
  .msg .bubble pre { background:#0f1419; padding:8px 10px; border-radius:4px; overflow-x:auto; }
  .msg .bubble blockquote { border-left:3px solid var(--accent); padding-left:10px; margin:6px 0; color:var(--muted); }
  .msg .bubble a { color:var(--accent); }
  .msg .bubble a.ref { color:#5b9dff; text-decoration:none; padding:1px 3px; border-radius:3px; background:rgba(91,157,255,0.1); font-weight:500; cursor:pointer; }
  .msg .bubble a.ref:hover { background:rgba(91,157,255,0.25); }

  .hits { margin-top:8px; padding-top:8px; border-top:1px dashed var(--line); font-size:12px; color:var(--muted); }
  .hits-head { color:var(--muted); margin-bottom:4px; font-size:11px; }
  .hit-group { margin-bottom:2px; }
  .hit-group > summary { cursor:pointer; padding:4px 8px; background:#161d26; border-radius:4px; list-style:none; display:flex; justify-content:space-between; align-items:center; font-size:12px; }
  .hit-group > summary::-webkit-details-marker { display:none; }
  .hit-group > summary::before { content:'▶ '; color:var(--muted); font-size:10px; transition:transform .15s; display:inline-block; }
  .hit-group[open] > summary::before { transform:rotate(90deg); }
  .hit-group .sec-name { color:var(--accent); font-weight:500; }
  .hit-group .sec-cnt { color:var(--muted); font-size:11px; }
  .hit { margin:3px 0 0 0; padding:6px 8px; background:#121820; border-radius:4px; scroll-margin-top:8px; transition:background 1.2s ease-out; }
  .hit.highlight { background:rgba(91,157,255,0.25); }
  .hit .path { color:var(--accent); }
  .hit .path .chap-link { color:var(--accent); text-decoration:none; cursor:pointer; padding:1px 4px; border-radius:3px; background:rgba(91,157,255,0.08); margin-left:4px; }
  .hit .path .chap-link:hover { background:rgba(91,157,255,0.25); }
  .hit .score { color:#5b9dff; font-weight:500; }
  .hit .src { color:var(--muted); font-size:11px; margin-top:1px; }
  .hit .snip { color:#9aa5b3; margin-top:3px; line-height:1.5; }

  .chapter-expand { margin:6px 0 8px; background:#0a0e13; border:1px solid var(--line); border-radius:6px; }
  .chapter-expand .head { display:flex; justify-content:space-between; align-items:center; padding:6px 12px; background:#161d26; border-bottom:1px solid var(--line); border-radius:6px 6px 0 0; }
  .chapter-expand .head .title { color:var(--accent); font-size:13px; font-weight:500; }
  .chapter-expand .head .close { background:transparent; border:0; color:var(--muted); cursor:pointer; font-size:14px; padding:0 4px; }
  .chapter-expand .head .close:hover { color:var(--fg); }
  .chapter-expand .body { padding:12px 16px; max-height:60vh; overflow-y:auto; line-height:1.7; }
  .chapter-expand .body h1, .chapter-expand .body h2, .chapter-expand .body h3, .chapter-expand .body h4 {
    color:var(--fg); margin:12px 0 6px; padding-top:6px;
  }
  .chapter-expand .body h1 { font-size:20px; border-bottom:1px solid var(--line); padding-bottom:6px; }
  .chapter-expand .body h2 { font-size:16px; color:var(--accent); scroll-margin-top:8px; }
  .chapter-expand .body h3 { font-size:14px; color:#9aa5b3; scroll-margin-top:8px; }
  .chapter-expand .body h4 { font-size:13px; color:var(--muted); }
  .chapter-expand .body p { margin:6px 0; }
  .chapter-expand .body ul, .chapter-expand .body ol { margin:6px 0; padding-left:24px; }
  .chapter-expand .body blockquote { border-left:3px solid var(--accent); padding-left:10px; margin:8px 0; color:var(--muted); }
  .chapter-expand .body code { background:#0f1419; padding:1px 5px; border-radius:3px; font-size:12px; }
  .chapter-expand .body pre { background:#0f1419; padding:8px 10px; border-radius:4px; overflow-x:auto; }
  .chapter-expand .body a { color:var(--accent); }
  .chapter-expand .body .highlight-anchor { background:rgba(91,157,255,0.25); padding:2px 4px; border-radius:3px; transition:background 1.5s ease-out; }

  .meta { color:var(--muted); font-size:11px; margin-top:8px; }

  footer { border-top:1px solid var(--line); padding:12px 20px; background:var(--panel); }
  .input-row { max-width:780px; margin:0 auto; display:flex; gap:8px; }
  textarea { flex:1; background:#0f1419; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:8px 10px; resize:none; font:inherit; height:60px; }
  textarea:focus { outline:none; border-color:var(--accent); }
  .send { background:var(--accent); color:#fff; border:none; padding:0 18px; border-radius:6px; cursor:pointer; font-weight:600; }
  .send:disabled { background:#3a4a5a; cursor:wait; }
  .empty { color:var(--muted); text-align:center; margin-top:40px; font-size:14px; }
  .empty code { background:var(--panel); padding:2px 6px; border-radius:3px; }
</style>
</head>
<body>

<aside>
  <div class="brand">📚 RAG on Book</div>
  <button class="newbtn" onclick="newChat()">+ 新对话</button>
  <div class="list" id="sessList"></div>
</aside>

<main>
  <header>
    <h1>DDIA 问答</h1>
    <span class="sub" id="sessId"></span>
    <span class="spacer"></span>
    <span class="sess-title" id="sessTitle" title="双击重命名">新对话</span>
  </header>
  <div class="chat" id="msgs">
    <div class="empty">
      试着问：<br><br>
      <code>DDIA 里事务隔离级别有哪些？</code><br>
      <code>LSM 树和 B+ 树的区别？</code><br>
      <code>两阶段提交有什么问题？</code><br>
      <code>第一章有哪些内容？</code>
    </div>
  </div>
  <footer>
    <div class="input-row">
      <textarea id="q" placeholder="输入问题，Enter 发送，Shift+Enter 换行" autofocus></textarea>
      <button class="send" id="btn" onclick="send()">发送</button>
    </div>
  </footer>
</main>

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

// ===== 章节缓存（key: chapter 数字 → 渲染好的 HTML + 锚点字典）=====
const chapterCache = {};  // ch -> { html, anchorMap }

// 把 markdown 的 heading 加上自定义 anchor id（slug 化中文）
function slugifyHeading(s) {
  // 简单中文 slug：保留汉字 + 拉丁字母数字，转 - ；md 本身已 headerIds 自动生成，
  // 我们只额外确保唯一
  return s.replace(/[^\w\u4e00-\u9fa5]+/g, '-').replace(/^-+|-+$/g, '');
}

// 从 heading_path（如"第8章 事务 > 弱隔离级别 > 读已提交"）抽出
// chapter（数字）和 target anchor（用于跳到该节）
function parseHeadingPath(hp) {
  const m = (hp || '').match(/第\s*(\d+)\s*章/);
  const chapter = m ? m[1] : null;
  // 跳过"第N章 ... "，剩下是层级路径，用 > 分隔
  const parts = (hp || '').split(' > ');
  const sectionPath = parts.slice(1).join(' > ');  // 去掉章
  return { chapter, sectionPath, full: hp || '' };
}

async function loadChapter(chNum) {
  if (chapterCache[chNum]) return chapterCache[chNum];
  const r = await fetch(`/t/ragbook/api/chapter/${chNum}`);
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.error || `HTTP ${r.status}`);
  }
  const d = await r.json();
  // marked 已给 heading 生成 id（headerPrefix: 'h-'），渲染完拿到 DOM 提取
  const wrapper = document.createElement('div');
  wrapper.innerHTML = md(d.markdown);
  // 收集 heading id 映射（h.text → h.id），用于定位
  const anchorMap = {};
  wrapper.querySelectorAll('h1, h2, h3, h4').forEach(h => {
    if (h.id) anchorMap[h.textContent.trim()] = h.id;
  });
  chapterCache[chNum] = { html: wrapper.innerHTML, anchorMap, title: wrapper.querySelector('h1')?.textContent || `第${chNum}章` };
  return chapterCache[chNum];
}

// 找到最匹配的 anchor（精确 > 包含 > 第一章/章引言兜底）
function findAnchor(chapterData, sectionPath) {
  const m = chapterData.anchorMap;
  if (!sectionPath) return null;
  // 精确匹配
  if (m[sectionPath]) return m[sectionPath];
  // 包含匹配
  for (const [text, id] of Object.entries(m)) {
    if (sectionPath.includes(text) || text.includes(sectionPath)) return id;
  }
  return null;
}

let SID = localStorage.getItem('ragbook_sid') || null;
const $list = document.getElementById('sessList');
const $msgs = document.getElementById('msgs');
const $q = document.getElementById('q');
const $btn = document.getElementById('btn');
const $title = document.getElementById('sessTitle');
const $sessId = document.getElementById('sessId');

$title.addEventListener('dblclick', () => {
  const old = $title.textContent;
  $title.contentEditable = 'true';
  $title.focus();
  const r = document.createRange(); r.selectNodeContents($title); const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
  const finish = () => {
    $title.contentEditable = 'false';
    $title.removeEventListener('blur', finish);
    $title.removeEventListener('keydown', onKey);
    const newTitle = ($title.textContent || '').trim() || old;
    $title.textContent = newTitle;
    if (SID && newTitle !== old) {
      fetch('/t/ragbook/api/sessions/' + SID + '/rename', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({title: newTitle})
      }).then(() => loadSessions());
    }
  };
  const onKey = e => { if (e.key === 'Enter') { e.preventDefault(); $title.blur(); } if (e.key === 'Escape') { $title.textContent = old; $title.blur(); } };
  $title.addEventListener('blur', finish);
  $title.addEventListener('keydown', onKey);
});

$q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

async function loadSessions() {
  const r = await fetch('/t/ragbook/api/sessions');
  const d = await r.json();
  $list.innerHTML = '';
  if (!d.sessions.length) {
    $list.innerHTML = '<div class="empty-list">还没有对话<br>点"+ 新对话"开始</div>';
    return;
  }
  d.sessions.forEach(s => {
    const div = document.createElement('div');
    div.className = 'sess-item' + (s.id === SID ? ' active' : '');
    div.dataset.sid = s.id;
    const preview = s.first_q ? s.first_q.slice(0, 30) : '空对话';
    const ago = relTime(s.updated_at);
    div.innerHTML = `
      <div class="title">
        <div>${escape(s.title)}</div>
        <small>${escape(preview)} · ${ago}</small>
      </div>
      <div class="actions">
        <button onclick="renameInline(event,'${s.id}')" title="重命名">✏️</button>
        <button onclick="delSession(event,'${s.id}')" title="删除">🗑</button>
      </div>
    `;
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
  const r = await fetch('/t/ragbook/api/session/' + sid);
  const d = await r.json();
  $msgs.innerHTML = '';
  if (!d.messages.length) {
    $msgs.innerHTML = '<div class="empty">开始你的第一个问题 ↓</div>';
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
  const r = await fetch('/t/ragbook/api/session/' + sid);
  const d = await r.json();
  $title.textContent = d.title || '新对话';
  $sessId.textContent = sid.slice(0,8);
  await loadSession(sid);
}

async function newChat() {
  const r = await fetch('/t/ragbook/api/sessions', {method:'POST'});
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
  await fetch('/t/ragbook/api/sessions/' + sid + '/rename', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({title: neu.trim().slice(0,80)})
  });
  if (sid === SID) $title.textContent = neu.trim().slice(0,80);
  await loadSessions();
}

async function delSession(ev, sid) {
  ev.stopPropagation();
  if (!confirm('删除这个会话？')) return;
  await fetch('/t/ragbook/api/sessions/' + sid, {method:'DELETE'});
  if (sid === SID) {
    SID = null;
    localStorage.removeItem('ragbook_sid');
    $msgs.innerHTML = '<div class="empty">点"+ 新对话"开始</div>';
    $title.textContent = '新对话';
    $sessId.textContent = '';
  }
  await loadSessions();
}

async function send() {
  const q = $q.value.trim();
  if (!q) return;
  if (!SID) {
    const r = await fetch('/t/ragbook/api/sessions', {method:'POST'});
    SID = (await r.json()).session_id;
  }
  $btn.disabled = true;
  $q.value = '';
  removeEmpty();
  appendMsg('user', q);
  const placeholder = appendMsg('assistant', '正在检索 + 生成回答…');

  const ctrl = new AbortController();
  const guard = setTimeout(() => ctrl.abort(), 90000);  // LLM 生成上限保护
  try {
    const r = await fetch('/t/ragbook/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query:q, session_id:SID, top_k:5 }),
      signal: ctrl.signal
    });
    // 先取文本再判断 JSON：空 body / 网关错误页不会让 json() 抛晦涩异常
    const raw = await r.text();
    let d = null;
    if (raw && (r.headers.get('content-type') || '').includes('application/json')) {
      try { d = JSON.parse(raw); } catch { d = null; }
    }
    placeholder.remove();
    if (!d) {
      // 网关断连 / 截断 / 502 等：回答可能已生成，重拉会话同步
      appendMsg('assistant', `⚠️ 服务响应异常（HTTP ${r.status}）。若回答已生成会自动显示，否则请稍后重试。`);
      try { await loadSession(SID); } catch {}
      $msgs.scrollTop = $msgs.scrollHeight;
    } else if (d.error) {
      appendMsg('assistant', '⚠️ ' + d.error, d.hits || []);
    } else {
      const intentTag = d.intent ? ` · ${d.intent}[${d.intent_src}]` : '';
      appendMsg('assistant', d.answer || '（无回答）', d.hits || [], `检索 ${d.search_ms}ms · LLM ${d.llm_ms}ms${intentTag}`);
      SID = d.session_id;
      localStorage.setItem('ragbook_sid', SID);
      await loadSessions();
      document.querySelectorAll('.sess-item').forEach(el => el.classList.toggle('active', el.dataset.sid === SID));
      const item = document.querySelector(`.sess-item[data-sid="${SID}"]`);
      if (item) { $title.textContent = item.querySelector('.title > div').textContent; $sessId.textContent = SID.slice(0,8); }
    }
  } catch (e) {
    placeholder.remove();
    if (e.name === 'AbortError') {
      appendMsg('assistant', '⏳ 生成超过 90 秒被中断。回答可能仍在生成：请稍等几秒后刷新页面查看，或重新发送问题。');
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
  let bodyHtml;
  if (role === 'assistant') {
    bodyHtml = linkifyRefs(md(text || ''));
  } else {
    bodyHtml = escape(text || '');
  }
  // 注意：模板必须单行紧凑拼接——user 气泡有 white-space:pre-wrap，
  // 跨行模板的换行/缩进会进入文本节点，把短消息撑成超高气泡
  const hitsHtml = hits && hits.length ? renderHits(hits) : '';
  const metaHtml = meta ? `<div class="meta">${meta}</div>` : '';
  div.innerHTML = `<div class="role">${role==='user'?'你':'书'}</div><div class="bubble">${bodyHtml}${hitsHtml}${metaHtml}</div>`;
  const bubble = div.querySelector('.bubble');
  // 引用跳转（LLM 答案里的 [N]）
  bubble.addEventListener('click', e => {
    const a = e.target.closest('a.ref');
    if (!a) return;
    e.preventDefault();
    const n = a.dataset.ref;
    jumpToRef(n, div);
  });
  // 章节展开（每条 hit 里的 📖 章节链接）
  bubble.addEventListener('click', e => {
    const link = e.target.closest('.chap-link');
    if (!link) return;
    e.preventDefault();
    e.stopPropagation();
    openChapter(link, div);
  });
  $msgs.appendChild(div);
  $msgs.scrollTop = $msgs.scrollHeight;
  return div;
}

// 在 hits 区域跳转高亮
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

// 点击 hit 的章节链接 → 在该 hit 下方展开整章 markdown
async function openChapter(linkEl, msgDiv) {
  const hitEl = linkEl.closest('.hit');
  const chNum = linkEl.dataset.chapter;
  const sectionPath = linkEl.dataset.section || '';
  // 同一 hit 已经展开过 → 关闭
  const existing = hitEl.nextElementSibling;
  if (existing && existing.classList && existing.classList.contains('chapter-expand') && existing.dataset.chapter === chNum) {
    existing.remove();
    linkEl.textContent = '📖 展开本章';
    return;
  }
  linkEl.textContent = '⏳ 加载中…';
  try {
    const data = await loadChapter(chNum);
    // 删除可能存在的旧展开（不同章节）
    if (existing && existing.classList && existing.classList.contains('chapter-expand')) {
      existing.remove();
    }
    const div = document.createElement('div');
    div.className = 'chapter-expand';
    div.dataset.chapter = chNum;
    div.innerHTML = `
      <div class="head">
        <span class="title">📖 ${escape(data.title)}</span>
        <button class="close" title="收起">✕</button>
      </div>
      <div class="body">${data.html}</div>
    `;
    div.querySelector('.close').onclick = () => {
      div.remove();
      linkEl.textContent = '📖 展开本章';
    };
    hitEl.after(div);
    // 跳到指定节
    const anchor = findAnchor(data, sectionPath);
    if (anchor) {
      const target = div.querySelector('#' + CSS.escape(anchor));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        target.classList.add('highlight-anchor');
        setTimeout(() => target.classList.remove('highlight-anchor'), 1800);
      }
    } else {
      div.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    linkEl.textContent = '📕 收起本章';
  } catch (e) {
    linkEl.textContent = '⚠️ ' + e.message;
  }
}

function renderHits(hits) {
  const groups = {};
  hits.forEach((h, i) => {
    const parts = (h.heading_path || '').split(' > ');
    const section = parts.length >= 2 ? parts[1] : parts[0] || '其他';
    if (!groups[section]) groups[section] = { items: [], indices: [] };
    groups[section].items.push(h);
    groups[section].indices.push(i + 1);
  });
  const sections = Object.keys(groups);
  let html = `<div class="hits"><div class="hits-head">📎 引用 ${hits.length} 条 · ${sections.length} 节</div>`;
  sections.forEach(sec => {
    const g = groups[sec];
    const idxRange = g.indices.length === 1 ? `[${g.indices[0]}]` : `[${g.indices[0]}-${g.indices[g.indices.length-1]}]`;
    html += `<details class="hit-group"><summary><span class="sec-name">${escape(sec)}</span><span class="sec-cnt">${idxRange} ${g.items.length}条</span></summary>`;
    g.items.forEach((h, k) => {
      const origIdx = g.indices[k];
      const parsed = parseHeadingPath(h.heading_path);
      const chapBtn = parsed.chapter
        ? `<a class="chap-link" data-chapter="${parsed.chapter}" data-section="${escape(parsed.sectionPath)}">📖 展开本章</a>`
        : '';
      const snipMd = md('…' + (h.snippet || '') + '…');
      html += `<div class="hit" id="ref-${origIdx}">
        <div class="path">[${origIdx}] ${escape(h.heading_path)} <span class="score">${h.score}%</span>${chapBtn}</div>
        <div class="src">${escape(h.source)}</div>
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

function escape(s) {
  return String(s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

(async () => {
  await loadSessions();
  if (SID) {
    const item = document.querySelector(`.sess-item[data-sid="${SID}"]`);
    if (item) {
      switchSession(SID);
    } else {
      SID = null;
      localStorage.removeItem('ragbook_sid');
    }
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 仅本机访问，由任务101 反代 /t/ragbook/ 过来
    app.run(host="127.0.0.1", port=82, debug=False, threaded=True)
