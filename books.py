#!/usr/bin/env python3
"""
RAG on Book — books.py：书清单与元数据的**单一来源**

为什么有这个模块（2026-11-18 改造）：
  改造前"这本书叫什么 / 有几章 / 第N章是哪些小节 / 提问时怎么称呼这本书"
  这些信息散落在 web.py 的硬编码字典 + ask.py 的 prompt 字符串里，换一本书要改 5 处。

现在每本书在 books/<id>/book.json 里声明自己的元数据，本模块负责：
  - 发现书（扫 books/ 目录，读 manifest）
  - 校验（slot 不重复、manifest 字段齐全）
  - 提供两个策略入口：元信息提问识别（章节式 / 主题式）

manifest 字段：
  id            书标识（目录名）
  title         书名（显示 + prompt 用）
  author        作者
  subtitle      副标题（可选）
  kind          解析策略：chapter_book（按章）| note_set（笔记集，按主题文件）
  source_dir    源 markdown 所在子目录（相对 books/<id>/）
  skip_files    跳过文件名列表（目录页/索引页等噪音）
  meta          元信息提问策略
    kind        chapter（"第N章"）| topic（书名/主题名）
    chapters    chapter 模式：{"1": {"title": ..., "sections": [...]}, ...}
    groups      topic 模式：[{"name": ..., "patterns": [...]}, ...]  文件名→一级分组

目录布局（2026-11-18：一本书一个目录，目录里装完这本书的一切）：
  books/<id>/
    book.json          manifest
    <source_dir>/*.md  完整书籍原文
    outline.json       目录树（parse.py 生成）
    parsed.jsonl       切块产物（parse.py 生成）
    vector/            这本书自己的 Milvus Lite（book_chunks.db）
  会话库是**全局**的（data/sessions.db）—— 会话跟书无关，不按书拆。
"""

import json
import os
import re

BOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books")

# 每本书的向量库放在自己目录下的 vector/ 子目录
VECTOR_SUBDIR = "vector"
VECTOR_DB_NAME = "book_chunks.db"

# 支持的解析策略（parse.py 按这个分派）
KIND_CHAPTER_BOOK = "chapter_book"
KIND_NOTE_SET = "note_set"
SUPPORTED_KINDS = (KIND_CHAPTER_BOOK, KIND_NOTE_SET)

# 支持的元信息提问策略
META_CHAPTER = "chapter"
META_TOPIC = "topic"
SUPPORTED_META = (META_CHAPTER, META_TOPIC)


class BookError(Exception):
    """书配置有问题（manifest 缺失/字段错/slot 冲突）。"""


def manifest_path(book_id: str) -> str:
    return os.path.join(BOOKS_DIR, book_id, "book.json")


def _read_manifest(book_id: str) -> dict:
    path = manifest_path(book_id)
    if not os.path.exists(path):
        raise BookError(f"书 {book_id} 缺少 manifest: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            m = json.load(f)
        except ValueError as e:
            raise BookError(f"书 {book_id} 的 manifest 不是合法 JSON: {e}") from e

    required = ("id", "title", "kind")
    missing = [k for k in required if k not in m]
    if missing:
        raise BookError(f"书 {book_id} 的 manifest 缺字段: {', '.join(missing)}")
    if m["id"] != book_id:
        raise BookError(f"书 {book_id} 的 manifest id 不匹配: {m['id']}")
    if m["kind"] not in SUPPORTED_KINDS:
        raise BookError(f"书 {book_id} 的 kind 不支持: {m['kind']}（支持 {SUPPORTED_KINDS}）")
    meta = m.setdefault("meta", {})
    if meta.get("kind") not in SUPPORTED_META:
        raise BookError(f"书 {book_id} 的 meta.kind 不支持: {meta.get('kind')}")
    return m


def list_book_ids() -> list[str]:
    """books/ 下所有含 book.json 的目录名（按目录名排序）。"""
    if not os.path.isdir(BOOKS_DIR):
        return []
    out = []
    for d in sorted(os.listdir(BOOKS_DIR)):
        if os.path.isdir(os.path.join(BOOKS_DIR, d)) and os.path.exists(manifest_path(d)):
            out.append(d)
    return out


def list_books() -> list[dict]:
    """所有书的 manifest（供前端下拉用）。带块数与索引状态。"""
    books = []
    for bid in list_book_ids():
        try:
            m = _read_manifest(bid)
        except BookError:
            continue
        m = dict(m)
        m["chunk_count"] = count_chunks(bid)
        m["source_dir"] = m.get("source_dir", "")
        m["subtitle"] = m.get("subtitle", "")
        m["author"] = m.get("author", "")
        m["indexed"] = has_vector_store(bid)
        books.append(m)
    return books


def get_book(book_id: str) -> dict:
    """取单本书 manifest。找不到抛 BookError。"""
    if not book_id:
        raise BookError("书 id 为空")
    return _read_manifest(book_id)


def count_chunks(book_id: str) -> int:
    """已解析的块数（parsed.jsonl 行数）；没解析过返回 0。"""
    path = os.path.join(BOOKS_DIR, book_id, "parsed.jsonl")
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def book_dir(book_id: str) -> str:
    """这本书的根目录（books/<id>/）。"""
    return os.path.join(BOOKS_DIR, book_id)


def vector_dir(book_id: str) -> str:
    """这本书自己的向量库目录（books/<id>/vector/）。"""
    return os.path.join(book_dir(book_id), VECTOR_SUBDIR)


def vector_db_path(book_id: str) -> str:
    """这本书自己的 Milvus Lite 文件（books/<id>/vector/book_chunks.db）。"""
    return os.path.join(vector_dir(book_id), VECTOR_DB_NAME)


def has_vector_store(book_id: str) -> bool:
    """这本书是否已有向量库文件。"""
    return os.path.exists(vector_db_path(book_id))


def pk_base(book_id: str) -> int:
    """这本书的主键起点。

    2026-11-18 起**每书一个独立向量库**，pk 在书内自增，起点恒为 0。
    （旧版多书共库时靠 slot*10亿 分段，现已不需要。）
    """
    return 0


def is_indexed(book_id: str) -> bool:
    """这本书是否已建索引（parsed.jsonl 存在且非空）。"""
    return count_chunks(book_id) > 0


# ---------- 元信息提问：章节式（"第N章"） ----------

_CN_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _cn_to_int(s: str) -> int | None:
    """中文数字转整数，支持 一~九十九（够用）。"""
    if not s:
        return None
    if s in _CN_NUM:
        return _CN_NUM[s]
    if len(s) == 2 and s[0] == "十":
        return 10 + _CN_NUM.get(s[1], 0)
    if len(s) == 2 and s[1] == "十":
        return _CN_NUM.get(s[0], 0) * 10
    if len(s) == 3 and s[1] == "十":
        return _CN_NUM.get(s[0], 0) * 10 + _CN_NUM.get(s[2], 0)
    return None


def detect_chapter_num(query: str) -> str | None:
    """从 query 里抽章号（"第3章"/"第三章"/"ch3"）。抽不到返 None。"""
    q = query.strip()
    m = re.search(r'第\s*(\d{1,3})\s*章', q)
    if m:
        return m.group(1)
    m = re.search(r'第([一二三四五六七八九十]+)\s*章', q)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return str(n)
    m = re.search(r'\bch\s*(\d{1,3})\b', q, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def fuzzy_match_section(section_hint: str, candidates: list[str]) -> str | None:
    """在白名单里模糊匹配用户说的小节名。返回标准 section 名。

    注意（2026-11-18 修正）：排除"什么/怎么/如何"这类**疑问词碎片**。
    原实现会拿 2 字中文片段盲配，"讲了什么"里的"什么"命中了"事务到底是什么？"——
    用户问整章却被精确到某一节，比不匹配更糟（答案范围错了一半）。
    """
    hint = (section_hint or "").strip()
    if not hint or not candidates:
        return None
    if hint in candidates:
        return hint
    for c in candidates:
        if hint in c or c in hint:
            return c
    # 取 2-4 字的中文片段去匹配（"弱隔离" → "弱隔离级别"）
    for length in (4, 3, 2):
        for i in range(len(hint) - length + 1):
            sub = hint[i:i + length]
            if not re.match(r'^[\u4e00-\u9fa5]+$', sub):
                continue
            # 疑问词碎片不是有效的主题词，跳过
            if _is_question_fragment(sub):
                continue
            for c in candidates:
                if sub in c:
                    return c
    return None


# 疑问词/泛指碎片：单独出现时不能用来定位小节
_GENERIC_FRAGMENTS = {
    "什么", "怎么", "如何", "为什么", "哪些", "哪个", "多少", "是不是", "有吗",
    "讲讲", "介绍", "总结", "内容", "主要", "大概", "简介", "一下", "详细",
}


def _is_question_fragment(sub: str) -> bool:
    """片段是否只是疑问词/泛指词（不能作为小节定位依据）。

    判据：片段本身在泛指词表里，或整串能按 2 字切成全是泛指词的块
    （如"是什么"= "是什"+"么" 不成立，但 "有多少" 这类能切出来）。
    """
    if sub in _GENERIC_FRAGMENTS:
        return True
    if len(sub) % 2 == 0:
        parts = [sub[i:i + 2] for i in range(0, len(sub), 2)]
        if all(p in _GENERIC_FRAGMENTS for p in parts):
            return True
    return False


# ---------- 元信息提问：主题式（书名/主题名） ----------

def match_topic(query: str, book: dict, outline: dict | None = None) -> str | None:
    """从 query 里认出主题名（分组名或文件名）。认不出返 None。

    命中后可以对该主题做精确召回（不调 embedding）。
    书的**自身标题**不算主题（问"CS-Notes 讲了什么"是问整本书，不是问某个主题）。
    """
    q = query.strip()
    if not q:
        return None
    book_title = (book.get("title") or "").strip()
    if book_title and book_title in q:
        return None   # 问的是整本书

    groups = book.get("meta", {}).get("groups") or []
    best = None
    for g in groups:
        name = (g.get("name") or "").strip()
        if name and len(name) >= 2 and name in q:
            # 长的优先（"计算机网络" 优于 "网络"）
            if best is None or len(name) > len(best):
                best = name
    if best:
        return best

    # 文件名级：outline 里每个文件有原始名和归一化 topic（去掉分组前缀）
    for item in (outline or {}).get("items") or []:
        for f in item.get("files") or []:
            if isinstance(f, dict):
                candidates = [f.get("topic") or "", f.get("name") or ""]
            else:
                candidates = [str(f)]
            for c in candidates:
                c = (c or "").strip()
                # 文件名里带 ".md" 的先剥掉
                if c.endswith(".md"):
                    c = c[:-3]
                if c and len(c) >= 2 and c in q:
                    if best is None or len(c) > len(best):
                        best = c
    return best
