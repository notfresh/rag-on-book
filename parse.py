#!/usr/bin/env python3
"""
RAG on Book — parse.py：把 Markdown 书拆成带章节路径的检索块

输入: books/<book>/<source_dir>/*.md
输出: books/<book>/parsed.jsonl      每行一个块 {pk, book, heading_path, content, source, ...}

定位链设计（书版相对 issue 版的关键差异）:
  issue 版: title 独立成块, 命中后按 number 聚合
  书版:     小节标题 + 正文一起 embed（前文后文语义强, 标题是块的一部分）,
            heading_path 记录完整定位链, 问答时直接引用

两种解析策略（由 books/<id>/book.json 的 kind 指定）:
  chapter_book —— 一本书 N 章，章号来自 frontmatter（DDIA）
  note_set     —— 一堆主题笔记文件，每文件一个主题（CS-Notes）

2026-11-18 改造：
  - 策略从 manifest 读，不再把 DDIA 的 frontmatter 字段写死
  - pk 跨书唯一（book.slot * 1e9 + 序号），此前两本书会从 1000000 重新开始撞主键
  - 章/主题清单自动生成到 parsed.jsonl 同目录的 outline.json，供 web 端做目录树
  - 章节小节表不再手写硬编码（原来 web.py 里维护一份 CHAPTER_SECTIONS，改解析器就得同步）

用法:
  python3 parse.py                    # 解析 books/ 下所有书
  python3 parse.py --book ddia        # 指定一本书
"""

import argparse
import fnmatch
import json
import os
import re
import sys

import books as books_mod

BOOKS_DIR = books_mod.BOOKS_DIR

# ---------- 切块参数（复用 service/index.py 验证过的 V1 策略） ----------
MAX_TOKENS = 480        # 块 token 上限
WINDOW_CHARS = 450      # 窗口兜底字符数
OVERLAP = 0.25          # 重叠比例
MIN_TOKENS = 10         # 碎片过滤

# ---------- 清理 ----------
HTML_TAG_RE = re.compile(r"<[^>]+>")
ANCHOR_RE = re.compile(r"\s*\{#[\w-]+\}\s*")           # {#sec_xxx} 锚点语法
FOOTNOTE_REF_RE = re.compile(r"\[\^\d+\]")              # 正文里的 [^1] 引用标记
FOOTNOTE_DEF_RE = re.compile(r"^\[\^\d+\]:.*$", re.MULTILINE)  # 章尾脚注定义行
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HR_RE = re.compile(r"^\s*(---|\*\*\*|___)\s*$")
COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
GFM_TOC_RE = re.compile(r"<!--\s*GFM-TOC\s*-->[\s\S]*?<!--\s*/?GFM-TOC\s*-->", re.IGNORECASE)


def estimate_tokens(text):
    """中文1字≈1token + 英文词×1.3 + 数字/符号占位。"""
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z0-9]+", text))
    other = len(re.findall(r"[^\u4e00-\u9fffA-Za-z0-9\s]", text))
    return cn + int(en * 1.3) + other // 2


def clean_line(line):
    """单行清理：HTML 标签、锚点语法、脚注引用标记、markdown 强调符。"""
    line = HTML_TAG_RE.sub("", line)
    line = ANCHOR_RE.sub("", line)
    line = FOOTNOTE_REF_RE.sub("", line)
    # 剥掉行内的 markdown 链接语法 [text](url) → text
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    # 图片 ![alt](url) → alt
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)
    # 剥掉强调/粗体符号，保留文字
    line = line.replace("**", "").replace("*", "").replace("`", "")
    line = line.replace("~~~", "")
    return line.strip()


def parse_frontmatter(text):
    """解析 YAML frontmatter 的关键字段（手写轻量解析，不依赖 yaml 库）。"""
    m = FRONTMATTER_RE.match(text)
    meta = {}
    if m:
        block = m.group(0)
        for line in block.strip("-\n").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"\'')
    return meta


def strip_frontmatter(text):
    return FRONTMATTER_RE.sub("", text, count=1)


def parse_chapter(path):
    """解析单个 md 文件 → (frontmatter, 小节列表)

    小节列表每项：{title, anchor, body: [...], subs: [{title, anchor, body}]}
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    meta = parse_frontmatter(raw)
    body = strip_frontmatter(raw)
    body = COMMENT_RE.sub("", body)
    # 章尾脚注定义区剔除（V1 简化；脚注信息量记入 Future.md）
    body = FOOTNOTE_DEF_RE.sub("", body)
    # GFM-TOC 目录块：纯导航噪音，入库只会污染召回
    body = GFM_TOC_RE.sub("", body)

    sections = []          # 顶层节(H2)列表，每节含 subs
    cur_sec = None
    lines = body.splitlines()
    for line in lines:
        if HR_RE.match(line) or not line.strip():
            continue
        hm = HEADING_RE.match(line)
        if hm:
            level, title = len(hm.group(1)), clean_line(hm.group(2))
            anchor = ""
            am = re.search(r"\{#([\w-]+)\}", line)
            if am:
                anchor = am.group(1)
            if not title or title.lower() in ("references", "参考文献"):
                # 参考文献节：内容入正文价值低, 标题保留但标记
                title = title or "参考文献"
            if level == 2:
                cur_sec = {"title": title, "anchor": anchor, "subs": [], "body": []}
                sections.append(cur_sec)
            elif level == 3 and cur_sec is not None:
                cur_sec["subs"].append({"title": title, "anchor": anchor, "body": []})
            elif level >= 4 and cur_sec is not None:
                if cur_sec["subs"]:
                    cur_sec["subs"][-1]["body"].append(title)
                else:
                    cur_sec["subs"].append({"title": title, "anchor": anchor, "body": []})
            elif level == 1:
                # H1 章标题：记录（一般 frontmatter 已有 title）
                if not meta.get("title"):
                    meta["title"] = title
            continue
        # 普通内容行
        cl = clean_line(line)
        if not cl:
            continue
        if cur_sec is not None:
            if cur_sec["subs"]:
                cur_sec["subs"][-1]["body"].append(cl)
            else:
                cur_sec["body"].append(cl)
        # 正文在第一个 H2 之前（章引言）——存为 intro
        else:
            meta.setdefault("intro", []).append(cl)

    return meta, sections


def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[。！？.!?；;])\s*|\n", text) if p.strip()]


def window_chunk(text):
    """窗口兜底：定长 + 重叠 + 就近标点落刀。"""
    if estimate_tokens(text) <= MAX_TOKENS:
        return [text]
    step = int(WINDOW_CHARS * (1 - OVERLAP))
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + WINDOW_CHARS]
        if len(chunk) >= WINDOW_CHARS:
            cut = max(chunk.rfind(c) for c in ("。", "！", "？", ".", "!", "\n"))
            if cut > WINDOW_CHARS // 2:
                chunk = chunk[:cut + 1]
        chunks.append(chunk.strip())
        i += step
        if i >= len(text):
            break
    return [c for c in chunks if c]


def chunk_text(text):
    """书版切块：小节内短段落贪心合并到大块（300~480 token），
    语义完整优先——一本书的检索粒度太碎会丢前后文。

    段落 → 累计合并至接近 MAX_TOKENS → 落块；超长单段 → 句号→窗口。
    """
    if not text:
        return []
    if estimate_tokens(text) <= MAX_TOKENS:
        return [text] if estimate_tokens(text) >= MIN_TOKENS else []

    result = []
    buf, buf_t = [], 0
    TARGET = MAX_TOKENS - 80  # 留标题前缀余量

    def flush():
        nonlocal buf, buf_t
        if buf:
            result.append("\n".join(buf))
            buf, buf_t = [], 0

    for para in split_paragraphs(text):
        t = estimate_tokens(para)
        if t <= TARGET:
            # 短段：并入当前缓冲
            if buf_t + t > TARGET:
                flush()
            buf.append(para)
            buf_t += t
        else:
            flush()
            # 超长单段：句号级切, 长句再窗口
            for sent in split_sentences(para):
                st = estimate_tokens(sent)
                if st <= TARGET:
                    if buf_t + st > TARGET:
                        flush()
                    buf.append(sent)
                    buf_t += st
                else:
                    flush()
                    result.extend(window_chunk(sent))
    flush()
    return [c for c in result if estimate_tokens(c) >= MIN_TOKENS]


def make_chunks(text, heading, heading_path, source):
    """标题 + 正文一起切块。

    注意（2026-11-18 修正）：**这里不再把 heading_path 拼进 content**。
    定位链只参与 embedding（index.py 编码时加前缀），不入库正文——
    否则入库正文会污染召回（用户搜"第3章"会召回正文恰好含"第3章"的不相关段落），
    也会让后续 FTS5 关键词路把定位链当正文词索引。
    原文上下文由 ask.py 的 prompt 单独提供（"[N] 出处: {heading_path}"）。
    """
    return chunk_text(text)


# ---------- 公共：文件收集 ----------

def _collect_files(book_dir: str, book: dict) -> list[str]:
    """按 manifest 的 source_dir + skip_files 收集 md 文件（跳过噪音页）。"""
    src = book.get("source_dir") or ""
    root = os.path.join(book_dir, src) if src else book_dir
    if not os.path.isdir(root):
        return []
    skip = book.get("skip_files") or []
    md_files = []
    for r, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            # skip_files 支持精确名和 glob（"算法 - 目录.md" / "*目录*"）
            if fn in skip or any(fnmatch.fnmatch(fn, pat) for pat in skip):
                continue
            md_files.append(os.path.join(r, fn))
    md_files.sort()
    return md_files


# ---------- 策略 1：chapter_book（一本书 N 章） ----------

def parse_chapter_book(book: dict) -> tuple[list[dict], list[str], dict]:
    """按章解析。返回 (chunks, md_files, outline)。"""
    book_id = book["id"]
    book_dir = os.path.join(BOOKS_DIR, book_id)
    md_files = _collect_files(book_dir, book)
    pk_base = books_mod.pk_base(book_id)

    chunks = []
    outline = {"kind": "chapter", "items": []}
    file_order = 0
    for path in md_files:
        meta, sections = parse_chapter(path)
        rel = os.path.relpath(path, book_dir).replace(os.sep, "/")
        file_order += 1

        chapter = meta.get("book_number", "")          # "1".."14" 或空(preface等)
        part = meta.get("book_part", "")               # I/II/III
        chapter_title = meta.get("title", "")
        # 章引言（第一个 H2 之前）作为独立小节
        intro = meta.get("intro", [])
        if intro:
            text = "\n".join(intro)
            heading_path = f"第{chapter}章 {chapter_title}" if chapter else chapter_title
            heading = "章引言"
            for c in make_chunks(text, heading, heading_path, f"{heading_path} > 章引言"):
                chunks.append({
                    "pk": pk_base + len(chunks),
                    "book": book_id, "part": part, "chapter": chapter,
                    "chapter_title": chapter_title, "section": heading,
                    "subsection": "", "heading": heading,
                    "heading_path": heading_path + " > 章引言",
                    "content": c, "source": rel, "chunk_idx": len(chunks),
                })

        chapter_sections = []
        if intro:
            chapter_sections.append("章引言")
        for sec in sections:
            sec_title = sec["title"]
            chapter_sections.append(sec_title)
            # 该节所有 subsection 的文本
            units = []  # [(sub_title, text)]
            if sec.get("subs"):
                for sub in sec["subs"]:
                    text = "\n".join(sub.get("body", [])).strip()
                    if text:
                        units.append((sub["title"], text))
            else:
                text = "\n".join(sec.get("body", [])).strip()
                if text:
                    units.append(("", text))
            if not units:
                continue  # 空节（如纯参考文献标题）
            for sub_title, text in units:
                heading = f"{sec_title} > {sub_title}" if sub_title else sec_title
                hp = f"第{chapter}章 {chapter_title} > {heading}" if chapter else f"{chapter_title} > {heading}"
                anchor = sec.get("anchor") or (sec.get("subs") and sec["subs"][0].get("anchor")) or ""
                source = f"{rel}#{anchor}" if anchor else rel
                for c in make_chunks(text, heading, hp, source):
                    chunks.append({
                        "pk": pk_base + len(chunks),
                        "book": book_id, "part": part, "chapter": chapter,
                        "chapter_title": chapter_title, "section": sec_title,
                        "subsection": sub_title, "heading": heading,
                        "heading_path": hp, "content": c,
                        "source": source, "chunk_idx": len(chunks),
                    })

        if chapter:
            outline["items"].append({
                "key": chapter,
                "title": chapter_title,
                "part": part,
                "sections": chapter_sections,
            })

    # 章按数字排序（DDIA 的 ch10 应在 ch9 之后）
    outline["items"].sort(key=lambda it: (it.get("part") or "", int(it["key"]) if it["key"].isdigit() else 0))
    return chunks, md_files, outline


# ---------- 策略 2：note_set（一堆主题笔记） ----------

def _match_group(filename: str, groups: list[dict]) -> str:
    """文件名 → 一级分组名。按 groups 顺序首个命中即返回；都没命中落 '其他'。"""
    base = os.path.basename(filename)
    stem = base[:-3] if base.endswith(".md") else base
    for g in groups:
        for pat in g.get("patterns") or []:
            if fnmatch.fnmatch(stem, pat):
                return g["name"]
    # 兜底：找名字里含 '其他' 的组
    for g in groups:
        if g["name"] == "其他":
            return g["name"]
    return "其他"


def parse_note_set(book: dict) -> tuple[list[dict], list[str], dict]:
    """按主题文件解析。定位链 = 主题分组 > 文件名 > 小节。

    与 chapter_book 的差别：
      - 没有"第N章"，主题名就是一级定位
      - 文件名本身是二级定位（H1 通常等于文件名）
      - H2/H3 成为三级定位
    """
    book_id = book["id"]
    book_dir = os.path.join(BOOKS_DIR, book_id)
    md_files = _collect_files(book_dir, book)
    pk_base = books_mod.pk_base(book_id)
    groups_cfg = book.get("meta", {}).get("groups") or []

    chunks = []
    group_map = {}   # 分组名 → [(文件名, 归一化 topic), ...]
    outline = {"kind": "topic", "items": []}

    for path in md_files:
        meta, sections = parse_chapter(path)
        rel = os.path.relpath(path, book_dir).replace(os.sep, "/")
        fname = os.path.basename(path)
        stem = fname[:-3] if fname.endswith(".md") else fname
        group = _match_group(fname, groups_cfg)

        # H1 优先（CS-Notes 的 H1 就是主题名），否则用文件名
        topic = meta.get("title") or stem
        # 主题名里剥掉分组前缀（"算法 - 排序" → "排序"），避免定位链冗长
        if group and topic.startswith(group + " - "):
            topic = topic[len(group) + 3:].strip() or topic
        elif group and topic.startswith(group + "-"):
            topic = topic[len(group) + 1:].strip() or topic
        group_map.setdefault(group, []).append({"name": stem, "topic": topic})

        hp0 = f"{group} > {topic}"

        # H1 之前的正文（简介）
        intro = meta.get("intro", [])
        if intro:
            text = "\n".join(intro)
            for c in make_chunks(text, topic, hp0, rel):
                chunks.append({
                    "pk": pk_base + len(chunks),
                    "book": book_id, "part": group, "chapter": "",
                    "chapter_title": topic, "section": "简介",
                    "subsection": "", "heading": topic,
                    "heading_path": hp0 + " > 简介",
                    "content": c, "source": rel, "chunk_idx": len(chunks),
                })

        for sec in sections:
            sec_title = sec["title"]
            units = []
            if sec.get("subs"):
                for sub in sec["subs"]:
                    text = "\n".join(sub.get("body", [])).strip()
                    if text:
                        units.append((sub["title"], text))
            else:
                text = "\n".join(sec.get("body", [])).strip()
                if text:
                    units.append(("", text))
            if not units:
                continue
            for sub_title, text in units:
                heading = f"{sec_title} > {sub_title}" if sub_title else sec_title
                hp = f"{hp0} > {heading}"
                for c in make_chunks(text, heading, hp, rel):
                    chunks.append({
                        "pk": pk_base + len(chunks),
                        "book": book_id, "part": group, "chapter": "",
                        "chapter_title": topic, "section": sec_title,
                        "subsection": sub_title, "heading": heading,
                        "heading_path": hp, "content": c,
                        "source": rel, "chunk_idx": len(chunks),
                    })

    # outline：分组 → 文件列表
    for g in groups_cfg:
        items = group_map.get(g["name"]) or []
        if items:
            outline["items"].append({"key": g["name"], "title": g["name"], "files": items})
    # 有文件但没落进任何组的（manifest 配漏了）→ 补一个"其他"组
    known = {g["name"] for g in groups_cfg}
    leftovers = [f for g, fs in group_map.items() if g not in known for f in fs]
    if leftovers:
        outline["items"].append({"key": "其他", "title": "其他", "files": leftovers})

    return chunks, md_files, outline


# ---------- 策略 3：ocr_book（OCR 版面 JSON 的一本书） ----------
# 载体与前两种不同：源不是 markdown，而是 PaddleOCR-VL 的版面 JSON。
# 章节/段落从版面信息还原（见 ocr_parser.py 顶部的实测事实）。

def parse_ocr_book(book: dict) -> tuple[list[dict], list[str], dict]:
    """按 OCR 版面 JSON 解析。返回 (chunks, md_files, outline)。"""
    import ocr_parser
    return ocr_parser.parse_ocr_book(book, chunk_text)


# ---------- 分派 ----------

PARSERS = {
    books_mod.KIND_CHAPTER_BOOK: parse_chapter_book,
    books_mod.KIND_NOTE_SET: parse_note_set,
    books_mod.KIND_OCR_BOOK: parse_ocr_book,
}


def parse_book(book_id: str):
    """解析一本书 → (chunks, md_files, outline)。按 manifest 的 kind 分派策略。"""
    book = books_mod.get_book(book_id)
    parser = PARSERS.get(book["kind"])
    if parser is None:
        raise books_mod.BookError(f"书 {book_id} 的 kind 没有对应解析器: {book['kind']}")
    return parser(book)


def main():
    ap = argparse.ArgumentParser(description="解析 Markdown 书为检索块")
    ap.add_argument("--book", default=None, help="指定书目录名（默认全部）")
    args = ap.parse_args()

    book_ids = [args.book] if args.book else books_mod.list_book_ids()
    if not book_ids:
        print(f"books/ 下没有书（需要在 books/<id>/book.json 声明 manifest）: {BOOKS_DIR}")
        return

    for book_id in book_ids:
        try:
            chunks, files, outline = parse_book(book_id)
        except books_mod.BookError as e:
            print(f"[{book_id}] 跳过：{e}")
            continue
        book_dir = os.path.join(BOOKS_DIR, book_id)
        out_path = os.path.join(book_dir, "parsed.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        # 目录树（web 端做侧边目录用）—— 从块反推，不再手写章节表
        with open(os.path.join(book_dir, "outline.json"), "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
        tokens = sum(estimate_tokens(c["content"]) for c in chunks)
        print(f"[{book_id}] {len(files)} 个 md 文件 → {len(chunks)} 块, 约 {tokens//1000}K tokens, 平均 {tokens//max(len(chunks),1)} token/块")
        print(f"  输出: {out_path}")


if __name__ == "__main__":
    main()
