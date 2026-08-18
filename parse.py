#!/usr/bin/env python3
"""
RAG on Book — parse.py：把 Markdown 书拆成带章节路径的检索块

输入: books/<book>/content/zh/*.md  (Vonng/ddia 结构: frontmatter + ##/### 小节)
输出: books/<book>/parsed.jsonl      每行一个块 {heading_path, content, source, ...}

定位链设计（书版相对 issue 版的关键差异）:
  issue 版: title 独立成块, 命中后按 number 聚合
  书版:     小节标题 + 正文一起 embed（前文后文语义强, 标题是块的一部分）,
            heading_path 记录完整定位链, 问答时直接引用

用法:
  python3 parse.py                    # 解析 books/ 下所有书
  python3 parse.py --book ddia        # 指定一本书
"""

import argparse
import json
import os
import re
import sys

BOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books")

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
    """解析单个 md 文件 → [{level, title, body, anchor}, ...] 小节列表"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    meta = parse_frontmatter(raw)
    body = strip_frontmatter(raw)
    body = COMMENT_RE.sub("", body)
    # 章尾脚注定义区剔除（V1 简化；脚注信息量记入 Future.md）
    body = FOOTNOTE_DEF_RE.sub("", body)

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
                cur_sec = {"title": title, "anchor": anchor, "subs": []}
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
                cur_sec.setdefault("body", []).append(cl)
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


def parse_book(book):
    """解析一本书 → 块列表。块结构：
    {pk, book, part, chapter, chapter_title, section, subsection,
     heading_path, heading, content, source, chunk_idx}
    """
    book_dir = os.path.join(BOOKS_DIR, book)
    if not os.path.isdir(book_dir):
        print(f"Error: 找不到书目录 {book_dir}")
        sys.exit(1)

    # 按文件名找章节 md（content/zh 或 content 下）
    # 排除站点页: 首页/目录/贡献者/版权/索引/分部目录页（噪音, 不入正文库）
    SKIP = {"_index.md", "toc.md", "contrib.md", "colophon.md", "indexes.md",
            "part-i.md", "part-ii.md", "part-iii.md"}
    md_files = []
    for root, _, files in os.walk(book_dir):
        for fn in sorted(files):
            if fn.endswith(".md") and fn not in SKIP:
                md_files.append(os.path.join(root, fn))
    md_files.sort()

    chunks = []
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
                    "pk": file_order * 1_000_000 + len(chunks),
                    "book": book, "part": part, "chapter": chapter,
                    "chapter_title": chapter_title, "section": heading,
                    "subsection": "", "heading": heading,
                    "heading_path": heading_path + " > 章引言",
                    "content": c, "source": rel, "chunk_idx": len(chunks),
                })

        for sec in sections:
            sec_title = sec["title"]
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
                        "pk": file_order * 1_000_000 + len(chunks),
                        "book": book, "part": part, "chapter": chapter,
                        "chapter_title": chapter_title, "section": sec_title,
                        "subsection": sub_title, "heading": heading,
                        "heading_path": hp, "content": c,
                        "source": source, "chunk_idx": len(chunks),
                    })
    return chunks, md_files


def make_chunks(text, heading, heading_path, source):
    """标题 + 正文一起切块：每块开头带小节标题（书版核心: 块自带上下文）。"""
    pieces = chunk_text(text)
    out = []
    for p in pieces:
        # 标题前置, 与正文一同 embed —— 检索时小节语义完整
        out.append(f"【{heading_path}】\n{p}")
    return out


def main():
    ap = argparse.ArgumentParser(description="解析 Markdown 书为检索块")
    ap.add_argument("--book", default=None, help="指定书目录名（默认全部）")
    args = ap.parse_args()

    books = [args.book] if args.book else [
        d for d in sorted(os.listdir(BOOKS_DIR))
        if os.path.isdir(os.path.join(BOOKS_DIR, d))
    ]
    for book in books:
        chunks, files = parse_book(book)
        out_path = os.path.join(BOOKS_DIR, book, "parsed.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        tokens = sum(estimate_tokens(c["content"]) for c in chunks)
        print(f"[{book}] {len(files)} 个 md 文件 → {len(chunks)} 块, 约 {tokens//1000}K tokens, 平均 {tokens//max(len(chunks),1)} token/块")
        print(f"  输出: {out_path}")


if __name__ == "__main__":
    main()
