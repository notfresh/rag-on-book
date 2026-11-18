#!/usr/bin/env python3
"""
RAG on Book — ocr_parser.py：把 PaddleOCR-VL 版面 JSON 解析成检索块

载体：OCR 版面 JSON（不是 markdown）。以《黄仁勋：英伟达之芯》的
PaddleOCR-VL-1.6 输出（301 页）为实测样本，版面事实：

  - JSON 结构：每页一个条目 {prunedResult:{parsing_res_list:[块]}, markdown:{text:整页文本}}
  - 页内块顺序 = 阅读顺序（301 页零违例，实测 bbox 的 y 坐标单调递增）
  - block_label 分布：text 1521 / paragraph_title 80 / doc_title 25 / table 8 /
    figure_title 3 / number 6 / header 1
  - bbox.x0 可区分版面角色：正文块 x0<=150（1243/1521）、居中块 190<=x0<=215（标题/引文/版权行）
  - 「第N章」标记有两种形态：
      text 块整行 = "第N章"（第1~11、15~18、20~22 章）
      paragraph_title 块 = "##### 第N章"（第12、13、14、19、23 章）
    只匹配 heading 会漏掉 18 章，只匹配 text 会漏掉 5 章 —— 两种都要认。
  - 跨页断段：上一页末行不以终止标点结尾 = 段落被页边界切断，需与下一页首行合并
    （实测 151 处；其中前段长度 <15 字的是居中标题/版权行，不能合）

产出与 parse.py 的其他策略同构：parsed.jsonl + outline.json，
外加每章一份 md（books/<id>/chapters/*.md），供 web 端「点击引用 → 展开原文」。

用法（由 parse.py 按 manifest 的 kind="ocr_book" 分派）：
  python3 parse.py --book huang-renxun
"""

import json
import os
import re
from collections import OrderedDict

import books as books_mod

# ---------- 版面常量（实测值，改这里就够） ----------
BODY_X_MAX = 150          # 正文块 x0 上限：x0<=BODY_X_MAX 视为正文
CENTER_X_MIN = 190        # 居中块 x0 区间（标题 / 引文 / 版权行）
CENTER_X_MAX = 215
CHAPTER_TITLE_MAX_LEN = 80  # 章节标题行的长度上限（超过就不是标题了）
CONTINUE_MIN_LEN = 15     # 跨页续接：前段长度下限（更短的是标题/版权行，不合）

TERMINAL_RE = re.compile(r'[。！？；：…”"』」）]$')   # 段落终止标点
CHAPTER_RE = re.compile(r'^\s*#{0,6}\s*第\s*(\d+)\s*章\s*$')
PART_RE = re.compile(r'^第([一二三四五六七八九十])部分$')
HTML_RE = re.compile(r'<[^>]+>')

HEADING_LABELS = ("doc_title", "paragraph_title", "figure_title", "number", "header")
# number 块（01~06）是**导读文章编号**，不是小节标题 —— 单独处理
NUMBER_LABELS = ("number",)


# ---------- 小工具 ----------

def _clean_text(s: str) -> str:
    """去 HTML 标签 + 压空白。"""
    return re.sub(r'\s+', ' ', HTML_RE.sub(' ', s or '')).strip()


def _clean_title(s: str) -> str:
    """标题清理：剥 markdown 井号 + HTML + 压空白。"""
    return _clean_text(re.sub(r'^\s*#{1,6}\s*', '', s or ''))


def _zone(x0: int) -> str:
    """版面角色：body（正文）/ center（居中：标题·引文·版权行）/ other（右侧·跨栏）。"""
    if x0 <= BODY_X_MAX:
        return "body"
    if CENTER_X_MIN <= x0 <= CENTER_X_MAX:
        return "center"
    return "other"


def _table_rows(html: str) -> list[str]:
    """OCR 表格 HTML → 「名：说明」文本行。

    表格块形如 <table ...><tr><td>洛丽·米尔斯</td><td>黄仁勋妻子……</td></tr>…</table>
    转成文本是为了让表格内容可检索（人物关系表有真实信息，不能丢）。
    """
    rows = []
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', html or '', re.S):
        cells = [HTML_RE.sub(' ', td).strip() for td in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
        cells = [re.sub(r'\s+', ' ', c) for c in cells if c]
        if not cells:
            continue
        if len(cells) == 1:
            rows.append(cells[0])
        else:
            rows.append("：".join(cells))
    return rows


# ---------- 读 JSON → 语义元素流 ----------

def load_pages(json_path: str) -> list[list[dict]]:
    """读 OCR JSON → 每页的块列表（保留 label / bbox / content）。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pages = []
    for page in data:
        blocks = []
        for b in (page.get("prunedResult", {}) or {}).get("parsing_res_list", []) or []:
            bbox = b.get("block_bbox") or [0, 0, 0, 0]
            blocks.append({
                "label": b.get("block_label"),
                "bbox": bbox,
                "content": (b.get("block_content") or "").strip(),
            })
        pages.append(blocks)
    return pages


def _build_items(pages: list[list[dict]]) -> list[dict]:
    """页/块 → 语义元素流：chapter_open / heading / part_marker / text / table。"""
    items = []
    for pi, blocks in enumerate(pages):
        for b in blocks:
            label = b["label"]
            content = b["content"]
            if not content:
                continue
            if label == "table":
                # 表格转成「名：说明」的文本行入正文流 —— 人物关系表有真实信息，
                # 直接丢会漏内容（致谢章的 5 张人物表）。表格本身另存结构化产物。
                for row in _table_rows(content):
                    items.append({"kind": "text", "page": pi, "content": row, "zone": "body"})
                continue
            if label in HEADING_LABELS:
                # 「第N章」的第二种形态：paragraph_title 块内容 = "##### 第12章"
                # （第12、13、14、19、23 章是这种；其余章是 text 块整行 = "第N章"）
                m = CHAPTER_RE.match(content.replace("\n", " ").strip())
                if m:
                    items.append({"kind": "chapter_open", "page": pi, "num": int(m.group(1))})
                    continue
                items.append({
                    "kind": "heading", "page": pi, "label": label,
                    "title": _clean_title(content), "raw": content,
                })
                continue
            # text 块：按行拆（OCR 里 text 块基本一行一段；偶有多行）
            x0 = b["bbox"][0] if b["bbox"] else 0
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = CHAPTER_RE.match(line)
                if m:
                    items.append({"kind": "chapter_open", "page": pi, "num": int(m.group(1))})
                    continue
                if PART_RE.match(line):
                    items.append({"kind": "part_marker", "page": pi, "title": line})
                    continue
                items.append({"kind": "text", "page": pi, "content": line, "zone": _zone(x0),
                              "raw": content})
    return items


def _resolve_chapter_titles(items: list[dict]) -> None:
    """给每个 chapter_open 补标题（章号行之后的第一个短文本行）。

    实测 23 章全部命中这一形态：章标题是**正文列里的短行**（如
    "“光速”之道：秘密开发CUDA，让游戏玩家为科学研发买单"），
    不是 doc_title 块 —— doc_title 是小节级标题。
    """
    for i, it in enumerate(items):
        if it["kind"] != "chapter_open":
            continue
        title = None
        for nx in items[i + 1:]:
            if nx["page"] != it["page"]:
                break
            if nx["kind"] == "text" and len(nx["content"]) <= CHAPTER_TITLE_MAX_LEN:
                title = nx["content"]
                nx["consumed"] = True      # 标题行不再当正文段
                break
        it["title"] = title


def _build_paragraphs(items: list[dict]) -> list[dict]:
    """元素流 → 段落列表（含跨页续接合并 + 章/小节归属）。

    跨页续接的判据（三个都要满足）：
      1. 前段不是终止标点结尾（= 段落被页边界切断）
      2. 前段是**单行**且长度 >= CONTINUE_MIN_LEN
         （更短的是居中标题/版权行；多行的是独立元素，不跨页）
      3. 下一段是正文列（zone=body）—— 即便前段是缩进段头（x0 190~215），
         它同样是正文被切断，只是首行有缩进

    缩进段头（x0 190~215 的正文首行）实测有 29 处被切断（如
    "……黄仁勋必须做好万全的准备。他得了" → "解，这种方法过去为何屡试屡败"），
    所以「前段必须 zone=body」是错的，会漏合这 29 处。

    章/小节归属在**同一次遍历**里定，不按页码反查 —— 同一页可能有多个标题
    （实测 p137 有 5 个小节标题），按页匹配会全部塌缩成最后一个。
    """
    paras = []
    cur = None
    cur_chapter = None      # 当前章 (num, title)
    cur_section = ""        # 当前小节标题

    def flush():
        nonlocal cur
        if cur:
            paras.append(cur)
            cur = None

    for it in items:
        kind = it["kind"]

        if kind == "chapter_open":
            flush()
            cur_chapter = (it["num"], it.get("title") or "")
            cur_section = ""
            continue

        if kind == "part_marker":
            continue

        # 导读换篇：清空上一篇遗留的小节标题（否则 04 篇会顶着 03 篇的小节名）
        if it.get("starts_article"):
            flush()
            cur_chapter = None
            cur_section = ""
            continue

        if kind == "heading":
            label = it["label"]
            if label in ("number", "figure_title", "header"):
                continue        # 导读编号 / 图注 / 落款 —— 不是小节标题
            title = it["title"]
            # 区域边界标题：脱离章归属（后记 / 致谢 / 导读手册）
            if title in ("后记", "致谢") or "导读手册" in title:
                flush()
                cur_chapter = None
                cur_section = ""
                continue
            # doc_title 和 paragraph_title 都是**小节标题**：
            #   doc_title      = "# 瓦解，与普里姆剑拔弩张"（markdown 一级，但它是小节）
            #   paragraph_title = "##### 只能选择坚强并勇往直前"
            # 章标题另有来源（第N章 之后的短文本行），不靠 heading 块 —— 23 章实测全是文本行。
            flush()
            cur_section = title
            continue

        if kind != "text" or it.get("consumed"):
            continue

        line = it["content"]
        if it["zone"] == "right":
            # 右侧元素（跨栏标题/引文）：独立成段，不参与合并
            flush()
            paras.append({"content": line, "pages": [it["page"]], "zone": it["zone"],
                          "chapter": cur_chapter, "section": cur_section})
            continue

        # 跨页续接判定：前段被切断 + 单行 + 够长 + 下一段是正文列
        if (cur and cur["zone"] != "right"
                and it["page"] == cur["pages"][-1] + 1
                and not TERMINAL_RE.search(cur["content"])
                and "\n" not in cur["raw"]
                and len(cur["content"]) >= CONTINUE_MIN_LEN
                and it["zone"] == "body"):
            cur["content"] += line
            cur["pages"].append(it["page"])
            continue
        flush()
        cur = {"content": line, "pages": [it["page"]], "zone": it["zone"],
               "raw": it.get("raw", ""),
               "chapter": cur_chapter, "section": cur_section}
    flush()
    return paras


# ---------- 区域划分：章 / 后记 / 致谢 / 导读 ----------

def _detect_boundaries(items: list[dict]) -> dict:
    """找后记 / 致谢 / 导读手册 的起始页。

    判据是标题文字本身（版面标题块），不靠页码硬编码。
    """
    out = {"afterword": None, "thanks": None, "guide": None}
    for it in items:
        if it["kind"] != "heading":
            continue
        t = it["title"]
        if out["afterword"] is None and t == "后记":
            out["afterword"] = it["page"]
        elif out["thanks"] is None and t == "致谢":
            out["thanks"] = it["page"]
        elif out["guide"] is None and "导读手册" in t:
            out["guide"] = it["page"]
    return out


def _part_of_page(items: list[dict]) -> dict:
    """页 → 部分名（第一部分 / 第二部分）。part_marker 行出现后生效，直到下一个。"""
    out = {}
    current = ""
    for it in items:
        if it["kind"] == "part_marker":
            current = it["title"]
        if it["page"] not in out:
            out[it["page"]] = current
    return out


# ---------- 组装：段落 → (章节, 小节) → 块 ----------

def _group_sections(paras, boundaries, guide_articles):
    """把段落按 (区域, 章/文章, 小节) 分组，产出可切块的文本单元。

    key = (region, chapter_key, article_key, section_title)
      region      = "chapter" | "前置" | "后记" | "致谢" | "导读"
      chapter_key = 章号（region=chapter 时有效）
      article_key = 导读文章编号（region=导读 时有效）

    章/小节来自段落自带的 chapter / section 字段（_build_paragraphs 单遍遍历时定的），
    不按页码反查 —— 同一页可能有多个小节标题，按页匹配会塌缩成最后一个。
    """
    groups = OrderedDict()
    for p in paras:
        page = p["pages"][0]
        chap = p.get("chapter")
        sec_title = p.get("section") or ""
        if chap:
            key = ("chapter", chap[0], "", sec_title)
        else:
            region = _region_of(page, boundaries)
            art = _guide_article_of(page, guide_articles)
            key = (region, "", art["num"] if art else "", sec_title)
        groups.setdefault(key, []).append(p)
    return groups


def _region_of(page, boundaries):
    if boundaries["guide"] is not None and page >= boundaries["guide"]:
        return "导读"
    if boundaries["thanks"] is not None and page >= boundaries["thanks"]:
        return "致谢"
    if boundaries["afterword"] is not None and page >= boundaries["afterword"]:
        return "后记"
    return "前置"


def _guide_article_of(page, guide_articles):
    """页 → 所属导读文章（取页码不超过该页的最后一篇）。"""
    art = None
    for a in guide_articles:
        if a["page"] <= page:
            art = a
        else:
            break
    return art


def _guide_articles(items, boundaries):
    """导读文章：number 块（01~06）开一篇，标题取紧随其后的标题行。

    标题有两种形态（实测）：
      doc_title 块 —— 01/03/05/06 篇（如 "如何从别人的传记中学到对自己有益的知识"）
      text 块     —— 02/04 篇（如 "为什么是英伟达和黄仁勋？"），且 x0 居中或偏右
    """
    if boundaries["guide"] is None:
        return []
    arts = []
    for i, it in enumerate(items):
        if it["kind"] != "heading" or it["label"] not in NUMBER_LABELS:
            continue
        title = ""
        for nx in items[i + 1:]:
            if nx["page"] != it["page"]:
                break
            # 标题 = number 块之后的第一个标题元素：
            #   doc_title 块（01/03/05/06 篇）或 text 块（02/04 篇）
            if nx["kind"] == "heading" and nx["label"] == "doc_title":
                title = nx["title"]
                nx["starts_article"] = True
                break
            if nx["kind"] == "text":
                title = nx["content"]
                nx["consumed"] = True
                nx["starts_article"] = True   # 换篇 → 上一篇的小节标题必须清空
                break
        arts.append({"page": it["page"], "num": it["title"], "title": title})
    return arts


# ---------- 写章节 md（web 端「展开原文」用） ----------

def _chapter_md_name(num: int) -> str:
    return f"ch{num:02d}.md"


def _write_chapter_md(out_dir: str, name: str, title: str, sections: "OrderedDict[str, list]") -> str:
    """写一份章节 md，返回相对路径（books/<id>/ 下）。

    sections = OrderedDict[小节标题 -> 段落列表]
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    lines = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    for sec_title, paras in sections.items():
        if sec_title:
            lines.append(f"## {sec_title}")
            lines.append("")
        for para in paras:
            lines.append(para["content"])
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return os.path.join("chapters", name)


# ---------- 主入口 ----------

def parse_ocr_book(book: dict, chunker) -> tuple[list[dict], list[str], dict]:
    """解析一本 OCR 版面书。返回 (chunks, md_files, outline)。

    chunker(text) -> list[str]：切块函数由 parse.py 注入（保持切块口径唯一）。
    """
    book_id = book["id"]
    book_dir = books_mod.book_dir(book_id)
    json_rel = book.get("ocr_json")
    if not json_rel:
        raise books_mod.BookError(f"书 {book_id} 的 manifest 缺 ocr_json 字段")
    json_path = os.path.join(book_dir, json_rel)
    if not os.path.exists(json_path):
        raise books_mod.BookError(f"书 {book_id} 的 OCR JSON 不存在: {json_path}")

    pages = load_pages(json_path)
    items = _build_items(pages)
    _resolve_chapter_titles(items)
    boundaries = _detect_boundaries(items)
    guide_articles = _guide_articles(items, boundaries)   # 必须先于段落构建（要标记 consumed）
    paras = _build_paragraphs(items)

    chapters = [it for it in items if it["kind"] == "chapter_open"]
    part_of = _part_of_page(items)

    groups = _group_sections(paras, boundaries, guide_articles)

    chunks = []
    outline = {"kind": "chapter", "items": [], "extras": []}
    md_files = []
    chapters_dir = os.path.join(book_dir, "chapters")

    # 章号 → 章标题（outline / heading_path 用）
    chap_titles = {c["num"]: (c.get("title") or "") for c in chapters}
    # 章号 → 该章的小节列表（outline 用）
    chap_sections = OrderedDict((c["num"], []) for c in chapters)

    # 1) 先把分组转成「文档 → 小节列表」结构（章/区域各一份 md）
    docs = OrderedDict()   # doc_key -> {"name", "title", "regions": OrderedDict[sec_title -> paras]}
    for (region, chap_key, art_key, sec_title), group_paras in groups.items():
        text = "\n\n".join(p["content"] for p in group_paras)
        if not text.strip():
            continue
        if region == "chapter":
            chap_num = int(chap_key)
            chap_title = chap_titles.get(chap_num, "")
            doc_key = ("chapter", chap_num)
            doc_name = _chapter_md_name(chap_num)
            doc_title = f"第{chap_num}章 {chap_title}".strip()
            sec_title = sec_title or "章引言"
        elif region == "导读":
            art = next((a for a in guide_articles if a["num"] == art_key), None)
            art_title = (art or {}).get("title") or f"导读 {art_key}"
            doc_key = ("导读", art_key)
            doc_name = f"guide_{art_key}.md"
            doc_title = art_title
        else:
            doc_key = (region, "")
            doc_name = f"{region}.md"
            doc_title = region
            sec_title = sec_title or region

        doc = docs.setdefault(doc_key, {"name": doc_name, "title": doc_title, "sections": OrderedDict()})
        doc["sections"].setdefault(sec_title, []).extend(group_paras)
        if region == "chapter":
            secs = chap_sections.setdefault(chap_num, [])
            if sec_title not in secs:
                secs.append(sec_title)

    # 2) 写每份文档 md（章/区域各一份，供 web 端「展开原文」）
    for doc_key, doc in docs.items():
        rel = _write_chapter_md(chapters_dir, doc["name"], doc["title"], doc["sections"])
        md_files.append(os.path.join(chapters_dir, doc["name"]))
        doc["rel"] = rel

    # 3) 逐小节切块 → chunks
    for doc_key, doc in docs.items():
        region, key = doc_key
        if region == "chapter":
            chap_num = int(key)
            chap_title = chap_titles.get(chap_num, "")
            hp_base = f"第{chap_num}章 {chap_title}" if chap_title else f"第{chap_num}章"
            chapter_field = str(chap_num)
            chapter_title = chap_title
        elif region == "导读":
            art = next((a for a in guide_articles if a["num"] == key), None)
            chapter_title = (art or {}).get("title") or f"导读 {key}"
            hp_base = f"导读 {chapter_title}"
            chapter_field = ""
        else:
            chapter_title = region
            hp_base = region
            chapter_field = ""

        # part = 卷，与 DDIA 对齐：只有正文章落在卷里，前置/后记/致谢/导读是
        # 书的附属部分、不属于任何一卷 —— DDIA 的「术语表」同样 part=""。
        part = part_of.get(_first_page(doc), "") if region == "chapter" else ""

        for sec_title, sec_paras in doc["sections"].items():
            text = "\n\n".join(p["content"] for p in sec_paras)
            pages_used = [pg for p in sec_paras for pg in p["pages"]]
            src = doc["rel"] + f"#p{min(pages_used)}-{max(pages_used)}"
            hp = f"{hp_base} > {sec_title}" if sec_title else hp_base
            for c in chunker(text):
                chunks.append({
                    "pk": len(chunks),
                    "book": book_id,
                    "part": part,
                    "chapter": chapter_field,
                    "chapter_title": chapter_title,
                    "section": sec_title,
                    "subsection": "",
                    "heading": sec_title,
                    "heading_path": hp,
                    "content": c,
                    "source": src,
                    "chunk_idx": len(chunks),
                })

    # 4) outline：章条目 + 附加条目（前置/后记/致谢/导读）
    # 附加条目也放进 items —— web.py 的 renderOutline() 只遍历 OUTLINE.items，
    # 放 extras 里前端就看不到（实测会整段消失）。用 kind 字段区分渲染方式。
    for num, title in chap_titles.items():
        if not title:
            continue
        first_page = _first_page(docs[("chapter", num)]) if ("chapter", num) in docs else 0
        outline["items"].append({
            "key": str(num),
            "title": title,
            "part": part_of.get(first_page, ""),
            "sections": chap_sections.get(num, []),
            "kind": "chapter",
        })
    for region in ("前置", "后记", "致谢"):
        if (region, "") not in docs:
            continue
        doc = docs[(region, "")]
        outline["items"].append({
            "key": region,
            "title": region,
            "part": "",          # 附属部分不属于任何一卷（同 DDIA 术语表）
            "sections": list(doc["sections"].keys()),
            "kind": "region",
        })
    for art in guide_articles:
        if ("导读", art["num"]) not in docs:
            continue
        doc = docs[("导读", art["num"])]
        outline["items"].append({
            "key": f"导读{art['num']}",
            "title": art["title"] or f"导读 {art['num']}",
            "part": "",
            "sections": list(doc["sections"].keys()),
            "kind": "guide",
        })

    return chunks, md_files, outline


def _first_page(doc: dict) -> int:
    """文档的首个页码。"""
    for paras in doc.get("sections", {}).values():
        for p in paras:
            return p["pages"][0]
    return 0
