# RAG on Book

多书本地 RAG 问答系统 —— 把 Markdown 书/笔记集变成可提问、可回溯原文的知识库。

> 基于 bge-small-zh-v1.5 向量检索 + Milvus Lite + LLM 生成，支持章节级/主题级元信息精确召回。

## 特性

- **多书**：每本书一份 manifest（`books/<id>/book.json`），加书 = 加目录 + 写 manifest，不改代码
- **两种解析策略**：`chapter_book`（按章，DDIA）/ `note_set`（按主题文件，CS-Notes）
- **元信息精确召回**：`第N章` / `第N章某节` / 主题名 → milvus `filter` 直查，**不调 embedding**
- **章节感知编码**：bge 编码时拼 `heading_path` 前缀，但定位链**不入库正文**（不污染召回）
- **多轮对话**：规则 + LLM 混合意图分类（chitchat / followup / question）
- **原文抽屉**：点引用直接看原文 markdown，定位到小节
- **会话持久化**：SQLite（WAL），跨设备不丢；会话绑定书
- **本地优先**：数据本地，向量库 Milvus Lite 嵌入式文件，无外部服务（除 LLM 调用）

## 快速开始

```bash
# 1. 依赖
python3 -m venv .venv && source .venv/bin/activate
pip install -U pymilvus sentence-transformers flask

# 2. 解析 + 建索引（books/ 下所有书）
python3 parse.py            # → books/<id>/parsed.jsonl + outline.json
python3 index.py --full     # → books/<id>/vector/book_chunks.db（每书一个库）

# 3a. 命令行问答
python3 ask.py "DDIA 里事务隔离级别有哪些？" --book ddia

# 3b. Web 端（监听 127.0.0.1:82）
python3 web.py
# 浏览器打开 http://127.0.0.1:82/
```

> ⚠️ **重建索引前必须先停 web.py** —— Milvus Lite 是单文件库，web.py 开着会锁文件，
> 报 `ConnectionConfigException: Open local milvus failed`。

## 项目结构

```
rag-on-book/
├── books.py          # 书清单单一来源：manifest 读取/校验/元信息识别
├── ask.py            # CLI 入口 + 复用函数（search/ask_llm/build_prompt）
├── index.py          # 块 → embedding → Milvus Lite 入库
├── parse.py          # Markdown → parsed.jsonl + outline.json（两种策略）
├── web.py            # Flask 后端（会话管理 + 意图分类 + 元信息过滤 + 前端）
├── books/
│   ├── ddia/
│   │   ├── book.json         # manifest：书名/作者/解析策略
│   │   ├── content/zh/*.md   # 完整书籍原文
│   │   ├── parsed.jsonl      # 解析产物（811 块）
│   │   ├── outline.json      # 目录树（自动生成）
│   │   └── vector/           # 这本书自己的向量库（gitignore）
│   └── cs-notes/
│       ├── book.json
│       ├── notes/*.md
│       ├── outline.json
│       └── vector/           # 这本书自己的向量库
├── data/             # 会话持久化 SQLite（全局，不按书拆，gitignore）
└── .venv/            # Python 虚拟环境（gitignore）
```

> **一书一库**：每本书的向量库放在自己目录下的 `vector/` 里，主键书内自增，
> 互不干扰（Milvus Lite 多实例可同时打开）。会话库是**全局**的 —— 会话跟书无关。

## 加一本新书

1. 把 markdown 放进 `books/<id>/`（子目录名写进 manifest 的 `source_dir`）
2. 写 `books/<id>/book.json`：

```json
{
  "id": "mybook",
  "title": "我的书",
  "author": "作者",
  "kind": "chapter_book",
  "source_dir": "content/zh",
  "skip_files": ["_index.md", "toc.md"],
  "meta": { "kind": "chapter" }
}
```

3. `python3 parse.py --book mybook && python3 index.py --book mybook`

每本书的向量库落在 `books/<id>/vector/book_chunks.db`，主键书内自增 —— 不需要跨书分段，
也不会跟别的书撞主键。加书就是加目录 + 写 manifest，代码一行不改。

### 两种解析策略

| kind | 适用 | 定位链 | 元信息提问 |
|------|------|--------|-----------|
| `chapter_book` | 有明确章节的书 | 章 > 节 > 小节 | `第N章` / `第N章某节` |
| `note_set` | 主题笔记集（无章节） | 分组 > 主题文件 > 小节 | 主题名 / 分组名 |

`note_set` 的分组在 manifest 的 `meta.groups` 里配（glob 模式 → 分组名）。

## Web 端架构

```
用户 query
  │
  ├─ 0. detect_meta_query：第N章 / 主题名？
  │    └─ 是 → milvus filter 精确召回（不调 bge）→ meta_prompt
  │    └─ 超范围（第99章）→ 直接返"共14章，没有第99章"
  │
  ├─ 1. 意图分类（规则初筛 + LLM 兜底）
  │    ├─ chitchat → chitchat_reply（纯对话，不检索）
  │    └─ question / followup → 检索
  │
  ├─ 2. bge 向量检索（top_k 块，filter=book）
  │
  └─ 3. LLM 生成回答（prompt 拼最近 2 轮问答 + 书名作者）
```

### Layer 1：元信息过滤

- 章节表**从 `outline.json` 自动读**（parse.py 生成），不再手写硬编码字典
- section fuzzy match：2-4 字中文片段匹配，**排除疑问词碎片**
  （"讲了什么"里的"什么"不能命中"事务到底是什么？"—— 问整章被精确到某节比不匹配更糟）
- 超范围章号直接返友好提示，不调 LLM

### Layer 2：编码增强

`index.py` 编码时把 heading_path 拼到编码文本头部：

```python
texts = [f"【{c['heading_path']}】\n{c['content']}" for c in chunks]
```

**定位链只参与编码、不入库正文** —— 入库会污染召回（搜"第3章"召回正文恰好含"第3章"的不相关段落）。

## 公网访问

通过任务101 门户站反代：`http://47.80.28.245:9616/t/ragbook/`

（任务101 监听 9616，web.py 监听 127.0.0.1:82）

## 配置

| 配置项 | 默认 | 位置 |
|--------|------|------|
| LLM provider | `minimax-cn`（MiniMax-M3） | `ask.py` 顶部 / `RAGBOOK_LLM_PROVIDER` 环境变量 |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5` | `index.py` |
| 向量维度 | 512 | `index.py` |
| Web 端口 | 82 | `web.py` 末尾 |
| 书清单 | `books/*/book.json` | `books.py` |

LLM key 从 `~/.hermes/.env` 读（MiniMax 用 `MINIMAX_CN_API_KEY`，DeepSeek 用 `DEEPSEEK_API_KEY`）。

## 开发

- 测试向量检索效果：`python3 ask.py "你的问题" --no-llm`（跳过 LLM，只看召回的块）
- 改 web.py 后必须重启（Flask 的 `INDEX_HTML` 是模块级常量）
- 改 index.py 的编码逻辑必须重建索引（`python3 index.py --full`，每本书各自重建）
- 改 parse.py 也要重建（`parse.py` → `index.py` → 重启 web.py）
- **重建索引前必须先停 web.py** —— Milvus Lite 是单文件锁，web.py 开着会报
  `Open local milvus failed`（每本书的库各自被锁）

## 路线图

详见 `Future.md` 和 `WEB_TODO.md`：

- **V2 检索质量**：混合检索（FTS5 + 向量 + RRF）、rerank 重排
- **V3 全书视角**：单章摘要、跨章主线、术语表联动
- **Web 端**：双栏问答/原文布局、历史导出

## 版本

- **V3（当前）**：多书支持 + 界面重写
- **V2**：Web 端 + 多轮对话 + 元信息过滤
- **V1**：CLI 单轮问答，见 [README.v1.md](./README.v1.md)

## 许可证

本仓库为个人研究项目，仅供学习参考。DDIA 内容版权归原作者所有。
CS-Notes 内容版权归原作者所有。
