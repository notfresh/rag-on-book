# RAG on Book

《设计数据密集型应用》(DDIA) 中文版的本地 RAG 问答系统。

> 基于 bge-small-zh-v1.5 向量检索 + Milvus Lite + DeepSeek LLM，对 DDIA 14 章内容做精确问答与章节级元信息检索。

## 特性

- **本地优先**：所有数据本地存储，向量库用 Milvus Lite（嵌入式），无外部服务依赖（除 LLM 调用）
- **Web 界面**：Flask 聊天 UI，支持会话管理（持久化）、侧栏历史、章节折叠引用
- **元信息过滤**：识别 `第N章` / `chN` 提问，用 milvus `filter` 精确召回整章或整节
- **多轮对话**：规则 + LLM 混合意图分类（chitchat / followup / question），闲聊不检索、追问带历史
- **章节感知编码**：bge 编码时拼 `heading_path` 前缀，让向量空间能区分章节

## 快速开始

```bash
# 1. 安装依赖（DDIA 的 markdown 已经在 books/）
python3 -m venv .venv
source .venv/bin/activate
pip install -U pymilvus sentence-transformers flask

# 2. 构建索引（首次必跑，811 块约 30 秒）
python3 index.py --full

# 3a. 命令行问答
python3 ask.py "DDIA 里事务隔离级别有哪些？"

# 3b. Web 端（监听 127.0.0.1:82）
python3 web.py
# 浏览器打开 http://127.0.0.1:82/
```

## 项目结构

```
rag-on-book/
├── ask.py            # CLI 入口 + 复用函数（search/ask_llm/build_prompt）
├── index.py          # 解析 + 入库（Milvus Lite，COLLECTION, DB_PATH, get_client）
├── parse.py          # DDIA markdown → parsed.jsonl（含 chapter/section/heading_path）
├── web.py            # Flask 后端（多轮管理 + 意图分类 + 元信息过滤）
├── README.md         # 项目门面（这个文件）
├── README.v1.md      # V1 历史版本说明
├── Future.md         # 未来想法（V2 混合检索 / V3 全书视角等）
├── WEB_TODO.md       # Web 端后续 TODO
├── books/ddia/       # DDIA 中文版 markdown（已结构化切块到 parsed.jsonl）
├── .vector/          # 向量库（Milvus Lite .db 文件，gitignore）
├── data/             # 会话持久化 SQLite（gitignore）
└── .venv/            # Python 虚拟环境（gitignore）
```

## Web 端架构

```
用户 query
  │
  ├─ 0. detect_meta_query：第N章 / chN 模式？
  │    └─ 是 → milvus filter 精确召回（不调 bge）→ meta_prompt
  │    └─ 超范围（ch>14）→ 直接返 "DDIA 共14章，没有第N章"
  │
  ├─ 1. 意图分类（规则初筛 + LLM 兜底）
  │    ├─ chitchat → chitchat_reply（纯对话，不检索）
  │    └─ question / followup → 检索
  │
  ├─ 2. bge 向量检索（top_k 块）
  │    └─ 检索文本包含 heading_path 前缀（Layer 2 增强）
  │
  └─ 3. LLM 生成回答
       └─ prompt 拼最近 2 轮问答（让 LLM 理解追问代词）
```

### Layer 1：元信息过滤

- 识别 `第N章` / `第N章某节` / `chN`（阿拉伯 + 中文数字）
- 用 milvus `filter` 精确召回：`chapter == 'N'` 或 `chapter == 'N' && section == '中文标题'`
- section fuzzy match：2-4 字中文片段匹配白名单
- 章号超范围（>14）直接返友好提示

### Layer 2：编码增强

`index.py index_chunks()` 编码时把 heading_path 拼到 content 头部：

```python
texts = [f"【{c['heading_path']}】\n{c['content']}" for c in chunks]
```

让 bge 在向量空间里能区分章节位置，**重建索引**：`rm -rf .vector && python3 index.py --full`

### 意图分类

- `rule_classify`：黑名单匹配（你好/hi/谢谢/你是谁 等）
- `llm_classify`：用 DeepSeek 做三选一分类（chitchat / followup / question）
- 失败回退到 `question`

## 公网访问

通过任务101 门户站反代：

```
http://47.80.28.245:9616/t/ragbook/
```

（任务101 监听 9616 端口，web.py 监听 127.0.0.1:82）

## 环境要求

- Python 3.10+
- ~1GB 内存（bge-small-zh-v1.5 模型）
- 首次构建索引需要联网下载模型（HF Hub）

## 配置

| 配置项 | 默认 | 位置 |
|--------|------|------|
| LLM 模型 | `deepseek-chat` | `ask.py` 常量 |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5` | `ask.py` 常量 |
| 向量维度 | 512 | `index.py` |
| Web 端口 | 82 | `web.py` 末尾 |
| DDIA 数据 | 14 章中文 markdown | `books/ddia/content/zh/` |

DeepSeek API key 从 `~/.hermes/.env` 的 `DEEPSEEK_API_KEY` 读（web.py 和 ask.py 都自动加载）。

## 开发

- 测试向量检索效果：`python3 ask.py "你的问题" --no-llm`（跳过 LLM，只看召回的 5 块）
- 看 web.py 启动日志：`web.py` 前台启动会输出 bge 加载、milvus 就绪、LLM 调用耗时
- 改 web.py 后必须重启（Flask app.py INDEX_HTML 是模块级常量）
- 改 index.py 的编码逻辑必须重建索引（`rm -rf .vector && python3 index.py --full`）

## 路线图

详见 `Future.md` 和 `WEB_TODO.md`：

- **V2 检索质量**：混合检索（FTS5 + 向量 + RRF）、rerank 重排
- **V3 全书视角**：单章摘要、跨章主线、术语表联动
- **Web 端**：Markdown 渲染、双栏问答/原文布局、历史导出

## 版本

- **V2（当前）**：Web 端 + 多轮对话 + 元信息过滤
- **V1**：CLI 单轮问答，见 [README.v1.md](./README.v1.md)

## 许可证

本仓库为个人研究项目，仅供学习参考。DDIA 内容版权归原作者所有。
