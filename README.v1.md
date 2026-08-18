# RAG on Book — V1 设计方案

> 把喜欢的书变成"可对话的知识库"：不用逐行读，问任何细节，它带引用回答，还能回到书里细看。
> V1 = **问答**：问细节答细节，答案附章节原文位置。
> 技术骨架继承 `github-repo-crawler/service`（issue 语义检索 V1，已验证），新增"书"这一侧：章节解析、定位链、书版切块。

---

## 1. 第一本书：《设计数据密集型应用》(DDIA) 中文版

- 仓库：`Vonng/ddia`（冯若航，23k+ stars，持续更新），完整第二版中文翻译
- 语料：`content/zh/*.md`，纯 Markdown，一章一个文件（Hugo 文档站结构）
- 体量：14 章 + 前言 + 术语表，正文约 1.3 MB / 300K tokens
- 为什么选它：数据系统圣经，与 RAG 底层原理、求职方向（agent×智能投研）直接相关——边做项目边读透 RAG 的"为什么"

## 2. 数据流总览

```
Vonng/ddia 仓库 content/zh/*.md   ← 原始语料（books/ddia/ 下, 24→16 个文件）
        │
        ▼
parse.py   frontmatter(章号/章名/部) + ##/### 小节 → 定位链块
        │   heading_path: "第8章 事务 > 弱隔离级别 > 快照隔离与可重复读"
        ▼
books/ddia/parsed.jsonl           ← 811 块, 平均 369 token/块
        │
        ▼
index.py   bge-small-zh-v1.5 embedding → milvus-lite 向量库
        ▼
.vector/book_chunks.db            ← collection: book_chunks (811 块)
        │
        ▼
ask.py     问题 → embedding → 检索 top-k → DeepSeek 生成回答 → 回答+引用
```

## 3. 技术栈（全部继承 service V1，已验证）

| 组件 | 选择 | 说明 |
|---|---|---|
| embedding | bge-small-zh-v1.5 本地 CPU | 512 维，中文为主，免费离线 |
| 向量库 | milvus-lite 嵌入式 | `.vector/book_chunks.db` 本地文件，无服务无 docker |
| 相似度 | COSINE + FLAT 索引 | |
| 生成 | DeepSeek API (`deepseek-chat`) | key 从 `~/.hermes/.env` 读取 |
| 负载控制 | 批次 32、推理线程 2、批间 sleep 0.3s | 4 核机器防卡死 |

## 4. 定位链设计（书版与 issue 版的核心差异）

| | issue 版 (service) | 书版 (本项目) |
|---|---|---|
| 聚合粒度 | number（issue 编号） | 章节路径（heading_path） |
| 标题处理 | title 独立成块，命中加权 ×1.5 | 小节标题 + 正文**一起 embed**（块自带上下文） |
| 上下文 | issue 相对独立 | 书前后文语义强，标题链是块的一部分 |
| 引用 | 链接到 GitHub issue | heading_path + 源文件锚点（可回原文） |
| 清洗 | 模板空字段/固定句剥离 | frontmatter/HTML/锚点/脚注引用剥离 |

块结构（parsed.jsonl 一行一块）：
```json
{
  "book": "ddia",
  "part": "II",
  "chapter": "8",
  "chapter_title": "事务",
  "section": "弱隔离级别",
  "subsection": "快照隔离与可重复读",
  "heading_path": "第8章 事务 > 弱隔离级别 > 快照隔离与可重复读",
  "source": "content/zh/ch8.md#sec_transactions_isolation_levels",
  "content": "【第8章 事务 > ...】\n正文...",
  "pk": 9000000
}
```

## 5. 切块策略（书版调整）

- 粒度：`##` 小节（H2）→ 内部按 `###`（H3）细分，每块 = 定位链标题 + 正文
- 书版核心调整：小节内**短段落贪心合并**到 300~480 token（issue 版"一段一块"会切碎书的短段落/列表）
- 超长段：句号级切 → 450 字窗口兜底（25% 重叠、就近标点落刀）
- 标题前置进 content：检索时小节语义完整，命中即知出处
- 排除噪音文件：首页/目录/贡献者/版权/索引/分部目录页（16 个 md 入库）

## 6. 使用方式

```bash
python3 parse.py --book ddia        # 解析 → parsed.jsonl（811 块）
python3 index.py --full             # 建库 → 811 块入库
python3 ask.py "事务的隔离级别有哪些"          # 问答（附引用）
python3 ask.py "什么是 LSM 树" -k 8            # 调召回块数
python3 ask.py "一致性" --no-llm              # 只看检索（调试）
```

## 7. 验证结果（2026-08-18 实测）

| 验证项 | 结果 |
|---|---|
| 解析 | 16 md → 811 块，平均 369 token/块，覆盖 14 章 + 前言 + 术语表 |
| 建库 | 811 块入库，模型加载 4.5s（缓存后） |
| 语义检索 | "事务的隔离级别有哪些" → 命中第8章 弱隔离级别（快照隔离/读已提交/写偏差） |
| 完整问答 | "事务的隔离级别有哪些？分别解决什么问题？" → LLM 正确回答读已提交/读未提交/快照隔离/可串行化 + [N] 引用编号 + 章节定位 |

## 8. 已知限制（V1 承认，V2+ 解决）

- [ ] 纯向量检索：精确术语/编号（如 "B+树"）召回不如关键词，混合检索见 Future.md V2
- [ ] 脚注被剥离（`[^n]` 定义区），脚注里不少补充信息丢了
- [ ] 无增量索引：换模型/加书需 `--full` 重建（811 块全量约 30s，可接受）
- [ ] 单书库结构：多书靠 `book` 字段区分，跨书检索是 Future.md 的内容
- [ ] `part-i/ii/iii.md` 分部目录页排除（召回噪音）
