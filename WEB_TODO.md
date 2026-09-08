# RAG-on-Book Web 端 — TODO

## 已实现（V1, 2026-09-08）
- [x] Flask 后端跑在 127.0.0.1:82
- [x] 任务101 反代 /t/ragbook/ → 127.0.0.1:82
- [x] 对话流界面（聊天气泡，连续追问）
- [x] 复用 ask.py 的 search() / ask_llm() / build_prompt()
- [x] 进程复用 bge 模型（避免每次请求现加载）

## TODO — V2 待办
- [ ] **双栏布局**（Q3-iii）：左问答区 / 右原文片段；引用可点击定位到 books/ddia 章节 markdown
- [ ] **聊天历史持久化**：现在只存内存，刷新即丢；可存 SQLite 或文件
- [ ] **历史会话列表**：侧栏能切换不同会话
- [ ] **新建对话 / 清空** 按钮
- [ ] **Markdown 渲染**：现在 LLM 回答当纯文本显示，应渲染 markdown
- [ ] **复制回答 / 引用** 按钮
- [ ] **DeepSeek 欠费/故障** 时的友好提示（实测验证后写）

## TODO — V3（与 Future.md 联动）
- [ ] 混合检索（FTS5 + 向量 + RRF）— Future.md V2
- [ ] rerank 重排 — Future.md V2
- [ ] 引用链接回跳到 ddia.vonng.com 锚点 — Future.md 工程化碎片
