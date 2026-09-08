# RAG-on-Book Buglog

事故 / Bug 记录。格式：日期 + 现象 → 排查 → 根因 → 修复 → 验证。与 WEB_TODO.md（功能规划）分开。

---

## 2026-09-08 · Bug#2 用户气泡高度异常偏高（142px）+ 服务重启窗口的"幽灵报错"

### 现象（用户报告，附截图）

1. 聊天界面里**用户自己的气泡高度异常**：只发"你好"两个字，气泡却撑成 ~142px 高（正常应 ~40px），上下大片空白。assistant 气泡正常。
2. 随后用户问"第一章的内容是？"，页面报 `⚠️ 网络错误：Failed to execute 'json' on 'Response': Unexpected end of JSON input`——但服务端日志显示该请求 **200 成功、回答已落库**。

### 排查过程

- 数据库里用户消息内容正常（`你好`，2 字符，无隐藏换行/特殊字符）→ 排除数据污染。
- `web.py` 有大量未提交改动（V2：markdown 渲染/章节展开/多轮会话），嫌疑在本次 V2 引入的代码。
- headless Chrome 实测 DOM：
  - 用户气泡 `getBoundingClientRect().height = 142px`，文本内容就是"你好"。
  - 气泡内文本节点实际为 `"\n      你好\n      \n      \n    "` —— **模板字符串的换行和缩进进了文本节点**。
- 用 OCR + 像素分析截图辅助定位（当时无视觉模型可用）。

### 根因（Bug A：气泡偏高）

`appendMsg()`（web.py）里 `.bubble` 用**跨行模板字符串**拼接：

```js
div.innerHTML = `
  <div class="role">...</div>
  <div class="bubble">
    ${bodyHtml}
    ${hits && hits.length ? renderHits(hits) : ''}
    ${meta ? `<div class="meta">${meta}</div>` : ''}
  </div>
`;
```

而 CSS 对用户气泡设了 `white-space:pre-wrap`（本意是保留用户输入里的换行）。pre-wrap 把模板自身的换行/缩进也当成了真实内容渲染 → 短消息被 4~5 个空行撑高到 142px。

assistant 气泡不受影响的原因：它没有 pre-wrap，且 markdown 渲染内容包在块级标签里，空白被折叠。旧版 V1 模板是单行拼接，所以没这个问题——**V2 改多行模板时引入的回归**。

### 根因（Bug B：幽灵网络报错）

前端 `send()` 拿到 fetch 响应后直接 `await r.json()`，不检查 `r.ok` / 响应体是否为空。用户提问恰逢**服务重启窗口**（修复 Bug A 时 kill 旧进程 → 新进程刚起），浏览器连接被切断/拿到空响应 → `r.json()` 抛晦涩的 `Unexpected end of JSON input`。而后端那次请求其实已处理完并落库 → 刷新页面能看到完整回答，但当时界面只显示报错。

### 修复

1. **Bug A**（web.py `appendMsg()`）：innerHTML 改为单行紧凑拼接，模板自身的换行/缩进不再进入文本节点；pre-wrap 保留（用户真实换行仍生效）。加注释说明为何不能改回跨行模板。
2. **Bug B**（web.py `send()`）：
   - fetch 加 90s AbortController 超时保护，超时给出明确中文提示（"生成超过 90 秒…请刷新查看"），不再抛晦涩异常；
   - 先 `r.text()` 再判断 content-type / JSON.parse，空 body、网关错误页都优雅降级；
   - 响应异常时自动重拉会话（`loadSession(SID)`），若回答实际已生成会直接显示；
   - 普通网络错误提示改为"请检查服务是否在线后重试"。

### 验证

- headless Chrome 复测：用户气泡 142px → **40px**（文本 26px + padding），assistant 气泡不受影响。
- 重启服务后 health 200；9616 反代入口 `/t/ragbook/` 正常。
- 用同样问题"第一章的内容是？"走完整链路复现：200 + 69KB 完整 JSON，回答正常生成。
- 修改后 web.py 语法检查通过，页面源码确认新错误处理已生效。

### 教训

- **HTML 注入点一律单行紧凑拼接**；任何跨行模板只要目标容器可能设 `white-space:pre-wrap`，空白就会变成真实内容。
- **前端 fetch 必须**：检查 `r.ok` → 检查响应体非空/是 JSON → 再解析；网络层错误信息对用户要可读。
- **重启服务前先确认无 in-flight 请求窗口**（或至少让前端错误提示能自愈：自动重拉会话）。

---
