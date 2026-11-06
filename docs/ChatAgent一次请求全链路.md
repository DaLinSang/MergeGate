# ChatAgent 一次请求的全链路

> 对照代码：`backend/app/api/routes/chat.py` + `backend/app/services/agents/chat_agent.py` + `backend/app/services/chat_memory.py`
> 行号基于 2026-10-25 的版本，注释工程完成后的状态。

---

## 0. 30 秒版

一次对话请求要穿过**五段**，每段的责任边界很清楚：

| 段 | 位置 | 干什么 | 关键点 |
|---|---|---|---|
| ① 入口 | `chat.py:2118` `POST /api/chat/stream` | 会话 upsert，返回 `StreamingResponse` | 先建会话再干活，后面所有写入都要 `conversation_id` |
| ② 准备 | `chat.py:1989 _conversation_sse` | 四步：压缩 → 写用户消息 → 重建证据 → 组装上下文 | **顺序不能乱**，反了会丢消息 |
| ③ 进 Agent | `chat_agent.py:176 run_stream` | 激活技能 → 建状态 → **四道短路** | 能不进 LLM 就不进，省钱 |
| ④ 推理循环 | `chat_agent.py:461 _build_agent_graph` | LangGraph：思考 ⇄ 行动，≤12 步 | 每轮只重建两条消息 |
| ⑤ 收尾 | `chat.py:2017` 收到 final 才落库 | 写消息 / AgentRun / 封存 → SSE 推回 | 中间事件直接透传，不落库 |

一句话记住：**上下文在进入 Agent 之前就已经在数据库里准备好了，Agent 本身不查数据库。**

---

## 1. 阶段①：入口

```python
@router.post("/stream")                                  # chat.py:2118
async def chat_stream(payload, db) -> StreamingResponse: # chat.py:2119
```

请求体是 `ChatRequest`：`repo_id`、`conversation_id`（可选）、`message`、`context`。

`_conversation_sse` 第一件事：

```python
repo = _repo_or_404(db, payload.repo_id)                          # 1991 找不到仓库 → 404
conversation = ensure_conversation(db, repo.id, payload.conversation_id)  # 1992
db.commit()                                                       # 1993
```

**为什么先 commit**：`conversation_id` 后面要写进 `chat_messages`、`agent_runs`、`chat_sessions` 三张表。如果会话还没落库就往下走，外键会失败。

---

## 2. 阶段②：四步准备（最容易看错的一段）

```python
# 1994  第 1 步：先压缩，腾出空间
compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)

# 1998  第 2 步：把用户这句话写进库
user_message = message_store.append(role="user", content=payload.message)

# 1999  第 3 步：重建"证据索引"（这一轮能查什么）
EvidenceStore(db).rebuild_repo(repo, conversation)
db.commit()

# 2001  第 4 步：组装上下文（历史 + 证据 + 记忆 + 摘要）
assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
```

### 2.1 为什么顺序绝对不能反

| 顺序 | 如果反了会怎样 |
|---|---|
| 压缩在写消息**之前** | 先写消息再压缩 → 刚插进去的那条用户消息可能被当成"旧消息"压掉，模型根本看不到用户问什么 |
| 写消息在组装**之前** | 先组装再写消息 → `assemble()` 里拿不到这句话，`current_message` 要单独算一遍 token，容易重复计入 |
| 重建证据在组装**之前** | 先组装再重建 → 用的是上一轮的证据快照，问"刚合并的那个 PR"会答成旧的 |

`assemble()` 里专门处理了这个问题：

```python
# chat_memory.py:900
current_message_in_history = bool(
    raw_history and raw_history[-1]["role"] == "user"
    and " ".join(raw_history[-1]["content"].split()) == " ".join(current_message.split())
)
pending_user_tokens = 0 if current_message_in_history else estimate_tokens(current_message)
```

**注释里的原话**：调用方可能已经把当前问题写进库了，这时不能再算一次 `pending_user_tokens`，否则虚高压力、误触发压缩。

### 2.2 第 1 步：`ensure_headroom` 在做什么

它判断"上下文还剩多少空间"，快满了就提前压缩，而不是等 LLM 报 `context_length_exceeded` 再补救。

失败时的兜底很有意思（`chat_memory.py:1748 _record_failure`）：连续失败 N 次后**不再尝试压缩**，直接放行 —— 因为压缩本身要调 LLM，一直失败会把每轮对话都拖慢。

### 2.3 第 3 步：`EvidenceStore.rebuild_repo`

这一步容易被忽略，但它是"Agent 能看到什么"的边界。

```python
# chat_memory.py:647 EvidenceStore.search
阶段 1：search_similar_documents()   # 向量 + BM25 混合检索，够数就返回
阶段 2：关键词兜底                    # 向量不够 / Milvus 挂了 → 在 evidence_items 里按词频捞最近 500 条
```

**关键**：`MilvusUnavailableError` 被静默吞掉，降级到关键词。**向量库挂了，对话不会挂。**

### 2.4 第 4 步：`ContextAssembler.assemble` 的三阶段预算

```python
# chat_memory.py:806
步骤 0  default_context_budget()          分预算（system / recent / evidence 各多少 token）
步骤 1  取记忆                              ConversationMemory 优先，否则 ThreadMemory
步骤 2  取上一次封存的会话摘要              ChatSession.status in ("sealed", "sealed_with_warnings")
步骤 3  取证                                EvidenceStore.search(limit=8)
步骤 4  取活动消息                          MessageStore.recent(limit=max(limit,30))  ← 多取，给压缩留选择空间
步骤 5  算原始 token（含 tool_calls）        tool_calls 也占预算，不能漏
步骤 6  第一层压缩 microcompact            只清旧的工具结果，不动消息本身
步骤 7  第二层压缩 compress_messages       按 recent 预算丢旧消息
步骤 8  拼 system_context                  build_context() 闭包
步骤 9  再算一次总压力                      还超且没 microcompact 过 → 补做一次
```

**步骤 9 是很多人看漏的地方**：`system_context` 自己也可能很大（记忆 + 证据 + 摘要），拼完必须重新称重。

### 2.5 组装出来的 `context` 长什么样

```python
# chat.py:2002-2013
context = {
    **_client_context(payload.context),   # 前端传来的，但已剔除 server_owned 键
    **_skill_runtime_context(),           # 技能注册表
    "repo_id": str(repo.id),
    "conversation_id": str(conversation.id),
    "server_context": assembled.system_context,     # 拼好的大字符串
    "server_messages": assembled.history_messages,  # 结构化历史
    "evidence": assembled.evidence,                 # 8 条预检索证据
    "compression_stats": assembled.compression_stats,
    "progressive_compaction": compaction_stats,
}
context["tools"] = _build_conversation_tools(repo, conversation, db)   # 14 个工具
```

**`_client_context` 是个安全点**：它剔除 `evidence` / `tools` 这类 `server_owned` 键。否则前端可以伪造证据，让模型"看到"不存在的文档。

---

## 3. 阶段③：`run_stream` 的骨架

```python
async def run_stream(self, input_data, context):   # chat_agent.py:176
    message = input_data["message"]
    runtime_context = dict(context)                # 浅拷贝，不污染调用方

    # 第 0 步：激活 Skill（匹配 SKILL.md，决定本轮加载哪些指令）
    active_skills = self._activate_skills(message, runtime_context)   # 186

    # 第 1 步：原始函数 → LangChain StructuredTool
    tools = self._build_langchain_tools(runtime_context["tools"], runtime_context)  # 209

    # 第 2 步：初始化 ConversationState（一个 TypedDict，全程在节点间传）
    state = { "message":..., "observations":[], "agent_events":[], "tool_calls":[],
              "pending_tool_calls":[], "step":0, "route":"direct_answer", ... }   # 212

    # 第 3 步：记录 RAG 策略（预检索 + 推理中补检索 = 混合 RAG）
    self._append_agent_event(state, "rag_strategy", {...})   # 290

    # 四条短路（见下）
    # 主路径：进 LangGraph
    graph = self._build_agent_graph(self._tool_model(tools))   # 433
    async for event in self._stream_agent_graph(graph, state):
        yield event
```

### 3.1 四道短路：能不进 LLM 就不进

| # | 触发条件 | 做法 | route 值 | 代码 |
|---|---|---|---|---|
| 1 | "我刚才说了什么" 这类只需最近消息 | 直接从 history 拼答案，**不调 LLM** | `direct_answer` | 314-340 |
| 2 | 问题明确点名某个记忆工具 | 直接预取那个工具，不让模型自己选 | 工具名 | 342-369 |
| 3 | 问"这个项目是干什么的" | 预读 README / manifest | `workspace_read_file` | 371-397 |
| 4 | 没配 LLM key | 用预检索证据拼本地答案 | `prefetched_evidence` / `direct_answer` | 399-428 |

**这四道闸的业务价值**：分支 1、2、3 省掉一次完整的 LLM 调用；分支 4 让"没配 key 也能演示"。

注意分支 1 的判断逻辑在 `_memory_context_policy`（`chat_agent.py:1251`），它返回 `mode` 和 `tool` 两个字段，分别喂给分支 1 和分支 2 —— 一次判断，两个用途。

---

## 4. 阶段④：LangGraph 推理循环

### 4.1 图的结构（只有三个节点）

```python
# chat_agent.py:474-490
graph.add_node("model_decision", model_decision_node)   # 思考：要不要调工具
graph.add_node("tool_action", self._graph_tool_action)  # 行动：执行一个工具
graph.add_node("max_steps_stop", self._graph_max_steps_stop)  # 预算耗尽兜底

START → model_decision
model_decision --有 tool_calls--> tool_action
model_decision --没有-----------> END
tool_action --还有待执行---------> tool_action
tool_action --步数超限且没答案---> max_steps_stop → END
tool_action --否则---------------> model_decision
```

**为什么 `tool_action` 一次只执行一个工具**（`638`：`pending[0]`，其余 `pending[1:]` 出队）：
模型可能一次返回 3 个 tool_calls，逐个执行才能逐个吐 `tool_use` / `tool_result` 事件，前端才能实时显示"正在调第 2 个工具"。并行执行会丢失过程。

### 4.2 `model_decision`：每轮只重建两条消息

这是整个设计里最值得学的一点。

```python
# chat_agent.py:542-545
messages = [
    SystemMessage(content=system_prompt),   # 固定不变
    HumanMessage(content=user_prompt),      # 每轮重新拼
]
```

**不是** `messages.append(...)` 的累积式对话。历史、证据、观察结果、工具索引全部拼进 `user_prompt` 这一个字符串里。

`_tool_calling_user_prompt`（`1111`）的拼装顺序是固定的：

```
使用规则 → 当前模型 → 可用工具 → 技能 → 团队 → 服务端上下文
→ 历史覆盖度 → 最近消息 → 已有观察结果 → 工具调用/结果索引 → 用户原话
```

**好处**：
- token 消耗可预测 —— 不会因为聊了 50 轮就把上下文撑爆
- 每一轮都能精确控制"给模型看多少"，而不是被动累积
- 压缩逻辑集中在拼装函数里，只有一处

**代价**：模型没有原生的多轮记忆，全靠这个字符串。所以拼装顺序本身就成了"提示工程"的一部分 —— 用户原话放在**最后**，因为 LLM 对末尾内容最敏感。

### 4.3 上下文溢出：一次激进重试

```python
# chat_agent.py:553
try:
    response = await tool_model.ainvoke(messages)   # 注意：ainvoke，不是 astream
except Exception as exc:
    if not is_context_overflow_error(exc) or state.get("context_overflow_recovery_attempted"):
        raise
    # 只重试一次，且用 aggressive 模式重建 prompt
    state["context_overflow_recovery_attempted"] = True
    user_prompt = self._tool_calling_user_prompt(state, aggressive=True)
```

`aggressive=True` 时（`1131`）：
- `server_context` 压到预算的 1/3
- `history` 压到 `recent_tokens` 的 1/2

**只重试一次**是关键 —— 无限重试会变成死循环。

### 4.4 三个决策出口

```python
if response_text and tool_calls:
    # 模型一边说话一边调工具 → 把这段"思考"也吐出去
    stream_events.append(self._event("thought", {"content": response_text}))   # 591-593

if not tool_calls:
    state["final_answer"] = response_text        # 没有工具调用 = 这就是最终答案
    self._stop_loop(state, "model_final_answer", "completed")   # 595-599
else:
    repeated = self._repeated_tool_call(tool_calls, state)      # 重复检测
    if repeated:
        # 同一工具同参数超过 2 次 → 强制停，防死循环烧钱
        state["route"] = "tool_loop_guard"
        state["final_answer"] = "我停止了这轮 Agent Loop，因为模型连续请求相同工具和参数..."
        return self._graph_state_update(state, stream_events)   # 602-634
    state["pending_tool_calls"] = tool_calls
```

### 4.5 `tool_action`：三道校验 + 分类重试

```python
# chat_agent.py:848 _complete_tool_call
# 校验 1：显式技能的工具白名单（861）
skill_violation = self._explicit_skill_tool_violation(public_action, state)
# → 越界直接拒绝，error_kind="permission_denied"，不重试

# 校验 2：工具是否存在（885）
elif safe_action not in tools:
# → error_kind="unknown_tool"，把可用工具列表回给模型，让它自己改

# 正常执行（909）
else:
    observation = await self._invoke_tool_with_recovery(...)
```

**校验 1 是"工具白名单即权限边界"的落地**：用户显式指定技能时，只能用该技能 `SKILL.md` 里 `tools` 字段声明的工具。**那不是建议，是沙箱。**

`_invoke_tool_with_recovery`（`984`）：

```python
max_attempts = 1 + self.tool_retry_attempts      # = 2
for attempt in 1..max_attempts:
    try:  return await tool.ainvoke(action_input)
    except Exception as exc:
        error_kind = self._classify_tool_error(exc)     # timeout / rate_limit / permission / ...
        retryable = self._is_retriable_tool_error(...)
        if not retryable: break
```

`_classify_tool_error`（`1051`）按异常文本归类：`timeout` / `rate_limit`（429）/ `permission_denied` / `unknown_tool` / `tool_runtime_error`。**只有超时和限流会重试** —— 参数错、权限错重试一万次也没用。

### 4.6 观察结果怎么变成"下一轮的输入"

```python
# chat_agent.py:968
state["citations"] = [*(state.get("citations") or []), *observation.get("citations", [])]
```

工具的返回结构是固定的四件套：

```python
{"route": "...", "tool_calls": [...], "answer": "...", "citations": [...]}
```

`answer` 进 `observations`，下一轮拼进 `user_prompt` 的"已有观察结果"段。**这就是 Agent 的"记忆"—— 不在消息列表里，在 prompt 字符串里。**

---

## 5. 事件流：后端吐什么，前端收什么

| SSE event | 触发时机 | data 关键字段 | 前端落点 |
|---|---|---|---|
| `thought` | 模型边说边调工具 | `content` | 执行链里的"公开思路" |
| `tool_use` | 工具开始执行 | `id` / `tool_name` / `arguments` | 执行链新增一条 running |
| `tool_result` | 工具执行完 | `tool_use_id` / `status` / `content` / `error` | 同 id 覆盖成 done / error |
| `delta` | 最终答案切片 | `content`（36 字符/片） | 正文逐字追加 |
| `final` | 一轮结束 | `answer` / `route` / `tool_calls` / `agent_events` / `model_usage` / `context_diagnostics` / `message_id` | 覆盖正文 + 渲染脚注 |

### 5.1 一个反直觉的点：`delta` 不是真流式

```python
# chat_agent.py:2336
async def _stream_answer(self, answer: str):
    """把最终答案切成 delta 事件流。
    注意：这不是真正的流式生成 —— 模型调用是 ainvoke（一次性拿完整结果），
    这里只是按 36 字符切片，给前端做打字机效果。
    原因：Agent 循环必须先拿到完整结果才能判断要不要继续调工具。
    """
    for part in self._chunk_text(answer):
        yield self._event("delta", {"content": part})
```

**为什么不做真流式**：Agent 必须先看到完整的 `response.tool_calls` 才能决定下一步。边生成边判断，等于要在流中途改主意。

代价是首字节延迟高（要等整个 LLM 调用结束）。这是"Agent 场景"和"纯聊天场景"的本质区别。

### 5.2 事件从图里怎么出来

```python
# chat_agent.py:492 _stream_agent_graph
async for graph_event in graph.astream(current_state, config={"recursion_limit": self.max_steps * 8 + 20}, stream_mode="debug"):
    if graph_event["type"] == "task" and node_name == "tool_action":
        yield self._event("tool_use", preview)     # 节点开始 → 先吐 tool_use
    if graph_event["type"] != "task_result": continue
    for event in result["stream_events"]:
        yield event                                # 节点结束 → 吐它攒下的事件
```

用 `stream_mode="debug"` 是为了拿到节点的输入输出 —— 这样"工具开始执行"（`task`）和"节点完成"（`task_result`）都能捕获。否则只能拿到最终状态，过程全丢。

---

## 6. 阶段⑤：只有 final 才落库

```python
# chat.py:2015-2053
async for event in ChatAgent().run_stream({"message": payload.message}, context):
    if event["event"] == "final":
        run_id = uuid.uuid4()
        event["data"]["run_id"] = str(run_id)
        event["data"]["compression_stats"] = assembled.compression_stats
        event["data"]["progressive_compaction"] = compaction_stats

        assistant_message = message_store.append(          # ① 写 assistant 消息
            role="assistant", content=answer, route=route,
            tool_calls=..., meta={"run_id":..., "agent_events":..., "model_usage":..., "context_diagnostics":...},
        )
        event["data"]["message_id"] = str(assistant_message.id)

        db.add(AgentRun(...))                              # ② 写执行记录
        touch_conversation(db, conversation, increment=2)  # ③ 更新会话计数与时间
        await SessionSealer(db).seal(...)                  # ④ 封存会话 → 抽长期记忆
        db.commit()
    yield encode_sse(event)                                # ⑤ 无论什么事件都推给前端
```

**为什么中间事件不落库**：一次对话可能产生 30 个 `tool_use` / `tool_result`，逐个写库会让 SSE 变慢，而且这些事件最后都打包在 `agent_events` 里跟着 `final` 一起落库了。

### 6.1 错误路径（响应头已发出，改不了状态码）

```python
except Exception as exc:
    db.rollback()                          # 回滚半截事务
    answer = f"请求失败：{exc}"
    failed_event = {"event": "final", "data": {"route": "error", ...}}
    # 仍然写一条 assistant 消息 + AgentRun（status="error"）+ 封存
    yield encode_sse(failed_event)
```

**为什么错误也要落库**：用户刷新页面时，这条"请求失败"要能看到，否则对话里会缺一段，看起来像丢消息。

### 6.2 `SessionSealer` 在做什么（数据流 5 的起点）

封存 = 把这一轮对话交给记忆系统：

```
会话 → 摘要 → memory_candidates（候选记忆）
                    ↓ 人工批准
                 documents → 进 RAG 向量库
```

**这就是"下次还能查到"的实现**。对话本身不是记忆，只有封存抽取出来的候选才是。

---

## 7. 三道刹车：防死循环烧钱

```python
# chat_agent.py:145-147
max_steps = 12                  # 最多 12 步
max_repeated_tool_calls = 2     # 同工具同参数最多 2 次
tool_retry_attempts = 1         # 仅网络类错误重试 1 次
```

| 刹车 | 触发后的行为 | route 值 |
|---|---|---|
| 步数用尽 | 走 `max_steps_stop`，基于已有观察给个交代 | `tool_loop_budget_exhausted` |
| 重复调用 | 直接停，返回"我停止了这轮 Agent Loop..." | `tool_loop_guard` |
| 工具报错 | 只对 timeout / 429 重试 1 次，其余直接回给模型 | `tool_error` |

**注意步数用尽时的行为**：不是抛异常，是**用已有证据给一个降级答案**。用户至少知道"它查到了什么、还差什么"。

---

## 8. 一次请求的数据流向总表

| 环节 | 数据从哪来 | 变成什么 | 落到哪 |
|---|---|---|---|
| 会话 upsert | `payload.conversation_id` | `Conversation` 行 | `conversations` 表 |
| 压缩 | `chat_messages` + 预算 | 压缩边界（`role=system` 的假消息） | `chat_messages` 表 |
| 写用户消息 | `payload.message` | `ChatMessage(role=user)` | `chat_messages` 表 |
| 重建证据 | `issues`/`pull_requests`/`workflow_runs`/`documents` | `evidence_items` 快照 | `evidence_items` 表 |
| 取证 | `evidence_items` + Milvus | 8 条证据（`title/snippet/score`） | `context["evidence"]`（内存） |
| 组装 | 记忆 + 摘要 + 证据 + 历史 | `system_context` 字符串 | `context["server_context"]`（内存） |
| 工具调用 | 模型返回的 `tool_calls` | `observation` 四件套 | `state["observations"]`（内存） |
| 最终答案 | `state["final_answer"]` | `ChatMessage(role=assistant)` | `chat_messages` 表 |
| 执行记录 | `route` / `tool_calls` / `answer` | `AgentRun` 行 | `agent_runs` 表 |
| 封存 | 整轮对话 | 摘要 + 候选记忆 | `chat_sessions` / `memory_candidates` |
| 事件流 | 每个节点产生 | SSE 事件 | 前端（不落库，随 final 打包） |

---

## 9. 五个值得记住的设计取舍

1. **上下文在 Agent 之外准备**
   Agent 不查数据库，只消费一个 `context` 字典。好处：Agent 可以被单独测试（喂假 context 就行），也便于评测复用同一套数据。

2. **投影而非累积**
   每轮只重建 System + Human 两条消息，历史全拼进字符串。token 可预测，不会被历史撑爆。

3. **四道短路优先于循环**
   能用规则回答的，不进 LLM。这是成本控制的第一层。

4. **工具白名单是沙箱不是建议**
   技能激活后越界调用直接被拦，`permission_denied` 且不重试。

5. **只有 final 落库**
   中间事件全部打包进 `agent_events`，跟着 final 一次写完。SSE 快，事务也干净。

---

## 10. 追代码的推荐顺序

如果你想自己顺着读一遍，按这个顺序最省力：

```
1. chat.py:2118            @router.post("/stream")          看入口签名
2. chat.py:1989            _conversation_sse                看四步准备（15 行，很短）
3. chat.py:2002-2013       context 字典                     看 Agent 到底收到什么
4. chat_agent.py:176       run_stream                       看骨架 + 四条短路
5. chat_agent.py:461       _build_agent_graph               看三个节点和边
6. chat_agent.py:525       _graph_model_decision            看"两条消息"和三个出口
7. chat_agent.py:638       _graph_tool_action               看一次只执行一个工具
8. chat_agent.py:848       _complete_tool_call              看三道校验
9. chat_agent.py:1111      _tool_calling_user_prompt         看 prompt 拼装顺序
10. chat.py:2017-2052      final 落库                       看四张表怎么写
```

**不建议**从头到尾顺序读 `chat_agent.py`（2,362 行），它的函数是按"调用顺序"排的，不是按"阅读顺序"。

---

## 11. 两个容易踩的认知误区

**误区 1：「Agent 有记忆」**
它没有。所谓记忆是三层拼出来的：`recent messages`（短期）+ `evidence`（检索）+ `ThreadMemory/ConversationMemory`（长期摘要）。每轮都重新拼进 prompt。

**误区 2：「delta 是模型在流式输出」**
不是。模型调用是 `ainvoke`（一次性拿全），`delta` 是后端把完整答案按 36 字符切片后推的，纯前端打字机效果。真流式要等"边生成边判断工具调用"这个问题被解决。
