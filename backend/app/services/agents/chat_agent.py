"""
对话 Agent（ChatAgent）—— 全项目最核心的文件
==========================================

【业务定位】业务主线"判断"这一步的实现：理解用户问题 → 选工具取证 → 综合回答。
    它是唯一"会思考"的 Agent；其余 12 个都是规则驱动的专家函数，由它调用。
【数据流向】
    上游：api/routes/chat.py 注入 context["tools"]（14 个工具）和上下文（历史/证据/记忆）
    下游：产出 SSE 事件流（thought / tool_use / tool_result / delta / final）
          → api/routes/chat.py 落库（ChatMessage + AgentRun）→ 前端渲染
【核心规则】
    1. 推理循环最多 12 步；同一工具同参数最多调 2 次；仅网络类错误重试 1 次。
    2. 四条短路分支（最近历史够答 / 强制记忆工具 / 项目概览 / 无 LLM）不进入循环。
    3. 模型调用是 ainvoke（非流式）；"流式"是最后按 36 字符切片模拟。
"""

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.services.agents.base import BaseAgent
from app.services.context_compression import (
    calculate_context_pressure,
    clip_to_token_budget,
    default_context_budget,
    estimate_tokens,
    is_context_overflow_error,
    prompt_cache_model_kwargs,
)
from app.services.skills import SkillRegistry, SkillRegistryError


# 工具函数的签名：接收（参数, 上下文），返回观察结果
ConversationTool = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]

# 工具错误分类：决定是否重试、前端怎么展示
ToolErrorKind = Literal[
    "unknown_tool",         # 模型调了不存在的工具
    "timeout",              # 超时
    "rate_limited",         # 限流
    "transient_network",    # 网络抖动
    "permission_denied",    # 权限不足（含 Skill 白名单拒绝）
    "data_not_found",       # 数据不存在
    "model_output_invalid", # 模型输出格式不对
    "tool_runtime_error",   # 其他运行时错误
]


class GenericToolArgs(BaseModel):
    """工具的通用参数 schema。

    【为什么用一个"万能参数类"而不是每个工具单独定义】
        14 个工具参数各异，逐个定义 schema 代码量巨大且难维护。
        这里用 extra="allow" 让模型可以传任意字段，再在
        _normalize_action_input() 里做统一清洗与默认值填充。
    """

    model_config = ConfigDict(extra="allow")

    message: str | None = Field(default=None, description="原始用户消息")          # 原始问题
    query: str | None = Field(default=None, description="搜索查询")                # 检索查询词
    keyword: str | None = Field(default=None, description="关键词过滤")            # 关键词
    limit: int | None = Field(default=None, description="结果数量上限")            # 条数上限
    repo_id: str | None = Field(default=None, description="仓库 UUID")             # 仓库
    repo_full_name: str | None = Field(default=None, description="仓库完整名称")   # owner/name
    conversation_id: str | None = Field(default=None, description="会话 UUID")     # 会话
    session_id: str | None = Field(default=None, description="会话片段 UUID")      # 封存会话
    message_id: str | None = Field(default=None, description="需要精确读取的消息 UUID")  # 指定消息
    before_message_id: str | None = Field(default=None, description="只读取该消息之前的记录")  # 分页
    after_message_id: str | None = Field(default=None, description="只读取该消息之后的记录")   # 分页
    offset: int | None = Field(default=None, description="从消息正文的字符偏移量开始读取")     # 分页
    max_chars: int | None = Field(default=None, description="本次最多返回的正文字符数")       # 限长
    include_transcript: bool | None = Field(default=None, description="读取封存会话时是否包含原始 transcript")
    route: str | None = Field(default=None, description="仓库路由或意图")
    path: str | None = Field(default=None, description="相对工作区的文件或目录路径")          # 文件路径
    pattern: str | None = Field(default=None, description="文本或 glob 模式")                # 匹配模式
    start_line: int | None = Field(default=None, description="读取文件时从第几行开始，行号从 1 开始")
    line_count: int | None = Field(default=None, description="最多读取多少行")


class ConversationState(TypedDict, total=False):
    """LangGraph 的状态对象。整个推理循环的所有数据都在这里流转。

    【为什么用 TypedDict 而不是 dataclass】
        LangGraph 要求状态是可合并的 dict；TypedDict 只是给 dict 加类型提示，
        运行时仍是普通 dict，兼容 graph 的合并语义。
    """

    message: str                                    # 用户原始问题
    context: dict[str, Any]                         # 运行上下文（tools / 证据 / 记忆 / 技能）
    tools: dict[str, StructuredTool]                # 已包装的 LangChain 工具
    tool_name_map: dict[str, str]                   # 安全名 → 公开名（点号被替换过）
    observations: list[dict[str, Any]]              # 每次工具调用的观察结果
    agent_events: list[dict[str, Any]]              # 细粒度事件（前端时间线）
    tool_calls: list[dict[str, Any]]                # 工具调用轨迹（给前端和评测）
    citations: list[dict[str, Any]]                 # 引用来源
    pending_tool_calls: list[dict[str, Any]]        # 待执行的工具调用队列
    stream_events: list[dict[str, Any]]             # 本轮要吐给前端的 SSE 事件
    route: str                                      # 走了哪条处理分支
    step: int                                       # 当前第几步（对应 max_steps）
    final_answer: str                               # 最终答案
    stop_reason: str                                # 为什么停止（用于诊断）
    completion_status: str                          # running / completed / blocked / budget_exhausted
    tool_call_counts: dict[str, int]                # 工具调用指纹计数（防重复）
    evaluation_trace: dict[str, Any]                # 评测模式下的证据轨迹
    model_usage: list[dict[str, Any]]               # 每次 LLM 调用的 token 用量
    prompt_diagnostics: list[dict[str, Any]]        # 每次 prompt 的 token 构成与压力
    runtime_context_stats: dict[str, Any]           # 运行期上下文投影统计
    context_overflow_recovery_attempted: bool       # 是否已经做过溢出重试（只重试一次）


class ChatAgent(BaseAgent):
    """使用原生工具调用的对话 Agent。"""

    name = "chat_agent"
    description = "使用原生工具调用的对话 Agent。LLM 选择 LangChain 工具，记忆工具通过 MCP 执行。"

    # 14 个工具 = Agent 的"手和眼睛"。定义在 api/routes/chat.py，这里只声明名字
    available_tools = [
        "devflow_search_evidence",              # MCP：搜持久记忆
        "devflow_get_thread_context",           # MCP：读当前会话原文
        "devflow_read_session_events",          # MCP：读封存会话
        "workspace.list_files",                 # 工作区：列文件
        "workspace.read_file",                  # 工作区：读文件
        "workspace.search_code",                # 工作区：搜代码
        "repo_health.generate_summary",         # 仓库健康度
        "rag.search_similar_documents",         # RAG：混合检索
        "issue_agent.analyze_issue",            # 专用 Agent：Issue 分诊
        "pr_review_agent.analyze_pr",           # 专用 Agent：PR 审查
        "ci_debug_agent.analyze_workflow_run",  # 专用 Agent：CI 排障
        "report_agent.generate_weekly_report",  # 专用 Agent：周报
        "safety_agent.classify_action_risk",    # 安全：判风险出草稿
        "workflow.run_engineering_review",      # 多 Agent 工作流
    ]

    # ── 三道安全刹车（防止无限循环烧钱）──
    max_steps = 12                  # 最多 12 步
    max_repeated_tool_calls = 2     # 同工具同参数最多 2 次
    tool_retry_attempts = 1         # 仅网络类错误重试 1 次

    def __init__(self, model: ChatOpenAI | None = None) -> None:
        """初始化。没有 LLM key 时 model 为 None，走"无模型降级"分支。"""
        self.model = model
        if self.model is None and settings.llm_api_key:
            self.model = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=0,                              # 分析类任务要稳定，不要创造性
                timeout=60,
                model_kwargs=prompt_cache_model_kwargs(),   # 启用上游 prompt 缓存省费用
            )

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """非流式入口：消费 run_stream，只取最后的 final 事件。"""
        final: dict[str, Any] | None = None
        async for event in self.run_stream(input_data, context):
            if event["event"] == "final":
                final = event["data"]
        return final or {
            "run_id": "",
            "route": "unknown",
            "tool_calls": [],
            "answer": "没有生成最终回答。",
            "citations": [],
        }

    async def run_stream(self, input_data: dict[str, Any], context: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """主入口：激活技能 → 建状态 → 四条短路分支 → LangGraph 循环 → 流式输出。

        返回的是 SSE 事件流，每个事件形如 {"event": "delta", "data": {...}}。
        """
        message = str(input_data.get("message") or "")
        runtime_context = dict(context)     # 浅拷贝，避免污染调用方传入的 context

        # ── 第 0 步：激活 Skill（匹配 SKILL.md，决定本轮加载哪些指令）──
        try:
            active_skills = self._activate_skills(message, runtime_context)
        except SkillRegistryError as exc:
            # 技能注册表出错就直接返回错误答案，不进入循环
            answer = f"Skill 激活失败：{exc}"
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "skill_error",
                    "tool_calls": [],
                    "agent_events": [{"type": "skill_activation_error", "error": str(exc)}],
                    "answer": answer,
                    "citations": [],
                    "completion": {"status": "blocked", "reason": "skill_activation_error", "steps": 0, "tool_call_count": 0, "tool_error_count": 0},
                },
            )
            return
        runtime_context["active_skills"] = active_skills

        # ── 第 1 步：把原始工具函数包装成 LangChain 工具 ──
        raw_tools: dict[str, ConversationTool] = runtime_context["tools"]
        tools = self._build_langchain_tools(raw_tools, runtime_context)

        # ── 第 2 步：初始化状态 ──
        state: ConversationState = {
            "message": message,
            "context": runtime_context,
            "tools": tools,
            "tool_name_map": self._tool_name_map(raw_tools),
            "observations": [],
            "agent_events": [],
            "tool_calls": [],
            "citations": [],
            "pending_tool_calls": [],
            "stream_events": [],
            "route": "direct_answer",
            "step": 0,
            "final_answer": "",
            "stop_reason": "",
            "completion_status": "running",
            "tool_call_counts": {},
            "model_usage": [],
            "prompt_diagnostics": [],
            "runtime_context_stats": {},
            "context_overflow_recovery_attempted": False,
        }
        if context.get("evaluation_mode"):
            state["evaluation_trace"] = self._initial_evaluation_trace(message, context)
        self._append_agent_event(
            state,
            "agent_loop_start",
            {
                "mode": "langgraph_tool_calling_loop",
                "max_steps": self.max_steps,
                "available_tools": list(self._tool_name_map(raw_tools).values()),
            },
        )
        # 记录每个激活的技能（前端"技能"面板会显示）
        for skill in active_skills:
            activation_event = {
                key: value
                for key, value in skill.items()
                if key
                in {
                    "skill_name",
                    "skill_title",
                    "skill_version",
                    "entrypoint",
                    "tools",
                    "workflow_steps",
                    "output_contract",
                    "safety_level",
                    "activation_mode",
                    "activation_reason",
                    "instruction_digest",
                }
            }
            activation_event["resources"] = [
                {key: value for key, value in resource.items() if key != "content"}
                for resource in skill.get("resources") or []
                if isinstance(resource, dict)
            ]
            self._append_agent_event(
                state,
                "skill_activated",
                activation_event,
            )

        # ── 第 3 步：记录 RAG 策略（调用前预检索 + 推理中可补检索 = 混合 RAG）──
        prefetched_evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
        pre_retrieval_enabled = "evidence" in context
        reasoning_retrieval_enabled = "rag.search_similar_documents" in raw_tools
        pre_retrieval_sources = [
            {
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "source_type": str(item.get("source_type") or "evidence"),
                "title": str(item.get("title") or "未命名证据"),
                "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            }
            for item in prefetched_evidence[:8]
            if isinstance(item, dict)
        ]
        self._append_agent_event(
            state,
            "rag_strategy",
            {
                "mode": "hybrid_rag" if pre_retrieval_enabled and reasoning_retrieval_enabled else "tool_calling",
                "pre_retrieval": {
                    "enabled": pre_retrieval_enabled,
                    "evidence_count": len(prefetched_evidence),
                    "query": message,
                    "sources": pre_retrieval_sources,
                },
                "reasoning_retrieval": {
                    "enabled": reasoning_retrieval_enabled,
                    "supports_query_rewrite": reasoning_retrieval_enabled,
                    "supports_multiple_searches": reasoning_retrieval_enabled,
                },
            },
        )

        # ════════════════════════════════════════════════════════
        # 四条短路分支：能不进 LLM 循环就不进（省钱省时间）
        # ════════════════════════════════════════════════════════

        # 分支 1：最近消息已经够回答（如"我刚才说了什么"）
        memory_policy = self._memory_context_policy(message, context)
        if memory_policy:
            self._append_agent_event(state, "context_policy", memory_policy)
        if memory_policy and memory_policy.get("mode") == "direct_recent_history":
            state["final_answer"] = await self._direct_answer_from_recent_history(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = self._local_recent_history_answer(state)
            self._stop_loop(state, "recent_history_available", "completed")
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "direct_answer",
                    "tool_calls": [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 分支 2：问题明确要求某个记忆工具 → 直接预取，不让模型自己选
        required_memory_tool = str((memory_policy or {}).get("tool") or "") or None
        if required_memory_tool and required_memory_tool in raw_tools:
            async for event in self._prefetch_memory_context(state, required_memory_tool):
                yield event
            state["final_answer"] = await self._final_from_observations(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = await self._fallback_answer(state)
            self._stop_loop(state, "required_memory_context_loaded", self._tool_completion_status(state))
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": state.get("route") or required_memory_tool,
                    "tool_calls": state.get("tool_calls") or [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 分支 3：问"这个项目是干什么的" → 预读 README/manifest
        if self._asks_project_overview(message):
            async for event in self._prefetch_project_context(state):
                yield event
            state["final_answer"] = await self._final_from_observations(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = await self._fallback_answer(state)
            self._stop_loop(state, "project_context_loaded", self._tool_completion_status(state))
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": state.get("route") or "workspace_read_file",
                    "tool_calls": state.get("tool_calls") or [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 分支 4：没有配置 LLM → 用预检索证据拼一个本地答案
        if self.model is None:
            answer = self._local_prefetched_evidence_answer(state) or (
                "当前没有取得足够证据。请先确认仓库已连接并完成同步；开放式对话还需要配置 LLM。"
            )
            evidence = (state.get("context") or {}).get("evidence")
            has_evidence = isinstance(evidence, list) and any(isinstance(item, dict) for item in evidence)
            self._stop_loop(
                state,
                "prefetched_evidence_available" if has_evidence else "llm_not_configured",
                "completed" if has_evidence else "blocked",
            )
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "prefetched_evidence" if has_evidence else "direct_answer",
                    "tool_calls": [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": self._prefetched_evidence_citations(state),
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # ════════════════════════════════════════════════════════
        # 主路径：进入 LangGraph 推理循环
        # ════════════════════════════════════════════════════════
        graph = self._build_agent_graph(self._tool_model(tools))
        async for event in self._stream_agent_graph(graph, state):
            yield event

        # 收尾：兜底答案 → 记录技能校验 → 流式输出 → final
        answer = str(state.get("final_answer") or "").strip()
        if not answer:
            answer = await self._fallback_answer(state)
        if state.get("completion_status") == "running":
            self._stop_loop(state, "fallback_answer_ready", self._tool_completion_status(state))
        self._record_skill_validation(state)
        async for chunk in self._stream_answer(answer):
            yield chunk
        yield self._event(
            "final",
            {
                "run_id": "",
                "route": state.get("route") or "direct_answer",
                "tool_calls": state.get("tool_calls") or [],
                "agent_events": state.get("agent_events") or [],
                "answer": answer,
                "citations": state.get("citations") or [],
                "completion": self._completion_summary(state),
                **self._runtime_metadata(state),
                **self._evaluation_result(state),
            },
        )

    def _build_agent_graph(self, tool_model: Any):
        """构建 LangGraph 状态图。只有两个工作节点 + 一个兜底节点。

        ┌──────────────┐
        │ model_decision │ ← 问 LLM：你要调工具吗？
        └───────┬──────┘
                │ 有 tool_calls → tool_action（执行一个工具）→ 回到 model_decision
                │ 没有          → END
                └ 步数超限     → max_steps_stop → END
        """
        async def model_decision_node(state: ConversationState) -> dict[str, Any]:
            return await self._graph_model_decision(state, tool_model)

        graph = StateGraph(ConversationState)
        graph.add_node("model_decision", model_decision_node)          # 思考
        graph.add_node("tool_action", self._graph_tool_action)         # 行动
        graph.add_node("max_steps_stop", self._graph_max_steps_stop)   # 预算耗尽兜底
        graph.add_edge(START, "model_decision")
        graph.add_conditional_edges(
            "model_decision",
            self._route_after_model_decision,
            {"tools": "tool_action", "end": END},
        )
        graph.add_conditional_edges(
            "tool_action",
            self._route_after_tool_action,
            {"tools": "tool_action", "model": "model_decision", "limit": "max_steps_stop"},
        )
        graph.add_edge("max_steps_stop", END)
        return graph.compile(name="chat_agent_tool_calling_graph")

    async def _stream_agent_graph(self, graph: Any, state: ConversationState) -> AsyncIterator[dict[str, Any]]:
        """跑图，并把图内部产生的事件转成 SSE 吐出去。

        用 stream_mode="debug" 能拿到每个节点的输入输出，
        这样"工具开始执行"（task）和"节点执行完"（task_result）都能捕获。
        """
        current_state: ConversationState = dict(state)
        config = {"recursion_limit": self.max_steps * 8 + 20}   # LangGraph 自身的递归保护
        async for graph_event in graph.astream(current_state, config=config, stream_mode="debug"):
            if not isinstance(graph_event, dict):
                continue
            payload = graph_event.get("payload") if isinstance(graph_event.get("payload"), dict) else {}
            node_name = str(payload.get("name") or "")
            # 工具开始执行时先吐一个 tool_use（让前端立刻显示"正在调用 XX"）
            if graph_event.get("type") == "task" and node_name == "tool_action":
                preview = self._preview_tool_use_event(payload.get("input"))
                if preview:
                    yield self._event("tool_use", preview)
                continue
            if graph_event.get("type") != "task_result":
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            # 把节点返回的状态合并回外层 state
            for key, value in result.items():
                if key != "stream_events":
                    current_state[key] = value
                    state[key] = value
            for event in result.get("stream_events") or []:
                if isinstance(event, dict):
                    yield event

    async def _graph_model_decision(self, state: ConversationState, tool_model: Any) -> dict[str, Any]:
        """【思考节点】调 LLM，决定下一步：给最终答案，还是调工具。

        关键实现：
            1. 每轮只重建两条消息（System + Human），不累积历史
               —— 历史/证据/观察结果都拼进 user_prompt 这个字符串里
            2. 上下文溢出时用 aggressive 模式重建 prompt 重试一次
            3. 检测重复调用（同工具同参数）→ 触发 loop guard
        """
        next_state: ConversationState = dict(state)
        step = int(next_state.get("step") or 0) + 1
        next_state["step"] = step
        self._append_agent_event(next_state, "agent_step", {"step": step, "phase": "model_decision"})

        system_prompt = self._tool_calling_system_prompt()
        user_prompt = self._tool_calling_user_prompt(next_state)
        # 【关键】只有两条消息。不是 messages.append(...) 的累积式对话
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        estimated_prompt_tokens = self._record_prompt_diagnostics(
            next_state,
            phase="model_decision",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        try:
            response = await tool_model.ainvoke(messages)   # 注意：ainvoke 不是 astream
        except Exception as exc:
            # 只有"上下文溢出"且没重试过，才用更激进的压缩重试一次
            if not is_context_overflow_error(exc) or next_state.get("context_overflow_recovery_attempted"):
                raise
            next_state["context_overflow_recovery_attempted"] = True
            next_state["runtime_context_stats"] = {
                **(next_state.get("runtime_context_stats") or {}),
                "reactive_compaction": "aggressive_retry",
                "overflow_error": str(exc),
            }
            self._append_agent_event(
                next_state,
                "context_overflow_recovery",
                {"phase": "model_decision", "strategy": "aggressive_runtime_projection"},
            )
            user_prompt = self._tool_calling_user_prompt(next_state, aggressive=True)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            estimated_prompt_tokens = self._record_prompt_diagnostics(
                next_state,
                phase="model_decision_reactive_retry",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await tool_model.ainvoke(messages)

        self._record_model_usage(
            next_state,
            response,
            phase="model_decision",
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        response_text = self._message_content(response).strip()
        raw_tool_calls = getattr(response, "tool_calls", None) or []
        tool_calls = [dict(call) for call in raw_tool_calls if isinstance(call, dict)]
        stream_events: list[dict[str, Any]] = []

        # 模型一边说话一边调工具 → 把这段"思考"也吐给前端
        if response_text and tool_calls:
            self._append_agent_event(next_state, "thought", {"content": response_text})
            stream_events.append(self._event("thought", {"content": response_text}))

        if not tool_calls:
            # 没有工具调用 → 这就是最终答案，循环结束
            next_state["final_answer"] = response_text
            next_state["pending_tool_calls"] = []
            self._stop_loop(next_state, "model_final_answer", "completed")
        else:
            # 有工具调用 → 先做重复检测
            repeated = self._repeated_tool_call(tool_calls, next_state)
            if repeated:
                # 同一工具同参数超过 2 次 → 强制停止（防死循环烧钱）
                next_state["pending_tool_calls"] = []
                next_state["route"] = "tool_loop_guard"
                next_state["final_answer"] = (
                    "我停止了这轮 Agent Loop，因为模型连续请求相同工具和参数，"
                    "继续执行只会重复消耗预算。请根据已有工具结果调整问题，或让我换一种检索/分析路径。"
                )
                next_state["observations"] = [
                    *(next_state.get("observations") or []),
                    {
                        "tool_name": repeated["tool_name"],
                        "arguments": repeated["arguments"],
                        "observation": {
                            "route": "tool_loop_guard",
                            "answer_preview": "Detected repeated identical tool call and stopped the loop.",
                            "repeat_count": repeated["repeat_count"],
                            "fingerprint": repeated["fingerprint"],
                        },
                    },
                ]
                self._stop_loop(
                    next_state,
                    "repeated_tool_call",
                    "blocked",
                    {
                        "tool_name": repeated["tool_name"],
                        "arguments": repeated["arguments"],
                        "repeat_count": repeated["repeat_count"],
                    },
                )
                return self._graph_state_update(next_state, stream_events)
            next_state["pending_tool_calls"] = tool_calls
        return self._graph_state_update(next_state, stream_events)

    async def _graph_tool_action(self, state: ConversationState) -> dict[str, Any]:
        """【行动节点】一次只执行一个工具调用（pending[0]）。"""
        next_state: ConversationState = dict(state)
        pending = list(next_state.get("pending_tool_calls") or [])
        if not pending:
            return self._graph_state_update(next_state, [])
        tool_call = pending[0]
        next_state["pending_tool_calls"] = pending[1:]      # 出队
        public_action, action_input, observation, _tool_message, _tool_use_event, tool_result_event = await self._execute_tool_call(tool_call, next_state)
        stream_events = [
            self._event("tool_result", tool_result_event),
            self._event("action", {"tool_name": public_action, "arguments": action_input}),
            self._event("observation", {"tool_name": public_action, "content": observation}),
        ]
        return self._graph_state_update(next_state, stream_events)

    async def _graph_max_steps_stop(self, state: ConversationState) -> dict[str, Any]:
        """【预算耗尽兜底】步数用完但还没答案 → 基于已有观察给个交代。"""
        next_state: ConversationState = dict(state)
        next_state["route"] = "tool_loop_budget_exhausted"
        next_state["final_answer"] = (
            "这轮 Agent Loop 已达到步骤预算上限。下面的结论只能基于已经完成的工具调用和观察结果，"
            "如果还需要继续，我会从当前证据继续缩小范围。"
        )
        self._stop_loop(next_state, "max_steps_exhausted", "budget_exhausted", {"max_steps": self.max_steps})
        return self._graph_state_update(next_state, [])

    def _route_after_model_decision(self, state: ConversationState) -> str:
        """条件边：有待执行工具就去行动，否则结束。"""
        return "tools" if state.get("pending_tool_calls") else "end"

    def _route_after_tool_action(self, state: ConversationState) -> str:
        """条件边：还有待执行工具就继续行动；步数超限且没答案就去兜底；否则回思考。"""
        if state.get("pending_tool_calls"):
            return "tools"
        if int(state.get("step") or 0) >= self.max_steps and not str(state.get("final_answer") or "").strip():
            return "limit"
        return "model"

    def _graph_state_update(self, state: ConversationState, stream_events: list[dict[str, Any]]) -> dict[str, Any]:
        """只把需要跨节点传递的字段回写给图（避免把 tools 这种大对象反复传递）。"""
        keys = [
            "observations",
            "agent_events",
            "tool_calls",
            "citations",
            "pending_tool_calls",
            "route",
            "step",
            "final_answer",
            "stop_reason",
            "completion_status",
            "tool_call_counts",
            "evaluation_trace",
            "model_usage",
            "prompt_diagnostics",
            "runtime_context_stats",
            "context_overflow_recovery_attempted",
        ]
        update = {key: state[key] for key in keys if key in state}
        update["stream_events"] = stream_events
        return update

    def _stop_loop(
        self,
        state: ConversationState,
        reason: str,
        completion_status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """统一记录"循环为什么停"，便于诊断和评测。"""
        state["stop_reason"] = reason
        state["completion_status"] = completion_status
        return self._append_agent_event(
            state,
            "agent_loop_stop",
            {"reason": reason, "completion_status": completion_status, **(extra or {})},
        )

    def _repeated_tool_call(self, tool_calls: list[dict[str, Any]], state: ConversationState) -> dict[str, Any] | None:
        """检测重复调用：给每次调用算指纹（工具名+参数），超过阈值就返回重复信息。"""
        counts = dict(state.get("tool_call_counts") or {})
        for tool_call in tool_calls:
            safe_action, public_action, _tool_use_id, action_input = self._tool_call_parts(tool_call, state)
            fingerprint = self._tool_call_fingerprint(safe_action, action_input)
            repeat_count = counts.get(fingerprint, 0) + 1
            counts[fingerprint] = repeat_count
            if repeat_count > self.max_repeated_tool_calls:
                state["tool_call_counts"] = counts
                return {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "fingerprint": fingerprint,
                    "repeat_count": repeat_count,
                }
        state["tool_call_counts"] = counts
        return None

    def _tool_call_fingerprint(self, safe_action: str, action_input: dict[str, Any]) -> str:
        """调用指纹：工具名 + 排序后的参数 JSON（保证同参数得到同指纹）。"""
        return f"{safe_action}:{json.dumps(action_input, ensure_ascii=False, sort_keys=True, default=str)}"

    def _completion_summary(self, state: ConversationState) -> dict[str, Any]:
        """生成完成状态摘要（前端显示"完成 / 部分完成 / 预算耗尽"）。"""
        tool_calls = [call for call in state.get("tool_calls") or [] if isinstance(call, dict)]
        error_calls = [call for call in tool_calls if call.get("status") == "error"]
        status = state.get("completion_status") or "completed"
        # 有工具报错但整体完成 → 单独标一个状态，避免掩盖问题
        if status == "completed" and error_calls:
            status = "completed_with_tool_errors"
        return {
            "status": status,
            "reason": state.get("stop_reason") or "model_final_answer",
            "steps": int(state.get("step") or 0),
            "tool_call_count": len(tool_calls),
            "tool_error_count": len(error_calls),
        }

    def _tool_completion_status(self, state: ConversationState) -> str:
        """全部工具都失败 → blocked；否则 completed。"""
        calls = [call for call in state.get("tool_calls") or [] if isinstance(call, dict)]
        if calls and not any(call.get("status") == "success" for call in calls):
            return "blocked"
        return "completed"

    def _preview_tool_use_event(self, raw_state: Any) -> dict[str, Any] | None:
        """在工具真正执行前，先给前端一个 tool_use 事件（让 UI 立刻有反馈）。"""
        if not isinstance(raw_state, dict):
            return None
        pending = raw_state.get("pending_tool_calls")
        if not isinstance(pending, list) or not pending or not isinstance(pending[0], dict):
            return None
        safe_action, public_action, tool_use_id, action_input = self._tool_call_parts(pending[0], raw_state)
        return {
            "type": "tool_use",
            "step": raw_state.get("step", 0),
            "id": tool_use_id,
            "tool_name": public_action,
            "arguments": action_input,
            **self._skill_event_metadata(public_action, raw_state),
        }

    async def _execute_tool_call_events(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行一个工具，并把 4 个事件（tool_use / tool_result / action / observation）吐出去。"""
        safe_action, public_action, tool_use_id, action_input, tool_use_event = self._prepare_tool_call(tool_call, state)
        yield self._event("tool_use", tool_use_event)
        observation, _tool_message, tool_result_event = await self._complete_tool_call(
            safe_action,
            public_action,
            tool_use_id,
            action_input,
            state,
        )
        yield self._event("tool_result", tool_result_event)
        yield self._event("action", {"tool_name": public_action, "arguments": action_input})
        yield self._event("observation", {"tool_name": public_action, "content": observation})

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> tuple[str, dict[str, Any], dict[str, Any], ToolMessage, dict[str, Any], dict[str, Any]]:
        """执行一个工具（不吐事件版，供图节点用）。"""
        safe_action, public_action, tool_use_id, action_input, tool_use_event = self._prepare_tool_call(tool_call, state)
        observation, tool_message, tool_result_event = await self._complete_tool_call(
            safe_action,
            public_action,
            tool_use_id,
            action_input,
            state,
        )
        return public_action, action_input, observation, tool_message, tool_use_event, tool_result_event

    def _prepare_tool_call(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        """解析工具调用并记录 tool_use 事件。"""
        safe_action, public_action, tool_use_id, action_input = self._tool_call_parts(tool_call, state)
        tool_use_event = self._append_agent_event(
            state,
            "tool_use",
            {"id": tool_use_id, "tool_name": public_action, "arguments": action_input, **self._skill_event_metadata(public_action, state)},
        )
        return safe_action, public_action, tool_use_id, action_input, tool_use_event

    def _tool_call_parts(
        self,
        tool_call: dict[str, Any],
        state: ConversationState | dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """拆解工具调用：安全名、公开名、调用 ID、清洗后的参数。"""
        safe_action = str(tool_call.get("name") or "").strip()
        public_action = self._public_tool_name(safe_action, state)  # type: ignore[arg-type]
        tool_use_id = str(tool_call.get("id") or f"tool_use_{len(state.get('agent_events') or []) + 1}")
        raw_args = tool_call.get("args") if isinstance(tool_call, dict) else {}
        # 模型有时把参数传成 JSON 字符串而不是 dict，这里兼容一下
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        action_input = self._normalize_action_input(public_action, raw_args if isinstance(raw_args, dict) else {}, state)  # type: ignore[arg-type]
        return safe_action, public_action, tool_use_id, action_input

    async def _complete_tool_call(
        self,
        safe_action: str,
        public_action: str,
        tool_use_id: str,
        action_input: dict[str, Any],
        state: ConversationState,
    ) -> tuple[dict[str, Any], ToolMessage, dict[str, Any]]:
        """执行工具并汇总结果（含三道前置校验）。"""
        tools = state.get("tools") or {}

        # ── 前置校验 1：显式技能的工具白名单 ──
        # 用户显式指定技能时，只能用该技能声明的工具，越界直接拒绝
        skill_violation = self._explicit_skill_tool_violation(public_action, state)
        if skill_violation:
            error = skill_violation
            error_kind: ToolErrorKind = "permission_denied"
            observation = {
                "route": "tool_error",
                "answer": error,
                "citations": [],
                "tool_calls": [
                    {
                        "tool_name": public_action,
                        "arguments": action_input,
                        "status": "error",
                        "error": error,
                        "error_kind": error_kind,
                        "retryable": False,
                        "attempts": 1,
                        "tool_use_id": tool_use_id,
                    }
                ],
                "error_kind": error_kind,
                "retryable": False,
            }
        # ── 前置校验 2：工具是否存在 ──
        elif safe_action not in tools:
            error = f"未知工具：{public_action}。可用工具：{', '.join(state.get('tool_name_map', {}).values())}"
            error_kind: ToolErrorKind = "unknown_tool"
            observation = {
                "route": "tool_error",
                "answer": error,
                "citations": [],
                "tool_calls": [
                    {
                        "tool_name": public_action,
                        "arguments": action_input,
                        "status": "error",
                        "error": error,
                        "error_kind": error_kind,
                        "retryable": False,
                        "attempts": 1,
                        "tool_use_id": tool_use_id,
                    }
                ],
                "error_kind": error_kind,
                "retryable": False,
            }
        # ── 正常执行（带重试）──
        else:
            observation = await self._invoke_tool_with_recovery(
                tools[safe_action],
                public_action,
                tool_use_id,
                action_input,
                state,
            )

        # 工具没返回 tool_calls 时，补一条"成功"记录，保证轨迹完整
        if not observation.get("tool_calls"):
            observation["tool_calls"] = [
                {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "status": "success",
                    "tool_use_id": tool_use_id,
                    **self._skill_event_metadata(public_action, state),
                }
            ]
        # 把 tool_use_id / 重试信息 / 技能信息补进每条记录
        for call in observation.get("tool_calls", []):
            if isinstance(call, dict):
                call.setdefault("tool_use_id", tool_use_id)
                if observation.get("attempts") is not None:
                    call.setdefault("attempts", observation.get("attempts"))
                if observation.get("error_kind") is not None:
                    call.setdefault("error_kind", observation.get("error_kind"))
                if observation.get("retryable") is not None:
                    call.setdefault("retryable", observation.get("retryable"))
                call.update(self._skill_event_metadata(str(call.get("tool_name") or public_action), state))

        self._capture_evaluation_observation(
            state,
            tool_name=public_action,
            arguments=action_input,
            observation=observation,
        )
        summary = self._summarize_observation(observation)
        status = "error" if any(call.get("status") == "error" for call in observation.get("tool_calls", []) if isinstance(call, dict)) else "success"
        tool_result_event = self._append_agent_event(
            state,
            "tool_result",
            {
                "tool_use_id": tool_use_id,
                "tool_name": public_action,
                "status": status,
                "route": observation.get("route"),
                "error_kind": observation.get("error_kind"),
                "retryable": observation.get("retryable"),
                "content": summary,
                **self._skill_event_metadata(public_action, state),
            },
        )
        # 把结果累加进状态（观察、轨迹、引用、事件）
        state["observations"] = [
            *(state.get("observations") or []),
            {"tool_name": public_action, "arguments": action_input, "observation": summary},
        ]
        state["tool_calls"] = [*(state.get("tool_calls") or []), *observation.get("tool_calls", [])]
        state["citations"] = [*(state.get("citations") or []), *observation.get("citations", [])]
        extra_events = observation.get("agent_events")
        if isinstance(extra_events, list):
            state["agent_events"] = [*(state.get("agent_events") or []), *[event for event in extra_events if isinstance(event, dict)]]
        observation_route = str(observation.get("route") or "")
        has_successful_tool = any(call.get("status") == "success" for call in state.get("tool_calls") or [])
        # 只在"没有成功调用"时才用 tool_error 覆盖 route（避免一次失败盖掉整体成功的路径）
        if observation_route and (observation_route != "tool_error" or not has_successful_tool):
            state["route"] = observation_route

        tool_message = ToolMessage(
            content=json.dumps(summary, ensure_ascii=False),
            tool_call_id=tool_use_id,
        )
        return summary, tool_message, tool_result_event

    async def _invoke_tool_with_recovery(
        self,
        tool: StructuredTool,
        public_action: str,
        tool_use_id: str,
        action_input: dict[str, Any],
        state: ConversationState,
    ) -> dict[str, Any]:
        """调工具，仅对可重试错误（超时/限流/网络）重试一次。"""
        max_attempts = 1 + self.tool_retry_attempts
        last_error: Exception | None = None
        last_error_kind: ToolErrorKind = "tool_runtime_error"
        retryable = False
        for attempt in range(1, max_attempts + 1):
            try:
                observation = await tool.ainvoke(action_input)
                if isinstance(observation, dict):
                    observation.setdefault("attempts", attempt)
                    return observation
                # 工具返回非 dict（如字符串）→ 包成标准结构
                return {
                    "route": public_action,
                    "answer": str(observation),
                    "citations": [],
                    "tool_calls": [],
                    "attempts": attempt,
                }
            except Exception as exc:
                last_error = exc
                last_error_kind = self._classify_tool_error(exc)
                retryable = self._is_retriable_tool_error(last_error_kind, public_action)
                if not retryable or attempt >= max_attempts:
                    break
                self._append_agent_event(
                    state,
                    "tool_retry",
                    {
                        "tool_use_id": tool_use_id,
                        "tool_name": public_action,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "error_kind": last_error_kind,
                        "error": str(exc),
                    },
                )
        error_text = str(last_error or "unknown tool failure")
        return {
            "route": "tool_error",
            "answer": f"工具 {public_action} 执行失败：{error_text}",
            "citations": [],
            "tool_calls": [
                {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "status": "error",
                    "error": error_text,
                    "error_kind": last_error_kind,
                    "retryable": retryable,
                    "attempts": max_attempts if retryable else 1,
                    "tool_use_id": tool_use_id,
                }
            ],
            "error_kind": last_error_kind,
            "retryable": retryable,
            "attempts": max_attempts if retryable else 1,
        }

    def _classify_tool_error(self, exc: Exception) -> ToolErrorKind:
        """按异常文本归类错误（决定是否重试 + 前端怎么展示）。"""
        if isinstance(exc, TimeoutError):
            return "timeout"
        text = str(exc).lower()
        if any(token in text for token in ["timeout", "timed out", "deadline"]):
            return "timeout"
        if any(token in text for token in ["rate limit", "429", "too many requests"]):
            return "rate_limited"
        if any(token in text for token in ["connection", "temporarily", "unavailable", "reset by peer", "503", "502", "504"]):
            return "transient_network"
        if any(token in text for token in ["permission", "forbidden", "unauthorized", "401", "403"]):
            return "permission_denied"
        if any(token in text for token in ["not found", "missing", "404"]):
            return "data_not_found"
        if any(token in text for token in ["validation", "schema", "json"]):
            return "model_output_invalid"
        return "tool_runtime_error"

    def _is_retriable_tool_error(self, error_kind: ToolErrorKind, public_action: str) -> bool:
        """只有超时/限流/网络抖动可重试；重活（工作流/报告/安全）不重试，避免重复副作用。"""
        if error_kind not in {"timeout", "rate_limited", "transient_network"}:
            return False
        if public_action in {"safety_agent.classify_action_risk", "report_agent.generate_weekly_report", "workflow.run_engineering_review"}:
            return False
        return True

    def _tool_model(self, tools: dict[str, StructuredTool]):
        """把工具 schema 绑定给模型（原生工具调用的关键一步）。"""
        if self.model is None:
            raise RuntimeError("ChatAgent requires an LLM for open-ended tool selection")
        return self.model.bind_tools(list(tools.values()))

    def _tool_calling_system_prompt(self) -> str:
        """系统提示词。内容稳定 → 可被上游 prompt 缓存命中，省费用。

        这里写的是"行为准则"，不是知识：
            怎么工作（先规划、调工具、看结果、修正）
            禁止什么（不许编造工具结果、不许把文档里的人当团队成员）
            什么时候必须用哪个工具（写操作先安全审查、复杂决策用工作流）
        """
        return (
            "你是 DevFlow AI 的对话控制器。这个稳定前缀会被缓存。"
            "请像长任务软件 Agent 一样工作：先规划，调用工具，读取 tool_result 观察结果，修正计划，"
            "持续推进直到用户请求真正完成，或步骤预算耗尽。"
            "当需要仓库、记忆、工作区代码或写操作安全上下文时，使用原生工具调用。"
            "绝不要编造工具结果。"
            "工程结论必须说明 WHY，并区分证据、推断和不确定项；证据不足时直接说缺什么，不要硬猜。"
            "当输出审查结论时，不要用 looks good、没什么问题这类空话代替检查；必须给出风险、级别和下一步。"
            "当回答交接、总结或工作流结论时，尽量保留 What、Why、Tradeoff、Open Questions、Next Action。"
            "当用户询问负责人、已分配人员、责任人或任务分配时，只能从明确的 DevFlow team_members "
            "或工具返回的现有 GitHub assignees 中推荐人员。仓库文档、Issue 文本、代码、示例、模型名、"
            "角色设定、团队比喻和记忆片段都不能当成人，除非它们同时出现在 DevFlow team_members 中。"
            "如果没有配置团队成员，请说明这一点，并要求用户先添加开发者后再分配任务。"
            "当用户请求 GitHub 写入、关闭、打标签、评论、发布、回复或发送时，必须先调用安全工具；"
            "除非存在单独的已确认写入工具，否则只返回草稿。"
            "对于合并就绪度、当前阻塞、优先级规划，或需要 Issue + PR + CI + RAG 综合判断的跨领域工程决策，"
            "优先使用 workflow.run_engineering_review 工具。"
        )

    def _tool_calling_user_prompt(self, state: ConversationState, *, aggressive: bool = False) -> str:
        """用户提示词 —— 这是一个"大拼盘"，把所有上下文按固定顺序拼成一个字符串。

        拼装顺序（重要）：
            使用规则 → 当前模型 → 可用工具 → 技能 → 团队 → 服务端上下文
            → 历史覆盖度 → 最近消息 → 已有观察结果 → 工具索引 → 用户原话

        aggressive=True 时（上下文溢出重试），把服务端上下文和历史压缩到 1/3 和 1/2。
        """
        prompt_observations, prompt_events, runtime_stats = self._runtime_tool_context(
            state,
            aggressive=aggressive,
        )
        state["runtime_context_stats"] = {
            **(state.get("runtime_context_stats") or {}),
            **runtime_stats,
        }
        context = state.get("context") or {}
        server_context = self._server_context(context)
        history = self._history_messages(context)
        if aggressive:
            budget = default_context_budget()
            server_context = clip_to_token_budget(server_context, max(512, budget.available_prompt_tokens // 3))
            history = self._compress_history_for_retry(history, max(512, budget.recent_tokens // 2))
        return (
            "上下文使用规则：如果服务端提供的最近消息已经包含回答所需的完整原文，直接基于这些消息回答，"
            "不要为了展示工具调用而重复读取记忆。只有目标消息不在最近窗口、内容被截断或压缩、用户明确要求"
            "核实原话，或者问题跨越会话时，才调用一个最匹配的 MCP 记忆工具。当前会话原始消息使用 "
            "devflow_get_thread_context；封存会话和跨会话摘要使用 devflow_read_session_events；项目长期记忆、"
            "历史决策和仓库证据使用 devflow_search_evidence。询问当前会话原话时，不要额外调用 "
            "devflow_search_evidence。\n\n"
            "硬性规则：如果用户询问这个项目/仓库做什么、技术栈、架构、启动方式、依赖或源码行为，"
            "回答前必须检查工作区文件。优先读取 README、package 或 manifest 文件；如果工作区文件不可用，要明确说明。\n\n"
            "涉及源码的问题，优先使用 workspace.search_code、workspace.read_file 或 workspace.list_files。"
            "涉及已索引的仓库历史和上传知识，使用 rag.search_similar_documents 或 devflow_search_evidence。\n\n"
            "混合 RAG 约定：服务端上下文中的 [Related Evidence] 是根据原始问题完成的调用前检索。"
            "如果这些证据不足、互相冲突，或问题包含多个子问题，请改写为更具体的 query 调用 "
            "rag.search_similar_documents；可以用不同 query 连续补查，并在最终回答前比较多次 tool_result。\n\n"
            "工作流硬性规则：如果用户请求多 Agent 审查、合并就绪度、当前阻塞、工程状态、跨 Agent 综合，"
            "或依赖 Issue、PR、CI 与仓库证据共同判断的决策，必须先调用 workflow.run_engineering_review，"
            "再给出最终建议。\n\n"
            "分配硬性规则：如果用户询问谁应该负责、处理、接手或被分配某个 Issue/PR/任务，"
            "可选人员只能来自下方列出的 DevFlow team_members，以及工具明确返回的现有 GitHub assignees。"
            "不要把仓库内容、项目角色设定、模型名、示例、作者或虚构标签当成团队成员。"
            "如果 team_members 为空且没有现有 assignee，请回答当前还没有添加项目成员，"
            "并要求用户在团队面板中添加开发者。可以说明可能需要的能力方向，但不要点名具体人员。\n\n"
            "循环约定：每次工具调用都会先记录为 tool_use，再记录为 tool_result。决定下一步前必须阅读之前的 tool_result 内容。"
            "如果目录列表里出现值得继续查看的 README、manifest、代码文件、Issue、PR 或 CI 记录，不要停在目录列表。"
            "除非新证据使重复调用有必要，不要用相同参数重复调用同一个工具。\n\n"
            f"当前 Agent 配置的运行模型：{settings.llm_model}\n\n"
            f"可用原生工具：\n{self._tool_help(state)}\n\n"
            f"已注册的 DevFlow 技能：\n{self._registered_skills_context(state.get('context') or {})}\n\n"
            f"本轮按需加载的 Skill 完整指令：\n{self._active_skill_prompt(state.get('context') or {})}\n\n"
            f"DevFlow 团队上下文：\n{self._team_assignment_context(state.get('context') or {})}\n\n"
            f"服务端上下文：\n{server_context}\n\n"
            f"上下文覆盖度：\n{json.dumps(self._history_coverage(context), ensure_ascii=False)}\n\n"
            f"最近消息：\n{json.dumps(history, ensure_ascii=False)}\n\n"
            f"已有观察结果（已按总预算投影，完整结果保存在会话转录）：\n{json.dumps(prompt_observations, ensure_ascii=False)}\n\n"
            f"工具调用/结果索引：\n{json.dumps(prompt_events, ensure_ascii=False)}\n\n"
            f"用户消息：{state.get('message') or ''}"
        )

    def _normalize_action_input(self, action: str, action_input: dict[str, Any], state: ConversationState) -> dict[str, Any]:
        """给工具参数补默认值（模型经常漏传 repo_id / query 这类必填项）。

        【为什么必须做】模型不一定每次都传全参数。
        比如调 rag.search_similar_documents 忘了传 query，工具会直接失败。
        这里用当前问题兜底，让工具至少能跑起来。
        """
        normalized = dict(action_input)
        message = str(state.get("message") or "")
        context = state.get("context") or {}

        # 大部分分析类工具都需要原始问题
        if action in {
            "repo_health.generate_summary",
            "rag.search_similar_documents",
            "issue_agent.analyze_issue",
            "pr_review_agent.analyze_pr",
            "ci_debug_agent.analyze_workflow_run",
            "report_agent.generate_weekly_report",
            "safety_agent.classify_action_risk",
            "workflow.run_engineering_review",
            "workspace.search_code",
        }:
            normalized.setdefault("message", message)
        if action == "workspace.search_code":
            normalized.setdefault("query", message)
        if action == "rag.search_similar_documents":
            normalized.setdefault("query", str(normalized.get("message") or message))
        # 问"项目是干什么的"但没指定文件 → 默认读 README
        if action == "workspace.read_file" and not normalized.get("path") and self._asks_project_overview(message):
            normalized["path"] = "README.md"
        if action == "devflow_search_evidence":
            normalized.setdefault("query", message)
        if action == "devflow_get_thread_context":
            normalized.setdefault("limit", 20)
        if action == "devflow_read_session_events":
            normalized.setdefault("limit", 5)
        # MCP 记忆工具强制注入 repo_id / conversation_id（防止跨仓库串数据）
        if action.startswith("devflow_") and context.get("repo_id"):
            normalized["repo_id"] = context["repo_id"]
            normalized.pop("repo_full_name", None)
        if action.startswith("devflow_") and context.get("conversation_id"):
            normalized["conversation_id"] = context["conversation_id"]
        return normalized

    def _asks_project_overview(self, message: str) -> bool:
        """判断是否在问"这个项目是干什么的"（这类问题走预读 README 的短路分支）。"""
        lowered = message.lower()
        # 先排除：提到 Issue/PR/CI 的肯定不是泛泛问项目
        if any(token in lowered for token in ["issue", "pull request", "pr #", "ci", "未关闭", "待处理"]):
            return False
        return any(
            token in lowered
            for token in [
                "what is this project",
                "what does this project",
                "project about",
                "repository about",
                "tech stack",
                "architecture",
                "setup",
                "项目介绍",
                "项目概览",
                "这个项目",
                "该项目",
                "这个仓库",
                "该仓库",
                "\u505a\u4ec0\u4e48",
                "\u67b6\u6784",
                "\u6280\u672f\u6808",
            ]
        )

    def _required_memory_tool(self, message: str, context: dict[str, Any] | None = None) -> str | None:
        """从记忆策略里取出"必须调用的工具名"。"""
        policy = self._memory_context_policy(message, context or {})
        return str((policy or {}).get("tool") or "") or None

    def _memory_context_policy(self, message: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """记忆策略：决定这轮要不要读记忆、读哪个。

        四种结果：
            None                        → 不需要记忆
            mode=required_tool          → 必须调指定工具
            mode=direct_recent_history  → 最近消息够答，不用调工具
        判定顺序：显式点名工具 > 项目历史 > 跨会话 > 近期回忆
        """
        lowered = message.lower()
        # 1. 用户直接点名了工具
        for tool_name in [
            "devflow_search_evidence",
            "devflow_get_thread_context",
            "devflow_read_session_events",
        ]:
            if tool_name in lowered:
                return {"mode": "required_tool", "tool": tool_name, "reason": "explicit_tool_request"}
        # 2. 问项目长期记忆/历史决策
        if any(token in lowered for token in ["历史证据", "历史决策", "项目记忆", "historical evidence", "past decision"]):
            return {"mode": "required_tool", "tool": "devflow_search_evidence", "reason": "project_history_request"}
        # 3. 问跨会话内容
        if any(
            token in lowered
            for token in [
                "封存会话",
                "会话摘要",
                "上次会话",
                "上个会话",
                "跨会话",
                "session digest",
                "session events",
                "previous session",
                "previous conversation",
            ]
        ):
            return {"mode": "required_tool", "tool": "devflow_read_session_events", "reason": "cross_session_request"}

        # 4. 问"刚才说了什么"这类近期回忆
        immediate_recall = any(
            token in lowered
            for token in [
                "刚刚说",
                "刚才说",
                "上一条消息",
                "上一句话",
                "上一句",
                "what did i just say",
                "my last message",
            ]
        )
        broad_recall = immediate_recall or any(
            token in lowered
            for token in [
                "之前说过",
                "上次讨论",
                "聊天历史",
                "会话历史",
                "你记得",
                "前面说过",
                "what did we say",
                "earlier in this chat",
            ]
        )
        if not broad_recall:
            return None

        # 要求逐字核实的，必须读原文，不能用摘要
        requires_exact_verification = any(
            token in lowered
            for token in ["原话", "逐字", "准确核实", "精确核实", "审计", "verbatim", "exact quote", "verify the record"]
        )
        if requires_exact_verification:
            return {"mode": "required_tool", "tool": "devflow_get_thread_context", "reason": "exact_verification_requested"}

        # 最近消息够答 → 不调工具，直接用历史回答（省一次 LLM + 工具调用）
        if self._recent_history_can_answer(message, context, immediate=immediate_recall):
            return {
                "mode": "direct_recent_history",
                "tool": None,
                "reason": "exact_recent_messages_available",
            }
        return {"mode": "required_tool", "tool": "devflow_get_thread_context", "reason": "recent_history_incomplete"}

    def _recent_history_can_answer(self, message: str, context: dict[str, Any], *, immediate: bool) -> bool:
        """判断"最近消息窗口"是否完整覆盖了用户要回忆的内容。"""
        history = self._history_messages(context)
        prior_user_index = self._last_prior_user_index(history, message)
        if prior_user_index is None:
            return False
        coverage = self._history_coverage(context)
        if not coverage:
            return True
        if immediate:
            # 问"上一条" → 只要精确后缀覆盖到那条就够
            exact_suffix = int(coverage.get("exact_suffix_messages") or 0)
            distance_from_end = len(history) - prior_user_index
            return exact_suffix >= distance_from_end
        return bool(coverage.get("complete_and_exact"))

    def _last_prior_user_index(self, history: list[dict[str, str]], message: str) -> int | None:
        """找"上一条用户消息"的下标（排除当前这条问题本身）。"""
        normalized_current = self._normalize_history_text(message)
        indexes = [index for index, item in enumerate(history) if item.get("role") == "user"]
        if indexes and self._normalize_history_text(history[indexes[-1]].get("content") or "") == normalized_current:
            indexes.pop()
        return indexes[-1] if indexes else None

    def _normalize_history_text(self, value: str) -> str:
        """归一化文本（去多余空白 + 转小写），用于比较两条消息是否相同。"""
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    async def _direct_answer_from_recent_history(self, state: ConversationState) -> str:
        """短路分支 1 的实现：只用最近消息让模型回答（不调任何工具）。"""
        if self.model is None:
            return self._local_recent_history_answer(state)
        history = self._history_messages(state.get("context") or {})
        try:
            system_prompt = (
                "请直接根据提供的近期原始消息回答，不要声称调用了任何工具。"
                "最后一条与当前问题相同的 user 消息是本轮问题，不要把它当成用户要回忆的上一条内容。"
                "只回答上下文能证明的内容；如果用户问上一条消息，优先简洁复述上一条 user 消息。"
            )
            user_prompt = (
                f"近期原始消息：{json.dumps(history, ensure_ascii=False)}\n"
                f"当前用户问题：{state.get('message') or ''}"
            )
            estimated = self._record_prompt_diagnostics(
                state,
                phase="direct_recent_history",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            self._record_model_usage(
                state,
                response,
                phase="direct_recent_history",
                estimated_prompt_tokens=estimated,
            )
            return self._message_content(response).strip() or self._local_recent_history_answer(state)
        except Exception:
            # LLM 失败就用本地兜底（直接复述上一条）
            return self._local_recent_history_answer(state)

    def _local_recent_history_answer(self, state: ConversationState) -> str:
        """不依赖 LLM 的兜底：直接从历史里摘出上一条用户消息。"""
        history = self._history_messages(state.get("context") or {})
        prior_user_index = self._last_prior_user_index(history, str(state.get("message") or ""))
        if prior_user_index is None:
            return "当前上下文里没有可确认的上一条用户消息。"
        content = history[prior_user_index].get("content") or ""
        return f"你刚刚说的是：{content}"

    def _history_coverage(self, context: dict[str, Any]) -> dict[str, Any]:
        """从压缩统计里取"历史覆盖度"（判断最近消息是否被截断过）。"""
        compression_stats = context.get("compression_stats") or {}
        if not isinstance(compression_stats, dict):
            return {}
        coverage = compression_stats.get("history_coverage") or {}
        return coverage if isinstance(coverage, dict) else {}

    async def _prefetch_memory_context(self, state: ConversationState, public_name: str) -> AsyncIterator[dict[str, Any]]:
        """短路分支 2 的实现：直接调指定的记忆工具（不让模型选）。"""
        tools = state.get("tools") or {}
        tool_names = state.get("tool_name_map") or {}
        safe_name = next((safe for safe, public in tool_names.items() if public == public_name), None)
        if not safe_name or safe_name not in tools:
            return
        message = str(state.get("message") or "")
        args: dict[str, Any]
        if public_name == "devflow_search_evidence":
            args = {"query": message, "limit": 5}
        elif public_name == "devflow_get_thread_context":
            args = {"keyword": "", "limit": 20}
        else:
            args = {"limit": 5}
        async for event in self._execute_tool_call_events(
            {"name": safe_name, "args": args, "id": f"required_{public_name}_{len(state.get('agent_events') or [])}"},
            state,
        ):
            yield event

    async def _prefetch_project_context(self, state: ConversationState) -> AsyncIterator[dict[str, Any]]:
        """短路分支 3 的实现：先列目录，再读 README / manifest。

        为什么不直接读 README：
            仓库不一定有 README.md（可能是 README.zh-CN.md、README.rst）。
            先列目录，从实际存在的文件里挑，命中率更高。
        """
        tools = state.get("tools") or {}
        tool_names = state.get("tool_name_map") or {}
        safe_by_public = {public: safe for safe, public in tool_names.items()}

        async def call(public_name: str, args: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
            safe_name = safe_by_public.get(public_name)
            if not safe_name or safe_name not in tools:
                return
            async for event in self._execute_tool_call_events(
                {"name": safe_name, "args": args, "id": f"prefetch_{public_name}_{len(state.get('agent_events') or [])}"},
                state,
            ):
                yield event

        async for event in call("workspace.list_files", {"path": ".", "limit": 120}):
            yield event
        # 从目录列表里解析出实际存在的文件名
        observed_paths: set[str] = set()
        for item in state.get("observations") or []:
            if not isinstance(item, dict) or not isinstance(item.get("observation"), dict):
                continue
            preview = str(item["observation"].get("answer_preview") or "")
            observed_paths.update(token.strip("'\"`[](),").lower() for token in preview.split())
        # 按优先级读第一个存在的说明文件
        for path in ["README.md", "README.zh-CN.md", "package.json", "SETUP.md"]:
            if path.lower() in observed_paths:
                async for event in call("workspace.read_file", {"path": path, "line_count": 220}):
                    yield event

    def _build_langchain_tools(self, raw_tools: dict[str, ConversationTool], context: dict[str, Any]) -> dict[str, StructuredTool]:
        """把 14 个原始工具函数包装成 LangChain 的 StructuredTool。

        两个要点：
            1. 工具名里的点号（workspace.read_file）不能直接给模型用，
               要转成下划线形式（_safe_tool_name），再用 tool_name_map 映射回来
            2. 用闭包把 context 捕获进去，工具函数就不需要模型传上下文
        """
        wrapped: dict[str, StructuredTool] = {}
        for name, func in raw_tools.items():
            async def call_tool(_func: ConversationTool = func, **kwargs: Any) -> dict[str, Any]:
                # 过滤 None 值：模型常把没传的参数填成 null
                input_data = {key: value for key, value in kwargs.items() if value is not None}
                return await _func(input_data, context)

            safe_name = self._safe_tool_name(name)
            call_tool.__name__ = safe_name
            wrapped[safe_name] = StructuredTool.from_function(
                coroutine=call_tool,
                name=safe_name,
                description=self._tool_description(name, context),
                args_schema=GenericToolArgs,     # 通用参数类，不逐个定义 schema
                infer_schema=False,
            )
        return wrapped

    def _tool_name_map(self, raw_tools: dict[str, ConversationTool]) -> dict[str, str]:
        """安全名 → 公开名的映射（日志和前端都显示公开名）。"""
        return {self._safe_tool_name(name): name for name in raw_tools}

    def _safe_tool_name(self, name: str) -> str:
        """把点号等非法字符转成双下划线（模型对工具名有字符限制）。"""
        return re.sub(r"[^a-zA-Z0-9_-]", "__", name)

    def _public_tool_name(self, safe_name: str, state: ConversationState) -> str:
        """安全名还原成公开名。"""
        return (state.get("tool_name_map") or {}).get(safe_name, safe_name)

    def _tool_description(self, name: str, context: dict[str, Any] | None = None) -> str:
        """工具描述 —— 这是"提示工程"的一部分。

        【为什么描述很重要】模型完全靠 description 判断"什么时候该调这个工具"。
        写得含糊，模型就会漏调或错调。
        """
        descriptions = {
            "devflow_search_evidence": "MCP 记忆工具。搜索持久记忆、会话摘要、仓库证据、Issue、PR、CI 和 Agent 运行记录。",
            "devflow_get_thread_context": "MCP 记忆工具。跨压缩边界读取当前仓库会话的完整原始聊天转录，支持消息 UUID 和字符偏移分页。",
            "devflow_read_session_events": "MCP 记忆工具。读取已封存的会话摘要、原始消息和工具事件。",
            "workspace.list_files": "列出所选仓库本地 checkout 下的文件。只读，用于探索源码树。",
            "workspace.read_file": "从所选仓库 checkout 中读取文本文件，可指定 start_line 和 line_count。只读。",
            "workspace.search_code": "在所选仓库 checkout 中搜索代码或文档文本。只读。",
            "repo_health.generate_summary": "总结当前仓库健康状态。",
            "rag.search_similar_documents": "搜索项目历史与非结构化知识，包括 Issue 描述、PR 说明与 Review 讨论、失败 CI 日志、项目文档、上传知识和已批准的长期记忆。可传入改写后的 query，并用不同 query 连续调用以补查和交叉验证。源码优先使用 workspace 工具，团队成员和实时状态使用对应结构化工具。",
            "issue_agent.analyze_issue": "分析最新同步的 Issue。",
            "pr_review_agent.analyze_pr": "分析最新同步的 PR。",
            "ci_debug_agent.analyze_workflow_run": "分析最新失败的 CI 运行。",
            "report_agent.generate_weekly_report": "生成仓库报告。",
            "safety_agent.classify_action_risk": "安全起草有风险的写操作或外部发送动作。",
            "workflow.run_engineering_review": "运行 Planner -> 专用 Agent -> Observer -> Synthesis 工作流，用于 PR 就绪度、阻塞项、优先级和多 Agent 仓库审查等跨领域工程决策。",
        }
        # 技能可以给工具追加描述（说明"这个工具在某个技能里怎么用"）
        suffixes = {}
        if context:
            raw_suffixes = context.get("skill_description_by_tool") or {}
            if isinstance(raw_suffixes, dict):
                suffixes = raw_suffixes
        return f"公开工具：{name}。{descriptions.get(name, 'DevFlow 工具。')}{suffixes.get(name, '')}"

    def _tool_help(self, state: ConversationState) -> str:
        """把工具清单格式化成 prompt 里的"可用工具"段。"""
        tools = state.get("tools") or {}
        return "\n".join(f"- {name}: {tool.description}" for name, tool in tools.items())

    async def _final_from_observations(self, state: ConversationState) -> str:
        """只基于已有观察结果生成最终答案（不调工具，用于短路分支收尾）。"""
        if self.model is None:
            return self._local_observation_answer(state)
        try:
            system_prompt = (
                "请使用用户的语言回答，并且只能基于观察结果、tool_result 记录和上下文。"
                "关键判断要说明 WHY；把证据、推断、不确定项和下一步区分开。"
                "如果是审查或合并建议，不能用 looks good、没什么问题一笔带过，必须保留风险和阻塞项。"
                "对于负责人、已分配人员、责任人或任务分配问题，只能从明确的 DevFlow team_members "
                "或工具返回的现有 GitHub assignees 中推荐人员。仓库文档、Issue 文本、代码、示例、"
                "角色设定、模型名和团队比喻都不能当成人。如果没有配置团队成员，也没有观察到现有 assignee，"
                "请说明当前还没有添加项目成员，并要求用户在团队面板中添加开发者。"
            )
            prompt_observations, prompt_events, runtime_stats = self._runtime_tool_context(state)
            state["runtime_context_stats"] = {**(state.get("runtime_context_stats") or {}), **runtime_stats}
            user_prompt = (
                f"用户消息：{state.get('message') or ''}\n"
                f"当前 Agent 配置的运行模型：{settings.llm_model}\n"
                f"DevFlow 团队上下文：{self._team_assignment_context(state.get('context') or {})}\n"
                f"上下文：{self._server_context(state.get('context') or {})}\n"
                f"观察结果：{json.dumps(prompt_observations, ensure_ascii=False)}\n"
                f"工具调用/结果索引：{json.dumps(prompt_events, ensure_ascii=False)}"
            )
            estimated = self._record_prompt_diagnostics(
                state,
                phase="final_from_observations",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            self._record_model_usage(
                state,
                response,
                phase="final_from_observations",
                estimated_prompt_tokens=estimated,
            )
            return self._message_content(response)
        except Exception:
            # LLM 失败 → 本地拼接兜底，保证永远有答案
            return self._local_observation_answer(state)

    def _local_observation_answer(self, state: ConversationState) -> str:
        """不依赖 LLM 的兜底答案：把观察结果拼成条目式回答。"""
        observations = state.get("observations") or []
        excerpts: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            value = item.get("answer") or item.get("content") or item.get("result") or item.get("message")
            if not value and isinstance(item.get("observation"), dict):
                nested = item["observation"]
                # 工具报错的观察结果不拼进答案（对用户没价值）
                if nested.get("route") == "tool_error":
                    continue
                value = nested.get("answer_preview") or nested.get("answer") or nested.get("content")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            text = " ".join(str(value or "").split())
            if text:
                excerpts.append(text[:600])
        if not excerpts:
            # 没有任何观察结果 → 退到预检索证据 → 再退到服务端上下文
            evidence_answer = self._local_prefetched_evidence_answer(state)
            if evidence_answer:
                return evidence_answer
            context_text = " ".join(self._server_context(state.get("context") or {}).split())
            if context_text:
                return "已读取当前仓库上下文。\n\n" + context_text[:1200]
            return "当前没有取得足够证据。请先确认仓库已连接并完成同步；开放式对话还需要配置 LLM。"
        return "已读取当前仓库证据。\n\n" + "\n\n".join(
            f"{index + 1}. {excerpt}" for index, excerpt in enumerate(excerpts[:4])
        )

    def _local_prefetched_evidence_answer(self, state: ConversationState) -> str:
        """用调用前预检索的证据拼答案（无 LLM 时的主要兜底）。

        针对"未关闭 Issue"这类高频问题做了专门格式化，比通用拼接更可读。
        """
        context = state.get("context") or {}
        raw_evidence = context.get("evidence")
        if not isinstance(raw_evidence, list):
            return ""
        evidence = [item for item in raw_evidence if isinstance(item, dict)]
        if not evidence:
            return ""

        message = str(state.get("message") or "").lower()
        asks_open_issues = "issue" in message and any(
            token in message for token in ["未关闭", "待处理", "open", "开放"]
        )
        # 专门处理"有哪些未关闭 Issue"：从证据的 metadata.state 里筛
        if asks_open_issues:
            open_issues = [
                item
                for item in evidence
                if str(item.get("source_type") or "") == "issue"
                and str((item.get("metadata") or {}).get("state") or "").lower() == "open"
            ]
            if open_issues:
                lines = ["根据本轮预检索证据，当前仓库有这些未关闭的 Issue："]
                for item in open_issues[:8]:
                    title = str(item.get("title") or "未命名 Issue").strip()
                    snippet = " ".join(str(item.get("snippet") or "").split())
                    lines.append(f"- {title}" + (f"：{snippet[:240]}" if snippet else ""))
                lines.append("\n数据来自当前仓库已同步的 Issue 证据；本轮没有调用工作区源码工具。")
                return "\n".join(lines)

        rendered = [self._render_evaluation_context(item) for item in evidence[:8]]
        rendered = [item for item in rendered if item]
        if not rendered:
            return ""
        return "根据当前仓库已同步的证据：\n\n" + "\n\n".join(
            f"{index + 1}. {item}" for index, item in enumerate(rendered)
        )

    def _prefetched_evidence_citations(self, state: ConversationState) -> list[dict[str, str]]:
        """把预检索证据转成引用列表（无 LLM 分支也要给引用，保持可追溯）。"""
        context = state.get("context") or {}
        evidence = context.get("evidence")
        if not isinstance(evidence, list):
            return []
        return [
            {
                "type": str(item.get("source_type") or "evidence"),
                "id": str(item.get("source_id") or item.get("id") or ""),
                "title": str(item.get("title") or "未命名证据"),
            }
            for item in evidence[:8]
            if isinstance(item, dict)
        ]

    def _append_agent_event(self, state: ConversationState, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """追加一条事件（前端时间线的原料）。"""
        event = {
            "type": event_type,
            "step": state.get("step", 0),
            **data,
        }
        state["agent_events"] = [*(state.get("agent_events") or []), event]
        return event

    def _tool_event_transcript(self, state: ConversationState) -> list[dict[str, Any]]:
        """只挑出与工具执行相关的事件（给 prompt 里的"工具索引"用）。"""
        return [
            event
            for event in state.get("agent_events") or []
            if event.get("type") in {"tool_use", "tool_result", "tool_retry", "agent_step", "agent_loop_stop"}
        ]

    def _runtime_tool_context(
        self,
        state: ConversationState,
        *,
        aggressive: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """把完整的工具观察结果"投影"成有 token 上限的 prompt 视图。

        【这是 ChatAgent 上下文管理的关键，务必理解】
            完整观察结果保留在 state 里（最终会被 SessionSealer 封存进转录）；
            只把"精简版"塞进 prompt。

            策略：从【最新】往前保留，超出预算的旧观察标 omitted。
            这样模型总能看到最近的工具结果，旧结果可以通过 session 工具重新读取。

        参数 aggressive=True 时预算降到 1/4（上下文溢出重试时用）。

        返回：（投影后的观察列表, 工具事件索引, 统计信息）
        """
        raw_observations = [item for item in state.get("observations") or [] if isinstance(item, dict)]
        budget = default_context_budget().tool_observation_tokens
        if aggressive:
            budget = max(512, budget // 4)
        # 每条观察的平均预算（160~1400 token 之间）
        per_item_budget = max(160, min(1400, budget // max(len(raw_observations), 1)))
        remaining = budget
        kept_reversed: list[dict[str, Any]] = []
        omitted = 0
        for item in reversed(raw_observations):     # 从最新往前遍历
            next_item = dict(item)
            observation = next_item.get("observation")
            if isinstance(observation, dict):
                next_observation = dict(observation)
                preview = str(
                    next_observation.get("answer_preview")
                    or next_observation.get("answer")
                    or next_observation.get("content")
                    or ""
                )
                if preview:
                    # 只保留 answer_preview，丢掉冗长的 answer/content 原文
                    next_observation["answer_preview"] = clip_to_token_budget(preview, per_item_budget)
                    next_observation.pop("answer", None)
                    next_observation.pop("content", None)
                next_item["observation"] = next_observation
            else:
                next_item["observation"] = clip_to_token_budget(str(observation or ""), per_item_budget)

            serialized = json.dumps(next_item, ensure_ascii=False, default=str)
            cost = estimate_tokens(serialized)
            if cost > remaining and kept_reversed:
                # 已经保留了更新的，这条就省略
                omitted += 1
                continue
            if cost > remaining:
                # 连一条都放不下 → 强行压到剩余预算里
                next_item = {
                    "tool_name": str(item.get("tool_name") or "tool"),
                    "observation": clip_to_token_budget(serialized, max(120, remaining)),
                    "compacted": True,
                }
                cost = estimate_tokens(json.dumps(next_item, ensure_ascii=False))
            kept_reversed.append(next_item)
            remaining = max(0, remaining - cost)
            if remaining <= 0:
                omitted += max(0, len(raw_observations) - len(kept_reversed))
                break

        prompt_observations = list(reversed(kept_reversed))
        if omitted:
            # 在开头插一条"有 N 条旧观察被省略，可从转录恢复"的提示
            prompt_observations.insert(
                0,
                {
                    "compacted": True,
                    "omitted_old_observations": omitted,
                    "recovery": "完整工具结果保存在本轮 ChatSession transcript，可按 session 读取。",
                },
            )

        # 工具事件索引：只保留关键字段，参数超长就截断
        prompt_events: list[dict[str, Any]] = []
        for event in self._tool_event_transcript(state):
            summary = {
                key: value
                for key, value in event.items()
                if key
                in {
                    "type",
                    "step",
                    "id",
                    "tool_use_id",
                    "tool_name",
                    "arguments",
                    "status",
                    "route",
                    "error_kind",
                    "retryable",
                    "reason",
                }
            }
            if "arguments" in summary:
                args_text = json.dumps(summary["arguments"], ensure_ascii=False, default=str)
                if estimate_tokens(args_text) > 240:
                    summary["arguments"] = clip_to_token_budget(args_text, 240)
            prompt_events.append(summary)

        original_tokens = estimate_tokens(json.dumps(raw_observations, ensure_ascii=False, default=str))
        prompt_tokens = estimate_tokens(json.dumps(prompt_observations, ensure_ascii=False, default=str))
        return prompt_observations, prompt_events, {
            "mode": "runtime_tool_projection",
            "aggressive": aggressive,
            "observation_budget_tokens": budget,
            "original_observation_tokens": original_tokens,
            "prompt_observation_tokens": prompt_tokens,
            "estimated_tokens_saved": max(0, original_tokens - prompt_tokens),
            "source_observations": len(raw_observations),
            "prompt_observations": len(prompt_observations),
            "omitted_observations": omitted,
            "tool_event_index_entries": len(prompt_events),
        }

    def _compress_history_for_retry(
        self,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """上下文溢出重试时压缩历史（从最新往前保留，单条最多 700 token）。"""
        kept: list[dict[str, str]] = []
        used = 0
        for item in reversed(history):
            content = clip_to_token_budget(item.get("content") or "", max(120, min(700, max_tokens // 3)))
            cost = estimate_tokens(item.get("role")) + estimate_tokens(content)
            if kept and used + cost > max_tokens:
                break
            kept.append({"role": item.get("role") or "user", "content": content})
            used += cost
        return list(reversed(kept))

    def _record_prompt_diagnostics(
        self,
        state: ConversationState,
        *,
        phase: str,
        system_prompt: str,
        user_prompt: str,
    ) -> int:
        """记录每次 prompt 的 token 构成和压力等级（用于诊断"为什么变慢/变贵"）。"""
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_prompt)
        tool_schema_tokens = self._tool_schema_estimated_tokens(state)
        total = system_tokens + user_tokens + tool_schema_tokens
        pressure = calculate_context_pressure(total)
        diagnostic = {
            "phase": phase,
            "estimated_prompt_tokens": total,
            "system_tokens": system_tokens,
            "user_context_tokens": user_tokens,
            "tool_schema_tokens": tool_schema_tokens,
            "pressure": pressure.as_dict(),
        }
        state["prompt_diagnostics"] = [*(state.get("prompt_diagnostics") or []), diagnostic]
        return total

    def _tool_schema_estimated_tokens(self, state: ConversationState) -> int:
        """估算 14 个工具 schema 占多少 token（这部分每次调用都要发给模型）。"""
        total = 0
        for tool in (state.get("tools") or {}).values():
            total += estimate_tokens(str(getattr(tool, "name", "")))
            total += estimate_tokens(str(getattr(tool, "description", "")))
            args_schema = getattr(tool, "args_schema", None)
            try:
                schema = args_schema.model_json_schema() if args_schema else {}
            except (AttributeError, TypeError, ValueError):
                schema = {}
            total += estimate_tokens(json.dumps(schema, ensure_ascii=False, default=str))
        return total

    def _record_model_usage(
        self,
        state: ConversationState,
        response: Any,
        *,
        phase: str,
        estimated_prompt_tokens: int,
    ) -> None:
        """记录本次 LLM 调用的 token 用量（成本核算）。"""
        usage = self._normalize_model_usage(response)
        usage.update(
            {
                "phase": phase,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                # 供应商没返回用量时，标记为估算值
                "source": "provider" if usage.get("input_tokens") is not None else "estimate",
            }
        )
        if usage.get("input_tokens") is None:
            usage["input_tokens"] = estimated_prompt_tokens
        if usage.get("total_tokens") is None:
            usage["total_tokens"] = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        state["model_usage"] = [*(state.get("model_usage") or []), usage]

    def _normalize_model_usage(self, response: Any) -> dict[str, Any]:
        """把不同供应商的用量字段统一成同一套键名。

        兼容三种来源：usage_metadata（LangChain 新版）/ response_metadata.token_usage（旧版）/ 都缺失。
        """
        raw = getattr(response, "usage_metadata", None)
        if raw is not None and not isinstance(raw, dict) and hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        raw = raw if isinstance(raw, dict) else {}
        response_metadata = getattr(response, "response_metadata", None)
        response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        token_usage = token_usage if isinstance(token_usage, dict) else {}
        input_details = raw.get("input_token_details") or {}
        input_details = input_details if isinstance(input_details, dict) else {}
        return {
            "input_tokens": raw.get("input_tokens", token_usage.get("prompt_tokens")),
            "output_tokens": raw.get("output_tokens", token_usage.get("completion_tokens")),
            "total_tokens": raw.get("total_tokens", token_usage.get("total_tokens")),
            "cache_read_input_tokens": input_details.get("cache_read", token_usage.get("cache_read_input_tokens", 0)),
            "cache_creation_input_tokens": input_details.get("cache_creation", token_usage.get("cache_creation_input_tokens", 0)),
            "model": response_metadata.get("model_name") or settings.llm_model,
        }

    def _runtime_metadata(self, state: ConversationState) -> dict[str, Any]:
        """运行期元信息（token 用量 + 上下文诊断），随 final 事件返回前端。"""
        return {
            "model_usage": state.get("model_usage") or [],
            "context_diagnostics": {
                "prompt_calls": state.get("prompt_diagnostics") or [],
                "runtime_tool_context": state.get("runtime_context_stats") or {},
                "reactive_overflow_retry": bool(state.get("context_overflow_recovery_attempted")),
            },
        }

    async def _fallback_answer(self, state: ConversationState) -> str:
        """兜底答案：基于已有观察结果再问一次模型（或本地拼接）。"""
        return await self._final_from_observations(state)

    def _history_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """从上下文里取历史消息（只保留 user/assistant 且内容非空的）。"""
        raw_messages = context.get("server_messages") or context.get("messages") or []
        if not isinstance(raw_messages, list):
            return []
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        return messages

    def _server_context(self, context: dict[str, Any]) -> str:
        """服务端上下文（由 ContextAssembler 在调用前拼好的系统信息 + 证据）。"""
        value = context.get("server_context")
        return str(value) if value else ""

    def _registered_skills_context(self, context: dict[str, Any]) -> str:
        """已注册技能的清单摘要（只给目录，不给完整正文，省 token）。"""
        value = context.get("registered_skills")
        return str(value) if value else "当前没有注册 DevFlow 技能。"

    def _activate_skills(self, message: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """激活技能：显式指定优先，否则按 triggers 关键词打分。"""
        registry = context.get("_skill_registry")
        if not isinstance(registry, SkillRegistry):
            return []
        requested_name = context.get("requested_skill") or context.get("skill_name")
        activations = registry.activate(
            message,
            requested_name=str(requested_name).strip() if requested_name else None,
        )
        return [activation.model_dump(mode="json") for activation in activations]

    def _active_skills(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """取本轮已激活的技能列表。"""
        skills = context.get("active_skills") or []
        return [item for item in skills if isinstance(item, dict)] if isinstance(skills, list) else []

    def _active_skill_prompt(self, context: dict[str, Any]) -> str:
        """把激活技能的完整指令拼进 prompt（上限 6000 token）。

        每个技能块包含：标题/版本、激活方式与原因、允许工具、输出契约、
        安全级别、指令正文、按需资源。
        """
        skills = self._active_skills(context)
        if not skills:
            return "本轮没有匹配到需要加载完整正文的 Skill；仅保留轻量目录摘要。"
        blocks = []
        for skill in skills:
            resources = skill.get("resources") if isinstance(skill.get("resources"), list) else []
            resource_sections = []
            for resource in resources:
                if not isinstance(resource, dict) or not str(resource.get("content") or "").strip():
                    continue
                resource_sections.append(f"### 资源：{resource.get('path')}\n{resource.get('content')}")
            resource_text = "\n\n按需读取的资源：\n" + "\n\n".join(resource_sections) if resource_sections else ""
            blocks.append(
                f"## {skill.get('skill_title')} ({skill.get('skill_name')} v{skill.get('skill_version')})\n"
                f"激活方式：{skill.get('activation_mode')}；原因：{skill.get('activation_reason')}\n"
                f"允许工具：{', '.join(str(item) for item in skill.get('tools') or []) or '无'}\n"
                f"输出契约：{skill.get('output_contract')}；安全级别：{skill.get('safety_level')}\n\n"
                f"{str(skill.get('instructions') or '').strip()}"
                f"{resource_text}"
            )
        return clip_to_token_budget("\n\n".join(blocks), 6000, suffix="\n[Skill 指令因上下文预算被截断]")

    def _explicit_skill_tool_violation(self, tool_name: str, state: ConversationState) -> str | None:
        """检查工具是否越界：显式技能只能用它声明的工具。

        【为什么需要】用户显式说"用 PR 审查技能"时，就不该让模型偷偷去调别的工具
        （比如去关 Issue）。这是权限边界，不是建议。
        """
        context = state.get("context") or {}
        explicit = [
            skill
            for skill in self._active_skills(context)
            if skill.get("activation_mode") == "explicit"
        ]
        if not explicit:
            return None
        allowed = {
            str(tool)
            for skill in explicit
            for tool in skill.get("tools") or []
        }
        if tool_name in allowed:
            return None
        skill_names = "、".join(str(skill.get("skill_name")) for skill in explicit)
        return f"工具 {tool_name} 不在显式 Skill（{skill_names}）声明的允许工具列表中，已拒绝执行。"

    def _record_skill_validation(self, state: ConversationState) -> None:
        """校验技能是否真的被执行了（不是"激活了但没用"）。

        四种结论：
            completed    → 入口工具成功执行，或走了委派任务
            failed       → 用了技能的工具但都失败了
            violated     → 用了白名单外的工具
            not_executed → 激活了但一个工具都没用
        """
        context = state.get("context") or {}
        active = self._active_skills(context)
        if not active:
            return
        observed_tools = [
            str(event.get("tool_name") or "")
            for event in state.get("agent_events") or []
            if isinstance(event, dict) and event.get("type") == "tool_use" and event.get("tool_name")
        ]
        successful_tools = [
            str(event.get("tool_name") or "")
            for event in state.get("agent_events") or []
            if isinstance(event, dict)
            and event.get("type") == "tool_result"
            and event.get("status") == "success"
            and event.get("tool_name")
        ]
        explicit_allowed = {
            str(tool)
            for skill in active
            if skill.get("activation_mode") == "explicit"
            for tool in skill.get("tools") or []
        }
        explicit_violations = [tool for tool in observed_tools if explicit_allowed and tool not in explicit_allowed]
        for skill in active:
            allowed = {str(tool) for tool in skill.get("tools") or []}
            used = [tool for tool in observed_tools if tool in allowed]
            entrypoint_called = str(skill.get("entrypoint") or "") in used
            successful_used = [tool for tool in successful_tools if tool in allowed]
            entrypoint_succeeded = str(skill.get("entrypoint") or "") in successful_used
            # 有些技能是通过工作流委派执行的，也要算"完成"
            delegated_tasks = [
                event
                for event in state.get("agent_events") or []
                if isinstance(event, dict)
                and event.get("type") == "workflow_task_result"
                and event.get("skill_name") == skill.get("skill_name")
                and event.get("status") == "success"
            ]
            if explicit_violations and skill.get("activation_mode") == "explicit":
                status = "violated"
            elif entrypoint_succeeded or successful_used or delegated_tasks:
                status = "completed"
            elif used:
                status = "failed"
            else:
                status = "not_executed"
            self._append_agent_event(
                state,
                "skill_validation",
                {
                    "skill_name": skill.get("skill_name"),
                    "skill_title": skill.get("skill_title"),
                    "skill_version": skill.get("skill_version"),
                    "status": status,
                    "instructions_loaded": bool(str(skill.get("instructions") or "").strip()),
                    "instruction_digest": skill.get("instruction_digest"),
                    "declared_steps": skill.get("workflow_steps") or [],
                    "observed_tools": used,
                    "entrypoint_called": entrypoint_called,
                    "entrypoint_succeeded": entrypoint_succeeded,
                    "delegated_task_ids": [str(event.get("task_id") or "") for event in delegated_tasks],
                    "tool_contract_violations": explicit_violations if skill.get("activation_mode") == "explicit" else [],
                    "step_verification": (
                        "entrypoint_completed"
                        if entrypoint_succeeded
                        else "delegated_entrypoint_completed"
                        if delegated_tasks
                        else "supporting_tools_observed"
                        if successful_used
                        else "skill_tool_failed"
                        if used
                        else "no_skill_tool_observed"
                    ),
                },
            )

    def _skill_event_metadata(self, tool_name: str, state: ConversationState) -> dict[str, Any]:
        """给工具事件附上"这个工具属于哪个技能"的追溯信息。"""
        context = state.get("context") or {}
        active = self._active_skills(context)
        # 优先级：工具就是技能入口 > 工具在技能白名单里 > 有显式技能
        selected = next((skill for skill in active if skill.get("entrypoint") == tool_name), None)
        if selected is None:
            selected = next((skill for skill in active if tool_name in (skill.get("tools") or [])), None)
        if selected is None:
            selected = next((skill for skill in active if skill.get("activation_mode") == "explicit"), None)
        if selected is not None:
            return {
                "skill_name": selected.get("skill_name"),
                "skill_title": selected.get("skill_title"),
                "skill_version": selected.get("skill_version"),
                "skill_steps": selected.get("workflow_steps") or [],
                "skill_activation_mode": selected.get("activation_mode"),
                "skill_activation_reason": selected.get("activation_reason"),
                "skill_instruction_digest": selected.get("instruction_digest"),
            }
        # 没激活技能时，看有没有预置的"工具→技能"映射
        mapping = context.get("skill_trace_by_tool") or {}
        if not isinstance(mapping, dict):
            return {}
        metadata = mapping.get(tool_name)
        if not isinstance(metadata, dict):
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key in {"skill_name", "skill_title", "skill_version", "skill_steps"}
        }

    def _team_assignment_context(self, context: dict[str, Any]) -> str:
        """团队成员上下文。空列表要显式说明，避免模型编造人名。"""
        members = self._team_members(context)
        if not members:
            return "team_members: []（当前仓库还没有配置 DevFlow 团队成员。）"
        return "team_members: " + json.dumps(members, ensure_ascii=False)

    def _team_members(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """清洗团队成员列表（最多 20 人，字段统一成驼峰给模型看）。"""
        raw_members = context.get("team_members") or []
        if not isinstance(raw_members, list):
            return []
        members: list[dict[str, str]] = []
        for member in raw_members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            members.append(
                {
                    "name": name,
                    "role": str(member.get("role") or "").strip(),
                    "strengths": str(member.get("strengths") or "").strip(),
                    "techStack": str(member.get("techStack") or member.get("tech_stack") or "").strip(),
                }
            )
        return members[:20]

    def _summarize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """把工具结果压缩成"预览 + 压缩统计"。

        为什么工作区工具给更大预算：
            读文件的结果是后续分析的主要原料，压太狠模型就看不到代码内容了。
        """
        answer = str(observation.get("answer") or "")
        budget = default_context_budget()
        limit = budget.tool_observation_tokens if str(observation.get("route") or "").startswith("workspace_") else max(300, budget.tool_observation_tokens // 3)
        compressed = clip_to_token_budget(answer, limit)
        return {
            "route": observation.get("route"),
            "answer_preview": compressed,
            "compression": {
                "mode": "tool_observation_budget",
                "original_estimated_tokens": estimate_tokens(answer),
                "compressed_estimated_tokens": estimate_tokens(compressed),
                "budget_tokens": limit,
            },
            "citations_count": len(observation.get("citations", [])),
        }

    def _initial_evaluation_trace(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        """评测模式：初始化证据轨迹（把调用前预检索的证据记成 prefetch 阶段）。"""
        evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
        contexts: list[str] = []
        sources: list[dict[str, Any]] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            rendered_context = self._render_evaluation_context(item)
            if rendered_context:
                contexts.append(rendered_context)
            sources.append(
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                }
            )
        retrievals = []
        if contexts or sources:
            retrievals.append(
                {
                    "stage": "prefetch",
                    "query": message,
                    "contexts": contexts,
                    "sources": sources,
                }
            )
        return {
            "target": "ChatAgent",
            "retrievals": retrievals,
            "tool_observations": [],
        }

    def _capture_evaluation_observation(
        self,
        state: ConversationState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        """评测模式：记录每次工具调用及其检索证据（供 RAGAS 算指标）。"""
        if not (state.get("context") or {}).get("evaluation_mode"):
            return
        trace = dict(state.get("evaluation_trace") or {})
        tool_observations = list(trace.get("tool_observations") or [])
        tool_observations.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "status": "error"
                if any(
                    call.get("status") == "error"
                    for call in observation.get("tool_calls", [])
                    if isinstance(call, dict)
                )
                else "success",
                "content": str(observation.get("answer") or ""),
            }
        )
        trace["tool_observations"] = tool_observations

        # 记录检索证据（Context Precision / Recall 的输入）
        raw_sources = observation.get("retrieval_results")
        sources = [dict(item) for item in raw_sources if isinstance(item, dict)] if isinstance(raw_sources, list) else []
        contexts = [self._render_evaluation_context(item) for item in sources]
        contexts = [item for item in contexts if item]
        if not contexts:
            contexts = [
                str(item).strip()
                for item in observation.get("retrieved_contexts", [])
                if str(item).strip()
            ]
        if tool_name == "rag.search_similar_documents" or contexts or sources:
            retrievals = list(trace.get("retrievals") or [])
            retrievals.append(
                {
                    "stage": "agent_tool",
                    "query": str(arguments.get("query") or arguments.get("message") or ""),
                    "contexts": contexts,
                    "sources": sources,
                }
            )
            trace["retrievals"] = retrievals
        state["evaluation_trace"] = trace

    @staticmethod
    def _render_evaluation_context(item: dict[str, Any]) -> str:
        """把一条证据渲染成 "[类型] 标题: 片段" 的文本（评测和本地答案共用）。"""
        source_type = str(item.get("source_type") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        heading = " ".join(part for part in ([f"[{source_type}]" if source_type else "", title]) if part)
        if heading and snippet:
            return f"{heading}: {snippet}"
        return heading or snippet

    def _evaluation_result(self, state: ConversationState) -> dict[str, Any]:
        """评测模式：汇总所有检索到的上下文（去重后）随 final 返回。"""
        if not (state.get("context") or {}).get("evaluation_mode"):
            return {}
        trace = dict(state.get("evaluation_trace") or {})
        contexts: list[str] = []
        seen: set[str] = set()
        for retrieval in trace.get("retrievals") or []:
            if not isinstance(retrieval, dict):
                continue
            for item in retrieval.get("contexts") or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    contexts.append(text)
        trace["retrieved_contexts"] = contexts
        trace["agent_events"] = self._tool_event_transcript(state)
        return {"evaluation_trace": trace}

    def _message_content(self, message: BaseMessage) -> str:
        """从 LangChain 消息里取出纯文本（兼容 str 和 list 两种 content）。"""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "".join(parts)
        return str(content or "")

    async def _stream_answer(self, answer: str) -> AsyncIterator[dict[str, Any]]:
        """把最终答案切成 delta 事件流。

        【注意】这不是真正的流式生成 —— 模型调用是 ainvoke（一次性拿完整结果），
        这里只是按 36 字符切片，给前端做打字机效果。
        原因：Agent 循环必须先拿到完整结果才能判断要不要继续调工具。
        """
        for part in self._chunk_text(answer):
            yield self._event("delta", {"content": part})

    def _chunk_text(self, text: str, size: int = 36) -> list[str]:
        """按固定字符数切片。"""
        if not text:
            return []
        return [text[index : index + size] for index in range(0, len(text), size)]

    def _event(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        """构造一个内部事件对象。"""
        return {"event": event, "data": data}


def encode_sse(event: dict[str, Any]) -> str:
    """把事件编码成 SSE 格式（event: xxx + data: json + 空行）。

    前端 apiPostStream 按 "\\n\\n" 切分事件块解析。
    """
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
