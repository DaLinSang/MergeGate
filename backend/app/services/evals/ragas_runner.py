"""
RAG 质量评测（RAGAS 指标 + 确定性硬规则）
======================================

【业务定位】
    用真实的 ChatAgent 跑评测集，输出两类判断：
        ① **语义指标**（RAGAS）：context_precision / context_recall / faithfulness /
           answer_relevancy / agent_goal_accuracy / tool_call_accuracy / tool_call_f1
        ② **硬规则**（确定性断言）：来源是否命中、必需工具是否调用、必须/禁止短语、
           正则格式。这一层不依赖 LLM，永远可复现。

    【★ 为什么必须两层都有】
        只看 RAGAS：分数会因 judge 模型波动而飘，且"错了但分数还行"的情况常见。
        只看硬规则：只能测"有没有"，测不出"答得好不好"。
        两层叠加 = 可复现的底线 + 有信息量的质量分。

【数据流向】
    上游（谁调它）：
        api/routes/evaluations.py         评测接口（带 progress_callback 做 SSE）
        services/evals/runner.py          基础评测里的可选一步
        services/evals/suites.py          提供 RagEvalCase 列表
    下游（它调谁）：
        ChatAgent                     ★ 跑真实的 Agent 链路（不是 mock）
        chat_memory.ContextAssembler  组装上下文
        chat_memory.EvidenceStore     重建证据库
        api/routes/chat._build_conversation_tools  复用生产的工具集
        rag.embeddings.embed_text     RAGAS 的 answer_relevancy 需要向量
        Ragas 0.4 Collections API     judge 打分

【核心规则】
    1. ★ **每条用例跑在独立的 Conversation 里，跑完 rollback**
       （评测不能污染用户的真实会话）
    2. ★ **judge 不可用不算失败，算"部分完成"**（status=partial）
       —— 硬规则仍然有效，不该因为没配 judge key 就整个评测报错
    3. ★ **失败要能定位到"哪一层坏了"**（诊断分类：检索 / 生成 / judge / agent）
    4. 长答案的 faithfulness 要分块打分（否则 judge 的结构化输出预算会被耗尽）

【本文件函数地图】

    ┌─ 一、常量与协议 ─────────────────────────────────
    │  CORE_RAGAS_METRICS / REQUIRED_RAGAS_METRICS / AGENT_RAGAS_METRICS
    │  RagEvalCase           ★ 一条评测用例
    │  SemanticScorer        Protocol（打分器接口，便于测试替换）
    │  ProgressCallback      进度回调类型
    │  _emit_progress()      发进度（同步/异步回调都支持）
    │
    ├─ 二、RAGAS 打分器 ───────────────────────────────
    │  _ensure_ragas_compatibility()  ★ 打补丁绕过 Ragas 的导入 bug
    │  RagasCollectionsScorer         ★ 核心类
    │      ├ _messages()                  转成 Ragas 的消息格式
    │      ├ _score_faithfulness()        ★ 分块打分 + 加权平均
    │      └ score()                     并发跑多个指标
    │
    ├─ 三、用例构造 ───────────────────────────────────
    │  _default_cases()      ★ 从文档/Issue/PR/CI 自动出题
    │
    ├─ 四、检索指标（纯数学，无 LLM）────────────────────
    │  _source_hit() / _dcg() / _retrieval_metrics()   recall/precision/MRR/NDCG
    │
    ├─ 五、硬规则与上下文 ─────────────────────────────
    │  _trace_rows()            从 trace 抽上下文和来源
    │  _hard_rules()            ★ 八项确定性断言
    │  _evaluation_project_context()  取团队/记忆（和生产一致）
    │  _run_chatagent_case()    ★ 隔离跑一条用例
    │
    ├─ 六、汇总与诊断 ─────────────────────────────────
    │  _thresholds() / _mean() / _metric_details()
    │  _case_diagnoses()        ★ 六类诊断（告诉用户"该修哪一层"）
    │  _diagnostic_summary()
    │  _finalize_result()       ★ 判定 status / quality_status
    │
    └─ 七、入口 ───────────────────────────────────────
       run_rag_quality_eval()    ★ 完整评测
       rescore_rag_result()      ★ 只重跑打分（不重跑 Agent）
"""

import asyncio
import copy
import importlib.metadata
import inspect
import math
import re
import sys
import time
import types
import uuid
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Conversation, Document, Issue, PullRequest, Repository, WorkflowRun
from app.services.agents.chat_agent import ChatAgent
from app.services.chat_memory import ContextAssembler, EvidenceStore
from app.services.rag.embeddings import embed_text
from app.services.rag.source_policy import RAG_SOURCE_TYPES


# ════════════════════════════════════════════════════════════════
# 一、指标清单（三层，用途不同）
# ════════════════════════════════════════════════════════════════

# 核心四指标：任何 RAG 评测都必须有这四个
CORE_RAGAS_METRICS = (
    "context_precision",     # 检索到的上下文里有多少是真正有用的（噪声比例）
    "context_recall",        # 该找到的证据有没有找全（漏召回）
    "faithfulness",          # 回答是否忠于证据（有没有编）
    "answer_relevancy",      # 回答是否切题（有没有跑偏）
)

# 必需指标：核心四 + Agent 目标达成
# 为什么把 agent_goal_accuracy 也算"必需"：
#   本项目的评测对象是 ChatAgent（不是纯 RAG），
#   它能答对但没遵守任务约束（如"用三句话"）也算失败。
REQUIRED_RAGAS_METRICS = (*CORE_RAGAS_METRICS, "agent_goal_accuracy")

# Agent 专属指标（需要 expected_tool_calls 才评）
AGENT_RAGAS_METRICS = (
    "agent_goal_accuracy",
    "tool_call_accuracy",    # 工具调用是否和预期一致
    "tool_call_f1",          # 工具调用的 F1（兼顾漏调和多调）
)

# faithfulness 分块大小（字符）
# 为什么是 700：judge 要先"抽取所有断言"再判断是否被证据支持。
# 答案太长会耗尽它的结构化输出预算，导致指标失败。
_FAITHFULNESS_CHUNK_CHARS = 700


def _split_faithfulness_response(text: str, max_chars: int = _FAITHFULNESS_CHUNK_CHARS) -> list[str]:
    """把长答案切成块，用于分块算 faithfulness（★ 不丢内容）。

    【调用链】被 RagasCollectionsScorer._score_faithfulness() 调用

    【数据流向】
        入：text（回答全文）、max_chars（每块上限）
        出：块列表（拼起来等于原文）

    Split a long answer without dropping content before native Ragas scoring.

    Faithfulness first asks the judge to extract all claims. Some OpenAI-compatible
    reasoning models can exhaust their structured-output budget even for a moderate
    answer. Scoring smaller semantic chunks keeps that native metric reliable; the
    final score is weighted by the amount of answer text in each chunk.

    【★ 三步切分，保证不丢内容】
        ① 先按句子切（句号/问号/感叹号/换行）
        ② 单个句子超长 → 硬切（while len(unit) > max_chars）
        ③ 把句子攒成块，直到接近 max_chars

    【★ 为什么第 ② 步要先 flush 当前块】
        如果 current 里已经攒了几句，这时来了一个超长句，
        必须先把 current 输出（否则顺序会乱），再单独处理超长句。

    【兜底：切完为空就返回 [原文]】
        极端情况（全是空白）下不要返回空列表 ——
        空列表会让下游 gather() 得到空结果，加权平均时分母为 0。
    """
    normalized = text.strip()
    if not normalized:
        return [""]

    # ① 按句子切
    units = [unit.strip() for unit in re.split(r"(?<=[。！？.!?])\s+|\n+", normalized) if unit.strip()]

    chunks: list[str] = []
    current = ""
    for unit in units:
        # ② 单句超长 → 先把 current 输出，再硬切这一句
        while len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:max_chars])
            unit = unit[max_chars:]

        # ③ 攒句子
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks or [normalized]


class RagasUnavailableError(RuntimeError):
    """Ragas 本身 / judge 模型 / 评测用 embedding 不可用。

    【为什么单独定义】
        上层要区分"评测失败"和"评测环境没配好"：
            前者要修代码/修 prompt，后者只要配个 key。
        有独立异常类型，route 层就能给出不同的提示。
    """
    pass


@dataclass
class RagEvalCase:
    """一条评测用例（★ 题库的最小单位）。

    【字段分两组】
        必填（question/reference）：没有这两项没法评
        可选（断言类）：不填就不检查对应项

    【断言类字段的用途】
        expected_source_ids  检索是否命中了预期来源（recall/precision 的基础）
        required_tools       必须调用哪些工具（漏调就是失败）
        expected_tool_calls  工具调用的标准答案（算 tool_call_accuracy/f1）
        must_include         回答里必须出现的短语（确定性断言）
        must_not_include     回答里禁止出现的短语（防幻觉/防泄露）
        answer_regex         回答必须匹配的格式（如"恰好三句话"）
        kind                 positive（该答）/ negative（该拒答）
    """

    question: str
    reference: str
    expected_source_ids: list[str]
    source_type: str | None = None
    required_tools: list[str] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "positive"
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    answer_regex: str | None = None
    source_file: str | None = None
    id: str = ""


class SemanticScorer(Protocol):
    """打分器接口（★ 用 Protocol 而不是基类，便于测试注入假实现）。

    【为什么需要这个抽象】
        真实打分器依赖 ragas + judge API（重、慢、要钱）。
        测试时注入一个"固定返回 0.9"的假打分器，
        就能单测"汇总/诊断/判定"这些纯逻辑，不必真的调模型。
    """

    version: str
    judge_model: str

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
        expected_tool_calls: list[dict[str, Any]],
        metric_names: set[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]: ...


# 进度回调：可以是同步函数，也可以是 async 函数
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


async def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    """发一条进度事件（★ 同时支持同步和异步回调）。

    【调用链】被 run_rag_quality_eval() / rescore_rag_result() 调用（多处）

    【★ 为什么用 inspect.isawaitable 判断】
        route 层传进来的可能是：
            同步函数（写日志）
            async 函数（往 SSE 队列里 put）
        统一在这里兼容，调用方不用关心自己是哪种。

    【callback 为 None 时直接返回】
        不传回调就是"不需要进度"，不必特殊处理。
    """
    if callback is None:
        return
    pending = callback(payload)
    # 如果回调返回了 awaitable，就 await 它
    if inspect.isawaitable(pending):
        await pending


# ════════════════════════════════════════════════════════════════
# 二、RAGAS 打分器
# ════════════════════════════════════════════════════════════════

def _ensure_ragas_compatibility() -> None:
    """打补丁绕过 Ragas 0.4.3 与 langchain-community 0.4 的导入冲突（★ 值得单独看）。

    【调用链】被 RagasCollectionsScorer.__init__() 调用

    【问题是什么】
        ragas 0.4.3 里有一句 `from langchain_community.chat_models.vertexai import ChatVertexAI`。
        但 langchain-community 0.4 删掉了这个模块 → ImportError。
        而 ragas 只是用它做一次 isinstance 检查，本项目根本不用 VertexAI。

    【解法：造一个假模块塞进 sys.modules】
        如果真实模块不存在，就创建一个只有 ChatVertexAI 空类的假模块。
        这样 ragas 的 import 能通过，isinstance 检查也永远返回 False（正确）。

    【★ 为什么不降级 langchain】
        降级会影响 langgraph / langchain-core 的版本链，风险大。
        一个 20 行的 shim 比动依赖树安全得多。

    【★ 三个细节】
        ① 先检查 `module_name in sys.modules` → 已经打过补丁就跳过（幂等）
        ② 用 warnings.catch_warnings 屏蔽 langchain 的 sunset 弃用警告
           （不屏蔽的话日志会被刷屏）
        ③ `if exc.name != module_name: raise` → 只处理"这个模块不存在"，
           其他 ImportError（比如它依赖的东西缺失）要原样抛出，不能掩盖真问题
    """
    module_name = "langchain_community.chat_models.vertexai"
    # ① 幂等：已经打过补丁
    if module_name in sys.modules:
        return

    try:
        with warnings.catch_warnings():
            # ② 屏蔽 langchain-community 的 sunset 警告
            warnings.filterwarnings(
                "ignore",
                message="`langchain-community` is being sunset.*",
                category=DeprecationWarning,
            )
            __import__(module_name)
    except ModuleNotFoundError as exc:
        # ③ 只处理"目标模块本身不存在"，其他 ImportError 继续抛
        if exc.name != module_name:
            raise
        # 造假模块：只要有一个 ChatVertexAI 类名就够（ragas 只用它做 isinstance）
        shim = types.ModuleType(module_name)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = shim


class RagasCollectionsScorer:
    """RAGAS 0.4 Collections API 的适配器（★ 把项目的模型客户端接进去）。

    【为什么需要"适配器"】
        RAGAS 要求传入 llm 和 embeddings 对象。
        本项目的 embedding 是自定义的（Qwen 本地 / OpenAI 兼容 / 哈希兜底），
        不是 RAGAS 内置的那几种。
        所以用 llm_factory + 自定义 BaseRagasEmbedding 桥接。
    """

    def __init__(self) -> None:
        _ensure_ragas_compatibility()
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AgentGoalAccuracyWithReference,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                ToolCallAccuracy,
                ToolCallF1,
            )
        except Exception as exc:  # pragma: no cover - depends on optional packages
            # ragas 是可选依赖（requirements-eval.txt）。没装就明确报"不可用"
            raise RagasUnavailableError(f"Unable to import Ragas 0.4 Collections API: {exc}") from exc

        class ProjectEmbedding(BaseRagasEmbedding):
            """把项目的 embed_text 桥接成 RAGAS 的 embedding 接口。

            【为什么同步和异步都要实现】
                RAGAS 内部有的路径调 embed_text，有的调 aembed_text。
                同步版直接调（会阻塞），异步版用 to_thread 包一层。
            """

            def embed_text(self, text: str, **_: Any) -> list[float]:
                return embed_text(text)

            async def aembed_text(self, text: str, **_: Any) -> list[float]:
                # embed_text 是同步阻塞的（可能走本地模型），丢线程池避免卡事件循环
                return await asyncio.to_thread(embed_text, text)

        # judge 的三要素：key / base_url / model（都有回退到通用 LLM 配置）
        api_key = settings.ragas_judge_api_key or settings.llm_api_key
        base_url = (settings.ragas_judge_base_url or settings.llm_base_url).rstrip("/")
        model = settings.ragas_judge_model or settings.llm_model

        if not api_key:
            raise RagasUnavailableError("RAGAS judge API key is not configured")
        if not base_url or not model:
            raise RagasUnavailableError("RAGAS judge base URL or model is not configured")

        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=settings.ragas_timeout_seconds)
            llm = llm_factory(
                model,
                client=client,
                # ★ temperature=0：评测必须可复现，不能有随机性
                temperature=0,
                max_tokens=settings.ragas_judge_max_tokens,
            )
            embedding = ProjectEmbedding()
            self.metrics = {
                "context_precision": ContextPrecision(llm=llm),
                "context_recall": ContextRecall(llm=llm),
                "faithfulness": Faithfulness(llm=llm),
                "answer_relevancy": AnswerRelevancy(
                    llm=llm,
                    embeddings=embedding,
                    # strictness：生成几个问题来测相关性，取平均。越大越稳但越慢
                    strictness=max(1, settings.ragas_answer_relevancy_strictness),
                ),
                "agent_goal_accuracy": AgentGoalAccuracyWithReference(llm=llm),
                # 工具指标不需要 llm（纯集合比较）
                # strict_order=False：不要求调用顺序一致（只比"调了哪些"）
                "tool_call_accuracy": ToolCallAccuracy(strict_order=False),
                "tool_call_f1": ToolCallF1(),
            }
            self.message_types = self._message_types()
            # 记录版本（报告里要写"用哪个版本的 ragas 评的"）
            self.version = importlib.metadata.version("ragas")
            self.judge_model = model
        except Exception as exc:
            raise RagasUnavailableError(f"Unable to initialize Ragas judge: {exc}") from exc

    @staticmethod
    def _message_types() -> tuple[Any, Any, Any, Any]:
        """延迟导入 Ragas 的消息类型（避免顶层 import 拖慢启动）。

        【调用链】被 __init__() 调用
        """
        from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

        return HumanMessage, AIMessage, ToolMessage, ToolCall

    def _messages(
        self,
        user_input: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
    ) -> list[Any]:
        """把一次对话转成 Ragas 的消息序列（★ agent_goal_accuracy 需要）。

        【调用链】被 score() 调用

        【数据流向】
            入：user_input、response、tool_calls、tool_observations
            出：[HumanMessage, AIMessage(tool_calls), ToolMessage..., AIMessage(response)]

        【★ 为什么要拼成"完整的 Agent 轨迹"而不是只有问答】
            agent_goal_accuracy 评的是"Agent 有没有完成任务"，
            它需要看到"调了哪些工具、工具返回了什么"。
            只给 question + answer，它没法判断过程是否合理。

        【★ 为什么 AIMessage(content="") 带 tool_calls】
            Ragas 的约定：工具调用挂在 assistant 消息上，
            工具结果作为独立的 ToolMessage 跟在后面。
        """
        HumanMessage, AIMessage, ToolMessage, ToolCall = self.message_types

        messages: list[Any] = [HumanMessage(content=user_input)]

        # 工具调用 → Ragas 的 ToolCall 对象
        ragas_calls = [
            ToolCall(
                name=str(item.get("tool_name") or item.get("name") or "unknown"),
                args=dict(item.get("arguments") or item.get("args") or {}),
            )
            for item in tool_calls
            if isinstance(item, dict)
        ]
        if ragas_calls:
            # content 为空：这条 assistant 消息只承载工具调用，没有文本
            messages.append(AIMessage(content="", tool_calls=ragas_calls))

        # 工具观察结果
        for item in tool_observations:
            if isinstance(item, dict):
                messages.append(ToolMessage(content=str(item.get("content") or "")))

        # 最终回答
        messages.append(AIMessage(content=response))
        return messages

    async def _score_faithfulness(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> float:
        """分块算 faithfulness，再按字符数加权平均（★ 本文件最精巧的一段）。

        【调用链】_split_faithfulness_response() → 本函数
        【被谁调用】score()

        【数据流向】
            入：user_input、response（完整回答）、retrieved_contexts
            出：加权平均分（0~1）

        【★ 为什么要分块】
            faithfulness 的实现是"先抽取回答里的所有断言，再逐条判断是否被证据支持"。
            答案太长时，judge 的结构化输出预算会被耗尽 → 指标直接失败。
            分块后每块都短，判官能稳定输出。

        【★ 为什么加权而不是简单平均】
            一块 600 字的段落和一块 20 字的句子，前者代表更多内容。
            按字符数加权，才反映"整篇答案的忠实度"。
            weights = max(1, len(chunk)) —— 至少 1，防止空块权重为 0 导致分母异常。

        【★ 为什么用 Semaphore(2) 限并发】
            faithfulness 要调 judge 很多次（每块至少 2 次 LLM 调用）。
            不限并发会瞬间打出几十个请求 → 触发限流或超时。
            2 个并发是"不浪费太多时间 + 不打爆 API"的折中。

        【★ 为什么用 return_exceptions=True 然后统一抛】
            先让所有块都跑完（而不是第一个失败就中断），
            然后统计"几块失败了"，错误信息里带上失败比例 ——
            这样能区分"偶发失败"（1/10）和"整体不可用"（10/10）。
        """
        chunks = _split_faithfulness_response(response)
        # 限并发 2
        semaphore = asyncio.Semaphore(2)

        async def score_chunk(chunk: str) -> Any:
            async with semaphore:
                return await asyncio.wait_for(
                    self.metrics["faithfulness"].ascore(
                        user_input=user_input,
                        response=chunk,
                        retrieved_contexts=retrieved_contexts,
                    ),
                    # 单块超时：judge 卡住时不能让整个评测挂住
                    timeout=settings.ragas_timeout_seconds,
                )

        results = await asyncio.gather(*(score_chunk(chunk) for chunk in chunks), return_exceptions=True)

        # 有任何一块失败 → 整体失败（但要报告失败比例）
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            first = failures[0]
            raise RagasUnavailableError(
                f"Faithfulness chunk scoring failed ({len(failures)}/{len(chunks)}): "
                f"{type(first).__name__}: {first}"
            ) from first

        # 按字符数加权平均
        weights = [max(1, len(chunk)) for chunk in chunks]
        return sum(float(result.value) * weight for result, weight in zip(results, weights, strict=True)) / sum(weights)

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
        expected_tool_calls: list[dict[str, Any]],
        metric_names: set[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """并发跑多个 RAGAS 指标（★ 只跑请求的那些）。

        【调用链】_messages() / _score_faithfulness() → 本函数
        【被谁调用】run_rag_quality_eval() / rescore_rag_result()

        【数据流向】
            入：问题/回答/上下文/参考答案/工具轨迹/期望工具调用/metric_names（可选，只跑这些）
            出：(scores, errors) —— 成功的分数 + 失败的原因

        【★ 为什么允许只跑部分指标（metric_names）】
            rescore 场景：只有 2 个指标失败了，重跑全部 7 个既慢又浪费钱。
            指定 metric_names 就只跑那几个。

        【★ 为什么所有指标并发（asyncio.gather）】
            7 个指标互相独立，串行跑要 7 倍时间。
            并发 + 单指标超时 + return_exceptions，既快又稳。

        【★ 为什么 tool_call_* 需要 expected_tool_calls 才跑】
            没有标准答案就没法算准确率。
            条件：`expected_tool_calls and (wants(accuracy) or wants(f1))`

        【★ 为什么 result 可能是 float 也可能是对象】
            RAGAS 不同版本/不同指标返回不同：有的返回 MetricResult（有 .value），
            有的直接返回 float。所以 `result if isinstance(result, (int, float)) else result.value`。

        【★ 单个指标失败不影响其他】
            失败记进 errors，成功的记进 scores。
            上层会根据"必需指标是否都有分数"来判断整体是否可用。
        """
        messages = self._messages(user_input, response, tool_calls, tool_observations)
        # metric_names 为 None 表示"全跑"
        wants = lambda name: metric_names is None or name in metric_names

        # ── 组装"要跑哪些指标"（还没 await）──
        calls: dict[str, Any] = {}

        if wants("context_precision"):
            calls["context_precision"] = self.metrics["context_precision"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("context_recall"):
            calls["context_recall"] = self.metrics["context_recall"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("faithfulness"):
            # 走分块逻辑（不用内置的直接打分）
            calls["faithfulness"] = self._score_faithfulness(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("answer_relevancy"):
            calls["answer_relevancy"] = self.metrics["answer_relevancy"].ascore(
                user_input=user_input,
                response=response,
            )
        if wants("agent_goal_accuracy"):
            # ★ 这里传的是 messages（完整轨迹），不是纯文本
            calls["agent_goal_accuracy"] = self.metrics["agent_goal_accuracy"].ascore(
                user_input=messages,
                reference=reference,
            )

        # 工具指标：必须有期望答案
        if expected_tool_calls and (wants("tool_call_accuracy") or wants("tool_call_f1")):
            _, _, _, ToolCall = self.message_types
            reference_calls = [
                ToolCall(
                    name=str(item.get("tool_name") or item.get("name") or "unknown"),
                    args=dict(item.get("arguments") or item.get("args") or {}),
                )
                for item in expected_tool_calls
            ]
            if wants("tool_call_accuracy"):
                calls["tool_call_accuracy"] = self.metrics["tool_call_accuracy"].ascore(
                    user_input=messages,
                    reference_tool_calls=reference_calls,
                )
            if wants("tool_call_f1"):
                calls["tool_call_f1"] = self.metrics["tool_call_f1"].ascore(
                    user_input=messages,
                    reference_tool_calls=reference_calls,
                )

        # ── 并发执行 + 单个超时 ──
        names = list(calls)
        results = await asyncio.gather(
            *(
                asyncio.wait_for(calls[name], timeout=settings.ragas_timeout_seconds)
                for name in names
            ),
            return_exceptions=True,
        )

        # ── 分离成功/失败 ──
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                # 记录异常类型 + 消息（便于排查是超时还是解析失败）
                errors[name] = f"{type(result).__name__}: {result}"
                continue
            # 兼容两种返回形态
            value = result if isinstance(result, (int, float)) else result.value
            scores[name] = round(float(value), 4)
        return scores, errors


# ════════════════════════════════════════════════════════════════
# 三、用例构造（自动出题）
# ════════════════════════════════════════════════════════════════

def _default_cases(db: Session, repo_id: str, limit: int | None = None) -> list[RagEvalCase]:
    """从仓库数据自动出题（★ 冒烟题库的构造逻辑）。

    【调用链】被 run_rag_quality_eval() 调用（没传 cases 时）

    【数据流向】
        入：db、repo_id、limit（最多几道题）
        出：RagEvalCase 列表

    【★ 三级优先，逐级降级】
        ① 优先用 documents 表（这是"已进 RAG 的知识"，最贴近真实评测目标）
        ② documents 为空 → 用 issues / pull_requests / workflow_runs
        ③ 都没有 → 返回空列表

    【★ 为什么要做第 ② 级降级】
        刚连接的仓库可能已经有 Issue/PR/CI 数据，但 Document 索引还没建。
        生产环境的 ChatAgent 会通过 EvidenceStore 预取这些记录 ——
        所以评测集也必须覆盖它们，否则"刚接入的仓库"跑评测会是 0 道题。

    【★ 题目为什么是模板化的英文问句】
        自动出的题不需要"像人问的"，只需要"能测出检索对不对"。
        模板问句（"What context exists for issue X?"）配合 reference（文档正文），
        足以测出 context_recall/precision —— 这才是这级评测的目标。

    【★ reference 为什么截断到 2000 字符】
        reference 会作为 judge 的输入。太长会挤占 judge 的上下文预算。
        2000 字符足够表达"这道题的正确答案是什么"。
    """
    repo_uuid = uuid.UUID(str(repo_id))
    case_limit = limit or settings.ragas_max_cases

    # ── ① 优先：Document（已进 RAG 的知识）
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_uuid, Document.source_type.in_(RAG_SOURCE_TYPES))
        .order_by(Document.created_at.desc())
        .limit(case_limit)
        .all()
    )
    cases: list[RagEvalCase] = []
    for index, doc in enumerate(docs, start=1):
        # 按来源类型给不同模板（问法要贴合内容性质）
        if doc.source_type == "issue":
            question = f"What context exists for issue {doc.title}?"
        elif doc.source_type == "pull_request":
            question = f"What changed in pull request {doc.title}?"
        elif doc.source_type == "workflow_run":
            question = f"What does the CI run {doc.title} show?"
        else:
            question = f"What project knowledge is available about {doc.title}?"

        cases.append(
            RagEvalCase(
                id=f"document-{index}",
                question=question,
                reference=doc.content[:2000],
                # 同时放 document.id 和 source_id（检索返回任一个都算命中）
                expected_source_ids=[str(doc.id), str(doc.source_id)],
                source_type=doc.source_type,
            )
        )
    if cases:
        return cases

    # ── ② 降级：Issue / PR / CI（结构化数据）
    # A freshly connected repository can have structured Issue/PR/CI data before
    # the reusable Document index is populated. Production ChatAgent prefetches
    # those records through EvidenceStore, so the default suite must cover them.
    issues = (
        db.query(Issue)
        .filter(Issue.repo_id == repo_uuid)
        # 先按更新时间、再按创建时间（最近活跃的优先）
        .order_by(Issue.updated_at.desc(), Issue.created_at.desc())
        .limit(case_limit)
        .all()
    )
    for issue in issues:
        # reference = 标题 + 正文 + 状态 + 标签（拼成"这道题的完整答案"）
        reference = "\n".join(
            value
            for value in [
                issue.title,
                issue.body or "",
                f"state: {issue.state}",
                f"labels: {', '.join(issue.labels or [])}",
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"issue-{issue.number}",
                question=f"What project context is recorded for issue #{issue.number}: {issue.title}?",
                reference=reference[:2000],
                expected_source_ids=[str(issue.id)],
                source_type="issue",
            )
        )
        # 够了就提前返回（不用继续查 PR/CI）
        if len(cases) >= case_limit:
            return cases

    # 剩余配额给 PR
    pull_requests = (
        db.query(PullRequest)
        .filter(PullRequest.repo_id == repo_uuid)
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        # ★ limit 用"剩余配额"，避免总数超标
        .limit(case_limit - len(cases))
        .all()
    )
    for pull_request in pull_requests:
        reference = "\n".join(
            value
            for value in [
                pull_request.title,
                pull_request.body or "",
                f"state: {pull_request.state}",
                f"base: {pull_request.base_branch}",
                f"head: {pull_request.head_branch}",
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"pull-request-{pull_request.number}",
                question=f"What changed in pull request #{pull_request.number}: {pull_request.title}?",
                reference=reference[:2000],
                expected_source_ids=[str(pull_request.id)],
                source_type="pull_request",
            )
        )
        if len(cases) >= case_limit:
            return cases

    # 剩余配额给"失败的 CI"（成功的没排障价值）
    workflow_runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == repo_uuid, WorkflowRun.conclusion == "failure")
        .order_by(WorkflowRun.created_at.desc())
        .limit(case_limit - len(cases))
        .all()
    )
    for workflow_run in workflow_runs:
        reference = "\n".join(
            value
            for value in [
                workflow_run.name,
                workflow_run.status,
                workflow_run.conclusion or "",
                # 日志只取前 1600 字符（和 reference 的 2000 上限配合）
                (workflow_run.logs_text or "")[:1600],
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"workflow-run-{workflow_run.github_run_id}",
                question=f"What does the failed CI run {workflow_run.name} show?",
                reference=reference[:2000],
                expected_source_ids=[str(workflow_run.id)],
                source_type="workflow_run",
            )
        )
    return cases


# ════════════════════════════════════════════════════════════════
# 四、检索指标（纯数学，不依赖 LLM）
# ════════════════════════════════════════════════════════════════

def _source_hit(result: dict[str, Any], expected_source_ids: set[str]) -> bool:
    """判断一条检索结果是否命中预期来源。

    【调用链】被 _retrieval_metrics() / _hard_rules() 调用

    【★ 为什么同时比 id 和 source_id】
        检索层可能返回 document_id（切片级）或 source_id（源对象级）。
        两个都试，兼容两种粒度。
    """
    return str(result.get("id")) in expected_source_ids or str(result.get("source_id")) in expected_source_ids


def _dcg(hits: list[int]) -> float:
    """折损累积增益（NDCG 的分子/分母共用）。

    【调用链】被 _retrieval_metrics() 调用

    【公式】Σ hit_i / log2(i + 2)
        i 从 0 开始，所以第一个位置的分母是 log2(2) = 1（不打折）
        第二个位置 log2(3) ≈ 1.58（打 63% 折）
        —— 位置越靠后，贡献越小。

    【为什么 +2 而不是 +1】
        log2(1) = 0，会除零。+2 保证分母最小是 1。
    """
    return sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))


def _retrieval_metrics(cases: list[RagEvalCase], rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    """算四个检索指标：recall@k / precision@k / MRR / NDCG。

    【调用链】_source_hit() / _dcg() → 本函数
    【被谁调用】run_rag_quality_eval()

    【数据流向】
        入：cases（用例）、rows（每条用例的检索结果）、k（只看前 k 条）
        出：{recall_at_k, precision_at_k, mrr, ndcg}（全部 0~1）

    【★ 四个指标各测什么】
        recall@k     该找的有没有找全（漏召回）
        precision@k  找到的有多少是有用的（噪声）
        MRR          第一个正确答案排第几（1/rank，越靠前越高）
        NDCG         综合"位置 + 命中"（排序质量）

    【★ 空 expected 的处理（三处都特殊照顾）】
        expected 为空表示"这道题没有指定正确答案"（如 negative 用例）。
        这时：
            recall   = 1.0（没有该找的，不算漏）
            MRR      = 1.0（同上）
            NDCG     = 1.0
            precision 仍按实际命中算（可能为 0，但那是检索噪声，合理）
        为什么这么处理：不能用"没有标准答案"来惩罚模型。
    """
    if not cases:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}

    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []

    for case, row in zip(cases, rows, strict=True):
        expected = set(case.expected_source_ids)
        # 只看前 k 条
        results = row["results"][:k]
        hits = [1 if _source_hit(result, expected) else 0 for result in results]

        # recall：命中任意一条就算 1（因为"该找的"通常只有一个来源）
        recall_values.append(1.0 if not expected or any(hits) else 0.0)
        # precision：命中数 / 返回数
        precision_values.append(sum(hits) / max(len(results), 1))

        # MRR：第一个命中的倒数排名
        first_hit = next((index + 1 for index, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1.0 / first_hit if first_hit else (1.0 if not expected else 0.0))

        # NDCG：理想排序是"第一条就命中"
        ideal = [1] + [0] * max(len(hits) - 1, 0)
        ndcg_values.append(_dcg(hits) / (_dcg(ideal) or 1.0) if expected else 1.0)

    return {
        "recall_at_k": round(sum(recall_values) / len(recall_values), 4),
        "precision_at_k": round(sum(precision_values) / len(precision_values), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "ndcg": round(sum(ndcg_values) / len(ndcg_values), 4),
    }


# ════════════════════════════════════════════════════════════════
# 五、硬规则与上下文
# ════════════════════════════════════════════════════════════════

def _trace_rows(trace: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """从执行轨迹里抽出"上下文列表"和"来源列表"（★ 两层去重）。

    【调用链】被 run_rag_quality_eval() 调用

    【数据流向】
        入：trace（ChatAgent 产出的 evaluation_trace）
        出：(contexts, sources)

    【★ 为什么要两层去重】
        一次 Agent 可能调 3 次检索工具，每次都返回部分重叠的结果。
        contexts 用"文本完全相同"去重（内容级）
        sources 用 (id, source_id, title) 三元组去重（实体级）
        —— 前者防"同一段话喂两遍"，后者防"同一文档算两个来源"。

    【★ 为什么用 seen_contexts 存全文而不是哈希】
        文本长度通常几百字，存全文占不了多少内存，
        而且能直接用于后续的 judge 输入（省一次转换）。
    """
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_contexts: set[str] = set()
    seen_sources: set[tuple[str, str, str]] = set()

    for retrieval in trace.get("retrievals") or []:
        if not isinstance(retrieval, dict):
            continue

        # 内容级去重
        for value in retrieval.get("contexts") or []:
            text = str(value).strip()
            if text and text not in seen_contexts:
                seen_contexts.add(text)
                contexts.append(text)

        # 实体级去重
        for raw_source in retrieval.get("sources") or []:
            if not isinstance(raw_source, dict):
                continue
            source = dict(raw_source)
            key = (
                str(source.get("id") or ""),
                str(source.get("source_id") or ""),
                str(source.get("title") or ""),
            )
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(source)
    return contexts, sources


def _hard_rules(case: RagEvalCase, result: dict[str, Any], sources: list[dict[str, Any]], contexts: list[str]) -> dict[str, Any]:
    """跑八项确定性断言（★ 不依赖 LLM，永远可复现）。

    【调用链】_source_hit() → 本函数
    【被谁调用】run_rag_quality_eval()

    【数据流向】
        入：case（含断言配置）、result（Agent 输出）、sources、contexts
        出：{passed, 各项明细}

    【★ 八项断言】
        ① source_hit              预期来源被检索到
        ② missing_tools           必需工具都调了（空列表 = 通过）
        ③ tool_errors             没有工具执行报错
        ④ answer_present          答案非空
        ⑤ contexts_present        有检索上下文（否则 RAGAS 无法评）
        ⑥ missing_phrases         必须出现的短语都出现了
        ⑦ forbidden_phrases       禁止的短语一个都没出现
        ⑧ answer_pattern_matched  答案符合正则（如"恰好三句话"）

    【★ 为什么这层比 RAGAS 更重要】
        RAGAS 的分数会飘（judge 换版本、temperature 微调都会变）。
        硬规则是"对/错"二值判断 —— 改坏了立刻能发现。
        所以最终判定里，硬规则不过 = 一定失败（不管 RAGAS 多少分）。

    【★ 为什么用 casefold() 而不是 lower()】
        casefold() 是"更彻底的小写"（处理 ß→ss 这类 Unicode 特例）。
        中英文混排的断言更可靠。
    """
    expected = set(case.expected_source_ids)

    # ① 来源命中（expected 为空视为通过）
    source_hit = not expected or any(_source_hit(item, expected) for item in sources)

    # ② 必需工具
    actual_tools = [
        str(item.get("tool_name") or "")
        for item in result.get("tool_calls") or []
        if isinstance(item, dict)
    ]
    missing_tools = [name for name in case.required_tools if name not in actual_tools]

    # ③ 工具报错
    tool_errors = [
        {
            "tool_name": item.get("tool_name"),
            "error": item.get("error"),
        }
        for item in result.get("tool_calls") or []
        if isinstance(item, dict) and item.get("status") == "error"
    ]

    answer = str(result.get("answer") or "").strip()
    answer_present = bool(answer)
    contexts_present = bool(contexts)

    # ⑥⑦ 短语断言
    missing_phrases = [value for value in case.must_include if value.casefold() not in answer.casefold()]
    forbidden_phrases = [value for value in case.must_not_include if value.casefold() in answer.casefold()]

    # ⑧ 正则断言（answer_regex 为 None 视为不检查）
    answer_pattern_matched = case.answer_regex is None or re.fullmatch(case.answer_regex, answer) is not None

    # ★ 八项全部满足才通过（and 关系，没有"部分通过"）
    passed = (
        source_hit
        and not missing_tools
        and not tool_errors
        and answer_present
        and contexts_present
        and not missing_phrases
        and not forbidden_phrases
        and answer_pattern_matched
    )

    return {
        "passed": passed,
        "source_hit": source_hit,
        "answer_present": answer_present,
        "contexts_present": contexts_present,
        "missing_tools": missing_tools,
        "tool_errors": tool_errors,
        "missing_phrases": missing_phrases,
        "forbidden_phrases": forbidden_phrases,
        "answer_pattern_matched": answer_pattern_matched,
        # 引用数量（不参与判定，但能看出"回答有没有标来源"）
        "citation_count": len(result.get("citations") or []),
    }


def _evaluation_project_context(db: Session, repo_id: uuid.UUID) -> dict[str, Any]:
    """取"项目自有上下文"（团队 + 记忆），和生产环境保持一致。

    【调用链】被 _run_chatagent_case() 调用

    Load the same project-owned context that the frontend sends to ChatAgent.

    【★ 为什么评测要模拟前端传的上下文】
        生产环境的 ChatAgent 会收到前端传来的 team_members / project_memory。
        评测如果不传，Agent 就少了这些信息 ——
        测出来的分数不能代表真实效果（会偏低）。

        所以这里从 Document 表里把这两类取出来，拼成和生产一样的结构。

    【★ 为什么从 Document 表取而不是直接查 ThreadMemory】
        前端传的就是"Document 里的 team_member / memory_note"。
        从同一个源取，才能保证结构完全一致。
    """
    documents = (
        db.query(Document)
        .filter(
            Document.repo_id == repo_id,
            Document.source_type.in_(("team_member", "memory_note")),
        )
        .order_by(Document.created_at.asc())
        .all()
    )

    team_members: list[dict[str, str]] = []
    project_memory: list[dict[str, str]] = []
    for document in documents:
        metadata = document.meta or {}
        if document.source_type == "team_member":
            team_members.append(
                {
                    # 兼容两种来源：metadata.name 或标题里的"团队成员：xxx"
                    "name": str(metadata.get("name") or document.title.replace("团队成员：", "")),
                    "role": str(metadata.get("role") or ""),
                    "strengths": str(metadata.get("strengths") or ""),
                    # tech_stack / techStack 两种命名都兼容
                    "techStack": str(metadata.get("tech_stack") or metadata.get("techStack") or ""),
                }
            )
        elif document.source_type == "memory_note":
            project_memory.append({"title": document.title, "content": document.content})
    return {"team_members": team_members, "project_memory": project_memory}


async def _run_chatagent_case(
    db: Session,
    repo: Repository,
    case: RagEvalCase,
    *,
    limit: int,
    agent_factory: Callable[[], ChatAgent],
) -> dict[str, Any]:
    """在隔离环境里跑一条用例（★ 跑完 rollback，绝不污染真实数据）。

    【调用链】追这个函数会进入：
        1. db.begin_nested()              → 开 SAVEPOINT
        2. Conversation(...) + flush      → 建一个临时会话
        3. EvidenceStore.rebuild_repo()   → 只给这个会话建证据
        4. ContextAssembler.assemble()    → 组装上下文（和生产一致）
        5. _build_conversation_tools()    → 复用生产的工具集
        6. agent_factory().run()          → 真正跑 Agent
    【被谁调用】run_rag_quality_eval()

    【数据流向】
        入：db、repo、case、limit、agent_factory
        出：Agent 的完整输出（含 evaluation_trace）

    【★ 为什么用 SAVEPOINT（begin_nested）而不是普通事务】
        评测可能在"已经打开的事务"里跑（比如从 route 调用）。
        普通 rollback 会把外层事务也回滚掉。
        SAVEPOINT 只回滚自己这一段 —— 精准隔离。

    【★ 为什么要建临时 Conversation】
        ChatAgent 依赖 conversation_id 来读历史、写记忆。
        不建的话它拿不到上下文，行为和线上不一致。
        标题带 [RAGAS Eval] 前缀，方便排查时区分。

    【★ 为什么 import 放在函数内部】
        `from app.api.routes.chat import ...` ——
        routes 层会 import services 层，顶层 import 会形成循环依赖。
        局部 import 把依赖推迟到调用时。

    【★ evaluation_mode=True 的作用】
        让 ChatAgent 知道"这是评测"，
        从而不写记忆、不发通知、不影响用户数据。
    """
    # Each trial gets a clean conversation and rolls back every runtime write.
    savepoint = db.begin_nested()
    try:
        conversation = Conversation(
            repo_id=repo.id,
            title=f"[RAGAS Eval] {case.id or case.question[:40]}",
            status="evaluation",       # ★ 用专门的状态，不混进用户的 active 会话
        )
        db.add(conversation)
        db.flush()

        # 只给这个临时会话建证据（不碰真实会话的）
        EvidenceStore(db).rebuild_repo(repo, conversation)
        assembled = ContextAssembler(db).assemble(repo, conversation, case.question)

        # 复用生产的工具集（评测必须和生产一致，否则测的不是同一个东西）
        from app.api.routes.chat import _build_conversation_tools, _skill_runtime_context

        context: dict[str, Any] = {
            # 项目上下文（团队 + 记忆）
            **_evaluation_project_context(db, repo.id),
            # 技能运行时上下文
            **_skill_runtime_context(),
            "repo_id": str(repo.id),
            "conversation_id": str(conversation.id),
            # 组装好的上下文（ChatAgent 直接用，不再自己组装）
            "server_context": assembled.system_context,
            "server_messages": assembled.history_messages,
            "evidence": assembled.evidence,
            "compression_stats": assembled.compression_stats,
            "source_type": case.source_type,
            "limit": limit,
            # ★ 标记评测模式
            "evaluation_mode": True,
        }
        # 工具集要单独注入（需要 db / repo / conversation）
        context["tools"] = _build_conversation_tools(repo, conversation, db)
        return await agent_factory().run({"message": case.question}, context)
    finally:
        # ★ 无论成功失败都回滚 —— 评测产生的所有写入（会话/消息/记忆）都丢弃
        if savepoint.is_active:
            savepoint.rollback()


# ════════════════════════════════════════════════════════════════
# 六、汇总与诊断
# ════════════════════════════════════════════════════════════════

def _thresholds() -> dict[str, float]:
    """各指标的门槛（从 settings 读，便于按项目调整）。

    【调用链】被 _metric_details() / _case_diagnoses() 调用

    【为什么门槛要可配置】
        不同项目的 RAG 难度不同。
        内部文档库可能 recall 0.9 很容易，公开 Issue 库 0.7 就不错。
        写死门槛会导致"换个仓库就全红"。
    """
    return {
        "context_precision": settings.ragas_context_precision_min,
        "context_recall": settings.ragas_context_recall_min,
        "faithfulness": settings.ragas_faithfulness_min,
        "answer_relevancy": settings.ragas_answer_relevancy_min,
        "agent_goal_accuracy": settings.ragas_agent_goal_accuracy_min,
        "tool_call_accuracy": settings.ragas_tool_call_accuracy_min,
        "tool_call_f1": settings.ragas_tool_call_f1_min,
    }


def _mean(values: list[float]) -> float | None:
    """求均值（★ 先过滤 NaN/Inf，全无效返回 None）。

    【调用链】被 _metric_details() 调用

    【★ 为什么要过滤非有限值】
        RAGAS 偶尔会返回 NaN（分母为 0 时）。
        NaN 参与 sum 会让整个结果变成 NaN，最终 JSON 序列化会失败。
        过滤后至少能给出"有效样本的均值"。

    【为什么全无效返回 None 而不是 0】
        None 表示"这个指标没测出来"，0 表示"测出来是 0 分"。
        两者含义完全不同 —— 前者是"环境问题"，后者是"质量差"。
    """
    finite = [value for value in values if math.isfinite(value)]
    return round(sum(finite) / len(finite), 4) if finite else None


def _metric_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """汇总每个指标：均值 + 门槛 + 是否通过 + 有效样本数。

    【调用链】_thresholds() / _mean() → 本函数
    【被谁调用】_finalize_result()

    【数据流向】
        入：rows（每条用例的结果）
        出：{指标名: {score, threshold, passed, scored_cases}}

    【★ 为什么每个指标都要 scored_cases】
        7 个指标里，有的可能只有 3 条用例有分数（其余失败）。
        只报均值会掩盖"样本太少不可信"。
        前端能据此提示"该指标样本不足"。

    【★ passed 为 None 的语义】
        score 为 None（完全没测出来）→ passed=None（不是 False）。
        这和"测出来但没达标"是两件事。
    """
    thresholds = _thresholds()
    # 初始化所有指标的容器（保证每个指标都有输出，哪怕是 None）
    values: dict[str, list[float]] = {
        name: [] for name in (*CORE_RAGAS_METRICS, *AGENT_RAGAS_METRICS)
    }
    for row in rows:
        for name, value in (row.get("ragas") or {}).items():
            # 只收已知指标 + 有限数值
            if name in values and isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[name].append(float(value))

    details: dict[str, dict[str, Any]] = {}
    for name, metric_values in values.items():
        score = _mean(metric_values)
        details[name] = {
            "score": score,
            "threshold": thresholds[name],
            "passed": None if score is None else score >= thresholds[name],
            "scored_cases": len(metric_values),
        }
    return details


def _case_diagnoses(row: dict[str, Any]) -> list[dict[str, str]]:
    """给一条用例生成"诊断结论"（★ 六类，告诉用户该修哪一层）。

    【调用链】_thresholds() → 本函数
    【被谁调用】_diagnostic_summary()

    【数据流向】
        入：row（一条用例的完整结果）
        出：[{category, severity, title, summary, action}] 列表

    【★ 六类诊断，按"该修哪里"分类】
        judge_error              打分器出问题 → 重试评分，别动 RAG
        retrieval                预期来源没进上下文 → 查切片/改写/Top-K
        generation               违反硬规则 → 收紧回答指令
        retrieval_noise          recall 达标但 precision 低 → 减噪声（rerank/Top-K）
        retrieval_recall         recall 低 → 查召回（切片/多文档）
        generation_faithfulness  faithfulness 低 → 回答没忠于证据
        generation_relevancy     relevancy 低 → 答非所问
        agent_goal               目标达成低 → 没遵守任务约束
        passed                   都没问题

    【★ 为什么 precision 和 recall 要分开诊断】
        "找全了但噪声多"和"没找全"的修法完全不同：
            前者要减（收紧 Top-K / 加强 rerank）
            后者要加（放宽 Top-K / 改切片 / 多路召回）
        混在一起给建议，用户会改错方向。

    【★ 为什么 precision 低但 recall 也低时只报 recall】
        条件写的是 `precision_low and not recall_low`。
        两个都低时，根本原因是"召回不全"，先修召回。
        同时报两条会让用户不知道先改哪个。

    【★ judge_error 为什么排第一】
        如果打分器本身出错了，其他指标的"低分"都不可信。
        先提示"这是环境问题，别急着调 RAG"。
    """
    scores = row.get("ragas") or {}
    errors = row.get("ragas_errors") or {}
    hard = row.get("hard_rules") or {}
    thresholds = _thresholds()
    diagnoses: list[dict[str, str]] = []

    # ── ① judge 出错（优先级最高：其他分数都不可信）──
    if errors:
        names = "、".join(errors)
        diagnoses.append(
            {
                "category": "judge_error",
                "severity": "error",
                "title": "Judge 指标计算异常",
                "summary": f"{names} 未得到可信分数；先仅重试评分，不要据此调整 RAG。",
                "action": "重试异常指标；若仍失败，再检查超时、模型结构化输出和答案长度。",
            }
        )

    # ── ② 检索层：来源没进上下文 ──
    if not hard.get("source_hit", True) or not hard.get("contexts_present", True):
        diagnoses.append(
            {
                "category": "retrieval",
                "severity": "failed",
                "title": "目标证据没有进入回答上下文",
                "summary": "预期来源未命中，或生产检索链没有返回可评分上下文。",
                "action": "检查切片、Query 改写、来源过滤、Top-K 和 Rerank。",
            }
        )

    # ── ③ 生成层：违反硬规则 ──
    rule_failures = (
        list(hard.get("missing_phrases") or [])
        + list(hard.get("forbidden_phrases") or [])
    )
    if rule_failures or hard.get("answer_pattern_matched") is False:
        diagnoses.append(
            {
                "category": "generation",
                "severity": "failed",
                "title": "回答违反确定性验收规则",
                "summary": "回答遗漏关键事实、包含禁用事实，或没有遵守要求的输出格式。",
                "action": "收紧回答指令，并审查 must_include、must_not_include 与格式规则。",
            }
        )

    # ── ④~⑦ 语义指标诊断 ──
    precision = scores.get("context_precision")
    recall = scores.get("context_recall")
    faithfulness = scores.get("faithfulness")
    relevancy = scores.get("answer_relevancy")
    goal = scores.get("agent_goal_accuracy")

    precision_low = isinstance(precision, (int, float)) and precision < thresholds["context_precision"]
    recall_low = isinstance(recall, (int, float)) and recall < thresholds["context_recall"]

    if precision_low and not recall_low:
        # 找全了但噪声多 → 该减
        diagnoses.append(
            {
                "category": "retrieval_noise",
                "severity": "failed",
                "title": "证据找全了，但上下文噪声过多",
                "summary": "Context Recall 达标而 Context Precision 未达标。",
                "action": "比较 Top-K、Rerank 和来源状态特征，减少旧资料或同主题噪声。",
            }
        )
    elif recall_low:
        # 没找全 → 该加（注意 elif：两个都低时只报 recall）
        diagnoses.append(
            {
                "category": "retrieval_recall",
                "severity": "failed",
                "title": "关键证据召回不完整",
                "summary": "Context Recall 未达到当前评测门槛。",
                "action": "检查切片、Query 改写、多文档召回和来源过滤。",
            }
        )

    if (
        isinstance(faithfulness, (int, float))
        and faithfulness < thresholds["faithfulness"]
    ):
        diagnoses.append(
            {
                "category": "generation_faithfulness",
                "severity": "failed",
                "title": "回答没有忠于检索证据",
                "summary": "回答包含证据无法支持的扩写、推断或错误选择。",
                "action": "缩短答案，要求逐项依据证据回答，并加强冲突来源处理。",
            }
        )

    if isinstance(relevancy, (int, float)) and relevancy < thresholds["answer_relevancy"]:
        diagnoses.append(
            {
                "category": "generation_relevancy",
                "severity": "failed",
                "title": "回答没有直接回应问题",
                "summary": "答案相关性低于当前门槛，可能存在过度展开或遗漏用户追问。",
                "action": "改进问题拆解和回答结构，删除无关背景与下一步建议。",
            }
        )

    if isinstance(goal, (int, float)) and goal < thresholds["agent_goal_accuracy"]:
        diagnoses.append(
            {
                "category": "agent_goal",
                "severity": "failed",
                "title": "ChatAgent 没有完成题目要求",
                "summary": "回答可能事实相关，但没有遵守句数、格式或任务约束。",
                "action": "让生成链优先执行用户明确约束，再补充必要解释。",
            }
        )

    # ── 兜底：一条诊断都没有 → 明确说"通过了" ──
    if not diagnoses:
        diagnoses.append(
            {
                "category": "passed",
                "severity": "passed",
                "title": "该用例未发现门禁问题",
                "summary": "硬规则和已得到的语义指标均符合当前门槛。",
                "action": "保留为回归基线。",
            }
        )
    return diagnoses


def _diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总所有用例的诊断（★ 注意：这里会原地写入 row["diagnoses"]）。

    【调用链】_case_diagnoses() → 本函数
    【被谁调用】_finalize_result()

    【数据流向】
        入：rows
        出：{failed_cases, error_cases, category_counts}

    【★ 为什么直接改 row 而不返回新结构】
        前端需要"每条用例的诊断详情"。
        在这里一次性算完写回 row，后面 cases 字段直接带出去，
        不用再传一遍。

    【★ failed_cases 和 error_cases 的区别】
        failed_cases  有非 passed 诊断的用例（可能是质量问题）
        error_cases   有 ragas_errors 的用例（是环境/judge 问题）
        两者可能重叠，但语义不同 —— 一个看质量，一个看可用性。
    """
    counts: dict[str, int] = {}
    failed_cases = 0
    error_cases = 0

    for row in rows:
        # 算诊断并写回
        row["diagnoses"] = _case_diagnoses(row)
        # 排除 passed，只要有问题就算 failed
        categories = {item["category"] for item in row["diagnoses"] if item["category"] != "passed"}
        if categories:
            failed_cases += 1
        if row.get("ragas_errors"):
            error_cases += 1
        # 按类别计数（前端画"问题分布"）
        for category in categories:
            counts[category] = counts.get(category, 0) + 1

    return {
        "failed_cases": failed_cases,
        "error_cases": error_cases,
        "category_counts": counts,
    }


def _finalize_result(
    *,
    rows: list[dict[str, Any]],
    retrieval: dict[str, float],
    performance: dict[str, Any],
    judge_requested: bool,
    scorer: SemanticScorer | None,
    judge_error: str | None,
    repo_id: str,
    limit: int,
    evaluation_set: dict[str, Any] | None,
    rescore_of: str | None = None,
) -> dict[str, Any]:
    """汇总成最终结果（★ 判定逻辑是本文件最需要理解的一段）。

    【调用链】_metric_details() / _diagnostic_summary() → 本函数
    【被谁调用】run_rag_quality_eval() / rescore_rag_result()

    【数据流向】
        入：rows、retrieval、performance、judge 状态、evaluation_set、rescore_of
        出：完整结果 dict（status / quality_status / 各指标 / 诊断 / 逐题明细）

    【★ 三层判定，逐步收紧】
        ① hard_passed        所有用例的硬规则都通过
        ② evaluation_complete judge 可用 + 所有必需指标都有分数 + 无用例级错误
        ③ ragas_passed       必需指标全部达标

    【★ status 的六种取值及含义】
        no_cases    没有用例（题库空）
        failed      硬规则不过，或语义指标不达标
        partial     没要求 judge（只跑了硬规则）
        error       judge 请求了但没完成（环境问题）
        passed      全通过

    【★ 为什么"硬规则不过"优先于"judge 没完成"】
        硬规则不过 = 确定有问题（比如答案里出现了禁用词）。
        这时 status 应该是 failed（质量失败），而不是 error（环境问题）。
        用户需要先修质量问题，而不是去查 judge 配置。

    【★ gated_names 为什么要把 tool_call_* 动态加进去】
        工具指标只有在"有用例配了 expected_tool_calls"时才有分数。
        没配就不该纳入判定（否则永远是 None，永远不通过）。
        条件：`metric_rows[name]["scored_cases"] > 0`

    【★ summary_metrics 为什么要扁平化一层】
        前端和报告模板都想要"指标名 → 分数"的直接映射。
        同时保留 metric_rows（含 threshold/passed），两种形态都给。
    """
    metric_rows = _metric_details(rows)

    # ── 硬规则通过率 ──
    hard_passed_cases = sum(1 for row in rows if (row.get("hard_rules") or {}).get("passed"))
    hard_pass_rate = round(hard_passed_cases / len(rows), 4) if rows else 0.0

    # ── judge 可用性判定 ──
    case_metric_errors = sum(bool(row.get("ragas_errors")) for row in rows)
    required_scored = all(metric_rows[name]["score"] is not None for name in REQUIRED_RAGAS_METRICS)
    ragas_available = bool(judge_requested and scorer is not None and required_scored)
    # "完整"= 可用 + 没有任何用例级错误
    evaluation_complete = bool(ragas_available and case_metric_errors == 0)

    # ── 纳入判定的指标清单（必需 + 实际有分的工具指标）──
    gated_names = list(REQUIRED_RAGAS_METRICS)
    gated_names.extend(
        name
        for name in ("tool_call_accuracy", "tool_call_f1")
        if metric_rows[name]["scored_cases"] > 0
    )
    semantic_failed = any(metric_rows[name]["passed"] is False for name in gated_names)
    ragas_passed = evaluation_complete and not semantic_failed

    hard_passed = bool(rows) and hard_passed_cases == len(rows)

    # ── 判定 status / quality_status ──
    if not rows:
        quality_status = "not_evaluated"
        status = "no_cases"
    elif not hard_passed or semantic_failed:
        # ★ 硬规则或语义不过 → failed（即使 judge 没完成也是 failed，不是 error）
        quality_status = "failed"
        status = "error" if judge_requested and not evaluation_complete else "failed"
    elif not judge_requested:
        # 没要求 judge → 只跑了硬规则，算"部分完成"
        quality_status = "partial"
        status = "partial"
    elif not evaluation_complete:
        # 要求了 judge 但没完成 → 环境问题
        quality_status = "incomplete"
        status = "error"
    else:
        quality_status = "passed"
        status = "passed"

    # ── 扁平化指标（只含有分数的）──
    summary_metrics = {
        name: value["score"]
        for name, value in metric_rows.items()
        if value["score"] is not None
    }
    summary_metrics.update(
        {
            "hard_gate_pass_rate": hard_pass_rate,
            "retrieval_recall_at_k": retrieval.get("recall_at_k", 0.0),
            "retrieval_precision_at_k": retrieval.get("precision_at_k", 0.0),
        }
    )

    diagnostics = _diagnostic_summary(rows)

    return {
        "kind": "chatagent_ragas",
        "status": status,
        "execution_status": "completed",
        "quality_status": quality_status,
        "evaluation_complete": evaluation_complete,
        "passed": status == "passed",
        # 重评 vs 完整跑（前端据此区分"这次只重打了分"）
        "run_mode": "rescore" if rescore_of else "full",
        "rescore_of": rescore_of,
        "target": {
            "name": "ChatAgent",
            "repo_id": str(repo_id),
            # ★ 明确说明"跑的是生产 Agent 还是只重打分"
            "execution": "stored_evidence_rescore" if rescore_of else "production_agent_with_isolated_conversation",
            "evaluation_mode": True,
        },
        # 没传题库信息就当作"自动冒烟"
        "evaluation_set": evaluation_set or {
            "id": "auto-document-smoke",
            "name": "auto-document-smoke",
            "label": "自动文档冒烟测试",
            "source": "auto_smoke",
            "sha256": None,
            "case_count": len(rows),
        },
        "pipeline": {"top_k": limit},
        **summary_metrics,
        "summary_metrics": summary_metrics,
        "retrieval": retrieval,
        "performance": performance,
        "hard_gate": {
            "passed": hard_passed,
            "pass_rate": hard_pass_rate,
            "passed_cases": hard_passed_cases,
            "total_cases": len(rows),
        },
        "ragas": {
            "requested": judge_requested,
            "available": ragas_available,
            "complete": evaluation_complete,
            # ★ 没完整时 passed 为 None（不是 False）—— 不能说"没通过"
            "passed": ragas_passed if evaluation_complete else None,
            "version": scorer.version if scorer is not None else None,
            "judge_model": scorer.judge_model if scorer is not None else None,
            "error": judge_error or (
                f"{case_metric_errors} evaluation case(s) contain Ragas metric errors"
                if case_metric_errors
                else None
            ),
            "case_errors": case_metric_errors,
            "metrics": metric_rows,
        },
        "diagnostics": diagnostics,
        "cases": rows,
    }


# ════════════════════════════════════════════════════════════════
# 七、入口
# ════════════════════════════════════════════════════════════════

async def run_rag_quality_eval(
    db: Session,
    repo_id: str,
    *,
    cases: list[RagEvalCase] | None = None,
    limit: int = 5,
    run_judge: bool | None = None,
    scorer: SemanticScorer | None = None,
    agent_factory: Callable[[], ChatAgent] | None = None,
    progress_callback: ProgressCallback | None = None,
    evaluation_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """【完整评测】逐条跑 ChatAgent，再逐条打分（★ 本文件的主入口）。

    【调用链】追这个函数会进入：
        1. _default_cases()               → 没传 cases 就自动出题
        2. RagasCollectionsScorer()       → 初始化 judge（失败不致命）
        3. _run_chatagent_case()          → 逐条跑 Agent（隔离 + rollback）
        4. _trace_rows() / _hard_rules()  → 抽上下文 + 跑硬规则
        5. scorer.score()                 → 逐条打分
        6. _retrieval_metrics()           → 检索指标
        7. _finalize_result()             → 汇总判定
    【被谁调用】api/routes/evaluations.py、services/evals/runner.py

    Evaluate the production ChatAgent path with Ragas and deterministic gates.

    【数据流向】
        入：db、repo_id、cases（可选）、limit（Top-K）、run_judge（可选覆盖）、
            scorer（可注入假实现）、agent_factory（可注入假 Agent）、
            progress_callback、evaluation_set
        出：_finalize_result 的完整结果

    【★ 为什么 judge 初始化失败不抛异常】
        `except RagasUnavailableError as exc: judge_error = str(exc)`
        没配 judge key 时，硬规则仍然能跑 —— 不该让整个评测失败。
        结果里 status=partial，用户知道"只跑了硬规则"。

    【★ 为什么逐条串行而不是并发跑 Agent】
        ChatAgent 会写数据库（消息、记忆）。并发写会互相干扰，
        而且 SAVEPOINT 在并发下行为复杂。
        串行慢一点，但结果可靠 —— 评测不是高频操作，慢几秒可以接受。

    【★ 三种跳过打分的理由（skipped_reason）】
        judge_disabled         没要求打分（run_judge=False）
        judge_unavailable      judge 初始化失败
        no_retrieved_contexts  这条用例没检索到任何上下文（没法算 recall/precision）
        前两个是环境问题，第三个是数据问题 —— 区分开才知道该修什么。
    """
    repo = db.get(Repository, uuid.UUID(str(repo_id)))
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")

    evaluation_cases = cases if cases is not None else _default_cases(db, repo_id)
    # run_judge 为 None 时用全局配置
    judge_requested = settings.ragas_enabled if run_judge is None else run_judge
    total_cases = len(evaluation_cases)

    await _emit_progress(
        progress_callback,
        stage="preparing",
        message=f"已准备 {total_cases} 条评测用例，正在初始化 Judge。",
        completed_cases=0,
        total_cases=total_cases,
    )

    # ── 初始化 judge（失败只记错误，不中断）──
    judge_error: str | None = None
    if judge_requested and scorer is None:
        try:
            scorer = RagasCollectionsScorer()
        except RagasUnavailableError as exc:
            judge_error = str(exc)

    rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    # ── 逐条跑 ──
    for index, case in enumerate(evaluation_cases, start=1):
        await _emit_progress(
            progress_callback,
            stage="chatagent",
            message=f"正在运行第 {index}/{total_cases} 条真实 ChatAgent 用例。",
            current_case=index,
            case_id=case.id or f"case-{index}",
            completed_cases=index - 1,
            total_cases=total_cases,
        )

        item_started = time.perf_counter()
        # ★ 跑真实的 ChatAgent（默认 ChatAgent，测试可注入假实现）
        result = await _run_chatagent_case(
            db,
            repo,
            case,
            limit=limit,
            agent_factory=agent_factory or ChatAgent,
        )

        # 抽上下文和来源
        trace = result.get("evaluation_trace") if isinstance(result.get("evaluation_trace"), dict) else {}
        contexts, sources = _trace_rows(trace)
        # 硬规则（不依赖 judge）
        hard_rules = _hard_rules(case, result, sources, contexts)

        semantic_scores: dict[str, float] | None = None
        semantic_errors: dict[str, str] = {}
        skipped_reason: str | None = None

        # ── 决定要不要打分（三种跳过理由）──
        if not judge_requested:
            skipped_reason = "judge_disabled"
        elif scorer is None:
            skipped_reason = "judge_unavailable"
        elif not contexts:
            skipped_reason = "no_retrieved_contexts"
        else:
            await _emit_progress(
                progress_callback,
                stage="ragas",
                message=f"ChatAgent 已完成第 {index}/{total_cases} 条，正在计算 RAGAS 指标。",
                current_case=index,
                case_id=case.id or f"case-{index}",
                completed_cases=index - 1,
                total_cases=total_cases,
            )
            semantic_scores, semantic_errors = await scorer.score(
                user_input=case.question,
                response=str(result.get("answer") or ""),
                retrieved_contexts=contexts,
                reference=case.reference,
                # 只传 dict（过滤掉脏数据）
                tool_calls=[dict(item) for item in result.get("tool_calls") or [] if isinstance(item, dict)],
                tool_observations=[
                    dict(item)
                    for item in trace.get("tool_observations") or []
                    if isinstance(item, dict)
                ],
                expected_tool_calls=case.expected_tool_calls,
            )

        latency_ms = round((time.perf_counter() - item_started) * 1000, 2)
        # 检索指标需要"每条的 sources"
        retrieval_rows.append({"results": sources})

        # ── 组装这条用例的完整现场（★ 后续 rescore 就靠这些现场）──
        rows.append(
            {
                "id": case.id or f"case-{index}",
                "question": case.question,
                "reference": case.reference,
                "expected_source_ids": case.expected_source_ids,
                "source_file": case.source_file,
                "kind": case.kind,
                # 断言配置也存下来（rescore 时不用重新加载题库）
                "must_include": case.must_include,
                "must_not_include": case.must_not_include,
                "answer_regex": case.answer_regex,
                "required_tools": case.required_tools,
                "expected_tool_calls": case.expected_tool_calls,
                # Agent 的实际输出
                "response": result.get("answer"),
                "route": result.get("route"),
                "citations": result.get("citations") or [],
                "tool_calls": result.get("tool_calls") or [],
                "tool_observations": trace.get("tool_observations") or [],
                "completion": result.get("completion") or {},
                # 检索现场
                "retrieved_contexts": contexts,
                "retrieved_sources": sources,
                "retrieval_trace": trace.get("retrievals") or [],
                # 判定结果
                "hard_rules": hard_rules,
                "ragas": semantic_scores,
                "ragas_errors": semantic_errors,
                "ragas_skipped_reason": skipped_reason,
                "latency_ms": latency_ms,
            }
        )

        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {index}/{total_cases} 条评测已完成。",
            current_case=index,
            case_id=case.id or f"case-{index}",
            completed_cases=index,
            total_cases=total_cases,
        )

    # ── 汇总 ──
    await _emit_progress(
        progress_callback,
        stage="finalizing",
        message="正在汇总指标并生成最终判定。",
        completed_cases=total_cases,
        total_cases=total_cases,
    )
    retrieval = _retrieval_metrics(evaluation_cases, retrieval_rows, limit)

    latencies = [float(row["latency_ms"]) for row in rows]
    performance = {
        "cases": len(rows),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }

    return _finalize_result(
        rows=rows,
        retrieval=retrieval,
        performance=performance,
        judge_requested=judge_requested,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str(repo.id),
        limit=limit,
        evaluation_set=evaluation_set,
    )


async def rescore_rag_result(
    result: dict[str, Any],
    *,
    eval_id: str,
    scorer: SemanticScorer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """【只重打分】不重跑 ChatAgent，只把失败/缺失的指标补上（★ 省钱省时间）。

    【调用链】scorer.score() / _finalize_result() → 本函数
    【被谁调用】api/routes/evaluations.py（"重试评分"按钮）

    Retry missing or failed Judge metrics without running ChatAgent again.

    【数据流向】
        入：result（上一次的完整结果）、eval_id（原评测 ID）、scorer、progress_callback
        出：新的完整结果（run_mode="rescore"，rescore_of=eval_id）

    【★ 为什么需要这个功能】
        跑一次完整评测要几分钟（每条用例都跑 Agent）。
        但"judge 超时"是很常见的偶发问题 ——
        重跑 Agent 完全没必要，因为 Agent 的输出（答案、上下文）都存着。
        直接拿存下来的现场重打分即可。

    【★ 怎么确定"要重试哪些指标"】
        两个来源合并：
            ① errors 里有的指标（上次失败了）
            ② REQUIRED 里没出现在 scores 的指标（上次没跑/没结果）
            ③ 如果这条用例有 expected_tool_calls，工具指标缺失也要补
        三者的并集就是 retry_names。

    【★ 为什么要 deepcopy(rows)】
        原 result 可能是数据库里读出来的、或者被别的请求共用。
        直接改会污染原数据。deepcopy 后随便改。

    【★ 为什么 retry_count == 0 要抛异常】
        "没有任何要重试的指标"说明用户点了按钮但没意义。
        抛异常让前端给出明确提示，而不是返回一个和原来一样的结果。
    """
    # ★ 深拷贝：不污染传入的 result
    rows = copy.deepcopy(result.get("cases") or [])
    if not rows:
        raise ValueError("原评测没有可用于重评分的逐题现场")

    # ── 逐条算出"要重试哪些指标" ──
    retry_by_case: list[set[str]] = []
    for row in rows:
        scores = row.get("ragas") or {}
        errors = row.get("ragas_errors") or {}
        # ① 上次失败的
        retry_names = {name for name in errors if name in (*CORE_RAGAS_METRICS, *AGENT_RAGAS_METRICS)}
        # ② 必需但没分数的
        retry_names.update(name for name in REQUIRED_RAGAS_METRICS if name not in scores)
        # ③ 工具指标（有期望答案才补）
        expected_calls = row.get("expected_tool_calls") or []
        if expected_calls:
            retry_names.update(
                name for name in ("tool_call_accuracy", "tool_call_f1") if name not in scores
            )
        retry_by_case.append(retry_names)

    retry_count = sum(bool(names) for names in retry_by_case)
    if retry_count == 0:
        raise ValueError("原评测没有失败或缺失的 Judge 指标")

    # ── 初始化 judge ──
    judge_error: str | None = None
    if scorer is None:
        try:
            scorer = RagasCollectionsScorer()
        except RagasUnavailableError as exc:
            judge_error = str(exc)

    started = time.perf_counter()
    completed = 0

    # ── 逐条重打分 ──
    for index, (row, retry_names) in enumerate(zip(rows, retry_by_case, strict=True), start=1):
        # 这条不需要重试 → 跳过（但 index 继续，保证 case_id 对得上）
        if not retry_names:
            continue

        await _emit_progress(
            progress_callback,
            stage="rescore",
            message=f"正在重试第 {completed + 1}/{retry_count} 条异常评分。",
            current_case=index,
            case_id=str(row.get("id") or f"case-{index}"),
            completed_cases=completed,
            total_cases=retry_count,
        )

        existing_errors = dict(row.get("ragas_errors") or {})

        # judge 不可用 → 把错误写进每条指标
        if scorer is None:
            for name in retry_names:
                existing_errors[name] = f"RagasUnavailableError: {judge_error or 'Judge unavailable'}"
            row["ragas_errors"] = existing_errors
            completed += 1
            continue

        # 没有上下文 → 无法打分（写明确错误）
        contexts = [str(value) for value in row.get("retrieved_contexts") or [] if str(value).strip()]
        if not contexts:
            for name in retry_names:
                existing_errors[name] = "ValueError: no retrieved contexts"
            row["ragas_errors"] = existing_errors
            completed += 1
            continue

        # ★ 只跑 retry_names 里的指标
        scores, errors = await scorer.score(
            user_input=str(row.get("question") or ""),
            response=str(row.get("response") or ""),
            retrieved_contexts=contexts,
            reference=str(row.get("reference") or ""),
            tool_calls=[dict(item) for item in row.get("tool_calls") or [] if isinstance(item, dict)],
            tool_observations=[
                dict(item) for item in row.get("tool_observations") or [] if isinstance(item, dict)
            ],
            expected_tool_calls=[
                dict(item) for item in row.get("expected_tool_calls") or [] if isinstance(item, dict)
            ],
            metric_names=retry_names,
        )

        # ── 合并：新分数覆盖旧的，成功的从 errors 里移除 ──
        merged_scores = dict(row.get("ragas") or {})
        merged_scores.update(scores)
        for name in retry_names:
            existing_errors.pop(name, None)
        existing_errors.update(errors)
        row["ragas"] = merged_scores
        row["ragas_errors"] = existing_errors
        # 重试后就不再是"跳过"状态了
        row["ragas_skipped_reason"] = None
        completed += 1

        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {completed}/{retry_count} 条异常评分已重试。",
            current_case=index,
            case_id=str(row.get("id") or f"case-{index}"),
            completed_cases=completed,
            total_cases=retry_count,
        )

    # ── 汇总（沿用原结果的 retrieval，只更新 performance）──
    performance = dict(result.get("performance") or {})
    performance.update(
        {
            # 标记"这次没重跑 Agent"
            "agent_reused": True,
            "rescore_cases": retry_count,
            "rescore_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
    top_k = int((result.get("pipeline") or {}).get("top_k") or 5)

    return _finalize_result(
        rows=rows,
        retrieval=dict(result.get("retrieval") or {}),
        performance=performance,
        # 重评分一定是"要求 judge"的（否则没意义）
        judge_requested=True,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str((result.get("target") or {}).get("repo_id") or ""),
        limit=top_k,
        evaluation_set=dict(result.get("evaluation_set") or {}),
        # ★ 标记这是一次重评分
        rescore_of=eval_id,
    )
