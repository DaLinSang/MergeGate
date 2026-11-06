"""
上下文压缩与 token 预算
======================

【业务定位】
    LLM 的上下文窗口是硬上限。对话一长、工具结果一多就会爆。
    本文件负责整条"省 token"链路：算 token → 算窗口 → 判压力 → 分预算 → 逐级压缩 → 提炼记忆。

    为什么需要这一层：
        Agent 一轮可能调 5 个工具，每个返回几千字。如果不裁剪，
        第 6 轮就必然超出窗口，模型直接报错，对话中断。

【数据流向】
    上游（谁调它）：
        services/chat_memory.py        → 组装上下文时做预算与裁剪（调用最多）
        services/agents/chat_agent.py  → 每轮算 pressure、判断是否溢出、取 prompt cache 参数
        services/rag/knowledge_base.py → 压缩检索结果
        services/rag/retrieval.py      → 压缩检索结果
        services/evals/runner.py       → 评测时算压缩率
        services/llm/client.py         → prompt cache 参数
    下游（它依赖谁）：
        app/core/config.py 的 settings → 所有阈值、比例、开关
    注意：本文件是【纯函数库】—— 不碰数据库、不发 HTTP、不读文件。
         所以它是整个 services 里最好测、最安全的一块。

【核心规则】
    1. 估算永远偏保守：宁可多裁一点，也不能算少了导致真实请求超限。
    2. 裁剪必须保持 tool_use / tool_result 成对，否则 OpenAI/Anthropic 直接 400。
       —— 这条由 adjust_keep_start_to_preserve_api_invariants() 保证。
    3. 压缩失败必须有兜底：fallback_compaction_summary() 纯规则生成摘要，
       绝不允许"因为压缩失败导致整轮对话挂掉"。

【本文件函数地图】（追源码时按 ①→⑨ 顺序读，就是一条完整数据流）
    ① 算 token      estimate_tokens()
    ② 算窗口        model_context_window_tokens() → effective_context_window_tokens()
    ③ 判压力        calculate_context_pressure() → _thresholds_for_window()
    ④ 分预算        default_context_budget() → ContextBudget
    ⑤ 轻量压缩      microcompact_messages() → is_regenerable_context_message()
    ⑥ 消息压缩      compress_messages() → clip_to_token_budget()
    ⑦ 结果压缩      compress_retrieval_results() → extract_relevant_snippet()
    ⑧ 摘要压缩      build_compaction_summary_payload()
                    → ensure_compaction_summary_preserves_key_facts()
                    → format_compaction_summary() / fallback_compaction_summary()
    ⑨ 记忆提炼      structured_memory_from_model() / fallback_memory_update()
                    → merge_memory() → render_structured_memory()

    辅助工具（被上面反复调用，建议先读这几个）：
        clip_to_token_budget()         截断到指定 token 预算
        _unique_preserve_order()       去重且保序（记忆字段到处在用）
        _message_meta() / _content_blocks() / _tool_use_ids() / _tool_result_ids()
                                       兼容多种消息格式的取值器
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import settings


# ════════════════════════════════════════════════════════════════
# ① 基础正则与常量
# ════════════════════════════════════════════════════════════════

# token 切分：一个中文字 / 一段英文单词 / 单个符号，各算一个"原子"
# 为什么要自己写而不装 tiktoken：不同 provider 的分词器不同，
# 装一堆依赖不如用一个"偏保守"的统一估算（见 estimate_tokens 的换算规则）。
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_\-]+|[^\s]")

# 词切分：用于关键词匹配（extract_relevant_snippet 里算句子相关性）
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+")

# 句子切分：中英文句号/问号/感叹号/换行后跟空白处断开
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?\n\u3002\uff01\uff1f])\s+")

# 各模型的【输入】上下文窗口（token）。
# 查不到时用 settings 里的保守默认值兜底，不要在这里瞎猜新模型。
MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
}

# 各模型的【输出】上限（token）。
# 用途：算"要给回答预留多少空间"，输入预算 = 窗口 - 预留。
MODEL_OUTPUT_WINDOWS = {
    "deepseek-v4-pro": 384_000,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4.1": 32_768,
    "gpt-4.1-mini": 32_768,
    "gpt-4.1-nano": 32_768,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "o1": 100_000,
    "o1-mini": 65_536,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
    "claude-3-5-sonnet": 8_192,
    "claude-3-7-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-opus-4": 64_000,
}

# 【可以清空的工具结果】白名单。
# 判断依据是"结果能靠重跑工具拿回来"，所以清掉不会丢信息。
# 反过来：写操作（action_drafts）、安全审查这类结果绝不能清，所以不在名单里。
COMPACTABLE_TOOL_HINTS = (
    "workspace.read_file",
    "workspace.search_code",
    "workspace.list_files",
    "rag.search_similar_documents",
    "devflow_search_evidence",
    "repo_health.generate_summary",
    "ci_debug_agent.analyze_workflow_run",
    "read",
    "grep",
    "search",
    "shell",
    "web_fetch",
    "webfetch",
)

# 压缩边界的消息标记：一条 role=system 且 meta.subtype=compact_boundary 的消息，
# 表示"这之前的对话已经被压缩成摘要了"。
# 有了它，后续只需要从边界往后读，不必重新解析整段历史。
COMPACT_BOUNDARY_SUBTYPE = "compact_boundary"

# 被清空的工具结果替换成的占位文本。
# 带 Recoverable 提示是刻意的：让模型知道"这不是没查到，是可以重跑拿回来的"。
MICROCOMPACT_MARKER = "[Old regenerable tool/context result cleared by microcompact]"


# 上下文压力等级（由轻到重）
#   normal        正常，什么都不用做
#   warning       提醒，但不压缩
#   auto_compact  自动压缩
#   aggressive    激进压缩（保留更少）
#   manual_path   走手动路径（用户点了"压缩"，或已接近硬上限）
PressureStage = Literal["normal", "warning", "auto_compact", "aggressive", "manual_path"]


# ════════════════════════════════════════════════════════════════
# ① 算 token：estimate_tokens
# ════════════════════════════════════════════════════════════════

def estimate_tokens(text: str | None) -> int:
    """估算一段文本的 token 数（不依赖任何分词器）。

    【调用链】无 —— 这是最底层的叶子函数，全文件都靠它。
    【被谁调用】几乎每个函数：clip_to_token_budget / compress_messages /
               microcompact_messages / compress_retrieval_results / 各统计逻辑

    【换算规则】（刻意偏保守，宁多不少）
        中文字符     → 1 字算 1 token（真实约 1~1.5，取小值 → 估少 → 有风险？
                                              不，中文一字一 token 已是下限，配合下面英文偏多算来平衡）
        英文/数字串  → 4 个字符算 1 token（向上取整，如 5 字符算 2）
        其他符号     → 1 个算 1 token

    为什么不用 tiktoken：
        项目支持 OpenAI / Claude / DeepSeek 多家，各家分词器不同。
        统一用一个保守估算，比"装 5 个分词器还各算各的"更稳、更简单。
    """
    # 空值直接返回 0（很多调用点会传 None）
    if not text:
        return 0

    total = 0
    # 逐个"原子"累加
    for token in TOKEN_PATTERN.findall(text):
        # 中文单字：1 token
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            total += 1
        # 英文单词 / 数字 / 下划线连写：按 4 字符 ≈ 1 token 换算，最少 1
        elif re.fullmatch(r"[a-zA-Z0-9_\-]+", token):
            total += max(1, (len(token) + 3) // 4)
        # 其余符号（括号、标点、emoji 片段）：1 token
        else:
            total += 1
    return total


def is_context_overflow_error(exc: Exception) -> bool:
    """判断一个异常是不是"上下文超长"。

    【调用链】无 —— 纯字符串匹配。
    【被谁调用】services/agents/chat_agent.py::_graph_model_decision()
               （捕获异常后判断：是超长就压缩重试一次，是别的错就直接抛）

    为什么要靠"消息文本"判断而不是异常类型：
        OpenAI 兼容网关（包括各种代理、企业网关）不会抛统一异常类型，
        只能靠错误文案里的特征词识别。所以这里刻意用一组"窄"的特征词，
        并且调用方只允许重试【一次】—— 判错也不会死循环。
    """
    message = str(exc or "").lower()
    # 各家 provider / 网关的典型文案（全部小写比较）
    markers = (
        "context_length_exceeded",
        "maximum context length",
        "prompt is too long",
        "prompt too long",
        "context window",
        "too many tokens",
        "request too large",
    )
    return any(marker in message for marker in markers)


# ════════════════════════════════════════════════════════════════
# ② 算窗口：这个模型到底能装多少
# ════════════════════════════════════════════════════════════════

def _normalize_model_name(model: str | None) -> str:
    """把模型名归一化，方便查表。

    【调用链】被 _model_lookup() 调用。
    【做什么】
        1. 空值 → 用 settings.llm_model，再不行返回 "unknown"
        2. 去掉 provider 前缀："openai/gpt-4o" → "gpt-4o"
        3. 去掉日期后缀："gpt-4o-2024-08-06" → "gpt-4o"

    为什么要去日期后缀：
        同一模型不同快照窗口一样，但名字带日期，不去掉就查不到表。
    """
    value = str(model or settings.llm_model or "").strip().lower()
    if not value:
        return "unknown"
    # 去掉 "provider/" 前缀
    value = value.split("/")[-1]
    # 去掉 "-2024-08-06" / "-20240806" 这类快照后缀
    value = re.sub(r"-(20\d{2}[-_]\d{2}[-_]\d{2}|\d{4}[-_]\d{2}[-_]\d{2})$", "", value)
    return value


def _model_lookup(model: str | None, table: dict[str, int]) -> int | None:
    """在窗口表里查模型，先精确匹配，再前缀匹配。

    【调用链】_normalize_model_name() → 本函数 → 返回 int 或 None

    为什么要有前缀匹配：
        "gpt-4o-2024-08-06" 归一化后仍可能带变体后缀，
        用 startswith 能兜住 "gpt-4o-audio-preview" 这类同窗口的变体。
    """
    normalized = _normalize_model_name(model)
    # 精确命中
    if normalized in table:
        return table[normalized]
    # 前缀命中（取第一个匹配，依赖字典插入顺序 —— 表里都是精确型号，顺序无影响）
    for key, value in table.items():
        if normalized.startswith(key):
            return value
    return None


def model_context_window_tokens(model: str | None = None) -> int:
    """取模型的输入窗口（token）。

    【调用链】_model_lookup() → 本函数
    【被谁调用】effective_context_window_tokens() / default_context_budget()

    优先级：
        1. settings.context_model_window_tokens（人工强制覆盖，>0 才生效）
        2. 内置窗口表
        3. settings 的 max_input + reserved_response 兜底
    最低不低于 4096 —— 防止配置写错导致预算变成 0，整个系统没法用。
    """
    # 人工覆盖优先（调试/特殊模型时用）
    if settings.context_model_window_tokens:
        return max(4096, settings.context_model_window_tokens)
    known = _model_lookup(model, MODEL_CONTEXT_WINDOWS)
    if known:
        return known
    # 兜底：配置里的输入上限 + 给回答预留的空间
    return max(4096, settings.context_max_input_tokens + settings.context_reserved_response_tokens)


def model_output_window_tokens(model: str | None = None) -> int:
    """取模型的输出上限（token）。

    【调用链】_model_lookup() → 本函数
    【被谁调用】effective_context_window_tokens()

    为什么需要它：输入预算 = 窗口 - 输出预留。
    给回答留的空间不够，模型会把答案截断在半句话。
    """
    known = _model_lookup(model, MODEL_OUTPUT_WINDOWS)
    if known:
        return known
    return max(512, settings.context_reserved_response_tokens)


def effective_context_window_tokens(model: str | None = None, *, for_compaction: bool = False) -> int:
    """算出"实际可用的输入预算"（= 模型窗口 - 给回答留的空间）。

    【调用链】
        model_context_window_tokens() → 拿总窗口
        model_output_window_tokens()  → 拿输出上限（for_compaction=True 时用）
        → 本函数返回可用输入预算

    【被谁调用】calculate_context_pressure()

    参数 for_compaction 的区别（重要，容易看错）：
        False（常规对话）：预留 = min(配置值, 窗口的一半)
            —— 保证回答有足够空间，但不会把一半窗口都占掉。
        True（准备压缩）：预留 = min(模型输出上限, 配置的压缩预留)
            —— 压缩场景希望"能用多少用多少"，所以预留取更小的值。
    """
    model_window = model_context_window_tokens(model)

    if for_compaction:
        # 压缩场景：预留尽量小，把窗口尽量让给输入
        reserved = min(model_output_window_tokens(model), max(512, settings.context_compact_reserved_output_tokens))
    else:
        # 常规对话：预留不超过窗口的一半（防止小窗口模型完全没有输入空间）
        reserved = max(512, min(settings.context_reserved_response_tokens, model_window // 2))

    dynamic_window = max(4096, model_window - reserved)

    # 开启动态窗口：可用预算至少是配置的 max_input（允许用满模型能力）
    if settings.context_dynamic_model_window_enabled:
        return max(settings.context_max_input_tokens, dynamic_window)
    # 关闭动态窗口：不超过配置上限（省钱/控成本模式）
    return max(4096, min(settings.context_max_input_tokens, dynamic_window))


# ════════════════════════════════════════════════════════════════
# ③ 判压力：当前用了多少、该不该压缩
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContextPressure:
    """一次"上下文压力"评估的快照（不可变）。

    【业务含义】告诉调用方：现在用了多少 token、处在哪个等级、还剩多少。
    【谁产出】calculate_context_pressure()
    【谁消费】
        services/agents/chat_agent.py → 决定要不要先压缩再提问
        前端"上下文用量"进度条 → as_dict() 序列化后直接返回

    为什么用 frozen dataclass：
        这是个纯数据快照，不允许被下游改动，避免"读完又被改"的隐蔽 bug。
    """

    stage: PressureStage              # 压力等级：normal / warning / auto_compact / aggressive / manual_path
    token_usage: int                  # 当前已用 token
    effective_window_tokens: int      # 实际可用预算（窗口 - 输出预留）
    remaining_tokens: int             # 剩余可用 token（最低 0）
    warning_threshold: int            # 进入 warning 的阈值
    auto_compact_threshold: int       # 进入 auto_compact 的阈值
    aggressive_threshold: int         # 进入 aggressive 的阈值
    manual_threshold: int             # 进入 manual_path 的阈值
    utilization: float                # 使用率 0~1（保留 4 位小数）

    @property
    def should_warn(self) -> bool:
        """要不要给用户提示"上下文快满了"。只要不是 normal 就提示。"""
        return self.stage != "normal"

    @property
    def should_auto_compact(self) -> bool:
        """要不要自动压缩。后三档都要压。"""
        return self.stage in {"auto_compact", "aggressive", "manual_path"}

    def as_dict(self) -> dict[str, Any]:
        """转成可直接 JSON 序列化的字典（给前端/日志用）。"""
        return {
            "stage": self.stage,
            "token_usage": self.token_usage,
            "effective_window_tokens": self.effective_window_tokens,
            "remaining_tokens": self.remaining_tokens,
            "warning_threshold": self.warning_threshold,
            "auto_compact_threshold": self.auto_compact_threshold,
            "aggressive_threshold": self.aggressive_threshold,
            "manual_threshold": self.manual_threshold,
            "utilization": self.utilization,
        }


def _thresholds_for_window(window: int) -> tuple[int, int, int, int]:
    """按窗口大小算出四级阈值。

    【调用链】被 calculate_context_pressure() 调用。
    【被谁调用】calculate_context_pressure()

    返回：(warning, auto_compact, aggressive, manual)

    设计思路 —— "从窗口往回留余量"而不是"从 0 往上按百分比"：
        窗口越大，绝对余量越该大（因为一段工具输出也更大）。
        但余量又不能超过窗口的 1/4，否则小窗口模型永远进不了 warning。

        每档的余量上限：warning 1/4、auto 1/5、aggressive 1/8、manual 1/16
        —— 越往后越贴近硬上限，所以余量越小。
    """
    # 各档余量：取"配置值"与"窗口比例"的较小者，并保证至少几百 token
    warning_buffer = min(settings.context_warning_buffer_tokens, max(512, window // 4))
    auto_buffer = min(settings.context_auto_compact_buffer_tokens, max(512, window // 5))
    aggressive_buffer = min(settings.context_aggressive_buffer_tokens, max(512, window // 8))
    manual_buffer = min(settings.context_manual_buffer_tokens, max(256, window // 16))

    # auto 阈值 = 窗口 - auto 余量（这是主基准线）
    auto_threshold = max(1024, window - auto_buffer)
    # warning 必须严格小于 auto（用 -1 保证单调递增，否则 stage 判定会乱）
    warning_threshold = max(512, min(auto_threshold - 1, auto_threshold - warning_buffer))
    # aggressive 必须严格大于 auto
    aggressive_threshold = max(auto_threshold + 1, window - aggressive_buffer)
    # manual 必须严格大于 aggressive
    manual_threshold = max(aggressive_threshold + 1, window - manual_buffer)

    return warning_threshold, auto_threshold, aggressive_threshold, manual_threshold


def calculate_context_pressure(token_usage: int, model: str | None = None) -> ContextPressure:
    """评估当前上下文压力等级。

    【调用链】追这个函数会进入：
        1. effective_context_window_tokens()  → 本文件，算可用预算
        2. _thresholds_for_window()           → 本文件，算四级阈值
        3. ContextPressure(...)               → 本文件，组装结果

    【被谁调用】
        services/agents/chat_agent.py  → 每轮提问前判断是否要先压缩
        services/chat_memory.py        → 组装上下文后回填 pressure 给前端

    【数据流向】
        入：token_usage（由 estimate_tokens 累加得出）+ model 名
        出：ContextPressure → as_dict() → 前端进度条 / 决定压缩策略

    【判定顺序】从高到低，命中即返回 —— 所以阈值必须严格递增（见 _thresholds_for_window）。
    """
    # 注意：这里用 for_compaction=True —— 评估压力是为了"要不要压缩"，
    # 所以按压缩场景的预算算（预留更小，更容易触发压缩，更安全）
    effective_window = effective_context_window_tokens(model, for_compaction=True)
    warning_threshold, auto_threshold, aggressive_threshold, manual_threshold = _thresholds_for_window(effective_window)

    # ── 从最高档往下判，命中即停 ──
    if token_usage >= manual_threshold:
        stage: PressureStage = "manual_path"
    elif token_usage >= aggressive_threshold:
        stage = "aggressive"
    elif token_usage >= auto_threshold:
        stage = "auto_compact"
    elif token_usage >= warning_threshold:
        stage = "warning"
    else:
        stage = "normal"

    return ContextPressure(
        stage=stage,
        token_usage=token_usage,
        effective_window_tokens=effective_window,
        remaining_tokens=max(0, effective_window - token_usage),
        warning_threshold=warning_threshold,
        auto_compact_threshold=auto_threshold,
        aggressive_threshold=aggressive_threshold,
        manual_threshold=manual_threshold,
        # 用率四舍五入到 4 位，避免前端显示 0.30000000000000004
        utilization=round(token_usage / effective_window, 4) if effective_window else 0.0,
    )


# ════════════════════════════════════════════════════════════════
# 压缩边界：标记"这之前的对话已经压成摘要了"
# ════════════════════════════════════════════════════════════════

def is_compact_boundary_message(message: dict[str, Any]) -> bool:
    """判断这条消息是不是"压缩边界"标记。

    【调用链】被 find_last_compact_boundary_index() / build_compaction_summary_payload() 调用

    边界的定义：role=system 且 meta.subtype="compact_boundary"。
    它是一条"假消息"，不参与对话，只作为分界线存在。
    """
    # 兼容 meta / metadata 两种字段名（不同来源的消息格式不一样）
    meta = message.get("meta") or message.get("metadata") or {}
    return str(message.get("role") or "") == "system" and meta.get("subtype") == COMPACT_BOUNDARY_SUBTYPE


def find_last_compact_boundary_index(messages: list[dict[str, Any]]) -> int:
    """从后往前找最后一条压缩边界，返回下标；找不到返回 -1。

    【调用链】is_compact_boundary_message() → 本函数
    【被谁调用】messages_after_compact_boundary()

    为什么从后往前：只需要最近的那条边界，从后往前找到就能立刻返回。
    """
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[index]):
            return index
    return -1


def messages_after_compact_boundary(messages: list[dict[str, Any]], *, include_boundary: bool = True) -> list[dict[str, Any]]:
    """只保留最近一次压缩边界之后的消息。

    【调用链】find_last_compact_boundary_index() → 本函数
    【被谁调用】services/chat_memory.py（组装上下文时，避免重复塞已压缩的旧历史）

    参数 include_boundary：
        True  → 边界消息本身也保留（摘要内容就在这条消息里）
        False → 只要边界之后的（纯对话部分）
    没有边界时原样返回 —— 表示"从来没压缩过"，全部历史都要带。
    """
    boundary_index = find_last_compact_boundary_index(messages)
    if boundary_index < 0:
        return messages
    start = boundary_index if include_boundary else boundary_index + 1
    return messages[start:]


# ════════════════════════════════════════════════════════════════
# 消息格式兼容层：不同 provider 的消息结构不一样，统一在这里取值
# ════════════════════════════════════════════════════════════════

def _message_meta(message: dict[str, Any]) -> dict[str, Any]:
    """安全取出消息的 meta 字典（兼容 meta / metadata，非 dict 一律当空）。"""
    meta = message.get("meta") or message.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _message_provider_id(message: dict[str, Any]) -> str | None:
    """取 provider 侧的消息 ID（OpenAI 的 message_id / Anthropic 的 message id）。

    【调用链】_message_meta() → 本函数
    【被谁调用】adjust_keep_start_to_preserve_api_invariants()

    用途：某些 provider 要求 assistant 消息 + 其 tool_calls 必须连在一起，
    靠 provider_id 识别"这几条属于同一次 assistant 输出"。
    """
    meta = _message_meta(message)
    value = (
        message.get("message_id")
        or message.get("provider_message_id")
        or meta.get("message_id")
        or meta.get("provider_message_id")
    )
    return str(value) if value else None


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """把 content 统一成"块列表"。

    【调用链】被 _tool_use_ids() / _tool_result_ids() 调用

    Anthropic 风格：content = [{type: "text", ...}, {type: "tool_use", id: ...}]
    OpenAI 风格：  content 是纯字符串，工具调用放在 message["tool_calls"]
    这里只处理"块列表"的情况，纯字符串返回空列表（调用方另有分支处理）。
    """
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _tool_use_ids(message: dict[str, Any]) -> set[str]:
    """收集这条消息里【发起】的工具调用 ID。

    【调用链】_content_blocks() / _message_meta() → 本函数
    【被谁调用】adjust_keep_start_to_preserve_api_invariants()

    要兼容三种写法（同一份数据可能来自 OpenAI、Anthropic 或本地缓存）：
        1. message["tool_calls"][*]["id"]          —— OpenAI 风格
        2. content 块里 type="tool_use" 的 id      —— Anthropic 风格
        3. meta 里的 tool_call_id / tool_use_id    —— 本地归一化后的格式
    """
    ids: set[str] = set()

    # 来源 1：OpenAI 风格 tool_calls
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            value = (
                call.get("id")
                or call.get("tool_call_id")
                or call.get("tool_use_id")
                or call.get("call_id")
            )
            if value:
                ids.add(str(value))

    # 来源 2：Anthropic 风格 content 块
    for block in _content_blocks(message):
        if block.get("type") == "tool_use" and block.get("id"):
            ids.add(str(block["id"]))

    # 来源 3：meta 字段（只对 assistant / tool_use 角色生效，避免误收 tool 消息的 ID）
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and str(message.get("role") or "") in {"assistant", "tool_use"}:
            ids.add(str(meta[key]))

    return ids


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
    """收集这条消息【回填】的工具结果 ID（即它在回答哪次工具调用）。

    【调用链】_content_blocks() / _message_meta() → 本函数
    【被谁调用】adjust_keep_start_to_preserve_api_invariants()

    与 _tool_use_ids 是一对：一个找"提问"，一个找"回答"。
    裁剪时如果只留下了 result 而把 use 裁掉，API 会报 400（孤儿工具结果）。
    """
    ids: set[str] = set()
    role = str(message.get("role") or "")

    # 来源 1：顶层字段（OpenAI 的 tool 消息就是这样）
    for key in ("tool_call_id", "tool_use_id"):
        value = message.get(key)
        if value:
            ids.add(str(value))

    # 来源 2：Anthropic 风格 content 块
    for block in _content_blocks(message):
        if block.get("type") == "tool_result" and block.get("tool_use_id"):
            ids.add(str(block["tool_use_id"]))

    # 来源 3：meta 字段（tool / tool_result / user 角色才可能承载工具结果）
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and role in {"tool", "tool_result", "user"}:
            ids.add(str(meta[key]))

    return ids


def adjust_keep_start_to_preserve_api_invariants(messages: list[dict[str, Any]], start_index: int) -> int:
    """把"保留起点"往左移，直到满足 API 的成对性约束。

    【调用链】追这个函数会进入：
        1. _tool_result_ids()  → 收集保留区里的"工具结果 ID"
        2. _tool_use_ids()     → 收集保留区里的"工具调用 ID"
        3. _message_provider_id() → 处理 assistant 连续段的合并
    【被谁调用】
        microcompact_messages()  → 计算保留区起点时调用
        services/chat_memory.py  → 硬裁剪历史时调用

    【要解决的两个 API 约束】
        约束 A：tool_result 必须有对应的 tool_use。
            如果裁剪线切在中间，留下 result 却把 use 裁掉了，API 直接 400。
            → 做法：找出"缺 use 的 result"，往左回溯把发起它的那条消息也纳入保留区。
        约束 B：某些 provider 要求 assistant 的连续多条（含 tool_calls）不能被拆开。
            → 做法：如果保留区第一条是 assistant，且它的 provider_id 与前面那条相同，
              就把前面那条也拉进来。

    【数据流向】
        入：messages（全量历史）、start_index（想从哪里开始保留）
        出：调整后的 start_index（只会变小，不会变大）

    返回值的边界：小于等于 0 或大于等于长度时，直接夹取到合法范围，不做调整。
    """
    # 边界情况：起点越界，直接夹取（0 或末尾），无需调整
    if start_index <= 0 or start_index >= len(messages):
        return max(0, min(start_index, len(messages)))

    adjusted = start_index

    # ── 步骤 1：统计保留区里已有的 tool_use / tool_result ──
    kept_results: set[str] = set()
    kept_uses: set[str] = set()
    for message in messages[adjusted:]:
        kept_results.update(_tool_result_ids(message))
        kept_uses.update(_tool_use_ids(message))

    # ── 步骤 2：处理约束 A —— 有结果但没调用，往左把调用方拉进来 ──
    missing_uses = kept_results - kept_uses
    if missing_uses:
        for index in range(adjusted - 1, -1, -1):
            uses = _tool_use_ids(messages[index])
            if uses & missing_uses:
                adjusted = index
                missing_uses -= uses
                # 全部补齐就停（不要无谓地扩大保留区，那会浪费预算）
                if not missing_uses:
                    break

    # ── 步骤 3：处理约束 B —— assistant 连续段不能被拆开 ──
    # 先收集保留区里所有 assistant 的 provider_id
    kept_provider_ids = {
        provider_id
        for message in messages[adjusted:]
        if str(message.get("role") or "") == "assistant"
        for provider_id in [_message_provider_id(message)]
        if provider_id
    }
    # 如果前一条也是 assistant 且属于同一批（provider_id 在保留区里），一起拉进来
    while adjusted > 0:
        previous = messages[adjusted - 1]
        if str(previous.get("role") or "") != "assistant":
            break
        previous_provider_id = _message_provider_id(previous)
        if not previous_provider_id or previous_provider_id not in kept_provider_ids:
            break
        adjusted -= 1

    return adjusted


# ════════════════════════════════════════════════════════════════
# 文本工具：切词、去重保序、按 token 截断
# ════════════════════════════════════════════════════════════════

def _words(text: str) -> list[str]:
    """切词并转小写（用于关键词相关性打分）。"""
    return WORD_PATTERN.findall((text or "").lower())


def _unique_preserve_order(items: list[str], limit: int = 12) -> list[str]:
    """去重并保持原顺序，最多返回 limit 条。

    【调用链】无 —— 纯工具函数，但被大量调用：
        microcompact_messages（工具名去重）
        structured_memory_from_model / merge_memory（记忆字段去重）
        fallback_compaction_summary（决策/任务/事实去重）

    为什么不用 set：
        set 会打乱顺序。记忆是有先后的（越晚的越重要），顺序不能丢。
    为什么用 lower() 做去重键：
        "Fix the bug" 和 "fix the bug" 应视为同一条记忆。
    为什么先 normalize 空白：
        多行文本合并后常有连续空格/换行，不去掉会导致"看起来一样但 key 不同"。
    """
    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        # 归一化：把任意连续空白压成单空格
        clean = " ".join(str(item or "").strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        # 达到上限就提前退出（避免处理无用数据）
        if len(output) >= limit:
            break

    return output


def clip_to_token_budget(text: str, max_tokens: int, suffix: str = "...") -> str:
    """把文本截断到指定 token 预算内，超出部分用 suffix 代替。

    【调用链】estimate_tokens() → 本函数（循环校验直到符合预算）
    【被谁调用】几乎每个压缩函数：compress_messages / extract_relevant_snippet /
               compress_retrieval_results / 各种摘要与记忆字段

    实现要点：
        1. 先用 4 字符≈1 token 粗切（比逐 token 试快得多）
        2. 再循环回退，每次砍 24 字符，直到加上 suffix 也不超预算
        3. 为什么每次砍 24：小步回退能贴近上限，不会一次砍太多浪费预算

    边界：
        max_tokens <= 0 或文本为空 → 返回空串
        本来就够短 → 原样返回（不做任何改动）
    """
    clean = str(text or "").strip()
    if max_tokens <= 0 or not clean:
        return ""
    # 已经够短，直接返回
    if estimate_tokens(clean) <= max_tokens:
        return clean

    # ── 粗切：按 4 字符 ≈ 1 token 估算初始长度（最少 32 字符）──
    approx_chars = max(32, max_tokens * 4)
    clipped = clean[:approx_chars].rstrip()

    # ── 精修：逐步回退，直到 suffix 也算进去仍不超预算 ──
    while clipped and estimate_tokens(clipped + suffix) > max_tokens:
        clipped = clipped[: max(0, len(clipped) - 24)].rstrip()

    return f"{clipped}{suffix}" if clipped else ""


# ════════════════════════════════════════════════════════════════
# ④ 分预算：把可用窗口切成 5 份
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContextBudget:
    """上下文预算分配方案（不可变）。

    【业务含义】把可用输入预算切给 5 个用途，防止某一类内容吃掉全部空间。
    【谁产出】default_context_budget()
    【谁消费】services/chat_memory.py（按份额裁剪各部分内容）

    五个份额对应 prompt 的五个部分：
        system          系统提示词（固定，不随对话变）
        memory          长期记忆摘要
        evidence        RAG 检索证据
        recent          最近对话
        tool_observation 工具调用结果
    """

    max_input_tokens: int            # 输入预算总上限
    reserved_response_tokens: int    # 给回答预留的空间
    system_tokens: int               # 系统提示词份额
    memory_tokens: int               # 记忆份额
    evidence_tokens: int             # 证据份额
    recent_tokens: int               # 最近对话份额
    tool_observation_tokens: int     # 工具结果份额

    @property
    def available_prompt_tokens(self) -> int:
        """真正能给 prompt 用的 token（= 总上限 - 回答预留），最低 512。"""
        return max(512, self.max_input_tokens - self.reserved_response_tokens)


def default_context_budget(model: str | None = None) -> ContextBudget:
    """按配置比例算出五个份额。

    【调用链】model_context_window_tokens() → 本函数
    【被谁调用】services/chat_memory.py::ContextAssembler、chat_agent.py

    算法：
        max_input = 动态窗口（开启动态时取模型窗口，否则取配置上限）
        reserved  = min(配置预留, max_input/2)   ← 预留不超过一半
        available = max_input - reserved
        各份额    = max(512, available × 配置比例)

    为什么每份都至少 512：
        比例可能很小（比如 system_ratio=0.02），算出来只有几十 token，
        连一句系统提示都放不下。设下限保证每部分都能表达完整意思。
    """
    # ── 步骤 1：确定输入上限 ──
    if settings.context_dynamic_model_window_enabled:
        max_input = max(settings.context_max_input_tokens, model_context_window_tokens(model))
    else:
        max_input = max(4096, settings.context_max_input_tokens)

    # ── 步骤 2：算出要给回答留多少（不超过上限一半）──
    reserved = max(512, min(settings.context_reserved_response_tokens, max_input // 2))
    available = max_input - reserved

    # ── 步骤 3：按比例切五份，每份保底 512 ──
    return ContextBudget(
        max_input_tokens=max_input,
        reserved_response_tokens=reserved,
        system_tokens=max(512, int(available * settings.context_system_ratio)),
        memory_tokens=max(512, int(available * settings.context_memory_ratio)),
        evidence_tokens=max(512, int(available * settings.context_evidence_ratio)),
        recent_tokens=max(512, int(available * settings.context_recent_ratio)),
        tool_observation_tokens=max(512, int(available * settings.context_tool_ratio)),
    )


# ════════════════════════════════════════════════════════════════
# ⑥ 消息压缩：按预算保留最近若干条
# ════════════════════════════════════════════════════════════════

def compress_messages(messages: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """把消息列表按预算裁剪，保留最近的若干条。

    【调用链】追这个函数会进入：
        1. clip_to_token_budget()  → 单条消息超长时截断
        2. estimate_tokens()       → 算每条消息的成本
    【被谁调用】
        services/chat_memory.py::ContextAssembler  ← 主路径（组装 prompt 前）
        services/evals/runner.py                   ← 评测时算压缩率

    【数据流向】
        入：messages（来自 chat_messages 表，已转成 dict 列表）、max_tokens（预算）
        出：(kept, stats)
            kept  → 直接拼进 prompt 的对话部分
            stats → 前端"上下文用量"面板 + 评测指标

    【关键规则】
        1. 从【后】往前保留 —— 最近的消息最重要，先丢最旧的。
        2. 单条消息最多占预算的 1/3（per_message_budget，96~900）——
           否则一条巨长的工具输出就能吃掉全部预算，其他消息全丢。
        3. 只处理 user / assistant 两类；tool 消息不进来（工具结果另有 microcompact 处理）。
        4. 至少保留 1 条（kept_reversed 为空时不 break），否则模型收到的对话是空的。
    """
    kept_reversed: list[dict[str, str]] = []
    used = 0
    truncated_messages = 0
    exact_suffix_messages = 0
    # exact_suffix_open 表示"目前保留的尾部还没有被截断过"，
    # 一旦出现截断就置 False —— 用于统计"最近几条是原文保留的"。
    exact_suffix_open = True

    # ── 从最新一条往前遍历，边算边装 ──
    for message in reversed(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")

        # 只保留 user / assistant，且必须有内容（空消息白占预算）
        if role not in {"user", "assistant"} or not content.strip():
            continue

        # ── 单条预算上限：min(900, 总预算/3)，最低 96 ──
        per_message_budget = max(96, min(900, max_tokens // 3))
        clipped = clip_to_token_budget(content, per_message_budget)

        # 成本 = 角色名的 token + 内容的 token
        cost = estimate_tokens(role) + estimate_tokens(clipped)

        # 已经装了至少一条，再加就超预算 → 停（保证至少留一条）
        if kept_reversed and used + cost > max_tokens:
            break

        kept_reversed.append({"role": role, "content": clipped})

        # ── 统计：区分"原文保留"和"被截断" ──
        if clipped == content.strip():
            # 只有连续的尾部原文才计入 exact_suffix
            if exact_suffix_open:
                exact_suffix_messages += 1
        else:
            truncated_messages += 1
            exact_suffix_open = False

        used += cost
        # 预算刚好用满，提前结束
        if used >= max_tokens:
            break

    # 因为是倒序装的，输出前要翻回来
    kept = list(reversed(kept_reversed))

    # 统计"本来有多少条可选"（用于算 omitted 数量）
    total_eligible_messages = sum(
        1
        for message in messages
        if str(message.get("role") or "") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    )

    return kept, {
        "input_messages": len(messages),                                  # 输入总条数
        "eligible_messages": total_eligible_messages,                     # 可参与压缩的条数
        "kept_messages": len(kept),                                       # 实际保留条数
        "omitted_messages": max(0, total_eligible_messages - len(kept)),  # 被整条丢弃的条数
        "truncated_messages": truncated_messages,                         # 被截断内容的条数
        "exact_suffix_messages": exact_suffix_messages,                   # 尾部原文保留的条数
        "estimated_tokens": used,                                         # 实际占用
        "budget_tokens": max_tokens,                                      # 预算
    }


# ════════════════════════════════════════════════════════════════
# ⑤ 轻量压缩：把"可重跑"的工具结果换成摘要
# ════════════════════════════════════════════════════════════════

def _tool_names(message: dict[str, Any]) -> list[str]:
    """取这条消息关联的工具名（route + 每个 tool_call 的名字）。

    【调用链】被 is_regenerable_context_message() / microcompact_messages() /
             build_compaction_summary_payload() 调用
    """
    names: list[str] = []
    # route 也当作"工具名"参与匹配（有些消息用 route 标记来源工具）
    route = str(message.get("route") or "").strip()
    if route:
        names.append(route)
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            name = str(call.get("tool_name") or "").strip()
            if name:
                names.append(name)
    return names


def is_regenerable_context_message(message: dict[str, Any]) -> bool:
    """判断这条消息是不是"可以清空、需要时重跑工具就能拿回来"。

    【调用链】_message_meta() / _tool_names() → 本函数
    【被谁调用】microcompact_messages()

    三个条件全部满足才算可清：
        1. 是工具结果（role=tool/tool_result，或 meta.subtype=tool_result）
        2. meta.regenerable 不是显式 False（调用方可以标记"这个别清"）
        3. 工具名命中 COMPACTABLE_TOOL_HINTS 白名单

    为什么用白名单而不是黑名单：
        黑名单会漏 —— 新增一个写操作工具忘了加黑名单，它的结果就被清掉了，
        模型会以为"操作没执行"。白名单默认安全。
    """
    role = str(message.get("role") or "")
    meta = _message_meta(message)

    # 条件 1：必须是工具结果
    if role not in {"tool", "tool_result"} and meta.get("subtype") != "tool_result":
        return False

    # 条件 2：调用方显式禁止清除
    if meta.get("regenerable") is False:
        return False

    # 条件 3：工具名命中白名单
    haystack = " ".join(_tool_names(message)).lower()
    if not haystack:
        return False
    return any(hint in haystack for hint in COMPACTABLE_TOOL_HINTS)


def microcompact_messages(
    messages: list[dict[str, Any]],
    query: str,
    *,
    keep_recent: int | None = None,
    max_excerpt_tokens: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """【轻量压缩】把旧的、可重跑的工具结果替换成"摘要 + 可恢复提示"。

    【调用链】追这个函数会进入：
        1. adjust_keep_start_to_preserve_api_invariants()
           → 算出"最近 N 条"的起点（并保证工具调用成对）
        2. is_regenerable_context_message()  → 判断某条能不能清
        3. extract_relevant_snippet()        → 从旧结果里抽一句最相关的留着
        4. _unique_preserve_order()          → 工具名去重

    【被谁调用】services/chat_memory.py（每轮组装上下文时先做轻量压缩）

    【数据流向】
        入：messages（全量历史）、query（当前问题，用来决定留哪句摘要）
        出：(新消息列表, 统计)
            新列表里旧工具结果的 content 被换成 MICROCOMPACT_MARKER + 摘要

    【和 compress_messages 的区别】（容易混，记住这句）
        microcompact：保留所有消息，只把旧工具结果【瘦身】—— 结构不变，成本更低
        compress_messages：直接【丢消息】—— 结构会变，用于最后兜底

    【为什么摘要要按 query 抽】
        清空后如果只留一句"已清空"，模型会丢关键线索。
        按当前问题抽最相关的 90 token，既省空间又不丢要点。
    """
    # 最近多少条不动（默认取配置）
    keep = keep_recent if keep_recent is not None else settings.context_microcompact_keep_recent_messages

    # ── 步骤 1：算保留区起点，并修正到满足 API 成对约束 ──
    keep_start = adjust_keep_start_to_preserve_api_invariants(
        messages,
        max(0, len(messages) - max(0, keep)),
    )

    output: list[dict[str, Any]] = []
    compacted = 0
    tokens_before = 0
    tokens_after = 0
    compacted_tools: list[str] = []

    # ── 步骤 2：逐条处理 ──
    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        before = estimate_tokens(content)
        tokens_before += before
        next_message = dict(message)

        # ── 步骤 3：三个条件同时满足才压缩 ──
        #   ① 在保留区之前（旧消息）
        #   ② 是可重跑的工具结果
        #   ③ 够长（> max_excerpt_tokens*2，太短的不值得动，避免无谓改写）
        if (
            index < keep_start
            and is_regenerable_context_message(message)
            and before > max_excerpt_tokens * 2
        ):
            names = _tool_names(message)
            # 按当前问题抽一句最相关的（留个线索）
            excerpt = extract_relevant_snippet(query, content, max_tokens=max_excerpt_tokens)
            tool_label = ", ".join(_unique_preserve_order(names, 4)) or "tool output"

            # 替换成占位文本：说明"清空了 + 来源工具 + 可以重跑拿回来"
            compacted_content = f"{MICROCOMPACT_MARKER}\nSource: {tool_label}\nRecoverable: rerun the corresponding workspace, RAG, or GitHub tool if exact details are needed."
            if excerpt:
                compacted_content += f"\nKey excerpt: {excerpt}"
            next_message["content"] = compacted_content

            compacted += 1
            compacted_tools.extend(names)

        tokens_after += estimate_tokens(str(next_message.get("content") or ""))
        output.append(next_message)

    return output, {
        "mode": "microcompact_regenerable_context",                        # 压缩模式标识
        "input_messages": len(messages),                                   # 输入条数
        "compacted_messages": compacted,                                   # 被瘦身的条数
        "tokens_before": tokens_before,                                    # 压缩前 token
        "tokens_after": tokens_after,                                      # 压缩后 token
        "estimated_tokens_saved": max(0, tokens_before - tokens_after),    # 节省量
        "kept_recent_messages": max(0, min(keep, len(messages))),          # 保留区条数
        "compacted_tools": _unique_preserve_order(compacted_tools, 12),    # 被清空结果的工具名
    }


# ════════════════════════════════════════════════════════════════
# ⑦ 结果压缩：按 query 抽取最相关的句子
# ════════════════════════════════════════════════════════════════

def extract_relevant_snippet(query: str, text: str, max_tokens: int = 220) -> str:
    """从长文本里抽出与 query 最相关的几句话（抽取式，不改写）。

    【调用链】追这个函数会进入：
        1. _words()              → 切词
        2. SENTENCE_SPLIT_PATTERN → 切句
        3. clip_to_token_budget() → 结果截断
    【被谁调用】
        microcompact_messages()      → 清空旧工具结果时留一句摘要
        compress_retrieval_results() → 压缩检索结果

    【数据流向】
        入：query（当前问题）、text（长文本，如一个文件内容/一条 Issue 正文）
        出：若干句拼成的短文本（≤ max_tokens）

    【打分规则】（纯关键词，不调模型 —— 便宜、快、可解释）
        基础分 = 命中的不同词数 × 3 + 各词出现次数之和
        额外分 = query 整串出现在句子里 +4（强信号：原文提到了问题本身）
        取分数 > 0 的前 4 句，再按【原顺序】拼回去 —— 顺序很重要，
        乱序拼出来的句子读起来是断的。

    【为什么要"抽取式"而不是"生成式"】
        这是每轮都要跑的路径，调模型会拖慢响应、增加成本。
        抽取式不产生幻觉，且能保留原始引用（可追溯）。
    """
    # 归一化：换行压成空格（后面按句子切）
    clean = " ".join(str(text or "").replace("\r", "\n").split())
    if not clean:
        return ""

    # 切出 query 的词；没有可用词就退化成"直接截断"
    terms = set(_words(query))
    if not terms:
        return clip_to_token_budget(clean, max_tokens)

    # ── 步骤 1：切句 ──
    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(clean) if part.strip()]
    if not sentences:
        sentences = [clean]

    # ── 步骤 2：给每句打分 ──
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        matched = [term for term in terms if term and term in lower]
        # 命中词种类越多越相关；同时考虑出现频次
        score = len(set(matched)) * 3.0 + sum(lower.count(term) for term in matched)
        # query 整串出现在句子里：强信号
        if query.lower().strip() and query.lower().strip() in lower:
            score += 4.0
        scored.append((score, index, sentence))

    # 一句都不相关 → 退化成"取开头"
    positive = [item for item in scored if item[0] > 0]
    if not positive:
        return clip_to_token_budget(clean, max_tokens)

    # ── 步骤 3：取 top4，再按原顺序还原 ──
    positive.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(positive[:4], key=lambda item: item[1])
    snippet = " ".join(sentence for _score, _index, sentence in selected)
    return clip_to_token_budget(snippet, max_tokens)


def compress_retrieval_results(
    query: str,
    items: list[dict[str, Any]],
    max_tokens_per_item: int = 260,
    max_total_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """压缩 RAG 检索结果（把每条证据的正文换成相关片段）。

    【调用链】追这个函数会进入：
        1. extract_relevant_snippet() → 单条抽取
        2. estimate_tokens()          → 算压缩前后 token

    【被谁调用】
        services/rag/knowledge_base.py  ← 检索流水线第 ⑤ 步 context_compression
        services/rag/retrieval.py       ← 检索后直接压缩
        services/evals/runner.py        ← 评测

    【数据流向】
        入：items（检索结果，含 content / _raw_content / snippet）
        出：新 items 列表，每条：
              snippet             ← 压缩后的片段
              compression         ← 压缩统计（比例、预算）
              retrieval.context_compression ← 同上（挂在检索明细里给前端看）
              _raw_content / content 被移除（避免二次进 prompt）

    【双重预算】
        每条：max_tokens_per_item
        总量：max_total_tokens（None 表示不限）
        总量快用完时提前 break —— 后面的条目直接不要，
        而不是塞一个只剩 10 token 的残句（那样等于噪声）。
    """
    compressed: list[dict[str, Any]] = []
    total_budget = max(0, int(max_total_tokens)) if max_total_tokens is not None else None
    used_tokens = 0

    for item in items:
        # ── 总量预算已用完 → 停 ──
        if total_budget is not None and used_tokens >= total_budget:
            break

        # ── 剩余空间太小（< min(24, 单条预算)）→ 停，不留残句 ──
        if (
            total_budget is not None
            and compressed
            and total_budget - used_tokens < min(24, max(1, int(max_tokens_per_item)))
        ):
            break

        # ── 单条预算 = min(单条上限, 剩余总量) ──
        item_budget = max(1, int(max_tokens_per_item))
        if total_budget is not None:
            item_budget = min(item_budget, total_budget - used_tokens)

        # 优先用 _raw_content（完整原文），其次 content，最后退回已有 snippet
        raw_text = str(item.get("_raw_content") or item.get("content") or item.get("snippet") or "")
        snippet = extract_relevant_snippet(query, raw_text, max_tokens=item_budget)
        original_tokens = estimate_tokens(raw_text)
        compressed_tokens = estimate_tokens(snippet)

        # ── 组装新条目：丢掉原始全文，只留片段 ──
        next_item = {key: value for key, value in item.items() if key not in {"_raw_content", "content"}}
        next_item["snippet"] = snippet
        next_item["compression"] = {
            "mode": "query_aware_extractive",                                  # 压缩方式
            "original_estimated_tokens": original_tokens,                      # 压缩前
            "compressed_estimated_tokens": compressed_tokens,                  # 压缩后
            "ratio": round(compressed_tokens / original_tokens, 4) if original_tokens else 1.0,  # 压缩比
            "item_budget_tokens": item_budget,                                 # 本条分到的预算
            "global_budget_tokens": total_budget,                              # 全局预算
            "global_tokens_before_item": used_tokens,                          # 处理本条前已用
        }

        # 同步写进 retrieval 明细（前端"检索明细"面板读这里）
        retrieval = dict(next_item.get("retrieval") or {})
        retrieval["context_compression"] = next_item["compression"]
        retrieval["selected_for_context"] = True
        next_item["retrieval"] = retrieval

        compressed.append(next_item)
        used_tokens += compressed_tokens

    return compressed


# ════════════════════════════════════════════════════════════════
# ⑧ 摘要压缩：把一整段历史压成"可继续工作的摘要"
# ════════════════════════════════════════════════════════════════

# 让模型写摘要的提示词。
# 关键设计：目标不是"减少 token"，而是"保住工作连续性"——
#   普通摘要会丢掉文件路径、命令、错误信息，导致压缩后 Agent 忘了在干什么。
# 所以这里明确要求保留 5 类内容，并要求"原文保留关键标识符"。
COMPACT_SUMMARY_SYSTEM_PROMPT = """请在上下文压缩后，为 DevFlow AI 创建一份可延续工作的摘要。
目标不是单纯减少 token，而是为软件工程 Agent 保留工作连续性。

请用纯文本返回以下部分：
1. 用户意图和明确要求。
2. 当前工作面：现在正在处理什么；如相关，请包含仓库、文件、函数、命令、测试和 UI 状态。
3. 持久决策和约束。
4. 重要事实、代码引用、错误和修复。
5. 待办任务和下一步，并以最近一次用户请求为依据。

当文件路径、标识符、日期、命令、错误消息和用户偏好很重要时，请保留原文。
不要编造细节。如果某个细节只存在于可以重新生成的工具输出中，请说明它可以通过相关工具恢复。"""

# 文件路径识别：如 "app/services/chat.py"、"C:\repo\a.py:42"
# 用途：这些是"绝不能丢"的信息，压缩后要校验摘要里还在不在。
PATH_LIKE_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/])?(?:[\w.@()+\- ]+[\\/])+[\w.@()+\- ]+\.[A-Za-z0-9_]+(?::\d+)?"
)

# 命令识别：出现这些词的行通常是"关键操作记录"（跑过什么、怎么复现）
COMMAND_HINT_PATTERN = re.compile(
    r"\b(pytest|npm|pnpm|yarn|git|docker|uvicorn|alembic|python|pip|ruff|mypy|pytest)\b",
    flags=re.IGNORECASE,
)

# 必须保留的"关键行"特征词（中英双语）。
# 命中任一关键词的行，都会被当成"压缩后必须还能找到"的内容。
PRESERVE_LINE_HINTS = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "decision",
    "decide",
    "agreed",
    "todo",
    "next step",
    "prefer",
    "preference",
    "错误",
    "失败",
    "异常",
    "决定",
    "确定",
    "待办",
    "下一步",
    "偏好",
)


def _must_preserve_candidates(messages: list[dict[str, Any]], limit: int = 18) -> list[str]:
    """从历史消息里挑出"压缩后必须还能找到"的关键片段。

    【调用链】PATH_LIKE_PATTERN / COMMAND_HINT_PATTERN / PRESERVE_LINE_HINTS
             + clip_to_token_budget() + _unique_preserve_order() → 本函数
    【被谁调用】compaction_preservation_report()

    两类候选：
        1. 文件路径（正则直接捞）
        2. 命中关键词的行（error / 决定 / todo / 命令 等）
    每条都先 clip 到 120 token —— 只留"锚点"，不需要整段。
    """
    candidates: list[str] = []

    for message in messages:
        content = str(message.get("content") or "")
        if not content:
            continue

        # ── 候选类型 1：文件路径 ──
        candidates.extend(match.group(0).strip() for match in PATH_LIKE_PATTERN.finditer(content))

        # ── 候选类型 2：命中关键词的行 ──
        for raw_line in re.split(r"[\r\n]+", content):
            line = " ".join(raw_line.strip(" -\t").split())
            if not line:
                continue
            lower = line.lower()
            if COMMAND_HINT_PATTERN.search(line) or any(hint in lower for hint in PRESERVE_LINE_HINTS):
                candidates.append(clip_to_token_budget(line, 120))

    # 去重保序，限制条数（太多会撑爆摘要预算）
    return _unique_preserve_order(candidates, limit)


def compaction_preservation_report(messages: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    """校验摘要有没有丢掉关键信息。

    【调用链】_must_preserve_candidates() → 本函数
    【被谁调用】ensure_compaction_summary_preserves_key_facts()

    【数据流向】
        入：原始 messages、模型生成的 summary
        出：报告 {required_count, missing_count, required, missing}
            missing 就是"摘要在压缩过程中丢掉的关键信息"

    这是"压缩质量"的唯一硬指标：missing_count 越少越好。
    """
    required = _must_preserve_candidates(messages)
    summary_lower = str(summary or "").lower()
    # 简单包含判断：只要摘要里出现过这个片段就算保留
    missing = [item for item in required if item.lower() not in summary_lower]
    return {
        "required_count": len(required),
        "missing_count": len(missing),
        "required": required,
        "missing": missing,
    }


def ensure_compaction_summary_preserves_key_facts(
    messages: list[dict[str, Any]],
    summary: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """保证摘要不丢关键信息：丢了就补一个"保真补充"附录。

    【调用链】追这个函数会进入：
        1. compaction_preservation_report() → 找出丢了什么
        2. clip_to_token_budget()           → 控制正文/附录/总长度
        3. compaction_preservation_report() → 补完再校验一次

    【被谁调用】services/chat_memory.py（生成压缩摘要后必经这一步）

    【数据流向】
        入：messages（原始历史）、summary（模型摘要）、max_tokens（摘要预算）
        出：(最终摘要, 报告) —— 报告里多一个 appended_missing_count

    【为什么要这一步】
        模型写摘要时经常会漏掉文件路径、命令、错误码。
        与其反复重试（贵且不一定好），不如直接把漏掉的片段追加到末尾 ——
        规则兜底比模型兜底更可靠。
    """
    report = compaction_preservation_report(messages, summary)
    missing = [str(item) for item in report.get("missing") or []]

    # 没有丢失 → 只做长度控制，直接返回
    if not missing:
        return clip_to_token_budget(summary, max_tokens), report

    # ── 组装"保真补充"附录（最多 12 条，每条 96 token）──
    appendix = "保真补充：\n" + "\n".join(
        f"- {clip_to_token_budget(item, 96)}" for item in missing[:12]
    )

    # ── 正文要给附录让出空间（至少留 160 给正文）──
    body_budget = max(160, max_tokens - estimate_tokens(appendix) - 12)
    combined = "\n\n".join([clip_to_token_budget(summary, body_budget), appendix]).strip()
    combined = clip_to_token_budget(combined, max_tokens)

    # ── 补完再校验一次（附录本身也可能被截断）──
    final_report = compaction_preservation_report(messages, combined)
    final_report["appended_missing_count"] = len(missing)
    return combined, final_report


def build_compaction_summary_payload(messages: list[dict[str, Any]], max_tokens: int = 6000) -> str:
    """把历史消息拼成"喂给摘要模型的输入"。

    【调用链】_tool_names() / is_compact_boundary_message() / clip_to_token_budget() → 本函数
    【被谁调用】services/chat_memory.py（调 LLM 写摘要前）

    【数据流向】
        入：messages（全量历史）、max_tokens（输入预算）
        出：纯文本，每行形如 "user tools=[...]: 内容"

    两个细节：
        1. 边界消息的 role 改写成 "compact_boundary" —— 让摘要模型知道
           "这之前已经压过一次"，避免重复总结。
        2. 每条消息的预算 = max_tokens / 条数（至少 160）——
           保证长对话不会因为前几条太长而把后面的全挤掉。
    """
    lines: list[str] = []

    for message in messages:
        role = str(message.get("role") or "unknown")
        # 边界消息特殊标记
        if role == "system" and is_compact_boundary_message(message):
            role = "compact_boundary"

        # 单条预算：平均分配，最少 160
        content = clip_to_token_budget(str(message.get("content") or ""), max(160, max_tokens // max(len(messages), 1)))
        if not content:
            continue

        # 把工具名附在行首，让摘要模型知道"这条消息涉及哪些工具"
        tool_names = _tool_names(message)
        suffix = f" tools={tool_names}" if tool_names else ""
        lines.append(f"{role}{suffix}: {content}")

    return clip_to_token_budget("\n\n".join(lines), max_tokens)


def format_compaction_summary(summary: str) -> str:
    """清洗模型返回的摘要（去掉思考过程，只留正文）。

    【调用链】无 —— 纯正则清洗
    【被谁调用】services/chat_memory.py（拿到模型摘要后第一步）

    处理三件事：
        1. 删掉 <analysis>...</analysis>（有些模型会先输出思考过程）
        2. 如果有 <summary>...</summary>，只取标签内的内容
        3. 连续 3 个以上换行压成 2 个（避免摘要里有大片空白）
    """
    formatted = str(summary or "").strip()
    # 去掉思考过程块
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted, flags=re.IGNORECASE).strip()
    # 有 <summary> 标签就只取标签内容
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted, flags=re.IGNORECASE)
    if match:
        formatted = "摘要：\n" + match.group(1).strip()
    # 压缩多余空行
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def fallback_compaction_summary(messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
    """【兜底】不调模型，用纯规则拼一份摘要。

    【调用链】clip_to_token_budget() / _unique_preserve_order() → 本函数
    【被谁调用】services/chat_memory.py —— 模型调用失败 / 超时 / 返回空时走这里

    【数据流向】
        入：messages、max_tokens（默认取 settings.context_compact_summary_tokens）
        出：固定五段式摘要（用户意图 / 助手上下文 / 决策 / 待办 / 事实）

    【为什么必须有】
        压缩是"关键路径"。如果模型挂了就没有摘要，整轮对话直接失败。
        纯规则摘要质量差，但保证对话能继续 —— 可用性优先于质量。

    【规则】按关键词把行分到三个桶：
        决策桶：decision / decide / agreed / 决定 / 确定
        待办桶：todo / next step / implement / fix / 待办 / 下一步
        事实桶：长度 > 20 的行（最多 10 条）
    """
    user_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    assistant_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "assistant"]

    decisions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []

    # ── 逐行分类 ──
    for content in [*user_messages, *assistant_messages]:
        for line in [part.strip(" -\t") for part in content.splitlines() if part.strip()]:
            lower = line.lower()
            if any(token in lower for token in ["decision", "decide", "agreed", "\u51b3\u5b9a", "\u786e\u5b9a"]):
                decisions.append(line)
            if any(token in lower for token in ["todo", "next step", "implement", "fix", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
                tasks.append(line)
            # 事实桶：够长的行才算"有信息量"，最多 10 条
            if len(line) > 20 and len(facts) < 10:
                facts.append(line)

    # ── 拼成五段式摘要 ──
    summary = "\n\n".join(
        [
            "This conversation was compacted. Continue from the preserved recent messages and this summary.",
            "User intent:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in user_messages[-8:]),
            "Assistant context:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in assistant_messages[-5:]),
            "Durable decisions:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(decisions, 12)),
            "Pending tasks:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(tasks, 12)),
            "Important facts:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(facts, 12)),
        ]
    )
    return clip_to_token_budget(summary, max_tokens or settings.context_compact_summary_tokens)


# ════════════════════════════════════════════════════════════════
# ⑨ 记忆提炼：把一轮对话变成结构化记忆
# ════════════════════════════════════════════════════════════════

def structured_memory_from_model(parsed: dict[str, Any]) -> dict[str, Any]:
    """把模型返回的 JSON 归一化成标准记忆结构。

    【调用链】_unique_preserve_order() → 本函数
    【被谁调用】
        services/chat_memory.py  ← 模型成功返回时
        fallback_memory_update() ← 规则兜底时（复用同一套归一化）

    【数据流向】
        入：parsed（模型原始输出，字段可能缺、可能是脏数据）
        出：固定 8 个字段的 dict —— 与 thread_memories / conversation_memories 表结构一致

    【为什么要归一化】
        模型输出不可信：字段可能缺失、列表里可能有 None、条数可能几十条。
        统一在这里做"转字符串 + 去重 + 截断条数"，下游就不用到处防御。
    """
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "facts": _unique_preserve_order([str(item) for item in parsed.get("facts") or []], 16),
        "decisions": _unique_preserve_order([str(item) for item in parsed.get("decisions") or []], 16),
        "open_questions": _unique_preserve_order([str(item) for item in parsed.get("open_questions") or []], 12),
        "tasks": _unique_preserve_order([str(item) for item in parsed.get("tasks") or []], 16),
        "user_preferences": _unique_preserve_order([str(item) for item in parsed.get("user_preferences") or []], 12),
        "repo_context": _unique_preserve_order([str(item) for item in parsed.get("repo_context") or []], 16),
        # citations 只留 dict（防止模型返回字符串数组导致下游崩）
        "citations": [item for item in parsed.get("citations") or [] if isinstance(item, dict)][:16],
    }


def fallback_memory_update(user_message: str, assistant_message: str, route: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """【兜底】不调模型，用规则从一轮对话里提炼记忆。

    【调用链】structured_memory_from_model() + clip_to_token_budget() → 本函数
    【被谁调用】
        services/chat_memory.py  ← 记忆提炼失败时
        services/evals/runner.py ← 评测时省 LLM 成本

    【数据流向】
        入：user_message、assistant_message、route（走的分支）、tool_calls（调了什么）
        出：标准记忆 dict（同 structured_memory_from_model 的输出结构）

    【规则】逐行打标签：
        含问号 / why / how / what  → 未决问题
        含 decide / 决定            → 决策
        含 todo / next step / fix   → 待办
        含 prefer / 偏好            → 用户偏好
        长度 > 20                   → 事实（最多 6 条）
    """
    joined = "\n".join([user_message or "", assistant_message or ""])
    lines = [line.strip(" -\t") for line in joined.splitlines() if line.strip()]

    decisions: list[str] = []
    open_questions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    preferences: list[str] = []

    # ── 逐行分类打标签 ──
    for line in lines:
        lower = line.lower()
        # 疑问句 → 未决问题
        if "?" in line or "\uff1f" in line or lower.startswith(("why ", "how ", "what ")):
            open_questions.append(line)
        if any(token in lower for token in ["decide", "decision", "agreed", "chosen", "\u786e\u5b9a", "\u51b3\u5b9a"]):
            decisions.append(line)
        if any(token in lower for token in ["todo", "next step", "action", "fix", "implement", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
            tasks.append(line)
        if any(token in lower for token in ["prefer", "preference", "\u559c\u6b22", "\u504f\u597d"]):
            preferences.append(line)
        # 事实桶：够长且有信息量，最多 6 条
        if len(line) > 20 and len(facts) < 6:
            facts.append(line)

    # ── 拼摘要：路由 + 请求 + 回答 + 工具 ──
    tools = ", ".join(str(call.get("tool_name") or "未知") for call in tool_calls) or "无"
    summary = (
        f"路由：{route}\n"
        f"用户请求：{clip_to_token_budget(user_message, 90)}\n"
        f"助手回答：{clip_to_token_budget(assistant_message, 140)}\n"
        f"工具：{tools}"
    )

    # 复用同一套归一化，保证输出结构与模型路径一致
    return structured_memory_from_model(
        {
            "summary": summary,
            "facts": facts,
            "decisions": decisions,
            "open_questions": open_questions,
            "tasks": tasks,
            "user_preferences": preferences,
            "repo_context": [f"使用路由：{route}"] if route else [],
            "citations": [],
        }
    )


def merge_memory(existing: dict[str, Any], update: dict[str, Any], digest: str, max_tokens: int) -> dict[str, Any]:
    """把"新提炼的记忆"合并进"已有记忆"。

    【调用链】_unique_preserve_order() / clip_to_token_budget() → 本函数
    【被谁调用】services/chat_memory.py（每轮结束更新 thread/conversation memory）

    【数据流向】
        入：existing（库里已有的记忆 dict）、update（本轮新提炼的）、
            digest（本轮摘要，update 没摘要时兜底）、max_tokens（摘要预算）
        出：合并后的记忆 dict（直接写回数据库）

    【合并策略】
        摘要：旧的 + 新的 拼起来再截断（保留演进过程）
        列表：旧 + 新 去重保序（新的在后面，因为 _unique_preserve_order 保留首次出现，
              所以同一条记忆以【旧版本】为准 —— 避免同一件事被反复追加）
        引用：只留最后 16 条（引用会一直增长，必须截断）
    """
    existing_summary = str(existing.get("summary") or "").strip()
    # update 没有摘要时用 digest（本轮对话摘要）兜底
    update_summary = str(update.get("summary") or digest or "").strip()
    summary_parts = [part for part in [existing_summary, update_summary] if part]
    summary = "\n\n".join(summary_parts)
    summary = clip_to_token_budget(summary, max_tokens)

    return {
        "summary": summary,
        "facts": _unique_preserve_order([*(existing.get("facts") or []), *(update.get("facts") or [])], 16),
        "decisions": _unique_preserve_order([*(existing.get("decisions") or []), *(update.get("decisions") or [])], 16),
        "open_questions": _unique_preserve_order([*(existing.get("open_questions") or []), *(update.get("open_questions") or [])], 12),
        "tasks": _unique_preserve_order([*(existing.get("tasks") or []), *(update.get("tasks") or [])], 16),
        "user_preferences": _unique_preserve_order([*(existing.get("user_preferences") or []), *(update.get("user_preferences") or [])], 12),
        "repo_context": _unique_preserve_order([*(existing.get("repo_context") or []), *(update.get("repo_context") or [])], 16),
        # 引用只留最近 16 条（用切片而不是去重：引用本来就允许重复指向同一来源）
        "citations": [*(existing.get("citations") or []), *(update.get("citations") or [])][-16:],
    }


def render_structured_memory(memory: Any, max_tokens: int | None = None) -> str:
    """把记忆对象渲染成给模型看的文本。

    【调用链】clip_to_token_budget() → 本函数
    【被谁调用】
        services/chat_memory.py::ContextAssembler  ← 每轮注入 prompt
        memory_document_text()                     ← 记忆晋升为文档时
        services/evals/runner.py                   ← 评测

    【数据流向】
        入：memory（ORM 对象或 dict，用 getattr 取字段，两种都能吃）
        出：Markdown 风格的分段文本

    为什么用 getattr 而不是 memory["facts"]：
        调用方可能传 ORM 对象（ThreadMemory）也可能传 dict，
        getattr 对两者都不报错（dict 没有属性会走默认值）。
        注意：传 dict 时字段会被当成"没有"—— 所以这里实际只支持对象。
    """
    sections: list[str] = []

    summary = str(getattr(memory, "summary", "") or "").strip()
    if summary:
        sections.append("摘要：\n" + summary)

    # 六个列表字段，有内容才渲染（空列表不占空间）
    for label, attr in [
        ("事实", "facts"),
        ("决策", "decisions"),
        ("未决问题", "open_questions"),
        ("任务", "tasks"),
        ("用户偏好", "user_preferences"),
        ("仓库上下文", "repo_context"),
    ]:
        values = getattr(memory, attr, None) or []
        if values:
            sections.append(label + ":\n" + "\n".join(f"- {item}" for item in values[:12]))

    text = "\n\n".join(sections)
    # max_tokens 为 None 表示不截断（写文档时要全文）
    return clip_to_token_budget(text, max_tokens) if max_tokens else text


def memory_document_text(memory: Any) -> str:
    """把记忆渲染成"可入库的文档全文"（不截断）。

    【调用链】render_structured_memory(max_tokens=None) → 本函数
    【被谁调用】services/memory_hub.py（记忆候选被批准后晋升为 Document 进 RAG）

    与 render_structured_memory 的唯一区别：不传 max_tokens，保留全文。
    """
    return render_structured_memory(memory, max_tokens=None)


# ════════════════════════════════════════════════════════════════
# 杂项：prompt cache 参数、压缩评测、紧凑 JSON
# ════════════════════════════════════════════════════════════════

def prompt_cache_model_kwargs() -> dict[str, Any]:
    """生成 prompt cache 相关的模型调用参数。

    【调用链】无 —— 只读 settings
    【被谁调用】services/llm/client.py（每次调模型时合并进 kwargs）
    【被谁间接使用】services/agents/chat_agent.py

    为什么需要：
        System prompt 每轮都一样，开启 cache 后 provider 会缓存这部分，
        命中缓存的价格通常只有 1/10。对长系统提示 + 高频调用很划算。

    返回空 dict 表示不启用（调用方直接 **kwargs 合并，不需要判断）。
    """
    if not settings.prompt_cache_enabled:
        return {}
    kwargs: dict[str, Any] = {}
    if settings.prompt_cache_key:
        kwargs["prompt_cache_key"] = settings.prompt_cache_key
    if settings.prompt_cache_retention:
        kwargs["prompt_cache_retention"] = settings.prompt_cache_retention
    return kwargs


def compression_eval_summary(before_text: str, after_text: str, retrieved: int, kept: int) -> dict[str, Any]:
    """生成一次压缩的评测指标。

    【调用链】estimate_tokens() → 本函数
    【被谁调用】services/evals/runner.py（评测报告里的 context_compression 段）

    【数据流向】
        入：压缩前文本、压缩后文本、检索条数、保留条数
        出：指标 dict → 写进 eval_runs.result_json

    compression_ratio < 1 表示压小了；= 1 表示没压（before 为 0 时也返回 1）。
    """
    before = estimate_tokens(before_text)
    after = estimate_tokens(after_text)
    return {
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compression_ratio": round(after / before, 4) if before else 1.0,
        "retrieved_items": retrieved,
        "kept_items": kept,
        "token_savings": max(0, before - after),
    }


def dumps_compact(value: Any) -> str:
    """紧凑 JSON 序列化（不缩进、不转义中文、遇到不可序列化对象转 str）。

    【调用链】无
    【被谁调用】services/chat_memory.py（把结构化数据写进 JSON 列 / 拼进 prompt）

    三个参数的作用：
        ensure_ascii=False   → 中文原样输出，不变成 \\uXXXX（省 token）
        separators=(",",":") → 去掉空格（进一步省 token）
        default=str          → datetime / UUID 等对象不报错，直接转字符串
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
