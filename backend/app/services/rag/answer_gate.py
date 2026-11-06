"""
回答准入闸门（Answer Gate）
==========================

【在 RAG 流水线中的位置】
    生成答案【之前】的最后一道关卡，位于六阶段流水线的第 ⑥ 步：

        retrieval 召回 → rerank 重排 → 父块扩展 → 阈值过滤 → 上下文压缩
             ↓
        ★ 本文件 evaluate_answer_gate()   ← 证据够不够？有没有冲突？问题太模糊？
             ↓
        ┌────────────┴────────────┐
     answer                   非 answer
        ↓                        ↓
     qa.py 调 LLM          qa.py 直接返回话术（★ 不调 LLM）

【业务定位】
    生成答案【之前】的判定：证据够不够、有没有冲突、问题是否太模糊。
    这是防幻觉的关键——不够就拒答，而不是让 LLM 编一个。

    【为什么"不调 LLM"很重要】
        如果让 LLM 自己判断"证据够不够"，它经常会强行回答（模型有"必须给出答案"的倾向）。
        用规则先判定，不够就根本不进 LLM —— 既省 token，又杜绝幻觉。

【数据流向】
    上游（谁调它）：
        services/rag/knowledge_base.py::retrieve_knowledge()  末尾调用
    下游（它调谁）：
        无 —— 纯规则判定，不碰数据库、不调模型
    消费方：
        services/rag/qa.py::answer_question()
            decision == "answer" → 继续调 LLM
            否则                 → 调 gate_response_message() 拿话术，直接返回

【核心规则】
    四种决策：answer / insufficient_evidence / conflict / ask_clarification。
    判定顺序从"最坏情况"开始，命中即返回。

【本文件函数地图】
    evaluate_answer_gate()     ★ 主判定（按优先级 6 步检查）
        ├ _strong_query_signals()  提取查询里的"精确锚点"
        ├ _evidence_text()         拼证据的可搜索文本
        ├ _detect_conflict()       检测极性冲突
        └ _is_ambiguous_query()    判断问题是否模糊
    gate_response_message()    把非 answer 的决策转成用户话术
    _normalise()               小写 + 压缩空白
"""

import re
from typing import Any


# 模糊指代词：出现这些词说明问题缺少明确对象
# 例："这个怎么办" —— 系统不知道"这个"指什么
AMBIGUOUS_MARKERS = ("这个", "那个", "它", "这段", "这里", "怎么办", "怎么处理")

# 极性冲突检测：同一主题出现"允许 vs 禁止"这类相反表述
# 结构：(冲突类型, 正向词, 反向词)
#
# 为什么只有三组：冲突检测靠词表，词表越多误判越多。
# 这三组是研发场景里最常见、也最容易造成严重后果的冲突。
CONFLICT_POLARITIES = (
    ("permission", ("允许", "可以", "enabled", "enable"), ("禁止", "不允许", "不能", "disabled", "disable")),
    ("outcome", ("成功", "通过", "success", "passed"), ("失败", "未通过", "failure", "failed")),
    ("requirement", ("必须", "需要", "required"), ("可选", "无需", "optional")),
)


def _normalise(text: str) -> str:
    """归一化：小写 + 压缩空白。

    【调用链】被 _evidence_text() 调用

    为什么：后面要做 "关键词 in 文本" 的子串匹配，
        大小写不一致会漏匹配（"Error" vs "error"），
        连续空白也会影响匹配（"timeout   error" vs "timeout error"）。
    """
    return " ".join(str(text or "").lower().split())


def _evidence_text(item: dict[str, Any]) -> str:
    """把一条证据的可搜索文本拼起来（标题+正文+路径+符号+分支）。

    【调用链】_normalise() → 本函数
    【被谁调用】_detect_conflict() / evaluate_answer_gate()

    【为什么要拼这么多字段，而不是只看正文】
        用户可能问"src/api/auth.py 里的超时"，
        这时"src/api/auth.py"只出现在 metadata.path 里，不在正文里。
        只看正文会导致"强信号全部缺席"误判。

    字段优先级：title > snippet/正文 > path > symbol > branch
    """
    metadata = item.get("metadata") or {}
    return _normalise(
        "\n".join(
            [
                str(item.get("title") or ""),
                # snippet 是压缩后的片段，_raw_content 是完整内容，二选一
                str(item.get("snippet") or item.get("_raw_content") or ""),
                str(metadata.get("path") or ""),
                str(metadata.get("symbol") or ""),
                str(metadata.get("branch") or metadata.get("base_branch") or ""),
            ]
        )
    )


def _strong_query_signals(query: str) -> list[str]:
    """提取查询里的"强信号"——必须在证据里出现的具体标识。

    【调用链】被 evaluate_answer_gate() 调用

    什么算强信号：
        带下划线/点/斜杠/横线的（auth_service、config.py、src/api）
        含数字的（#42、v2、500）
        以 Error/Exception 结尾的（TimeoutError）
    【为什么】这些是"精确匹配"的锚点。如果全都找不到，说明检索结果跟问题无关。

    【为什么这些算"强"，普通词不算】
        "超时"这种词几乎每篇文档都有，命中它说明不了什么。
        但 "auth_service.py" 这种标识符，命中了就基本确定相关。
        所以只用强信号做"跑偏检测"，不用普通词。

    【返回值】去重保序的小写列表（dict.fromkeys 保序）
    """
    # 候选：英文标识符（允许 _ . / -）、#数字、3 位以上数字
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_./\-]*|#\d+|\b\d{3,}\b", query)

    signals: list[str] = []
    for candidate in candidates:
        if (
            any(marker in candidate for marker in ("_", ".", "/", "-"))   # 像路径/标识符
            or any(char.isdigit() for char in candidate)                  # 含数字
            or candidate.endswith(("Error", "Exception"))                 # 异常类名
        ):
            signals.append(candidate.lower())

    # 去重且保持原顺序
    return list(dict.fromkeys(signals))


def _detect_conflict(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """检测高分证据之间的极性冲突（如一份文档说"允许"、另一份说"禁止"）。

    【调用链】_evidence_text() → 本函数
    【被谁调用】evaluate_answer_gate()

    只在两条以上高置信证据（score>=0.6）之间比较，避免噪声误判。

    【为什么阈值定 0.6】
        分数低本来就是"不确定"的证据，它们之间矛盾很正常（都是噪声）。
        只有高分证据互相矛盾，才说明知识库本身有问题（比如文档过期了）。

    【数据流向】
        入：候选证据列表
        出：冲突详情 dict 或 None
            {type, positive_evidence_ids, negative_evidence_ids}

    【判定逻辑（以 permission 为例）】
        negative_ids = 含"禁止/不能/disabled"的证据
        positive_ids = 含"允许/可以/enabled" 且【不含】反向词的证据
        两者都非空 且 不是同一批证据 → 判定冲突

    为什么 positive 要排除含反向词的：
        一份文档可能同时写"允许 X，禁止 Y"，它既不是纯正向也不是纯反向。
        把它算进 positive 会造成假冲突。
    """
    # 只比较高置信证据
    high_confidence = [item for item in items if float(item.get("score") or 0.0) >= 0.6]
    if len(high_confidence) < 2:
        return None

    # (证据 ID, 可搜索文本)
    texts = [(str(item.get("id") or item.get("source_id") or ""), _evidence_text(item)) for item in high_confidence]

    for conflict_type, positive_markers, negative_markers in CONFLICT_POLARITIES:
        # 含反向词的证据
        negative_ids = [item_id for item_id, text in texts if any(marker in text for marker in negative_markers)]
        # 含正向词、且不含反向词的证据（纯正向）
        positive_ids = [
            item_id
            for item_id, text in texts
            if any(marker in text for marker in positive_markers)
            and not any(marker in text for marker in negative_markers)
        ]

        # 两侧都有，且不是同一批证据 → 冲突
        if positive_ids and negative_ids and set(positive_ids) != set(negative_ids):
            return {
                "type": conflict_type,
                "positive_evidence_ids": positive_ids,
                "negative_evidence_ids": negative_ids,
            }
    return None


def _is_ambiguous_query(query: str, items: list[dict[str, Any]]) -> bool:
    """判断问题是否太模糊（含指代词 + 多来源 或 问题太短）。

    【调用链】被 evaluate_answer_gate() 调用

    【为什么"多来源"也算模糊】
        问"这个怎么办"，如果证据来自 3 个不同文件，说明系统也不确定用户指哪个。
        这时候硬答等于瞎猜，不如反问一句。

    【两个条件必须同时满足】
        ① 含指代词（这个/那个/它…）
        ② 证据来自多个源对象，或者问题本身很短（≤8 字符）

    为什么要 ②：如果只含"这个"但证据只有一个来源，
    那"这个"显然就是指它，可以直接答，不必反问。
    """
    # 去掉所有空白（"这 个" 也要能匹配到"这个"）
    compact = re.sub(r"\s+", "", query)

    # 不含指代词 → 不模糊
    if not any(marker in compact for marker in AMBIGUOUS_MARKERS):
        return False

    # 证据来自几个不同的源对象
    source_ids = {str(item.get("source_id") or "") for item in items if item.get("source_id")}
    return len(source_ids) > 1 or len(compact) <= 8


def evaluate_answer_gate(
    query: str,
    items: list[dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    threshold_enabled: bool = True,
) -> dict[str, Any]:
    """【主判定】按优先级依次检查，返回四种决策之一。

    【调用链】追这个函数会进入：
        1. _strong_query_signals()  → 提取强信号
        2. _evidence_text()         → 拼证据文本（判断强信号是否缺席）
        3. _detect_conflict()       → 检测极性冲突
        4. _is_ambiguous_query()    → 判断问题是否模糊
    【被谁调用】services/rag/knowledge_base.py::retrieve_knowledge() 末尾

    【数据流向】
        入：query、items（已重排+扩展+压缩的证据）、score_threshold、threshold_enabled
        出：dict，至少含 decision / reason / evidence_count / top_score
            decision = answer / insufficient_evidence / conflict / ask_clarification

    检查顺序（先排除最坏情况）：
        1. 没有证据              → insufficient_evidence
        2. 最高分低于阈值         → insufficient_evidence
        3. 强信号全部缺席         → insufficient_evidence（检索跑偏了）
        4. 高分证据互相矛盾       → conflict
        5. 问题太模糊            → ask_clarification
        6. 都通过                → answer

    【为什么这个顺序】
        从"最确定的问题"到"最需要人工介入"：
        没证据是硬伤（一定不能答）；冲突是知识库问题（要提示用户核对）；
        模糊是用户表达问题（要反问）。顺序错了会给出误导性的原因。
    """
    # 最高分（空列表时 default=0.0）
    top_score = max((float(item.get("score") or 0.0) for item in items), default=0.0)

    # 所有决策都会带上的基础信息（便于前端展示和排查）
    base = {
        "evidence_count": len(items),
        "top_score": round(top_score, 4),
        "score_threshold": round(float(score_threshold), 4) if threshold_enabled else None,
    }

    # ── 检查 1：完全没有证据 ──
    if not items:
        return {**base, "decision": "insufficient_evidence", "reason": "没有检索到满足条件的证据"}

    # ── 检查 2：最高分没到阈值 ──
    if threshold_enabled and top_score < score_threshold:
        return {**base, "decision": "insufficient_evidence", "reason": "最高证据分数未达到回答门槛"}

    # ── 检查 3：强信号全缺席 → 检索结果和问题无关 ──
    # 典型表现：问 auth.py 却召回一堆无关文档（向量检索只匹配到"超时"这种泛词）
    signals = _strong_query_signals(query)
    evidence_text = "\n".join(_evidence_text(item) for item in items)
    missing_signals = [signal for signal in signals if signal not in evidence_text]
    # 注意条件：signals 非空（没有问题就没信号，不能因此拒答）且【全部】缺席
    if signals and len(missing_signals) == len(signals):
        return {
            **base,
            "decision": "insufficient_evidence",
            "reason": "查询中的关键名称没有出现在候选证据中",
            "required_signals": signals,
            "missing_signals": missing_signals,
        }

    # ── 检查 4：高分证据互相矛盾 ──
    conflict = _detect_conflict(items)
    if conflict:
        return {**base, "decision": "conflict", "reason": "高排名证据给出了相反结论", "conflict": conflict}

    # ── 检查 5：问题太模糊 ──
    if _is_ambiguous_query(query, items):
        return {
            **base,
            "decision": "ask_clarification",
            "reason": "问题缺少可定位的对象或范围",
            # 告诉用户该补充什么（前端可以直接渲染成提示）
            "suggested_fields": ["文件或模块", "错误名称", "分支或版本"],
        }

    # ── 检查 6：全部通过 → 可以答 ──
    return {
        **base,
        "decision": "answer",
        "reason": "证据达到分数门槛，关键名称可以在候选中核对",
        # 命中了哪些强信号（用于前端展示"依据"）
        "matched_signals": [signal for signal in signals if signal in evidence_text],
    }


def gate_response_message(gate: dict[str, Any]) -> str:
    """把非 answer 的决策转成给用户看的话术（不调 LLM，直接返回）。

    【调用链】被 services/rag/qa.py::answer_question() 调用

    【为什么要给三种不同话术，而不是统一一句"资料不足"】
        用户看到"资料不足"不知道该做什么。
        说清"是问题太模糊"还是"是知识库有冲突"，用户才知道下一步怎么办：
            模糊 → 补充文件/错误名
            冲突 → 核对来源或版本
            不足 → 补充资料 / 换问法
    """
    decision = str(gate.get("decision") or "insufficient_evidence")

    if decision == "ask_clarification":
        return "当前问题的对象或范围不够明确。请补充文件、模块、错误名称、分支或版本后再试。"
    if decision == "conflict":
        return "当前知识库中的高排名证据存在冲突，暂时无法给出确定结论。请先核对对应来源或版本。"
    # 兜底：insufficient_evidence 及任何未知决策
    return "当前知识库资料不足，无法确认。请补充相关资料、检查检索范围，或换一种更具体的问法。"
