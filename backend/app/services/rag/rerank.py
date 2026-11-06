"""
启发式重排（Rerank 兜底方案）
============================

【在 RAG 流水线中的位置】
    查询链路的第 ② 步（六阶段流水线的 rerank 阶段）：

        混合检索召回 50 条（粗排）
             ↓
        ★ 本文件 rerank_documents()   ← 精排，取前 8 条
             ↓
        父块扩展 → 阈值过滤 → 压缩 → 准入 → 生成

【业务定位】
    RAG 查询链路的重排环节。首选 Qwen3-Rerank API（在 knowledge_base.py 的 _qwen_rerank），
    本文件是 API 不可用时的纯本地兜底。

    【为什么要重排】
        向量检索 + BM25 都是"粗排"，只能保证"大致相关"。
        重排用更精细的信号（词覆盖率、标题命中、新鲜度…）重新打分，
        把真正该排第一的挑出来。这是提升答案质量最划算的一步。

    【为什么要有兜底】
        重排要调外部 API，会失败（超时/限流/没配 key）。
        失败时不能让整个查询挂掉 —— 用启发式规则顶上，效果差一点但能用。

【数据流向】
    上游（谁调它）：
        services/rag/knowledge_base.py（Qwen rerank 失败后降级到这里）
    下游（它调谁）：
        app.core.config.settings → rag_rerank_top_n（只重排前多少条）

【核心规则】
    1. 只重排前 top_n 条（省算力），尾部原样拼接（不丢结果）。
    2. 分数永远夹在 0~1（NaN/Inf 归零）。
    3. 必须给出 rank_reason（"为什么排这里"），便于调试和前端展示。

【本文件函数地图】
    rerank_documents()      ★ 主入口：重排 + 排序 + 拼回尾部
        ├ _heuristic_score()    单条打分（11 项加权）
        │    ├ _tokens()            分词
        │    ├ _candidate_text()    拼可比较文本
        │    ├ _freshness_bonus()   新鲜度加分
        │    └ _normalise()         分数夹到 0~1
        └ （排序）
"""

import math
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


def _tokens(text: str) -> list[str]:
    """分词：英文按词，中文按单字 + 双字组合。

    【调用链】被 _heuristic_score() 调用

    【为什么中文要切双字】中文没有空格，单字太碎（"用"会匹配到一切），
        双字（"用户"）更接近语义单位。这是无分词库时的常用近似做法。

    【为什么单字和双字都要】
        单字保证召回（"超时"里的"超"能匹配到"超时设置"），
        双字保证精度（"用户"不会匹配到"使用"）。
        两者一起进 terms，覆盖率算的是"命中的不同 term 数"，天然平衡。
    """
    lowered = text.lower()
    # 英文/数字/下划线/连字符：按词切
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    # 中文：每段连续汉字 → 单字 + 相邻双字
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)                                                      # 单字
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))  # 双字
    return tokens


def _normalise(value: float) -> float:
    """把分数夹到 0~1（NaN/Inf 归零）。

    【调用链】被 _heuristic_score() / rerank_documents() 调用

    【为什么要处理 NaN/Inf】
        上游分数可能来自除法（如 compression ratio），分母为 0 时会产生 inf/nan。
        nan 参与任何比较都是 False，会导致排序结果乱掉（sort 行为未定义）。
        所以必须在这里清掉。
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _candidate_text(candidate: dict[str, Any]) -> str:
    """拼出候选的可比较文本（标题 + 路径 + 片段）。

    【调用链】被 _heuristic_score() 调用

    【为什么路径要单独拼进来】
        用户常问"src/api/auth.py 里的超时"，路径是强信号。
        如果只看标题+正文，会漏掉这个精确匹配。

    【为什么不拼 metadata 里的所有字段】
        字段太多会稀释信号（"作者是 xxx"这种无关信息也会被匹配到）。
        只取 path 这一个高价值字段。
    """
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "")
    return "\n".join(
        [
            str(candidate.get("title") or ""),
            path,
            str(candidate.get("snippet") or ""),
        ]
    ).lower()


def _freshness_bonus(metadata: dict[str, Any]) -> float:
    """新鲜度加分：30 天内 +0.04，180 天内 +0.02，更早不加分。

    【调用链】被 _heuristic_score() 调用

    【业务理由】研发场景里"最近的"往往更相关（最新 CI 失败、最新 PR 讨论）。

    【为什么按顺序试四个时间字段】
        Issue 有 created/updated/closed，PR 还有 merged。
        取第一个有值的（通常是 updated_at），不叠加。
        —— 叠加会让"同时有 4 个时间字段"的文档莫名多 0.08 分，不公平。

    【为什么 0.04 / 0.02 这么小】
        新鲜度是"锦上添花"，不能盖过内容相关性。
        总分里各项权重加起来约 0.9，0.04 只是微调。

    返回 0.0 的情况：没有时间字段、时间格式解析失败、超过 180 天。
    """
    for key in ("updated_at", "created_at", "merged_at", "closed_at"):
        value = metadata.get(key)
        if not value:
            continue
        try:
            # GitHub 用 "Z" 表示 UTC，Python 的 fromisoformat 不认，要换成 +00:00
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            # 没有时区信息的当作 UTC（否则下面相减会抛 TypeError）
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue        # 格式不对就试下一个字段

        age_days = max(0, (datetime.now(timezone.utc) - parsed).days)
        if age_days <= 30:
            return 0.04
        if age_days <= 180:
            return 0.02
        return 0.0

    return 0.0


def _heuristic_score(query: str, candidate: dict[str, Any]) -> tuple[float, str]:
    """启发式打分：多信号加权求和，并返回"为什么排在这里"的可读理由。

    【调用链】_tokens() / _candidate_text() / _freshness_bonus() / _normalise() → 本函数
    【被谁调用】rerank_documents()

    【数据流向】
        入：query、candidate（含 score / vector_score / keyword_score / metadata）
        出：(分数 0~1, 理由字符串)

    权重设计（总和约 0.9，留出加分的余量）：
        0.31 × 基础分（混合检索的融合分）—— 最可信，权重最高
        0.20 × 向量分
        0.16 × 关键词分
        0.17 × 词覆盖率（查询里的词有多少出现在候选里）
        0.04 × 标题命中
        0.02 × 路径命中
        + 短语整句命中 0.12 / 多词命中 0.08 / 双路召回 0.05 / 新鲜度 0.04 / 来源类型匹配 0.03

    【为什么基础分权重最高（0.31）】
        基础分是"向量+关键词"两路融合的结果，已经包含了两方面信息。
        重排只是在此之上做微调，不应该推翻它。

    【为什么 query 没有可用词时直接返回基础分】
        纯符号查询（如 "???"）没法做词匹配，硬算会得出全是 0 的分数，
        反而不如直接用基础分（至少保留了召回时的排序信息）。
    """
    terms = _tokens(query)
    # 没有可用词 → 退化成"只用基础分"
    if not terms:
        return _normalise(float(candidate.get("score") or 0.0)), "base score only"

    text = _candidate_text(candidate)
    title = str(candidate.get("title") or "").lower()
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "").lower()

    # ── 四个基础统计量 ──
    matched = [term for term in terms if term in text]
    # 词覆盖率：命中的【不同】词数 / 查询里的【不同】词数
    coverage = len(set(matched)) / max(len(set(terms)), 1)
    title_hits = sum(1 for term in terms if term in title)
    path_hits = sum(1 for term in terms if term in path)

    # ── 三个奖励项 ──
    # 整句命中：查询原串出现在候选里 → 强信号
    phrase_bonus = 0.12 if query.lower().strip() and query.lower().strip() in text else 0.0
    # 多词命中：至少命中 2 个不同的词 → 说明不是碰巧
    source_bonus = 0.08 if len({term for term in terms if term in text}) >= 2 else 0.0

    # ── 三个上游分数 ──
    base = _normalise(float(candidate.get("score") or 0.0))
    keyword = _normalise(float(candidate.get("keyword_score") or 0.0))
    vector = _normalise(float(candidate.get("vector_score") or 0.0))

    # 双路召回：向量和关键词都召回了同一条 → 强信号
    sources = set((candidate.get("retrieval") or {}).get("sources") or [])
    dual_recall_bonus = 0.05 if {"vector", "keyword"}.issubset(sources) else 0.0

    freshness_bonus = _freshness_bonus(metadata)

    # 来源类型匹配：查询里的词出现在 source_type 里（如问 "PR" 命中 pull_request）
    source_type = str(candidate.get("source_type") or "").lower()
    source_type_bonus = 0.03 if source_type and any(term in source_type for term in terms) else 0.0

    score = (
        0.31 * base
        + 0.20 * vector
        + 0.16 * keyword
        + 0.17 * coverage
        # min(...,3)/3 把命中数归一到 0~1，避免标题里堆满关键词就能刷分
        + 0.04 * min(title_hits, 3) / 3
        + 0.02 * min(path_hits, 3) / 3
        + phrase_bonus
        + source_bonus
        + dual_recall_bonus
        + freshness_bonus
        + source_type_bonus
    )

    # 生成可读的排序理由（前端/调试能看到"为什么这条排第一"）
    reasons = []
    if coverage:
        reasons.append(f"matched {len(set(matched))}/{len(set(terms))} query terms")
    if title_hits:
        reasons.append("title match")
    if path_hits:
        reasons.append("path match")
    if vector:
        reasons.append("vector candidate")
    if keyword:
        reasons.append("keyword candidate")
    if dual_recall_bonus:
        reasons.append("dual recall")
    if freshness_bonus:
        reasons.append("fresh evidence")
    if source_type_bonus:
        reasons.append("source type match")

    # 一个理由都没有 → 说明只是弱匹配
    return _normalise(score), ", ".join(reasons) or "弱关键词匹配"


def rerank_documents(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """对候选重新打分排序，返回前 limit 条。

    【调用链】_heuristic_score() → 排序 → 本函数
    【被谁调用】services/rag/knowledge_base.py::retrieve_knowledge()（Qwen rerank 降级时）

    【数据流向】
        入：query、candidates（混合检索的候选，已按融合分排序）、limit
        出：重排后的列表，每条多了三个字段：
              score         ← 覆盖成重排分
              rerank_score  ← 同上（保留原名，避免下游混淆）
              rank_reason   ← 为什么排这里
              retrieval.rerank_mode = "heuristic"（标记走了兜底）

    【为什么只重排前 top_n 条】
        重排要算每个候选的多项特征，全量重排成本高但收益递减
        （排在第 100 位的候选本来也不会被选中）。
        尾部直接拼在最后，保证不丢结果。

    【top_n 怎么算】
        min(max(settings.rag_rerank_top_n, limit), len(candidates))
        —— 至少覆盖 limit（否则选不出 limit 条），最多等于候选总数。
    """
    if not candidates:
        return []

    top_n = min(max(settings.rag_rerank_top_n, limit), len(candidates))
    head = candidates[:top_n]      # 要重排的部分
    tail = candidates[top_n:]      # 原样保留的部分

    ranked: list[dict[str, Any]] = []
    for candidate in head:
        heuristic_score, heuristic_reason = _heuristic_score(query, candidate)
        ranked.append(
            {
                **candidate,
                "score": round(heuristic_score, 4),
                "rerank_score": round(heuristic_score, 4),
                "rank_reason": heuristic_reason,
                # 同时写进 retrieval 明细（前端"检索明细"面板读这里）
                "retrieval": {
                    **(candidate.get("retrieval") or {}),
                    "rerank_mode": "heuristic",
                    "rerank_score": round(heuristic_score, 4),
                    "rank_reason": heuristic_reason,
                },
            }
        )

    # 重排分相同则用融合分兜底排序
    # （两个候选重排分一样时，融合分高的应该在前 —— 保持稳定性）
    ranked.sort(key=lambda item: (float(item.get("rerank_score") or item.get("score") or 0.0), float(item.get("combined_score") or 0.0)), reverse=True)

    # 重排后的头部 + 原始尾部，再取前 limit 条
    return [*ranked, *tail][:limit]
