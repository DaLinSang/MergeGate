"""
混合检索（向量 + BM25 双路召回 + 加权融合）
==========================================

【在 RAG 流水线中的位置】
    查询链路的第 ③ 步（六阶段流水线的 hybrid_recall + 后面的去重/聚合）：

        用户问题
             ↓
        embed_knowledge_query()  算查询向量
             ↓
        ★ 本文件 search_similar_documents()
             ① 向量召回（Milvus）  ┐
             ② 关键词召回（BM25）  ├→ ③ 加权融合 → ④ 去重 → ⑤ 父块占用限制
                                  ┘
             ↓
        rerank（本文件内部会调）→ 压缩 → 准入 → 生成

【业务定位】
    RAG 查询的"召回"环节：把相关文档从库里捞出来。这是检索质量的地基。

    【为什么必须两路都做】
        向量召回（语义）：能理解"登录失败" ≈ "认证报错"，但会漏掉精确的标识符
        关键词召回（BM25）：能精确匹配 "auth_service.py"，但不懂同义词
        两路互补 → 融合后既懂语义又不丢精确匹配。

【数据流向】
    上游（谁调它）：
        services/rag/knowledge_base.py::retrieve_knowledge()   六阶段流水线的第 ① 步
        services/agents/chat_agent.py 的 rag 工具               Agent 直接检索
    下游（它调谁）：
        vector_store.search_milvus_documents()   向量召回
        PostgreSQL documents 表                  关键词召回（BM25 自己算）
        rerank.rerank_documents()                重排
        context_compression.compress_retrieval_results()  压缩

【核心规则】
    1. 向量召回（语义） + BM25 召回（关键词）双路，按 0.62 / 0.30 加权融合，双路都命中 +0.08。
    2. Milvus 挂了时 hybrid 自动退化为纯关键词；vector 模式则直接抛异常。
    3. 同一父块的切片最多保留 2 个，防止上下文被同一段内容占满。

【本文件函数地图】

    ┌─ 主流程（先读这个）──────────────────────────────
    │  search_similar_documents()   ★ 总入口，7 步串联
    │      ├ ① _milvus_vector_candidates()   向量召回
    │      ├ ② _keyword_candidates()         关键词召回（BM25）
    │      ├ ③ _merge_candidates()           ★ 加权融合（0.62/0.30/0.08）
    │      ├ ④ _deduplicate_candidates()     两层去重
    │      ├ ⑤ _limit_parent_occupancy()     父块占用限制
    │      ├ ⑥ rerank.rerank_documents()     重排
    │      └ ⑦ compress_retrieval_results()  压缩
    │
    ├─ 召回内部 ────────────────────────────────────
    │  _bm25_document_stats()   ★ 字段加权（title 4.0 / symbol 4.5 / content 1.0）
    │  _bm25_score()            ★ 标准 BM25 公式
    │  _document_result()        ORM → 候选 dict
    │
    ├─ 去重与聚合 ──────────────────────────────────
    │  _normalised_fingerprint()   精确指纹（去标点后哈希）
    │  _candidate_term_set()       词集（Jaccard 用）
    │  _same_duplicate_scope()     同作用域才允许判重
    │  _limit_parent_occupancy()   每父块最多 2 片
    │
    ├─ 父块扩展 ────────────────────────────────────
    │  expand_parent_context()  ★ 把小切片拼回大上下文
    │      └ _same_version()        不同版本不能混拼
    │
    └─ 工具 ───────────────────────────────────────
       cosine_similarity() / _as_uuid() / _parse_datetime() / _metadata_time()
       _matches_metadata()   元数据过滤（JSON 列只能在应用层做）
       _tokens() / _clip() / _normalise_scores()
"""

import heapq
import hashlib
import math
import re
import uuid
from collections import Counter
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.context_compression import compress_retrieval_results
from app.services.rag.rerank import rerank_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES, is_searchable_source, normalize_source_types
from app.services.rag.vector_store import MilvusUnavailableError, search_milvus_documents


# ════════════════════════════════════════════════════════════════
# 参数：BM25 与融合权重（改这些就是在调检索效果）
# ════════════════════════════════════════════════════════════════

# ── BM25 参数 ──
BM25_K1 = 1.5      # 词频饱和参数：越大越"奖励高频词"，1.2~2.0 是常用区间
BM25_B = 0.75      # 长度归一化强度：0=不归一化，1=完全按长度惩罚

# ── 融合权重（三者相加约 1.0）──
VECTOR_FUSION_WEIGHT = 0.62    # 向量分权重最高：语义匹配更可靠
KEYWORD_FUSION_WEIGHT = 0.30   # 关键词分次之：精确匹配兜底
DUAL_RECALL_BONUS = 0.08       # 双路都召回：强信号，额外加分

# ── 去重与聚合 ──
MAX_CHUNKS_PER_PARENT = 2      # 同一父块最多保留几个切片
NEAR_DUPLICATE_JACCARD = 0.9   # 词集重合度 >= 0.9 视为近似重复

# 【为什么向量权重 0.62 远高于关键词 0.30】
#   向量召回对"换个说法"更稳，是主力；
#   关键词召回主要是"兜底精确匹配"，不该主导排序。
#   0.62 + 0.30 + 0.08 = 1.0，保证 combined_score 天然在 0~1。


# ════════════════════════════════════════════════════════════════
# 基础工具
# ════════════════════════════════════════════════════════════════

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量已归一化时等价于点积，但这里保留完整实现以防万一）。

    【调用链】无 —— 纯数学函数
    【被谁调用】测试和调试（生产路径用 Milvus 的 COSINE 度量，不走这里）

    【为什么保留完整实现】
        向量在 embeddings.py 里已经 L2 归一化，点积就等于余弦相似度。
        但万一将来有未归一化的向量（比如换了 provider），
        这个函数还能算对 —— 属于"防御性实现"。

    分母的 or 1.0：避免零向量导致除零。
    """
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """字符串转 UUID。

    【调用链】被 _keyword_candidates() / expand_parent_context() / search_similar_documents() 调用

    为什么要这个：API 层传来的 repo_id 是字符串，SQLAlchemy 比较 UUID 列时需要真 UUID 对象。
    """
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _parse_datetime(value: str | None, end_of_day: bool = False) -> datetime | None:
    """解析日期/时间字符串。

    【调用链】被 _metadata_time() / _matches_metadata() 调用

    end_of_day=True 时把纯日期（如 "2026-10-22"）补到 23:59:59.999999，
    这样"截止到某天"能包含当天全天。

    【为什么这个细节重要】
        用户筛选"到 10 月 22 日"，如果按 00:00:00 算，
        10 月 22 日当天的数据全被排除 —— 用户会觉得"筛选漏了"。
    """
    if not value:
        return None
    try:
        # 长度 10 说明是纯日期（YYYY-MM-DD），没有时间部分
        if len(value) == 10:
            parsed_date = datetime.fromisoformat(value).date()
            return datetime.combine(parsed_date, time.max if end_of_day else time.min, tzinfo=timezone.utc)
        # 完整时间：GitHub 的 "Z" 要换成 "+00:00"
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # 没有时区信息 → 当 UTC（否则和带时区的时间比较会抛异常）
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        # 格式不对 → 当作"没有这个时间"（不抛异常，因为过滤条件不该让查询失败）
        return None


def _metadata_time(doc_meta: dict) -> datetime | None:
    """从元数据里取"最相关的时间"（更新 > 创建 > 合并 > 关闭）。

    【调用链】_parse_datetime() → 本函数
    【被谁调用】_matches_metadata()（时间区间过滤）

    为什么按这个优先级：updated_at 最能代表"这条记录的最新状态"。
    """
    for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
        parsed = _parse_datetime(doc_meta.get(key))
        if parsed:
            return parsed
    return None


def _matches_metadata(doc_meta: dict, filters: dict | None) -> bool:
    """元数据过滤（在应用层做，因为 metadata 是 JSON 列）。

    【调用链】_parse_datetime() / _metadata_time() → 本函数
    【被谁调用】_milvus_vector_candidates() / _keyword_candidates()

    支持：标签、状态、路径/模块、时间区间、分支、任意键值。
    值为列表时做"包含"判断。

    【为什么在应用层而不是 SQL】
        metadata 是 JSON 列，key 是动态的。
        PostgreSQL 虽然支持 JSON 查询，但跨数据库兼容写法很麻烦。
        数据量不大时（单仓库几十到几百条），应用层过滤更简单可靠。

    【过滤顺序：先便宜后昂贵】
        标签/状态（直接比较）→ 路径（字符串包含）→ 时间（需要解析）→ 其他键值
        这样大部分不匹配的记录在前面就被排除了。
    """
    if not filters:
        return True

    # ── 标签：值必须在这个列表里 ──
    label = filters.get("label")
    if label and label not in (doc_meta.get("labels") or []):
        return False

    # ── 状态：精确匹配 ──
    state = filters.get("state")
    if state and doc_meta.get("state") != state:
        return False

    # ── 路径过滤：优先匹配 files 列表，没有 files 就在所有元数据值里找 ──
    path = (filters.get("path") or filters.get("path_prefix") or filters.get("module") or "").lower()
    if path:
        files = [str(item).lower() for item in doc_meta.get("files") or []]
        if files and not any(path in filename for filename in files):
            return False
        # 没有 files 字段（如文档类）→ 在所有元数据值里搜（兜底策略）
        if not files and path not in " ".join(str(value).lower() for value in doc_meta.values()):
            return False

    # ── 时间区间过滤 ──
    start_at = _parse_datetime(filters.get("start_date") or filters.get("updated_after"))
    end_at = _parse_datetime(filters.get("end_date") or filters.get("updated_before"), end_of_day=True)
    doc_time = _metadata_time(doc_meta)
    # 有区间但文档没有时间 → 排除（无法证明它在区间内）
    if start_at and (doc_time is None or doc_time < start_at):
        return False
    if end_at and (doc_time is None or doc_time > end_at):
        return False

    # ── 其余键值：逐一精确匹配 ──
    for key, value in filters.items():
        # 上面已处理过的键跳过
        if key in {
            "label",
            "path",
            "path_prefix",
            "module",
            "start_date",
            "end_date",
            "updated_after",
            "updated_before",
        } or value in (None, ""):
            continue
        # 分支特殊处理：PR 有 base/head 两个分支字段
        if key == "branch":
            current = doc_meta.get("branch") or doc_meta.get("base_branch") or doc_meta.get("head_branch")
            if current != value:
                return False
            continue
        current = doc_meta.get(key)
        # 值是列表 → 做"包含"判断（如 labels / assignees）
        if isinstance(current, list):
            if value not in current:
                return False
        elif current != value:
            return False
    return True


def _tokens(text: str) -> list[str]:
    """分词：英文按词，中文按单字 + 双字。

    【调用链】被 _bm25_document_stats() / _candidate_term_set() / _keyword_candidates() 调用

    【为什么中文要切单字 + 双字】
        中文无空格，单字保证召回、双字保证精度。详见 rerank.py 的同名函数注释。
    """
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        # 把中文拆成【单字】，"我爱计算机" → ["我", "爱", "计", "算", "机"]
        tokens.extend(segment)
        # 滑动窗口拿【连续两个字】
        # "我爱计算机" 截取：
        # 0:我+爱 → 我爱
        # 1:爱+计 → 爱计
        # 2:计+算 → 计算
        # 3:算+机 → 算机
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _clip(text: str, limit: int = 260) -> str:
    """压缩空白并截断（用于 snippet 预览）。

    【调用链】被 _document_result() 调用

    limit-1 是给省略号留位置。用 "…" 而不是 "..." 是因为单字符更省。
    """
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _normalise_scores(items: list[dict[str, Any]], field: str) -> None:
    """把某一列分数按最大值归一化到 0~1（让两路召回的分数可比）。

    【调用链】被 _milvus_vector_candidates() / _keyword_candidates() 调用

    【为什么必须归一化】BM25 的原始分可能是 0~30，向量余弦是 0~1。
        不归一化直接加权，BM25 会完全压过向量分。

    【为什么用"除以最大值"而不是 min-max】
        min-max 会把最低分也拉到 0，丢掉了"最低分有多低"的信息。
        除以最大值保留了相对比例，而且所有分都变到 0~1。

    【注意】这是原地修改（直接改 item[field]），不返回新列表。
    """
    if not items:
        return
    values = [max(0.0, float(item.get(field) or 0.0)) for item in items]
    # max_value 为 0 时用 1.0 兜底，避免除零
    max_value = max(values) or 1.0
    for item, value in zip(items, values):
        item[field] = round(value / max_value, 4)


def _document_result(doc: Document, score: float, source: str, score_field: str) -> dict[str, Any]:
    """把 ORM 对象转成统一的候选结构。

    【调用链】_clip() → 本函数
    【被谁调用】_keyword_candidates()（向量召回走 Milvus，结构本来就对）

    【统一的候选结构（两路必须一致，否则融合时字段对不上）】
        id / title / source_type / source_id / score / metadata
        snippet       预览（截断，给用户看）
        _raw_content  完整内容（父块扩展、去重要用）
        <score_field> 该路的分数（vector_score 或 keyword_score）
        retrieval     检索明细（各阶段分数，前端"检索明细"面板读它）

    【为什么 snippet 和 _raw_content 都要】
        snippet 省 token（给 prompt 用），_raw_content 保精度（去重/扩展用）。
    """
    metadata = doc.meta or {}
    result = {
        "id": str(doc.id),
        "title": doc.title,
        "source_type": doc.source_type,
        "source_id": str(doc.source_id),
        "score": round(max(0.0, score), 4),
        "metadata": metadata,
        "snippet": _clip(doc.content),        # 预览（截断）
        "_raw_content": doc.content,          # 完整内容（父块扩展/去重要用）
        score_field: round(max(0.0, score), 4),
        "retrieval": {"sources": [source], score_field: round(max(0.0, score), 4)},
    }
    return result


# ════════════════════════════════════════════════════════════════
# ① 向量召回
# ════════════════════════════════════════════════════════════════

def _milvus_vector_candidates(
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """向量召回（走 Milvus）。

    【调用链】追这个函数会进入：
        1. vector_store.search_milvus_documents()  → Milvus 检索
        2. _matches_metadata()                     → 应用层元数据过滤
        3. _normalise_scores()                     → 分数归一化
    【被谁调用】search_similar_documents()（第 ① 步）

    【数据流向】
        入：repo_id、query、source_types、fetch_limit、metadata_filters、query_embedding
        出：候选列表，每条带 vector_score（已归一化）+ vector_rank

    【为什么 source_types 分两种情况传】
        单一类型用 == 过滤，多类型用 in 过滤 ——
        Milvus 对 == 的优化更好，表达式也更简单。

    【为什么元数据过滤要在应用层做】
        Milvus 只支持有限的标量过滤，复杂的 JSON 条件做不到。
        所以先让 Milvus 按 repo_id + source_type 粗筛，
        再在 Python 里按 metadata 精筛。
    """
    milvus_results = search_milvus_documents(
        repo_id,
        query,
        # 单一类型用 == 过滤，多类型用 in 过滤（表达式更简单，Milvus 也更快）
        source_type=source_types[0] if len(source_types) == 1 else None,
        source_types=source_types if len(source_types) > 1 else None,
        limit=fetch_limit,
        query_embedding=query_embedding,
    )

    candidates = []
    for item in milvus_results:
        # Milvus 不支持复杂 JSON 过滤，元数据过滤在应用层做，这里就要进行源数据过滤
        if not _matches_metadata(item.get("metadata") or {}, metadata_filters):
            continue
        score = max(0.0, float(item.get("score") or 0.0))
        candidates.append(
            {
                # 字典解包，把字典 `item` 里面所有的 key-value，全部摊开，塞到这个新字典里
                **item,
                # Milvus 返回的字段叫 document_id，这里统一成 id
                "id": str(item.get("id") or item.get("source_id") or ""),
                "score": score, # 对外统一的分数
                "vector_score": score,
                "vector_score_raw": score,      # 保留原始分（调试用）
                # retrieval标记这条候选是从哪里召回的
                "retrieval": {
                    "sources": ["vector"],
                    "vector_score_raw": score,
                    "vector_backend": "milvus",
                },
            }
        )

    # 归一化到 0~1（让两路分数可比）
    _normalise_scores(candidates, "vector_score")

    # 记录排名（调试时能看"向量路排第几"）
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["vector_score"]
        item["retrieval"]["vector_score"] = item["vector_score"]
        item["vector_rank"] = rank
        item["retrieval"]["vector_rank"] = rank
    return candidates


# ════════════════════════════════════════════════════════════════
# ② 关键词召回（BM25）
# ════════════════════════════════════════════════════════════════

def _bm25_document_stats(doc: Document, query_terms: list[str]) -> dict[str, Any]:
    """计算单个文档的 BM25 中间量（词频、命中词、文档长度）。

    【调用链】_tokens() → 本函数
    【被谁调用】_keyword_candidates()

    【字段加权是这里的关键设计】
        title 4.0 / symbol 4.5 / path 3.0 / labels 2.5 / content 1.0
        含义：同一个词出现在标题里，比出现在正文里重要 4 倍。
        对代码检索尤其有效 —— 函数名命中几乎一定相关。

    【数据流向】
        入：Document 对象、query_terms（去重后的查询词）
        出：{doc, weighted_tf, matched_terms, document_length}

    【weighted_tf 怎么算】
        一个词在多个字段出现时，权重累加。
        比如 "auth" 在 title 出现 1 次（×4.0）+ 在 content 出现 3 次（×1.0）= 7.0
    """
    metadata = doc.meta or {}

    # 五个字段各自分词
    field_tokens = {
        "title": _tokens(doc.title or ""),
        "path": _tokens(str(metadata.get("path") or "")),
        "symbol": _tokens(str(metadata.get("symbol") or metadata.get("name") or "")),
        "labels": _tokens(" ".join(str(item) for item in metadata.get("labels") or [])),
        "content": _tokens(doc.content or ""),
    }
    # symbol权重最大，因为在代码/类中出现是最直接的
    field_weights = {"title": 4.0, "path": 3.0, "symbol": 4.5, "labels": 2.5, "content": 1.0}

    # 每个字段的词频统计
    # 例如：content 分词结果是`["auth", "login", "auth"]` → `counters["content"]["auth"] = 2`
    counters = {field: Counter(tokens) for field, tokens in field_tokens.items()}

    # 加权词频：一个词在多个字段出现，权重累加
    weighted_tf = {
        term: sum(float(counters[field].get(term, 0)) * field_weights[field] for field in field_tokens)
        for term in query_terms
    }

    # 跨字段累加，同一个词在标题 + 正文同时出现，分数相加
    matched_terms = [term for term in query_terms if weighted_tf.get(term, 0.0) > 0.0]
    # 文档长度 = 所有字段的词数之和（用于 BM25 的长度归一化）
    document_length = max(1, sum(len(tokens) for tokens in field_tokens.values()))

    return {
        "doc": doc,
        "weighted_tf": weighted_tf,
        "matched_terms": matched_terms,
        "document_length": document_length,
    }


def _bm25_score(
    stats: dict[str, Any],
    *,
    document_count: int,
    document_frequency: Counter[str],
    average_document_length: float,
) -> float:
    """标准 BM25 打分公式。

    【调用链】被 _keyword_candidates() 调用

    score = Σ idf(term) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × len/avg_len))

    三段含义：
        idf  → 稀有词权重高（"the" 几乎没权重，"auth_service" 权重很高）
        tf   → 词频饱和（出现 10 次不等于 10 倍重要）
        长度归一化 → 长文档不占便宜

    【为什么用 log1p 而不是 log】
        标准 BM25 的 idf = log((N - df + 0.5) / (df + 0.5))。
        当 df 接近 N（词几乎出现在所有文档里）时，括号内接近 1，log 接近 0；
        但如果 df 超过 N（理论上不会，但脏数据可能出现），会得到负数 → 负分。
        log1p(x) = log(1+x)，保证结果非负，更稳健。

    【为什么 tf 要饱和】
        一个词出现 1 次和 2 次差别很大，但 10 次和 11 次几乎没差别。
        (tf × (k1+1)) / (tf + length_norm) 这个式子天然实现了饱和。
    """
    score = 0.0
    document_length = float(stats["document_length"])

    # 长度归一化项：文档越长，这个值越大，分数越低
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / max(average_document_length, 1.0))

    for term, weighted_tf in stats["weighted_tf"].items():
        if weighted_tf <= 0:
            continue
        df = float(document_frequency.get(term, 0))
        # log1p 保证 df 接近总数时 idf 仍为正（避免负分）
        idf = math.log1p((document_count - df + 0.5) / (df + 0.5))
        score += idf * (weighted_tf * (BM25_K1 + 1.0)) / (weighted_tf + length_norm)

    return max(0.0, score)


def _keyword_candidates(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
) -> list[dict[str, Any]]:
    """关键词召回（BM25，直接查 PostgreSQL）。

    【调用链】追这个函数会进入：
        1. _tokens()                → 查询分词
        2. _matches_metadata()      → 元数据过滤
        3. _bm25_document_stats()   → 算每个文档的词频
        4. _bm25_score()            → 打分
        5. _document_result()       → 转候选结构
        6. _normalise_scores()      → 归一化
    【被谁调用】search_similar_documents()（第 ② 步）

    【数据流向】
        入：db、repo_id、query、source_types、fetch_limit、metadata_filters
        出：候选列表，每条带 keyword_score（已归一化）+ keyword_rank

    【为什么用 heapq 而不是全排序】
        仓库可能有几万条文档，全排序 O(n log n) 且占内存。
        用大小为 fetch_limit 的最小堆，只需 O(n log k)，内存也只占 k。

    【为什么用 yield_per(500)】
        一次把几万条 ORM 对象读进内存会爆。分批取，处理完就释放。

    【注意：这里算的是"全库 BM25"，不是"候选集 BM25"】
        因为 idf 需要全局文档频率（df）才有意义。
        所以必须扫一遍全仓库的文档 —— 这是 BM25 的固有成本。
    """
    # 查询词去重保序（重复词不该重复计分）
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms:
        return []

    q = db.query(Document).filter(
        Document.repo_id == _as_uuid(repo_id),
        Document.source_type.in_(source_types),
    )
    # 能在 SQL 层过滤的先过滤掉（state 是最常见的过滤条件）
    state = (metadata_filters or {}).get("state")
    if state:
        q = q.filter(Document.meta["state"].as_string() == str(state))

    corpus: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    total_document_length = 0

    # yield_per 分批取，避免一次把整个仓库的文档读进内存
    for doc in q.yield_per(500):
        # 剩下的复杂条件在应用层过滤，应用层过滤元数据
        if not _matches_metadata(doc.meta or {}, metadata_filters):
            continue
        stats = _bm25_document_stats(doc, terms)
        corpus.append(stats)
        total_document_length += int(stats["document_length"])
        # 文档频率：这个词出现在多少个文档里（用 set 保证一个文档只算一次）
        document_frequency.update(set(stats["matched_terms"]))

    if not corpus:
        return []

    average_document_length = total_document_length / len(corpus)

    # ── 用最小堆维护 top-K ──
    # 堆里存 (score, sequence, item)，sequence 用于同分时稳定排序
    top_candidates: list[tuple[float, int, dict[str, Any]]] = []
    for sequence, stats in enumerate(corpus):
        doc = stats["doc"]
        score = _bm25_score(
            stats,
            document_count=len(corpus),
            document_frequency=document_frequency,
            average_document_length=average_document_length,
        )
        # 0 分说明一个查询词都没命中，直接丢
        if score <= 0:
            continue

        item = _document_result(doc, score, "keyword", "keyword_score")
        item["bm25_score"] = round(score, 4)
        item["bm25_matched_terms"] = list(stats["matched_terms"])
        item["retrieval"].update(
            {
                "keyword_backend": "bm25",
                "bm25_score": round(score, 4),
                "bm25_matched_terms": list(stats["matched_terms"]),
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
            }
        )

        # 维护大小为 fetch_limit 的最小堆
        entry = (score, sequence, item)
        if len(top_candidates) < fetch_limit:
            heapq.heappush(top_candidates, entry)
        elif score > top_candidates[0][0]:
            # 比堆顶（当前最小的）大 → 替换掉
            heapq.heapreplace(top_candidates, entry)

    # 堆里是乱序的，要排好（分数降序，同分按 sequence 升序保证稳定）
    candidates = [
        entry[2]
        for entry in sorted(top_candidates, key=lambda entry: (entry[0], entry[1]), reverse=True)
    ]

    _normalise_scores(candidates, "keyword_score")

    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["keyword_score"]
        item["retrieval"]["keyword_score"] = item["keyword_score"]
        item["keyword_rank"] = rank
        item["retrieval"]["keyword_rank"] = rank
    return candidates


# ════════════════════════════════════════════════════════════════
# ③ 加权融合（混合检索的核心）
# ════════════════════════════════════════════════════════════════

def _merge_candidates(vector_items: list[dict[str, Any]], keyword_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把两路召回结果合并、加权、排序 —— 混合检索的核心。

    【调用链】被 search_similar_documents() 调用（第 ③ 步）

    合并键的优先级：chunk_id > id > (source_type:source_id:title)
        用 chunk_id 是为了让"同一个切片"在两路里被识别为同一条。

    加权公式：
        combined = min(1.0, 0.62 × vector + 0.30 × keyword + 0.08 × 双路命中)

    【数据流向】
        入：vector_items（向量路候选）、keyword_items（关键词路候选）
        出：融合后的候选列表（按 combined_score 降序），每条多两个字段：
              combined_score  融合分（= 最终 score）
              fusion_rank     融合排名

    【为什么要"取最大值"而不是"相加"】
        同一条切片被两路召回时，它在两路的分数不该叠加
        （那会让"被两路召回"这一事实被计算两次：一次是取max，一次是 +0.08 奖励）。
        所以各项分数取 max，只用一个显式的 dual_recall_bonus 表达"双路命中"。

    【举例验证】
        vector=0.9, keyword=0.8 →
        0.62×0.9 + 0.30×0.8 + 0.08 = 0.558 + 0.24 + 0.08 = 0.878
    """
    merged: dict[str, dict[str, Any]] = {}

    for source_name, items in [("vector", vector_items), ("keyword", keyword_items)]:
        for item in items:
            metadata = item.get("metadata") or {}
            # 合并键：优先用 chunk_id（切片级别唯一）
            key = str(
                metadata.get("chunk_id")
                or item.get("id")
                or f"{item.get('source_type')}:{item.get('source_id')}:{item.get('title')}"
            )

            current = merged.get(key)
            if current is None:
                # 第一次见到这条 → 直接放进去
                current = {**item, "retrieval": {**(item.get("retrieval") or {})}}
                current["retrieval"]["sources"] = list(dict.fromkeys(current["retrieval"].get("sources") or [source_name]))
                merged[key] = current
            else:
                # ── 两路都命中：各项分数取最大值，内容取更长的那个 ──
                current["vector_score"] = max(float(current.get("vector_score") or 0.0), float(item.get("vector_score") or 0.0))
                current["keyword_score"] = max(float(current.get("keyword_score") or 0.0), float(item.get("keyword_score") or 0.0))
                current["vector_score_raw"] = max(
                    float(current.get("vector_score_raw") or 0.0), float(item.get("vector_score_raw") or 0.0)
                )
                current["bm25_score"] = max(float(current.get("bm25_score") or 0.0), float(item.get("bm25_score") or 0.0))
                # 内容取更长：Milvus 里是截断版，PG 里是完整版 → 取完整的
                current["snippet"] = max([str(current.get("snippet") or ""), str(item.get("snippet") or "")], key=len)
                current["_raw_content"] = max(
                    [str(current.get("_raw_content") or ""), str(item.get("_raw_content") or "")], key=len
                )
                # 记录"这条被哪几路召回"
                current["retrieval"]["sources"] = list(dict.fromkeys([*(current["retrieval"].get("sources") or []), source_name]))
                # 合并检索明细（sources 已在上面处理，跳过）
                for trace_key, trace_value in (item.get("retrieval") or {}).items():
                    if trace_key == "sources":
                        continue
                    current["retrieval"][trace_key] = trace_value

    # ── 逐条算融合分 ──
    output = []
    for item in merged.values():
        vector_score = float(item.get("vector_score") or 0.0)
        keyword_score = float(item.get("keyword_score") or 0.0)
        # 双路都召回 → 额外加分（强信号）
        both_bonus = DUAL_RECALL_BONUS if vector_score and keyword_score else 0.0
        combined = min(1.0, VECTOR_FUSION_WEIGHT * vector_score + KEYWORD_FUSION_WEIGHT * keyword_score + both_bonus)
        item["combined_score"] = round(combined, 4)
        item["fusion_score"] = round(combined, 4)
        item["score"] = round(combined, 4)
        item["retrieval"].update(
            {
                "fusion_method": "weighted",
                "fusion_weights": {
                    "vector": VECTOR_FUSION_WEIGHT,
                    "keyword": KEYWORD_FUSION_WEIGHT,
                    "dual_recall_bonus": DUAL_RECALL_BONUS,
                },
                "dual_recall_bonus": both_bonus,
                "combined_score": item["combined_score"],
                "fusion_score": item["fusion_score"],
            }
        )
        output.append(item)

    # 按融合分降序
    output.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
    for rank, item in enumerate(output, start=1):
        item["fusion_rank"] = rank
        item["retrieval"]["fusion_rank"] = rank
    return output


# ════════════════════════════════════════════════════════════════
# ④ 去重
# ════════════════════════════════════════════════════════════════

def _normalised_fingerprint(text: str) -> str:
    """内容指纹：去掉所有标点空白后算 SHA-256（用于精确重复检测）。

    【调用链】被 _deduplicate_candidates() / expand_parent_context() 调用

    【为什么要先去掉标点空白】
        同一段话可能因为换行、空格、标点差异而"看起来不同"，
        但语义完全一样。去掉这些差异才能识别为重复。
    """
    normalised = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", str(text or "").lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest() if normalised else ""


def _candidate_term_set(item: dict[str, Any]) -> set[str]:
    """取候选的词集（用于 Jaccard 相似度）。

    【调用链】_tokens() → 本函数
    【被谁调用】_deduplicate_candidates()
    """
    text = str(item.get("_raw_content") or item.get("snippet") or "")
    return set(_tokens(text))


def _same_duplicate_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两条候选是否"同一作用域"（只有同作用域才允许判重）。

    【调用链】被 _deduplicate_candidates() 调用

    【为什么需要】不同仓库/不同来源可能有完全相同的文本（如两处都写"请先安装依赖"），
        那不算重复，不该去重。

    【判定顺序】
        ① 两边都有 parent_id → 比 parent_id（最精确，同一父块的切片才算同作用域）
        ② 否则 → 比 source_type + source_id（同一源对象）
    """
    left_meta = left.get("metadata") or {}
    right_meta = right.get("metadata") or {}
    left_parent = str(left_meta.get("parent_id") or "")
    right_parent = str(right_meta.get("parent_id") or "")
    if left_parent and right_parent:
        return left_parent == right_parent
    return (
        str(left.get("source_type") or "") == str(right.get("source_type") or "")
        and str(left.get("source_id") or "") == str(right.get("source_id") or "")
    )


def _deduplicate_candidates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """两层去重：精确指纹 + 近似 Jaccard。

    【调用链】_normalised_fingerprint() / _candidate_term_set() / _same_duplicate_scope() → 本函数
    【被谁调用】search_similar_documents()（第 ④ 步）

    第一层：内容归一化后哈希完全相同 → 精确重复
    第二层：词集 Jaccard 相似度 >= 0.9 → 近似重复
        （典型场景：同一段话被切在两个相邻切片的重叠区里）

    【数据流向】
        入：候选列表（已按融合分排序）
        出：(保留的, 被去掉的)
            被去掉的记录 {id, duplicate_of}
            保留的那条在 retrieval.duplicates_removed 里记下"合并了谁"（可追溯）

    【为什么保留"最先出现的"而不是"分数最高的"】
        输入已经按融合分降序，所以先出现的就是分数最高的 —— 天然正确。

    【复杂度】第二层是 O(n²)（每条和已保留的每条比）。
        但 n 通常只有几十到两百（fetch_limit 上限 200），可接受。
    """
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    # 缓存词集，避免重复分词
    term_sets: dict[str, set[str]] = {}

    for item in items:
        item_id = str(item.get("id") or "")
        raw_text = str(item.get("_raw_content") or item.get("snippet") or "")

        # ── 第一层：精确指纹 ──
        fingerprint = _normalised_fingerprint(raw_text)
        duplicate_of: dict[str, Any] | None = fingerprints.get(fingerprint) if fingerprint else None
        # 指纹相同但不同作用域 → 不算重复
        if duplicate_of is not None and not _same_duplicate_scope(item, duplicate_of):
            duplicate_of = None

        # ── 第二层：近似 Jaccard ──
        if duplicate_of is None:
            current_terms = _candidate_term_set(item)
            # 近似重复：和已保留的每条比 Jaccard
            for existing in kept:
                if not _same_duplicate_scope(item, existing):
                    continue
                existing_id = str(existing.get("id") or "")
                existing_terms = term_sets.get(existing_id) or _candidate_term_set(existing)
                if not current_terms or not existing_terms:
                    continue
                union = current_terms | existing_terms
                similarity = len(current_terms & existing_terms) / max(len(union), 1)
                if similarity >= NEAR_DUPLICATE_JACCARD:
                    duplicate_of = existing
                    break
            term_sets[item_id] = current_terms

        # ── 判为重复 → 丢弃，但留下追溯记录 ──
        if duplicate_of is not None:
            # 记录"谁被谁合并了"（可追溯）
            duplicate_id = str(duplicate_of.get("id") or "")
            duplicate_trace = duplicate_of.setdefault("retrieval", {})
            duplicate_trace.setdefault("duplicates_removed", []).append(item_id)
            removed.append({"id": item_id, "duplicate_of": duplicate_id})
            continue

        kept.append(item)
        # 只记录"第一个"用这个指纹的（后续同指纹的都算它的重复）
        if fingerprint:
            fingerprints[fingerprint] = item

    return kept, removed


def _limit_parent_occupancy(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """限制同一父块的切片数量（最多 2 个）。

    【调用链】被 search_similar_documents() 调用（第 ⑤ 步）

    【为什么需要】一个 10 页的章节切成 20 块，如果全被召回，
        上下文会被这一章占满，其他相关文档挤不进来。
        限制每个父块最多 2 块，保证来源多样性。

    【数据流向】
        入：去重后的候选（已排序）
        出：(保留的, 被去掉的)，被去掉的记录 {id, parent_id}

    没有 parent_id 的候选（如整条 Issue）不受限制，全部保留。
    """
    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for item in items:
        metadata = item.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")
        # 该父块已经满了 → 丢掉
        if parent_id and counts[parent_id] >= MAX_CHUNKS_PER_PARENT:
            removed.append({"id": str(item.get("id") or ""), "parent_id": parent_id})
            continue
        kept.append(item)
        if parent_id:
            counts[parent_id] += 1
    return kept, removed


# ════════════════════════════════════════════════════════════════
# 父块扩展（"小切片检索，大上下文回答"）
# ════════════════════════════════════════════════════════════════

def _same_version(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两份元数据是否属于同一版本（分支/提交/校验和一致）。

    【调用链】被 expand_parent_context() 调用

    【为什么需要】同一个文件可能在不同分支/不同提交下被索引过多次。
        把不同版本的切片拼在一起，会得到自相矛盾的上下文。
        所以扩展时必须校验版本一致性。

    【逻辑】只要有一方缺字段就跳过比较（无法判断时倾向于允许）。
        所有共同字段都相同才算同版本。
    """
    for key in ("branch", "commit_sha", "version", "checksum", "knowledge_document_id"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    return True


def expand_parent_context(
    db: Session,
    repo_id: str | uuid.UUID,
    items: list[dict[str, Any]],
    *,
    neighbor_window: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """父块扩展：把命中切片的相邻兄弟块拼回来。

    【调用链】追这个函数会进入：
        1. _as_uuid()                  → ID 转换
        2. _same_version()             → 版本一致性校验
        3. _normalised_fingerprint()   → 拼接时去重
    【被谁调用】services/rag/knowledge_base.py::retrieve_knowledge()（六阶段第 ③ 步）

    【为什么需要】切片会切断语义。命中"第 5 块"时，第 4、6 块往往是同一段论述，
        拼回来能让模型看到完整上下文。
        这就是"父子块"设计的价值：用小切片检索（准），用大上下文回答（全）。

    【数据流向】
        入：db、repo_id、items（重排后的候选）、neighbor_window（前后各取几块）
        出：(扩展后的候选列表, 阶段统计)
            扩展后的 _raw_content 变成"相邻块拼接"的内容
            统计 {name, input_count, output_count, expanded_items, additional_chunks}

    【三种"不扩展"的情况（都会写明 reason，便于排查）】
        ① 没有 parent 元数据（如整条 Issue 入库，本身就是一个完整块）
        ② source_id 不是合法 UUID
        ③ 找不到兄弟块 / 兄弟块内容全空

    【为什么要按 child_index 排序后再拼】
        数据库返回顺序不保证。按 child_index 排才能保证
        "第 4、5、6 块"按正确顺序拼成一段通顺的上下文。
    """
    expanded: list[dict[str, Any]] = []
    expanded_items = 0
    expanded_chunks = 0

    for item in items:
        metadata = dict(item.get("metadata") or {})
        parent_id = str(metadata.get("parent_id") or "")
        child_index = metadata.get("child_index")
        source_id = item.get("source_id")
        source_type = str(item.get("source_type") or "")
        retrieval = dict(item.get("retrieval") or {})

        # ── 情况 ①：没有父块信息（比如 Issue 这种整条入库的）→ 不扩展 ──
        if not parent_id or child_index is None or not source_id or not source_type:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "parent metadata unavailable"}
            expanded.append({**item, "retrieval": retrieval})
            continue

        try:
            source_uuid = _as_uuid(source_id)
        except (TypeError, ValueError):
            # ── 情况 ②：source_id 不是合法 UUID ──
            retrieval["parent_expansion"] = {"expanded": False, "reason": "invalid source id"}
            expanded.append({**item, "retrieval": retrieval})
            continue

        # 取同一来源的所有切片（一次查完，避免 N+1）
        siblings = (
            db.query(Document)
            .filter(
                Document.repo_id == _as_uuid(repo_id),
                Document.source_type == source_type,
                Document.source_id == source_uuid,
            )
            .all()
        )

        selected: list[Document] = []
        current_index = int(child_index)
        for sibling in siblings:
            sibling_meta = sibling.meta or {}
            # 必须同一个父块
            if str(sibling_meta.get("parent_id") or "") != parent_id:
                continue
            if not _same_version(metadata, sibling_meta):
                continue        # 不同版本不能混拼（否则会把旧版内容带进来）
            sibling_index = int(sibling_meta.get("child_index") or 0)
            # 前后 neighbor_window 块之内
            if abs(sibling_index - current_index) <= max(0, neighbor_window):
                selected.append(sibling)

        # 按 child_index 排序，保证拼接顺序正确
        selected.sort(key=lambda row: int((row.meta or {}).get("child_index") or 0))

        # ── 情况 ③：没有兄弟块 ──
        if not selected:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "no matching siblings"}
            expanded.append({**item, "retrieval": retrieval})
            continue

        # ── 拼接相邻块内容（去重）──
        parts: list[str] = []
        seen_parts: set[str] = set()
        expanded_ids: list[str] = []
        for sibling in selected:
            content = str(sibling.content or "").strip()
            # 用指纹去重：重叠区的内容会在相邻块里重复出现
            fingerprint = _normalised_fingerprint(content)
            if not content or fingerprint in seen_parts:
                continue
            seen_parts.add(fingerprint)
            parts.append(content)
            expanded_ids.append(str(sibling.id))

        if not parts:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "empty sibling content"}
            expanded.append({**item, "retrieval": retrieval})
            continue

        # 只有真的多拼进了内容才算"扩展了"
        parent_expanded = len(expanded_ids) > 1 or str(item.get("id") or "") not in expanded_ids

        retrieval["parent_expansion"] = {
            "expanded": parent_expanded,
            "parent_id": parent_id,
            "chunk_ids": expanded_ids,
            "neighbor_window": max(0, neighbor_window),
        }
        # 把扩展进来的块 ID 也写进 metadata（便于追溯引用来源）
        next_metadata = {**metadata, "expanded_chunk_ids": expanded_ids}

        expanded.append(
            {
                **item,
                "metadata": next_metadata,
                "_raw_content": "\n\n".join(parts),     # 拼接后的完整上下文
                "parent_expanded": parent_expanded,
                "retrieval": retrieval,
            }
        )

        if parent_expanded:
            expanded_items += 1
            # 净增的块数（去掉自己那一块）
            expanded_chunks += max(0, len(expanded_ids) - 1)

    return expanded, {
        "name": "parent_expansion",
        "input_count": len(items),
        "output_count": len(expanded),
        "expanded_items": expanded_items,
        "additional_chunks": expanded_chunks,
    }


# ════════════════════════════════════════════════════════════════
# ★ 总入口
# ════════════════════════════════════════════════════════════════

def search_similar_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 5,
    metadata_filters: dict | None = None,
    query_embedding: list[float] | None = None,
    retrieval_method: str = "hybrid",
    apply_rerank: bool = True,
    apply_compression: bool = True,
    source_types: list[str] | None = None,
    pipeline_trace: dict[str, Any] | None = None,
) -> list[dict]:
    """【混合检索总入口】

    【调用链】追这个函数会进入（按执行顺序）：
        ① _milvus_vector_candidates()      向量召回 → vector_store.search_milvus_documents()
        ② _keyword_candidates()            关键词召回
              ├ _bm25_document_stats()     算词频
              ├ _bm25_score()              BM25 打分
              └ _document_result()         转候选结构
        ③ _merge_candidates()              加权融合 0.62/0.30/0.08
        ④ _deduplicate_candidates()        两层去重
        ⑤ _limit_parent_occupancy()        父块占用限制
        ⑥ rerank.rerank_documents()        重排
        ⑦ context_compression.compress_retrieval_results()  压缩

    【被谁调用】
        services/rag/knowledge_base.py::retrieve_knowledge()  ← 主路径
        services/agents/chat_agent.py 的 rag 工具             ← Agent 直接检索

    【数据流向】
        入：db、repo_id、query、limit、过滤条件、retrieval_method（hybrid/vector/fulltext）、
            apply_rerank / apply_compression 两个开关、pipeline_trace（可选，出参）
        出：候选列表（已重排+压缩），每条含 snippet / score / retrieval 明细

    完整链路：
        ① 向量召回（Milvus）  ┐
        ② 关键词召回（BM25）  ├→ ③ 加权融合 → ④ 去重 → ⑤ 父块占用限制 → ⑥ 重排 → ⑦ 压缩
                              ┘

    pipeline_trace 会把每一步的候选数量都记下来，排查"为什么没结果"时非常有用。

    【三个模式的区别】
        hybrid    → 两路都跑，融合（默认，最准）
        vector    → 只跑向量（Milvus 挂了直接报错）
        fulltext  → 只跑 BM25（不需要 Milvus，纯数据库）
    """
    # 没指定来源类型时，默认搜所有"可检索"类型
    requested_types = normalize_source_types([source_type, *(source_types or [])]) or sorted(RAG_SOURCE_TYPES)
    # 再过滤一遍：只有"可检索"的才留下（source_policy 决定）
    searchable_types = list(dict.fromkeys(item for item in requested_types if is_searchable_source(item)))
    if not searchable_types:
        return []

    # 召回候选数 = limit × 8，夹在 40~200（要留足够余量给融合和重排）
    # 为什么 ×8：两路各召回一批，融合后会掉一些，重排后再选 limit 条
    fetch_limit = min(max(limit * 8, 40), 200)

    method = retrieval_method.strip().lower()
    vector_types = searchable_types
    vector_candidates: list[dict[str, Any]] = []
    keyword_candidates: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    # ════════════════════════════════════════════════════════════
    # ── ① 向量召回 ──
    # ════════════════════════════════════════════════════════════
    vector_status = "skipped"
    if method in {"vector", "hybrid"} and vector_types:
        try:
            vector_candidates = _milvus_vector_candidates(
                repo_id,
                query,
                vector_types,
                fetch_limit,
                metadata_filters,
                query_embedding=query_embedding,
            )
            vector_status = "success"
        except MilvusUnavailableError:
            vector_status = "unavailable"
            # hybrid 模式下 Milvus 挂了还能用关键词兜底；纯 vector 模式只能报错
            if method == "vector":
                raise

    stages.append(
        {
            "name": "vector_recall",
            "status": vector_status,
            "candidate_count": len(vector_candidates),
            "backend": "milvus",
        }
    )

    # ════════════════════════════════════════════════════════════
    # ── ② 关键词召回 ──
    # ════════════════════════════════════════════════════════════
    keyword_types = searchable_types if method in {"fulltext", "hybrid"} else []
    if keyword_types:
        keyword_candidates = _keyword_candidates(
            db,
            repo_id,
            query,
            keyword_types,
            fetch_limit,
            metadata_filters,
        )
    stages.append(
        {
            "name": "keyword_recall",
            "status": "success" if keyword_types else "skipped",
            "candidate_count": len(keyword_candidates),
            "backend": "bm25",
        }
    )

    # ════════════════════════════════════════════════════════════
    # ── ③ 融合 ──
    # ════════════════════════════════════════════════════════════
    if method == "vector":
        merged = vector_candidates
    elif method == "fulltext":
        merged = keyword_candidates
    else:
        merged = _merge_candidates(vector_candidates, keyword_candidates)

    stages.append(
        {
            "name": "weighted_fusion" if method == "hybrid" else "single_retriever_selection",
            "method": "weighted" if method == "hybrid" else method,
            "candidate_count": len(merged),
            "weights": {
                "vector": VECTOR_FUSION_WEIGHT,
                "keyword": KEYWORD_FUSION_WEIGHT,
                "dual_recall_bonus": DUAL_RECALL_BONUS,
            },
        }
    )

    # ════════════════════════════════════════════════════════════
    # ── ④⑤ 去重 + 父块占用限制 ──
    # ════════════════════════════════════════════════════════════
    deduplicated, duplicates_removed = _deduplicate_candidates(merged)
    aggregated, parent_overflow_removed = _limit_parent_occupancy(deduplicated)

    stages.append(
        {
            "name": "deduplication_and_parent_aggregation",
            "input_count": len(merged),
            "output_count": len(aggregated),
            "duplicates_removed": duplicates_removed,
            "parent_overflow_removed": parent_overflow_removed,
        }
    )

    # ════════════════════════════════════════════════════════════
    # ── ⑥ 重排 ──
    # ════════════════════════════════════════════════════════════
    ranked = rerank_documents(query, aggregated, limit) if apply_rerank else aggregated[:limit]
    if apply_rerank:
        stages.append(
            {
                "name": "rerank",
                "input_count": len(aggregated),
                "output_count": len(ranked),
                # 记录走的哪种重排（heuristic / qwen），便于判断是否降级了
                "mode": ranked[0].get("retrieval", {}).get("rerank_mode") if ranked else None,
            }
        )

    # 把 trace 写回调用方传进来的 dict（出参，不改返回值结构）
    if pipeline_trace is not None:
        pipeline_trace.update(
            {
                "retrieval_method": method,
                "fusion_method": "weighted" if method == "hybrid" else method,
                "stages": stages,
            }
        )

    # ════════════════════════════════════════════════════════════
    # ── ⑦ 压缩 ──
    # ════════════════════════════════════════════════════════════
    if not apply_compression:
        return ranked

    compressed = compress_retrieval_results(query, ranked, max_tokens_per_item=260)

    if pipeline_trace is not None:
        pipeline_trace["stages"].append(
            {
                "name": "context_compression",
                "input_count": len(ranked),
                "output_count": len(compressed),
            }
        )
    return compressed
