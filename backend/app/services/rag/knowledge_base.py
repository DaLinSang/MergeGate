"""
知识库配置与检索总编排
====================

【在 RAG 流水线中的位置】
    这是**查询链路的"总导演"**——把前面所有零件串成一条六阶段流水线：

        用户问题
             ↓
        ★ 本文件 retrieve_knowledge()
             ① 混合召回（向量 + BM25）      → retrieval.search_similar_documents()
             ② 重排（Qwen API，失败降级）    → _qwen_rerank() / rerank.rerank_documents()
             ③ 父块扩展（补回完整章节）      → retrieval.expand_parent_context()
             ④ 分数阈值过滤
             ⑤ 上下文压缩（按 token 预算）   → context_compression.compress_retrieval_results()
             ⑥ 准入判定（够不够回答）        → answer_gate.evaluate_answer_gate()
             ↓
        (items, trace, duration_ms)  → qa.answer_question()

    【读这个文件 = 看懂整个 RAG 查询】
        因为它是唯一一个"能看到全部阶段"的地方。
        其他文件都是某一个阶段的实现细节。

【业务定位】
    RAG 查询链路的"总导演"：把召回→重排→父块扩展→阈值→压缩→准入判定串起来。

【数据流向】
    上游（谁调它）：
        services/rag/qa.py::answer_question()      问答主路径
        services/agents/chat_agent.py 的 rag 工具   Agent 直接检索
        api/routes/rag.py                          召回测试页面
    下游（它调谁）：
        retrieval.search_similar_documents()      ① 召回
        _qwen_rerank() / rerank.rerank_documents() ② 重排
        retrieval.expand_parent_context()         ③ 父块扩展
        context_compression.compress_retrieval_results() ⑤ 压缩
        answer_gate.evaluate_answer_gate()        ⑥ 准入
        embeddings.embed_knowledge_query()        算查询向量
        embeddings.validate_embedding_contract()  契约校验

【核心规则】
    1. 六阶段流水线，每阶段都记录 trace（可诊断"哪一步把结果筛掉了"）。
    2. 重排失败自动降级启发式（API 挂了不能让整个查询失败）。
    3. 参数优先级：显式传入 > 仓库配置 > 全局默认。

【本文件函数地图】
    ┌─ 配置管理 ─────────────────────────────────────
    │  get_or_create_config()  ★ 取配置 + 每次校验契约
    │  config_dict()           配置 → dict（给前端/trace）
    │  model_catalog()         可用模型清单（含 available 标记）
    │  runtime_status()        当前配置能不能用（诊断页）
    │
    ├─ 重排实现 ─────────────────────────────────────
    │  _qwen_rerank()          调 Qwen3-Rerank API（async）
    │
    ├─ ★ 核心编排 ───────────────────────────────────
    │  retrieve_knowledge()    六阶段流水线（async）
    │
    └─ 留档 ────────────────────────────────────────
       record_retrieval_test() 召回测试记录（调参对比用）
"""

import importlib.util
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import KnowledgeBaseConfig, RetrievalTestRun
from app.services.context_compression import compress_retrieval_results, estimate_tokens
from app.services.rag.answer_gate import evaluate_answer_gate
from app.services.rag.embeddings import (
    embed_knowledge_query,
    embedding_contract,
    normalize_embedding_provider,
    validate_embedding_contract,
)
from app.services.rag.rerank import rerank_documents
from app.services.rag.retrieval import expand_parent_context, search_similar_documents
from app.services.rag.source_policy import normalize_source_types, should_vectorize


# ════════════════════════════════════════════════════════════════
# 配置管理
# ════════════════════════════════════════════════════════════════

def as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """字符串转 UUID。

    【调用链】被 get_or_create_config() / record_retrieval_test() 调用
    """
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def get_or_create_config(db: Session, repo_id: str | uuid.UUID, *, commit: bool = True) -> KnowledgeBaseConfig:
    """取仓库的知识库配置，没有就按全局契约创建一条。

    【调用链】追这个函数会进入：
        1. embedding_contract()          → 拿当前 .env 的向量空间契约
        2. KnowledgeBaseConfig(...)      → 按契约创建配置
        3. validate_embedding_contract() → ★ 每次调用都校验
    【被谁调用】
        retrieve_knowledge()                ← 每次检索
        document_processing.process_knowledge_document()  ← 每次入库（取切片参数）

    【数据流向】
        入：db、repo_id、commit
        出：KnowledgeBaseConfig（一定通过契约校验，否则抛异常）

    【关键】最后一定会执行 validate_embedding_contract：
        即使数据库里存的是旧配置（比如换了 .env 里的模型），也会在这里被拦住，
        而不是等到检索时才发现向量空间不匹配。

    【为什么"每次"都要校验，而不只在新建时校验】
        配置漂移：数据库里存的是 Qwen 1024 维，但 .env 改成了 OpenAI 1536 维。
        这时候如果只在新建时校验，检索会拿着 1536 维的向量去查 1024 维的 collection ——
        报错还算好，最怕的是 Milvus 不报错但结果全错。
    """
    repo_uuid = as_uuid(repo_id)
    config = db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_uuid).one_or_none()

    if config is None:
        # 首次：按全局契约创建（保证新仓库和 Milvus collection 一致）
        contract = embedding_contract()
        config = KnowledgeBaseConfig(
            repo_id=repo_uuid,
            embedding_provider=contract["provider"],
            embedding_model=contract["model"],
            embedding_dimensions=contract["dimensions"],
        )
        db.add(config)
        db.flush()      # 拿 id
        if commit:
            db.commit()
            db.refresh(config)

    # 每次都校验（不只是新建时）—— 防止配置漂移
    validate_embedding_contract(
        config.embedding_provider,
        config.embedding_model,
        config.embedding_dimensions,
    )
    return config


def config_dict(config: KnowledgeBaseConfig) -> dict[str, Any]:
    """把配置对象转成 dict（返回给前端 / 存进 trace）。

    【调用链】被 retrieve_knowledge() 调用（trace 里带完整配置，便于复盘）

    【为什么要把完整配置放进 trace】
        排查"为什么这次检索效果不好"时，第一件事就是看当时用的什么参数。
        不记下来就永远说不清。
    """
    return {
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "retrieval_method": config.retrieval_method,
        "rerank_enabled": config.rerank_enabled,
        "rerank_provider": config.rerank_provider,
        "rerank_model": config.rerank_model,
        "top_k": config.top_k,
        "score_threshold_enabled": config.score_threshold_enabled,
        "score_threshold": config.score_threshold,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


def model_catalog() -> dict[str, Any]:
    """可用模型清单 + 每个模型"当前能不能用"。

    【调用链】embedding_contract() / normalize_embedding_provider() → 本函数
    【被谁调用】runtime_status() / api/routes/rag.py（配置页面的下拉选项）

    【为什么要检查 available】用户看到下拉选项，但依赖没装/没配 key 时调了会失败。
        提前标出 available=False，前端可以灰掉。

    【为什么 embedding 只列一个模型】
        整个 Milvus collection 只能用一套向量空间。
        列出多个会让用户以为能选 —— 选了就崩。所以只展示当前生效的那个，
        并标注"固定向量空间"。
    """
    # sentence-transformers 是可选依赖，没装时 find_spec 返回 None
    sentence_transformers_ready = importlib.util.find_spec("sentence_transformers") is not None
    # Qwen rerank 需要 key + base_url 都配好
    qwen_api_ready = bool((settings.rerank_api_key or settings.dashscope_api_key) and settings.rerank_base_url)

    contract = embedding_contract()
    provider = normalize_embedding_provider(str(contract["provider"]))

    # 按 provider 判断依赖是否就绪
    if provider in {"qwen_local", "local"}:
        embedding_available = sentence_transformers_ready
    elif provider == "dashscope":
        embedding_available = bool(settings.dashscope_api_key)
    elif provider == "openai_compatible":
        embedding_available = bool(settings.embedding_api_key or settings.llm_api_key)
    else:
        # deterministic（哈希兜底）永远可用
        embedding_available = provider == "deterministic"

    return {
        # 【业务】embedding 只列当前配置的那一个 —— 因为整个 Milvus collection 只能用一套向量空间
        "embedding_models": [
            {
                "provider": provider,
                "model": contract["model"],
                "label": f"{contract['model']}（固定向量空间）",
                "dimensions": contract["dimensions"],
                "available": embedding_available,
                "recommended": True,
                "fallback": None,       # embedding 没有兜底：模型不对，向量就全废
            },
        ],
        "rerank_models": [
            {
                "provider": "qwen_api",
                "model": "qwen3-rerank",
                "label": "Qwen3-Rerank（百炼 API）",
                "available": qwen_api_ready,
                "recommended": True,
                "fallback": "heuristic",       # 不可用时降级到本地启发式
            },
            {
                "provider": "heuristic",
                "model": "heuristic-reranker",
                "label": "本地启发式重排",
                "available": True,             # 纯本地计算，永远可用
                "recommended": False,
                "fallback": None,
            },
        ],
        "deepseek": {
            "embedding_available": False,
            "rerank_available": False,
            "note": "DeepSeek 官方 API 当前未提供 Embedding 或 Rerank 接口。",
        },
    }


def runtime_status(config: KnowledgeBaseConfig) -> dict[str, Any]:
    """当前配置的运行时可用性（诊断页用）。

    【调用链】model_catalog() → 本函数
    【被谁调用】retrieve_knowledge()（写进 trace）

    【做什么】拿配置里的 provider/model 去 catalog 里查，得出"能不能用 + 有没有兜底"。

    【为什么用 next(..., None) 而不是下标查找】
        配置里存的模型可能已经不在 catalog 里了（比如下线了）。
        用 next 带默认值 None，不会抛 StopIteration。
    """
    catalog = model_catalog()

    embedding = next(
        (
            item
            for item in catalog["embedding_models"]
            if item["provider"] == config.embedding_provider and item["model"] == config.embedding_model
        ),
        None,
    )
    rerank = next(
        (
            item
            for item in catalog["rerank_models"]
            if item["provider"] == config.rerank_provider and item["model"] == config.rerank_model
        ),
        None,
    )

    return {
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "dimensions": config.embedding_dimensions,
            "available": bool(embedding and embedding["available"]),
            "fallback": (embedding or {}).get("fallback"),
        },
        "rerank": {
            "enabled": config.rerank_enabled,
            "provider": config.rerank_provider,
            "model": config.rerank_model,
            "available": bool(rerank and rerank["available"]),
            "fallback": (rerank or {}).get("fallback"),
        },
    }


# ════════════════════════════════════════════════════════════════
# 重排：Qwen3-Rerank API
# ════════════════════════════════════════════════════════════════

async def _qwen_rerank(
    query: str,
    items: list[dict[str, Any]],
    config: KnowledgeBaseConfig,
    limit: int,
) -> list[dict[str, Any]]:
    """调 Qwen3-Rerank API 做语义重排（比启发式准，但要网络 + 花钱）。

    【调用链】被 retrieve_knowledge() 调用（第 ② 步，仅当 rerank_provider == "qwen_api"）
    【下游】失败时由调用方降级到 rerank.rerank_documents()

    请求体里三个关键点：
        documents  → 每条截断到 4000 字符（API 有长度限制）
        top_n      → 只取前 N 条
        instruct   → 任务指令，告诉模型"这是检索任务"，能提升效果

    【数据流向】
        入：query、items（候选）、config（含 rerank_model）、limit
        出：重排后的候选列表（按 API 返回的 relevance_score 排序）

    【为什么 documents 要截断到 4000 字符】
        Rerank API 对单条文档有长度上限（通常 4096 token）。
        超了会报 400。截断虽然丢信息，但比整个重排失败好。

    【为什么拼 "标题\n路径\n正文"】
        和入库时向量化的内容保持一致（标题+正文）。
        重排模型看到的信息结构一致，打分才稳定。
    """
    api_key = settings.rerank_api_key or settings.dashscope_api_key
    base_url = (settings.rerank_base_url or "").rstrip("/")
    if not api_key or not base_url:
        raise RuntimeError("Qwen rerank API is not configured")

    # endpoint 可能是 base_url 本身（已含 /reranks）或需要拼上
    endpoint = base_url if base_url.endswith("/reranks") else f"{base_url}/reranks"

    # 每条文档拼成 "标题\n路径\n正文"，截断到 4000 字符
    documents = [
        "\n".join(
            part
            for part in (
                str(item.get("title") or ""),
                str((item.get("metadata") or {}).get("path") or ""),
                str(item.get("_raw_content") or item.get("snippet") or ""),
            )
            if part
        )[:4000]
        for item in items
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": config.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": min(max(1, limit), len(documents)),
                "instruct": settings.qwen_retrieval_instruction,
            },
        )
        response.raise_for_status()
        payload = response.json()

    # 兼容两种响应结构：{results:[]} 和 {output:{results:[]}}
    # （百炼 API 和 DashScope 原生 API 的结构不同）
    rows = payload.get("results") or payload.get("output", {}).get("results") or []

    ranked: list[dict[str, Any]] = []
    for row in rows:
        # index 是"这条对应输入里的第几条"—— 用它把分数映射回原候选
        index = int(row.get("index", -1))
        if index < 0 or index >= len(items):
            continue        # 越界的索引直接忽略（防止 API 返回脏数据）

        # 两种字段名都兼容
        score = float(row.get("relevance_score") or row.get("score") or 0.0)
        item = items[index]
        ranked.append(
            {
                **item,
                "score": round(score, 4),
                "rerank_score": round(score, 4),
                "rank_reason": "qwen3-rerank",
                "retrieval": {
                    **(item.get("retrieval") or {}),
                    "rerank_mode": "qwen_api",
                    "rerank_model": config.rerank_model,
                    "rerank_score": round(score, 4),
                },
            }
        )
    return ranked


# ════════════════════════════════════════════════════════════════
# ★ 六阶段流水线
# ════════════════════════════════════════════════════════════════

async def retrieve_knowledge(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    *,
    top_k: int | None = None,
    retrieval_method: str | None = None,
    rerank_enabled: bool | None = None,
    score_threshold_enabled: bool | None = None,
    score_threshold: float | None = None,
    source_types: list[str] | None = None,
    metadata_filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """【六阶段检索流水线】—— RAG 查询的核心。

    【调用链】追这个函数会进入（按执行顺序）：
        0. get_or_create_config()              取配置 + 契约校验
        0. embed_knowledge_query()             算查询向量（is_query=True）
        1. search_similar_documents()          混合召回（向量 + BM25 + 融合 + 去重）
        2. _qwen_rerank() 或 rerank_documents() 重排（失败降级）
        3. expand_parent_context()             父块扩展
        4. （阈值过滤，本函数内联）
        5. compress_retrieval_results()        上下文压缩
        6. evaluate_answer_gate()              准入判定
    【被谁调用】
        services/rag/qa.py::answer_question()       ← 主路径
        services/agents/chat_agent.py 的 rag 工具    ← Agent 直接检索
        api/routes/rag.py                           ← 召回测试页面

    【数据流向】
        入：db、repo_id、query、各种可选参数（None 表示用仓库配置）
        出：(items, trace, duration_ms)

    返回：（结果列表, 完整 trace, 耗时毫秒）
    trace 里每阶段都记录了 input_count / output_count，
    排查"为什么没结果"时能一眼看出是哪一步筛空的。

    ┌─ ① 混合召回（向量 + BM25）
    ├─ ② 重排（Qwen API，失败降级启发式）
    ├─ ③ 父块扩展（补回完整章节上下文）
    ├─ ④ 分数阈值过滤
    ├─ ⑤ 上下文压缩（按 token 预算裁剪）
    └─ ⑥ 准入判定（够不够回答？）

    【参数优先级：显式传入 > 仓库配置】
        这样同一个函数既能被"配置页"调用（用仓库配置），
        也能被"召回测试页"调用（临时试参数，不改库）。
    """
    started = time.perf_counter()

    # ── 阶段 0：取配置（同时校验向量空间契约）──
    config = get_or_create_config(db, repo_id)

    # 参数优先级：显式传入 > 仓库配置；top_k 夹在 1~20
    resolved_top_k = min(max(top_k or config.top_k, 1), 20)
    method = retrieval_method or config.retrieval_method
    use_rerank = config.rerank_enabled if rerank_enabled is None else rerank_enabled
    threshold_on = config.score_threshold_enabled if score_threshold_enabled is None else score_threshold_enabled
    threshold = config.score_threshold if score_threshold is None else score_threshold

    requested_source_types = normalize_source_types(source_types)
    effective_source_types = requested_source_types or None

    # 如果限定的来源类型全都不进 RAG（如只看 team_member），就没必要算向量
    # —— 省一次 embedding 调用（本地模型几百毫秒，API 还要钱）
    has_vector_sources = not effective_source_types or any(
        should_vectorize(source_type) for source_type in effective_source_types
    )

    query_embedding = None
    if method in {"vector", "hybrid"} and has_vector_sources:
        # 注意：这里用 embed_knowledge_query（is_query=True，带查询专用前缀）
        query_embedding = embed_knowledge_query(
            query,
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )

    # ════════════════════════════════════════════════════════════
    # ── ① 混合召回 ──
    # ════════════════════════════════════════════════════════════
    # 候选数 = top_k × 倍数：要重排就多取（8 倍），不重排取少（2 倍）。
    # 上限 100，防止一次拉太多把上下文撑爆。
    candidate_limit = min(max(resolved_top_k * (8 if use_rerank else 2), 20), 100)

    pipeline_trace: dict[str, Any] = {}
    items = search_similar_documents(
        db,
        repo_id,
        query,
        limit=candidate_limit,
        query_embedding=query_embedding,
        retrieval_method=method,
        apply_rerank=False,          # 重排在这里统一做，不让 retrieval 内部再做
        apply_compression=False,     # 压缩也在这里统一做
        source_types=effective_source_types,
        metadata_filters=metadata_filters,
        pipeline_trace=pipeline_trace,
    )

    # ════════════════════════════════════════════════════════════
    # ── ② 重排（含降级）──
    # ════════════════════════════════════════════════════════════
    rerank_stage: dict[str, Any] = {
        "name": "rerank",
        "input_count": len(items),
        "requested": "qwen_api" if use_rerank and config.rerank_provider == "qwen_api" else "heuristic",
    }

    if use_rerank and items:
        if config.rerank_provider == "qwen_api":
            try:
                items = await _qwen_rerank(query, items, config, resolved_top_k)
                # API 返回空也要降级（可能全部 index 越界了）
                if not items:
                    raise RuntimeError("Qwen rerank returned no valid results")
                rerank_stage.update({"mode": "qwen_api", "fallback": False})
            except Exception as exc:
                # API 挂了不能影响整体可用性 → 降级本地启发式，并记录原因
                items = rerank_documents(query, items, resolved_top_k)
                rerank_stage.update(
                    {
                        "mode": "heuristic",
                        "fallback": True,
                        "fallback_reason": type(exc).__name__,   # 只记异常类型，不记完整堆栈（trace 要塞进 JSON）
                    }
                )
        else:
            # 配置的就是启发式 → 直接用
            items = rerank_documents(query, items, resolved_top_k)
            rerank_stage.update({"mode": "heuristic", "fallback": False})
    elif use_rerank:
        # 开了重排但没有候选 → 不跑
        rerank_stage.update({"mode": "not_run", "fallback": False, "reason": "no candidates"})
    else:
        rerank_stage.update({"mode": "disabled", "fallback": False})

    # 重排后只留 top_k 条（后面所有阶段都基于这个更小的集合）
    items = items[:resolved_top_k]
    pipeline_trace.setdefault("stages", []).append({**rerank_stage, "output_count": len(items)})

    # ════════════════════════════════════════════════════════════
    # ── ③ 父块扩展 ──
    # ════════════════════════════════════════════════════════════
    # 切片时可能把一个大章节切成父子块，这里把父块补回来，让模型看到完整上下文
    items, parent_stage = expand_parent_context(db, repo_id, items, neighbor_window=1)
    pipeline_trace["stages"].append(parent_stage)

    # ════════════════════════════════════════════════════════════
    # ── ④ 分数阈值过滤 ──
    # ════════════════════════════════════════════════════════════
    before_threshold = len(items)
    if threshold_on:
        # 低于阈值直接丢 —— 宁缺毋滥，防止把不相关的内容塞进 prompt
        items = [item for item in items if float(item.get("score") or 0.0) >= threshold]

    pipeline_trace["stages"].append(
        {
            "name": "score_threshold",
            "enabled": threshold_on,
            "threshold": threshold if threshold_on else None,
            "input_count": before_threshold,
            "output_count": len(items),
        }
    )

    # ════════════════════════════════════════════════════════════
    # ── ⑤ 上下文压缩 ──
    # ════════════════════════════════════════════════════════════
    # 预算 = top_k × 260，夹在 520~1800 token（太少信息不够，太多挤占对话上下文）
    context_budget_tokens = min(1800, max(520, resolved_top_k * 260))

    compression_input_count = len(items)
    items = compress_retrieval_results(
        query,
        items,
        max_tokens_per_item=260,
        max_total_tokens=context_budget_tokens,
    )
    # 统计实际占用的 token（trace 里展示"预算 vs 实际"）
    compressed_tokens = sum(estimate_tokens(str(item.get("snippet") or "")) for item in items)

    pipeline_trace["stages"].append(
        {
            "name": "context_compression",
            "input_count": compression_input_count,
            "output_count": len(items),
            "budget_tokens": context_budget_tokens,
            "selected_tokens": compressed_tokens,
        }
    )

    # ════════════════════════════════════════════════════════════
    # ── ⑥ 准入判定 ──
    # ════════════════════════════════════════════════════════════
    gate = evaluate_answer_gate(
        query,
        items,
        score_threshold=threshold,
        threshold_enabled=threshold_on,
    )

    # 把判定结果写回每条证据（前端能看到"这条是否被选中进上下文"）
    for item in items:
        retrieval = dict(item.get("retrieval") or {})
        retrieval["answer_gate"] = gate["decision"]
        retrieval["selected_for_context"] = gate["decision"] == "answer"
        item["retrieval"] = retrieval

    pipeline_trace["stages"].append({"name": "answer_gate", **gate})

    # ════════════════════════════════════════════════════════════
    # 收尾：组装完整 trace
    # ════════════════════════════════════════════════════════════
    duration_ms = round((time.perf_counter() - started) * 1000)
    trace = {
        # 配置全量（复盘时看"当时用的什么参数"）
        **config_dict(config),
        "top_k": resolved_top_k,
        "source_types": requested_source_types,
        "filters": {
            "repo_id": str(repo_id),
            "source_types": requested_source_types,
            "metadata": metadata_filters or {},
        },
        "runtime": runtime_status(config),
        "fusion_method": "weighted" if method == "hybrid" else method,
        "context_budget_tokens": context_budget_tokens,
        "answer_gate": gate,
        "pipeline": pipeline_trace,
    }
    return items, trace, duration_ms


def record_retrieval_test(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    retrieval_config: dict[str, Any],
    results: list[dict[str, Any]],
    duration_ms: int,
) -> RetrievalTestRun:
    """记录一次"召回测试"（用户在调参页面每次试跑都留档，便于对比）。

    【调用链】被 api/routes/rag.py 调用（召回测试接口）

    【数据流向】
        入：db、repo_id、query、retrieval_config（本次参数）、results、duration_ms
        出：RetrievalTestRun（已 commit）

    【为什么要留档】
        调 RAG 参数是"试出来的"：改 top_k、改阈值、改检索方式。
        不留档就永远说不清"上次那个效果好的配置是什么"。
    """
    run = RetrievalTestRun(
        repo_id=as_uuid(repo_id),
        query=query,
        retrieval_config=retrieval_config,
        results_json=results,
        duration_ms=duration_ms,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
