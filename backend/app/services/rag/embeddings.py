"""
向量化（Embedding）：文本 → 向量
==============================

【在 RAG 流水线中的位置】
    入库侧和查询侧都要经过这里，而且是【同一个函数】：

        入库：chunk 文本 ──→ ★ embed_knowledge_texts(is_query=False) ──→ 写入 Milvus
        查询：用户问题   ──→ ★ embed_knowledge_query(is_query=True)   ──→ 去 Milvus 搜

    为什么要强调"同一个函数、不同的 is_query"：
        很多 embedding 模型（Qwen3-Embedding 等）对"文档"和"查询"用不同的 prompt 前缀。
        入库时不加前缀、查询时加 "query" 前缀，检索效果才最好。
        混用会让召回率明显下降，而且不会报错 —— 属于"静默变差"的坑。

【业务定位】
    把文本变成定长浮点向量。向量之间的余弦相似度 = 语义相似度。

【数据流向】
    上游（谁调它）：
        services/rag/document_processing.py → embed_knowledge_texts()（用户上传的文件）
        services/rag/indexing.py            → embed_knowledge_texts()（Issue/PR/CI 等）
        services/rag/knowledge_base.py      → embed_knowledge_query()（检索前）
        services/rag/retrieval.py           → embed_knowledge_query()（检索前）
        services/rag/vector_store.py        → embed_text()（单条补充）
    下游（它调谁）：
        sentence-transformers（本地模型，懒加载 + 进程级缓存）
        httpx（远程 OpenAI 兼容 /embeddings 接口）
        内置哈希实现（deterministic，仅测试/演示兜底）

【核心规则】
    ★ 向量空间契约：provider + model + dimensions 三者绑定，全库必须一致。
      不一致时【直接抛异常】，绝不截断/补零/自动降维 —— 理由见 validate_embedding_contract()。

【本文件函数地图】（按"你会怎么读"排列）
    ┌─ 先读这三个，理解契约 ─────────────────────────────
    │  validate_embedding_contract()   ← 最重要的防错机制
    │  normalize_embedding_provider()  ← 别名归一
    │  embedding_contract()            ← 当前生效的契约（前端展示）
    │
    ├─ 再读统一入口（对外只有 4 个函数）───────────────
    │  embed_knowledge_texts()   ← 总入口：校验契约 → 按 provider 分发
    │      ├─ _embed_local()               本地 sentence-transformers
    │      │     └ _local_sentence_transformer()  @lru_cache 单例加载
    │      ├─ _embed_openai_compatible()   远程 HTTP（含 DashScope）
    │      │     └ _embedding_payload()    构造请求体
    │      └─ deterministic_embedding()    哈希兜底（无语义！）
    │  embed_knowledge_query()  ← 查询专用（is_query=True）
    │  embed_texts() / embed_text()  ← 用全局配置的便捷封装
    │
    └─ 校验工具 ─────────────────────────────────────
       _validate_vector_dimensions()  ← 维度不符直接报错
"""

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import settings


# ════════════════════════════════════════════════════════════════
# 兜底实现：确定性哈希向量（无语义，仅供测试/演示）
# ════════════════════════════════════════════════════════════════

def _deterministic_tokens(text: str) -> list[str]:
    """分词（英文按词，中文按单字 + 双字）。

    【调用链】被 deterministic_embedding() 调用

    【为什么中文要拆成"单字 + 双字"】
        中文没有空格分词。拆单字能保证覆盖，拆双字能保留一点词序信息
        （"登录" 和 "录登" 的单字集合相同，但双字不同）。
        这只是哈希向量的粗略补偿，不是真正的分词。

    【为什么全部转小写】
        "Login" 和 "login" 应该产生相同的 token，否则同一概念会有两个哈希位置。
    """
    lowered = (text or "").lower()
    # 英文/数字/下划线/连字符：按词切
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    # 中文：每段连续汉字 → 拆成单字 + 相邻双字
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)                                                   # 单字
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))  # 双字
    return tokens


def deterministic_embedding(text: str, dimensions: int = 128) -> list[float]:
    """确定性哈希向量 —— 仅用于演示/测试兜底。

    【调用链】_deterministic_tokens() → 本函数
    【被谁调用】embed_knowledge_texts()（provider="deterministic" 时）

    【重要】这个向量没有任何语义！
        "登录" 和 "认证" 在它眼里完全不相关。
        它只是把词哈希到固定维度上，让流程能跑通（比如没有 API key 的 CI 环境）。
        生产环境必须换成真正的 embedding 模型。

    【为什么叫"确定性"】
        同一段文本永远得到同一个向量（哈希是确定的，没有随机性）。
        这样测试可以断言"相同输入 → 相同向量"，也能离线复现问题。

    【算法】
        1. 每个 token 做 SHA-256
        2. 取前 2 字节转成整数，对维度取模 → 得到落点下标
        3. 该位置 +1（相当于词袋计数）
        4. 最后做 L2 归一化（让余弦相似度 = 点积）
    """
    # 本地演示和 Eval 的兜底实现；生产环境请替换为 LLM embedding API。
    vector = [0.0] * dimensions
    for token in _deterministic_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # 前 2 字节 = 16 位 → 0~65535，对维度取模得到落点
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    # L2 归一化：||v|| 为 0 时用 1.0 兜底，避免除零
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


# ════════════════════════════════════════════════════════════════
# 契约校验：本项目最重要的防错机制
# ════════════════════════════════════════════════════════════════

def _validate_vector_dimensions(vector: list[float], dimensions: int, provider: str) -> list[float]:
    """维度校验：不一致直接报错，绝不截断或补零。

    【调用链】被 _embed_openai_compatible() / _embed_local() 调用

    【为什么不截断/补零】维度不同意味着向量空间不同。
        截断后的向量和原向量不在同一个空间，算出来的相似度毫无意义。
        而且这种错误不会报错，只会让检索结果变成"随机挑几条" —— 最难排查的 bug。

    【为什么错误信息里带 provider】
        出错时要能立刻知道是哪个 provider 返回了错的维度（本地模型？某个网关？）。
    """
    if len(vector) != dimensions:
        raise ValueError(
            f"Embedding provider '{provider}' returned {len(vector)} dimensions; expected {dimensions}. "
            "Refusing to truncate or pad vectors."
        )
    return vector


def normalize_embedding_provider(provider: str) -> str:
    """把各种别名归一成标准 provider 名（大小写/连字符/别名都兼容）。

    【调用链】无
    【被谁调用】validate_embedding_contract() / embedding_contract() / embed_knowledge_texts()

    【为什么需要别名表】
        .env 里可能写 "openai"、"api"、"remote"、"openai-compatible"，
        但它们指的是同一件事。归一化后只比较 4 个标准名，逻辑才清晰。

    【标准名只有 4 个】
        openai_compatible  远程 OpenAI 兼容接口（含各种网关）
        local              本地 sentence-transformers
        qwen_local         本地 Qwen3-Embedding（走 local 分支，但保留独立名以便区分）
        dashscope          阿里云 DashScope（有独立 api_key / base_url）
        deterministic      哈希兜底

    【为什么未知 provider 直接抛异常而不是默认某个值】
        静默回退到默认 provider 会导致"以为在用 Qwen，其实在用哈希向量"，
        检索质量暴跌却毫无提示。宁可启动失败。
    """
    # 统一格式：去空格、转小写、连字符换下划线（"openai-compatible" → "openai_compatible"）
    provider = (provider or "").strip().lower().replace("-", "_")
    aliases = {
        "api": "openai_compatible",
        "openai": "openai_compatible",
        "remote": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "sentence_transformer": "local",
        "sentence_transformers": "local",
        "local_sentence_transformers": "local",
        "local": "local",
        "qwen_local": "qwen_local",
        "dashscope": "dashscope",
        "hash": "deterministic",
        "deterministic": "deterministic",
    }
    resolved = aliases.get(provider)
    if resolved is None:
        raise ValueError(f"Unsupported embedding provider: {provider or '<empty>'}")
    return resolved


def validate_embedding_contract(provider: str, model: str, dimensions: int) -> None:
    """向量空间契约校验 —— 这是本项目最重要的防错机制之一。

    【调用链】normalize_embedding_provider() → 本函数
    【被谁调用】embed_knowledge_texts()（每次向量化前都校验）

    【为什么必须有】
        不同模型产生的向量不在同一个空间。如果把 Qwen 的向量和 OpenAI 的向量
        混在同一个 Milvus collection 里，余弦相似度算出来是随机的，
        检索结果会完全错乱，而且不会报错——这是最难排查的 bug。

        所以：请求的 (provider, model, dimensions) 必须与 .env 配置完全一致，
        不一致就直接抛异常，宁可跑不起来也不要给出错误答案。

    【为什么"每个仓库"也要遵守同一套】
        Milvus collection 是全局共享的（不按仓库隔离 collection，靠 metadata 过滤）。
        所以所有仓库必须用同一个向量空间。这也是 knowledge_base_configs 表里
        embedding 三件套不能随便改的原因 —— 改了就要全库重建索引。
    """
    # 当前 .env 配置的契约
    configured = (
        normalize_embedding_provider(settings.embedding_provider),
        settings.embedding_model.strip(),
        settings.embedding_dimensions,
    )
    # 本次请求的契约
    requested = (normalize_embedding_provider(provider), model.strip(), dimensions)

    if requested != configured:
        raise ValueError(
            "Embedding contract mismatch: "
            f"requested provider={requested[0]}, model={requested[1]}, dimensions={requested[2]}; "
            f"configured provider={configured[0]}, model={configured[1]}, dimensions={configured[2]}. "
            "All repositories sharing the Milvus collection must use one vector space."
        )


def embedding_contract() -> dict[str, Any]:
    """返回当前生效的向量空间契约（前端展示/诊断用）。

    【调用链】normalize_embedding_provider() → 本函数
    【被谁调用】api/routes/rag.py（配置页面展示"当前用的什么模型"）

    注意：返回的是【配置值】，不是实际检测到的值。
    如果配置写错，这里也会显示错的值 —— 所以它只是"展示"，不是"校验"。
    """
    return {
        "provider": normalize_embedding_provider(settings.embedding_provider),
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }


def embedding_provider_name() -> str:
    """当前 provider 的标准名（归一化后）。

    【调用链】normalize_embedding_provider() → 本函数
    【被谁调用】api/routes/rag.py（状态展示）
    """
    return normalize_embedding_provider(settings.embedding_provider)


# ════════════════════════════════════════════════════════════════
# 远程实现：OpenAI 兼容 /embeddings 接口
# ════════════════════════════════════════════════════════════════

def _embedding_payload(texts: list[str], dimensions: int, model: str) -> dict[str, Any]:
    """构造 OpenAI 兼容的 /embeddings 请求体。

    【调用链】被 _embed_openai_compatible() 调用

    【为什么 dimensions 是可选的】
        OpenAI 的 text-embedding-3 系列支持指定输出维度（降维），
        但很多兼容网关不支持这个参数，传了会报 400。
        所以这里只在 dimensions 为真值时才加进去。
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,       # 支持批量：一次传多条，比逐条调用快得多
    }
    if dimensions:
        payload["dimensions"] = dimensions
    return payload


def _embed_openai_compatible(
    texts: list[str],
    dimensions: int,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """调 OpenAI 兼容的 /embeddings 接口（也用于 DashScope 兼容模式）。

    【调用链】_embedding_payload() → HTTP POST → _validate_vector_dimensions() → 本函数
    【被谁调用】embed_knowledge_texts()（provider=openai_compatible / dashscope）

    【数据流向】
        入：texts（一批文本）、dimensions（期望维度）、可选的 model/key/url
        出：向量列表，顺序与 texts 严格一致

    【凭证优先级】（从高到低）
        显式传入 > embedding 专用配置 > OPENAI_API_KEY 环境变量 > LLM 的 key
        为什么要兜到 LLM key：很多服务（DashScope、各类网关）一个 key 通吃所有接口，
        用户只配一个就够，不必配两遍。

    【为什么返回前要按 index 排序】
        /embeddings 接口不保证返回顺序与输入一致（并发处理时会乱）。
        必须按 index 重排，否则第 1 条文本会拿到第 3 条的向量 ——
        这种错位不会报错，只会让检索莫名其妙地不准。
    """
    # 凭证优先级：显式传入 > embedding 专用 > 环境变量 > LLM key（很多服务共用一个 key）
    api_key = api_key or settings.embedding_api_key or os.getenv("OPENAI_API_KEY") or settings.llm_api_key
    # base_url 去掉末尾斜杠，避免拼出 "//embeddings"
    base_url = (base_url or settings.embedding_base_url or settings.llm_base_url).rstrip("/")
    model = model or settings.embedding_model

    if not api_key:
        raise RuntimeError("embedding API key is not configured")

    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json=_embedding_payload(texts, dimensions, model),
        )
        response.raise_for_status()      # 4xx/5xx 直接抛，不静默

        # 按 index 排序，保证返回顺序与输入一致（接口不保证顺序）
        rows = sorted(response.json().get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in rows]

        # 条数对不上说明响应异常，不能继续（否则后续按位置对应会全错）
        if len(vectors) != len(texts):
            raise ValueError("embedding response length mismatch")

        # 逐条校验维度（服务端可能悄悄换了模型）
        return [
            _validate_vector_dimensions([float(value) for value in vector], dimensions, "openai_compatible")
            for vector in vectors
        ]


# ════════════════════════════════════════════════════════════════
# 本地实现：sentence-transformers
# ════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4)
def _local_sentence_transformer(model_name: str, device: str | None) -> Any:
    """加载本地 sentence-transformers 模型。

    【调用链】无（只在第一次调用时真正加载）
    【被谁调用】_embed_local()

    【为什么用 lru_cache】模型加载要几秒 + 占几百 MB 内存。
        缓存后整个进程只加载一次，后续调用直接复用。
        maxsize=4 允许同时存在几个不同模型（切换配置时不必反复重载）。

    【注意 lru_cache 的 key 是参数元组】
        (model_name, device) 相同才命中缓存。
        所以 device=None 和 device="cpu" 会被当成两个不同的模型各加载一份。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional local dependency
        # 本地依赖是可选的（requirements-local-embedding.txt），没装时给明确指引
        raise RuntimeError(
            "sentence-transformers is not installed; install backend/requirements-local-embedding.txt"
        ) from exc

    # device 为空时不传这个参数，让库自己选（通常是 CPU，有 GPU 会用 GPU）
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def _embed_local(
    texts: list[str],
    dimensions: int,
    *,
    model_name: str | None = None,
    is_query: bool = False,
) -> list[list[float]]:
    """用本地模型向量化。

    【调用链】_local_sentence_transformer() → model.encode() → _validate_vector_dimensions() → 本函数
    【被谁调用】embed_knowledge_texts()（provider=local / qwen_local）

    【is_query 的作用】Qwen3-Embedding 这类模型对"查询"和"文档"用不同的 prompt 前缀。
        查询时传 prompt_name="query"，能显著提升检索效果。
        入库时（is_query=False）不加前缀。

    【为什么只对 qwen3-embedding 加前缀】
        这是该模型的训练方式（非对称检索）。
        其他模型（如 all-MiniLM）没有这个机制，传 prompt_name 会报错。
        所以先判断模型名，再决定加不加。

    【normalize_embeddings=True 的意义】
        归一化后向量模长为 1，余弦相似度 = 点积，Milvus 可以直接用 IP 度量，更快。
    """
    resolved_model = model_name or settings.local_embedding_model
    model = _local_sentence_transformer(resolved_model, settings.local_embedding_device)

    encode_kwargs: dict[str, Any] = {}
    # 只有 Qwen3-Embedding 系列支持 prompt_name="query"
    if is_query and "qwen3-embedding" in resolved_model.lower():
        encode_kwargs["prompt_name"] = "query"

    vectors = model.encode(
        texts,
        batch_size=32,                  # 批大小：控制显存/内存峰值
        normalize_embeddings=True,      # 归一化后可直接用余弦相似度
        convert_to_numpy=True,
        show_progress_bar=False,        # 后台任务不需要进度条（会污染日志）
        **encode_kwargs,
    )

    output: list[list[float]] = []
    for vector in vectors:
        # 兼容 numpy 数组（有 tolist）和普通列表
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        output.append(_validate_vector_dimensions([float(value) for value in values], dimensions, "local"))
    return output


# ════════════════════════════════════════════════════════════════
# 对外入口（业务代码只用下面 4 个函数）
# ════════════════════════════════════════════════════════════════

def embed_texts(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    """用【全局配置】向量化一批文本。

    【调用链】embed_knowledge_texts() → 本函数
    【被谁调用】services/rag/vector_store.py（补写向量时）

    这只是个便捷封装：把 settings 里的 provider/model 传下去，
    调用方不用每次都写三个参数。
    """
    dimensions = dimensions or settings.embedding_dimensions
    return embed_knowledge_texts(
        texts,
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=dimensions,
    )


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    """向量化单条文本（embed_texts 的单条版）。

    【调用链】embed_texts() → 本函数
    【被谁调用】services/rag/vector_store.py

    注意：单条调用没有批处理优势。如果循环里调它，应该改成先收集再 embed_texts()。
    """
    return embed_texts([text], dimensions=dimensions)[0]


def embed_knowledge_texts(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
    is_query: bool = False,
) -> list[list[float]]:
    """【总入口】统一的向量化：先校验契约，再按 provider 分发。

    【调用链】追这个函数会进入：
        1. validate_embedding_contract()      → 契约校验（不一致直接抛）
        2. normalize_embedding_provider()     → 别名归一
        3. 按 provider 分发：
           qwen_local / local   → _embed_local()
           dashscope            → _embed_openai_compatible()（用 dashscope 的 key/url）
           openai_compatible    → _embed_openai_compatible()
           deterministic        → deterministic_embedding()
    【被谁调用】
        services/rag/document_processing.py::_embed_in_batches()  ← 用户上传文件
        services/rag/indexing.py                                  ← Issue/PR/CI 索引
        services/rag/knowledge_base.py / retrieval.py              ← 检索前（is_query=True）

    【数据流向】
        入：texts（一批文本）、provider/model/dimensions（契约）、is_query（入库还是查询）
        出：向量列表，顺序与 texts 一致

    【关键顺序：先校验，后执行】
        校验放在分发之前，意味着即使 provider 写错也会先报"契约不匹配"，
        而不是先跑起来再出错。这让配置问题在最早的地方暴露。

    【空列表直接返回】
        避免下游对空批做无意义的 HTTP 调用 / 模型推理。
    """
    # 把 None 统一成空串（下游模型不接受 None）
    clean_texts = [text or "" for text in texts]
    if not clean_texts:
        return []

    validate_embedding_contract(provider, model, dimensions)     # 先校验，后执行

    resolved = normalize_embedding_provider(provider)

    # ── 本地模型（qwen_local 走同一个分支，只是名字不同）──
    if resolved in {"qwen_local", "local"}:
        return _embed_local(clean_texts, dimensions, model_name=model, is_query=is_query)

    # ── DashScope：走 OpenAI 兼容协议，但用自己的 key 和 base_url ──
    if resolved == "dashscope":
        return _embed_openai_compatible(
            clean_texts,
            dimensions,
            model=model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_embedding_base_url,
        )

    # ── 通用 OpenAI 兼容接口 ──
    if resolved == "openai_compatible":
        return _embed_openai_compatible(clean_texts, dimensions, model=model)

    # ── 哈希兜底（测试/演示，无语义）──
    if resolved == "deterministic":
        return [deterministic_embedding(text, dimensions) for text in clean_texts]

    # 理论上到不了这里（normalize 已经挡过未知 provider）
    raise ValueError(f"Unsupported embedding provider: {resolved}")


def embed_knowledge_query(text: str, *, provider: str, model: str, dimensions: int) -> list[float]:
    """向量化"查询"（带 is_query=True，走查询专用前缀）。

    【调用链】embed_knowledge_texts(is_query=True) → 本函数
    【被谁调用】
        services/rag/knowledge_base.py  ← 检索流水线第一步
        services/rag/retrieval.py       ← 混合检索的向量分支

    【为什么要和入库分开】入库用文档前缀、查询用查询前缀，
        这是 Qwen3-Embedding 等模型的训练方式，混用会降低召回率。

    【返回单条向量】
        查询永远只有一条，所以直接取 [0]，调用方不用再拆包。
    """
    return embed_knowledge_texts(
        [text],
        provider=provider,
        model=model,
        dimensions=dimensions,
        is_query=True,
    )[0]
