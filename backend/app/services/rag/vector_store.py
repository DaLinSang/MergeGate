"""
向量库读写（Milvus + PostgreSQL 双写）
====================================

【在 RAG 流水线中的位置】
    存储层 —— 入库的最后一步，也是检索的第一站：

        入库侧：
            document_processing / indexing
                ↓
            ★ 本文件 add_document(s)()
                ├→ PostgreSQL documents 表（完整全文 = 事实源）
                └→ Milvus collection（向量 + 截断正文 = 索引副本）
        查询侧：
            retrieval / knowledge_base
                ↓
            ★ 本文件 search_milvus_documents()
                └→ Milvus 返回 document_id + 分数
                     ↓
                  回 PostgreSQL 取完整正文

【业务定位】
    RAG 的存储层。PostgreSQL 存全文（事实源），Milvus 存向量（索引副本）。

    【一句话记住两者的关系】
        PG 是真相，Milvus 是可重建的索引。
        Milvus 挂了 → 检索降级；PG 挂了 → 数据全丢。
        所以所有"完整内容"只写 PG，Milvus 里存的是截断版。

【数据流向】
    上游（谁调它）：
        services/rag/document_processing.py → add_documents / delete_document_group（上传文件）
        services/rag/indexing.py            → replace_documents（Issue/PR/CI 建索引）
        services/rag/retrieval.py           → search_milvus_documents（向量检索分支）
        services/memory_hub.py              → add_document（记忆候选批准后晋升）
        app/main.py                         → reset_milvus_store_cache（启动时）
    下游（它调谁）：
        embeddings.embed_text / embed_texts  算向量
        source_policy.should_vectorize       决定要不要写向量
        pymilvus.MilvusClient                真正连 Milvus

【核心规则】
    1. 源码（code_file）禁止入 RAG —— add_document / add_documents 会直接抛异常拦住。
    2. Milvus 不可用时抛 MilvusUnavailableError，由 main.py 转成 503 degraded
       （不是整个系统 500 —— 降级比崩掉好）。
    3. 向量维度必须与契约一致，绝不截断/补零。
    4. 删 PG 和删 Milvus 必须一起做，否则留下孤儿向量。

【本文件函数地图】
    ┌─ 数据结构与工具 ────────────────────────────────
    │  DocumentPayload            写入前的中间结构（dto）
    │  stable_source_id()         uuid5 稳定 ID（重复入库不产生重复文档）
    │  _clip_utf8()               ★ 按字节截断（Milvus 的坑，必读）
    │  _json_dumps()              元数据 → JSON 字符串（带长度限制）
    │  _escape_filter_value()     Milvus 表达式转义（防注入）
    │
    ├─ Milvus 客户端封装 ─────────────────────────────
    │  MilvusVectorStore
    │      ├ _ensure_collection()  ★ 建表 + 维度校验（含 Milvus/MySQL 概念对照）
    │      ├ upsert_documents()    批量写入（按 pk 覆盖）
    │      ├ delete_documents()    按条件删
    │      └ search()              ★ 向量检索（repo_id 过滤是仓库隔离的物理保证）
    │  _collection_dimensions()    从 collection 描述里读维度
    │  _milvus_store()             @lru_cache 进程级单例
    │  reset_milvus_store_cache()  强制重连
    │  milvus_runtime_status()     诊断用，不抛异常
    │
    ├─ 对外读写（业务代码用这些）────────────────────
    │  upsert_milvus_documents()   写向量（含维度校验 + 类型过滤）
    │  delete_milvus_documents()   删向量（strict 两种语义）
    │  search_milvus_documents()   查向量
    │
    └─ 双写封装（最常被调用的四个）──────────────────
       add_document()                  单条：PG + Milvus
       add_documents()                 批量：支持传入已算好的向量
       replace_documents()             整组替换（全量快照语义）
       replace_documents_by_metadata() 按元数据精确替换（局部更新）
       delete_document_group()         删一个源对象的全部切片
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document
from app.services.rag.embeddings import embed_text, embed_texts, embedding_contract, embedding_provider_name
from app.services.rag.source_policy import policy_metadata, should_vectorize

logger = logging.getLogger(__name__)


class MilvusUnavailableError(RuntimeError):
    """Milvus 不可用（未安装依赖/未启动/collection 维度不匹配）。

    【为什么单独定义】main.py 注册了这个异常的全局处理器，
        转成 503 + status=degraded，让前端知道"向量检索暂不可用"，
        而不是整个系统 500。

    【为什么继承 RuntimeError 而不是 Exception】
        RuntimeError 语义上表示"运行时环境问题"（服务没起、连不上），
        和"代码写错了"（ValueError/TypeError）区分开。
        上层可以用 except MilvusUnavailableError 精确捕获，做降级处理。
    """
    pass


try:  # pymilvus 在测试和本地演示模式中是可选依赖。
    from pymilvus import DataType, MilvusClient
except Exception:  # pragma: no cover - 依赖缺失时会走到这里。
    # 装不上也不影响其他功能（关键词检索、直查都能用），
    # 只是向量/混合检索会降级。所以这里不 raise，置 None 后面按需报错。
    DataType = None  # type: ignore[assignment]
    MilvusClient = None  # type: ignore[assignment]


# ════════════════════════════════════════════════════════════════
# 数据结构与工具函数
# ════════════════════════════════════════════════════════════════

@dataclass
class DocumentPayload:
    """待入库的一条文档（写入前的中间结构）。

    【为什么要有这个 dto，而不是直接用 Document ORM 对象】
        1. 写库前还没有 id（要 flush 后才有），ORM 对象字段是必填的
        2. 调用方（document_processing / indexing）不该关心 ORM 细节
        3. 可以把"业务参数"和"数据库字段"解耦

    【字段顺序的坑】
        metadata 有默认值、source_id 也有默认值，所以它们必须排在
        source_type / title / content 后面 —— Python dataclass 不允许
        "有默认值的字段"出现在"无默认值字段"之前。
    """

    source_type: str                                    # issue / pull_request / knowledge_file ...
    title: str                                          # 标题（引用展示用）
    content: str                                        # 正文
    metadata: dict[str, Any] = field(default_factory=dict)   # 元数据（路径/页码/章节等）
    source_id: uuid.UUID | str | None = None            # 源对象 ID（为空则自动生成）


def _as_uuid(value: uuid.UUID | str | None) -> uuid.UUID:
    """转 UUID；None 则随机生成一个。

    【调用链】被 add_document / add_documents / replace_documents / delete_document_group 调用

    【为什么允许传 str】
        API 层收到的 ID 是字符串（JSON 里没有 UUID 类型）。
        在这里统一转换，业务代码就不用到处判类型。
    """
    if value is None:
        return uuid.uuid4()
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def stable_source_id(repo_id: uuid.UUID | str, source_type: str, key: str) -> uuid.UUID:
    """生成稳定的 source_id（uuid5 确定性哈希）。

    【调用链】无 —— 纯哈希计算
    【被谁调用】services/rag/indexing.py（为 Issue/PR/CI 生成稳定的 source_id）

    【为什么需要】同一份内容重复入库时必须得到同一个 ID，
        否则会不断产生重复文档。uuid5 保证"输入相同 → 输出相同"。

    【和 uuid4 的区别】
        uuid4 = 随机（每次都不一样）→ 适合"每次都是新对象"
        uuid5 = 确定性（同样的输入永远同样的输出）→ 适合"同一个业务对象重复处理"

    【为什么用 NAMESPACE_URL + 自定义前缀 "devflow:"】
        避免和其他系统的 uuid5 撞车；前缀里带 repo_id 保证跨仓库不冲突。
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"devflow:{repo_id}:{source_type}:{key}")


def _escape_filter_value(value: str) -> str:
    """转义 Milvus 过滤表达式里的字符串（防止表达式注入）。

    【调用链】被 delete_documents() / search() 调用

    【为什么必须转义】
        Milvus 的 filter 是"表达式字符串"，形如 repo_id == "xxx"。
        如果 xxx 里含双引号，就能闭合字符串并注入额外条件 ——
        比如 source_type 传 `a" or repo_id == "b`，就能跨仓库读数据。
        先转义反斜杠再转义引号（顺序不能反），才能彻底安全。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _clip_utf8(value: str, max_bytes: int) -> str:
    """按【字节数】截断 UTF-8 字符串。

    【调用链】被 _json_dumps() / upsert_documents() 调用

    【关键陷阱】Milvus 的 VARCHAR max_length 单位是**字节**，不是字符。
        一个中文字符占 3 字节，所以 max_length=8192 实际只能存约 2700 个汉字。
        如果按字符数截断，中英文混合内容会超长报错。
        这也是为什么 Milvus 里的 content 是截断版，完整正文只在 PostgreSQL。

    【为什么用 errors="ignore"】
        按字节切可能把某个汉字的 3 字节切成 2 字节，
        decode 会抛 UnicodeDecodeError。ignore 直接丢掉残缺的那半个字符。
    """
    data = value.encode("utf-8")
    if len(data) <= max_bytes:
        return value
    # errors="ignore" 避免把多字节字符截断成半个导致解码失败
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _json_dumps(value: dict[str, Any], max_chars: int = 4096) -> str:
    """把元数据序列化成 JSON 字符串并截断（Milvus 的 metadata_json 字段有长度限制）。

    【调用链】_clip_utf8() → 本函数
    【被谁调用】MilvusVectorStore.upsert_documents()

    三个参数的作用：
        ensure_ascii=False → 中文原样存（省字节，也便于人工排查）
        default=str        → datetime/UUID 等不可序列化对象转字符串，不报错
        最后 _clip_utf8    → 防止超长（metadata_json 上限 4096 字节）
    """
    text = json.dumps(value or {}, ensure_ascii=False, default=str)
    return _clip_utf8(text, max_chars)


# ════════════════════════════════════════════════════════════════
# Milvus 客户端封装
# ════════════════════════════════════════════════════════════════

class MilvusVectorStore:
    """Milvus 客户端封装（collection 管理 + 增删查）。

    【这个类只管"和 Milvus 说话"】
        不管业务规则（该不该向量化 → source_policy）、
        不管 PG 写入（→ add_document）、不管分数融合（→ retrieval）。
        保持它足够薄，将来换向量库（Qdrant/Weaviate）只需重写这个类。
    """

    def __init__(self) -> None:
        # 依赖缺失时立刻报错（而不是等第一次查询才炸）
        if MilvusClient is None or DataType is None:
            raise RuntimeError("pymilvus is not installed")
        self.collection_name = settings.milvus_collection
        self.dimensions = settings.embedding_dimensions
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        # 连上后立刻校验 collection（存在则查维度，不存在则建）
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保 collection 存在且维度匹配。

        【调用链】_collection_dimensions() → 本函数（仅在 collection 已存在时）
        【被谁调用】MilvusVectorStore.__init__()

        【维度校验是硬门槛】collection 建好后维度不可改。
            如果配置的模型维度变了（比如从 1024 换成 768），必须换一个新 collection 名重建索引，
            而不是往旧 collection 里塞不同维度的向量——那会让所有检索结果失去意义。
        """
        # ── 情况 A：collection 已存在 → 校验维度 ──
        if self.client.has_collection(self.collection_name):
            description = self.client.describe_collection(self.collection_name)
            actual_dimensions = _collection_dimensions(description)
            if actual_dimensions != self.dimensions:
                raise RuntimeError(
                    f"Milvus collection '{self.collection_name}' has dimension {actual_dimensions}; "
                    f"the configured embedding contract requires {self.dimensions}. "
                    "Use a new collection name and rebuild the index."
                )
            return

        # ── 情况 B：collection 不存在 → 建表 ──
        #
        # Milvus 与 MySQL 的概念对照（理解这一张表就够了）：
        #   Milvus 概念          MySQL 对应概念      本项目里的含义
        #   Collection（集合）   Table（数据表）     一张向量大表，本函数创建的就是它
        #   Field（字段）        Column（列）        pk / repo_id / document_id / title / content / embedding
        #   Entity（实体）       Row（一行记录）     一条 chunk，一行 = 一个切片的全部数据
        #   Schema               表结构             定义字段类型、主键、向量维度
        #   Index（向量索引）    B+ 树索引          专门给 embedding 建的索引，用于快速相似度检索
        #   Partition            分区               可选，大表拆分用，本项目未使用
        #
        # 一句话：Milvus 的 Collection ≈ MySQL 的一张数据表，Field 就是表里的列。
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

        # 主键：用 PostgreSQL 里 documents.id 的字符串形式（不是自增）
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64)    # 唯一主键
        # 仓库隔离：检索时【必须】带这个条件，否则会跨仓库召回
        schema.add_field(field_name="repo_id", datatype=DataType.VARCHAR, max_length=64)          # 仓库隔离用
        # 对应 PG 的 documents.id（同一个文档的多个切片共享 source_id，document_id 各自不同）
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)      # 对应 PG 的 documents.id
        schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=80)
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        # 注意：max_length 单位是【字节】不是字符 —— 8192 字节 ≈ 2700 汉字
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)        # 字节！约 2700 汉字
        # 元数据，json 字符串，存放额外信息（页码、章节、子块序号等）
        schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=4096)
        # 这里就是 Dense 检索用的稠密向量
        # 核心向量字段：FLOAT_VECTOR 浮点向量，dim 是向量维度，如 1024/768
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimensions)

        # AUTOINDEX + COSINE：向量已归一化，余弦相似度最合适
        # 创建索引
        # 索引作用：没有索引的话，Milvus 检索是暴力全量比对，海量数据极慢。建索引之后做 ANN 近似最近邻检索。
        #
        # 【为什么是 COSINE 而不是 L2】
        #   我们的向量在 embeddings.py 里做过 L2 归一化（模长=1）。
        #   归一化后 余弦相似度 与 内积 等价，但 COSINE 语义更清晰、不受模长影响。
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_documents(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """批量写入/更新向量（按 pk 覆盖）。

        【调用链】_clip_utf8() / _json_dumps() / should_vectorize() → 本函数
        【被谁调用】upsert_milvus_documents()

        【为什么用 upsert 而不是 insert】
            pk 是 documents.id。同一条文档重新入库时 pk 相同，
            upsert 会覆盖旧向量，不会产生重复行。insert 会报主键冲突。

        【两个校验】
            ① 条数一致：documents 和 vectors 必须一一对应
            ② 维度一致：单条维度不对就整批失败（不静默跳过）
        """
        if len(documents) != len(vectors):
            raise ValueError("document/vector count mismatch")

        rows = []
        for doc, vector in zip(documents, vectors):
            # 双重保险：不该向量化的类型直接跳过
            # （source_policy 已经过滤过一次，这里再拦一次防止调用方绕过）
            if not should_vectorize(doc.source_type):
                continue
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"document {doc.id} has {len(vector)} embedding dimensions; expected {self.dimensions}"
                )
            rows.append(
                {
                    "pk": str(doc.id),
                    "repo_id": str(doc.repo_id),
                    "document_id": str(doc.id),
                    "source_type": doc.source_type,
                    "source_id": str(doc.source_id),
                    # title 和 content 都要按字节截断（VARCHAR 限制是字节）
                    "title": _clip_utf8(doc.title, 500),
                    "content": _clip_utf8(doc.content or "", 8192),
                    "metadata_json": _json_dumps(doc.meta or {}),
                    "embedding": vector,
                }
            )
        # 空列表不调用（Milvus 会报错）
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)

    def delete_documents(
        self,
        *,
        repo_id: uuid.UUID | str | None = None,
        source_types: list[str] | None = None,
        source_id: uuid.UUID | str | None = None,
        document_ids: list[uuid.UUID | str] | None = None,
    ) -> None:
        """按条件删除。多个条件用 and 组合。

        【调用链】_escape_filter_value() → 本函数
        【被谁调用】delete_milvus_documents()

        【四个条件都是可选的，用 and 串起来】
            document_ids → 精确删某几条（按元数据替换时用）
            repo_id      → 整个仓库（删仓库时用）
            source_types → 某几类（整组替换时用，如重建所有 issue）
            source_id    → 某个源对象的所有切片（重传文件时用）

        【注意：一个条件都不给时什么都不做】
            这是刻意的 —— 防止误传空参数导致"删库"。
        """
        expressions: list[str] = []

        if document_ids:
            # in [...] 语法：把 ID 列表拼成 Milvus 表达式
            ids = ", ".join(f'"{_escape_filter_value(str(item))}"' for item in document_ids)
            expressions.append(f"document_id in [{ids}]")
        if repo_id is not None:
            expressions.append(f'repo_id == "{_escape_filter_value(str(repo_id))}"')
        if source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        if source_id is not None:
            expressions.append(f'source_id == "{_escape_filter_value(str(source_id))}"')

        if expressions:
            self.client.delete(collection_name=self.collection_name, filter=" and ".join(expressions))

    def search(
        self,
        *,
        repo_id: uuid.UUID | str,
        query_embedding: list[float],
        source_type: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """向量检索。

        【调用链】_escape_filter_value() → Milvus search → 组装结果
        【被谁调用】search_milvus_documents()

        【repo_id 过滤是必须的】这是"仓库隔离"的物理保证——
            不加这个条件，A 仓库的问题会召回 B 仓库的文档。

        【数据流向】
            入：query_embedding（已算好的查询向量）、repo_id、类型过滤、limit
            出：dict 列表，每条含 id/title/score/metadata/snippet/_raw_content

        【为什么要返回 _raw_content】
            下游 compress_retrieval_results() 要用它做"按 query 抽取片段"。
            这是 Milvus 里的截断版（≤2700 汉字），不是完整正文。
        """
        # repo_id 是唯一必填的过滤条件
        expressions = [f'repo_id == "{_escape_filter_value(str(repo_id))}"']
        # source_type（单个）和 source_types（多个）二选一，单个优先
        if source_type:
            expressions.append(f'source_type == "{_escape_filter_value(source_type)}"')
        elif source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],        # 批量接口，这里只传一条
            filter=" and ".join(expressions),
            limit=limit,
            # 只取需要的字段：不取 embedding（几 KB 一条，浪费带宽）
            output_fields=["document_id", "source_type", "source_id", "title", "content", "metadata_json"],
        )

        hits = results[0] if results else []
        output: list[dict[str, Any]] = []

        for hit in hits:
            entity = hit.get("entity") or {}

            # metadata_json 解析失败不能影响主流程（元数据是辅助信息）
            metadata_text = entity.get("metadata_json") or "{}"
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                metadata = {}       # 元数据损坏不影响主流程

            output.append(
                {
                    "id": str(entity.get("document_id") or hit.get("id")),
                    "title": str(entity.get("title") or ""),
                    "source_type": str(entity.get("source_type") or ""),
                    "source_id": str(entity.get("source_id") or ""),
                    # Milvus 返回的 distance 在 COSINE 下就是相似度（越大越相关）
                    "score": round(float(hit.get("distance") or 0.0), 4),
                    "metadata": metadata,
                    "snippet": str(entity.get("content") or "")[:260],       # 预览用（截断）
                    "_raw_content": str(entity.get("content") or ""),         # 完整内容（Milvus 里的截断版）
                }
            )
        return output


def _collection_dimensions(description: dict[str, Any]) -> int:
    """从 collection 描述里读出 embedding 字段的维度（兼容不同版本的字段位置）。

    【调用链】被 MilvusVectorStore._ensure_collection() 调用

    【为什么要兼容两种位置】
        pymilvus 不同版本把 dim 放在不同地方：
            老版本：field["params"]["dim"]
            新版本：field["type_params"]["dim"]
            某些版本：field["dim"]（直接平铺）
        三个位置都试一遍，升级依赖时不会突然崩。

    【读不到就报错】
        拿不到维度就无法校验，继续跑等于埋雷。宁可启动失败。
    """
    for schema_field in description.get("fields") or []:
        if schema_field.get("name") != "embedding":
            continue
        params = schema_field.get("params") or schema_field.get("type_params") or {}
        value = params.get("dim") or schema_field.get("dim")
        if value is not None:
            return int(value)
    raise RuntimeError("Milvus collection does not expose an embedding dimension")


@lru_cache(maxsize=1)
def _milvus_store() -> MilvusVectorStore:
    """全局单例（进程内只连一次 Milvus）。

    【调用链】MilvusVectorStore() → 本函数
    【被谁调用】所有对外读写函数（upsert / delete / search / milvus_runtime_status）

    【为什么缓存】每次 new 一个 MilvusClient 都会建连接，代价高。
        maxsize=1 保证整个进程只有一个实例。
        连接失效时调 reset_milvus_store_cache() 清缓存重连。

    【为什么在这里把异常转成 MilvusUnavailableError】
        把"依赖缺失/连不上/维度不匹配"统一成一种异常，
        上层只需捕获一种，就能做降级（转 503 或退回关键词检索）。
    """
    if not settings.milvus_enabled:
        raise MilvusUnavailableError("Milvus is disabled; vector and hybrid RAG are unavailable")
    try:
        return MilvusVectorStore()
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError(f"Milvus is unavailable: {exc}") from exc


def reset_milvus_store_cache() -> None:
    """清掉单例缓存（强制下次重连）。

    【调用链】_milvus_store.cache_clear()
    【被谁调用】app/main.py（启动时清一次，避免热重载后拿到失效连接）

    注意：lru_cache 的 cache_clear 会丢掉旧实例，但不会主动关闭它的连接。
    生产环境如果要优雅重连，应该在清缓存前显式 close。
    """
    _milvus_store.cache_clear()


def milvus_runtime_status() -> dict[str, Any]:
    """Milvus 运行时状态（诊断页用，不会抛异常）。

    【调用链】_milvus_store() / embedding_contract() → 本函数
    【被谁调用】services/memory_hub.memory_status()（记忆面板展示向量库健康）

    【为什么这个函数绝不抛异常】
        它是给"诊断页面"用的。诊断页面本身挂了就没法排查问题了。
        所以这里把异常转成 status="degraded" + error 信息返回。
    """
    try:
        store = _milvus_store()
    except MilvusUnavailableError as exc:
        return {
            "backend": "milvus",
            "status": "degraded",       # 降级：关键词检索仍可用
            "available": False,
            "error": str(exc),
            "collection": settings.milvus_collection,
            "embedding": embedding_contract(),
        }
    return {
        "backend": "milvus",
        "status": "ready",
        "available": True,
        "error": None,
        "collection": store.collection_name,
        "embedding": embedding_contract(),
    }


# ════════════════════════════════════════════════════════════════
# 对外读写：写 / 删 / 查
# ════════════════════════════════════════════════════════════════

def upsert_milvus_documents(documents: list[Document], vectors: list[list[float]]) -> bool:
    """写入 Milvus（先做维度校验，再过滤不该向量化的类型）。

    【调用链】should_vectorize() → _milvus_store() → MilvusVectorStore.upsert_documents() → 本函数
    【被谁调用】add_document() / add_documents()

    【校验顺序很重要：先校验全部，再过滤】
        如果先过滤再校验，那些"不向量化"的文档维度错了也不会被发现。
        现在这个顺序能保证：只要有一条第维度不对，整批立刻失败。

    【返回 True 的两种含义】
        ① 真的写成功了
        ② 没有任何一条需要写（eligible_documents 为空）
        调用方不用区分这两种情况 —— 都是"没有失败"。
    """
    if len(documents) != len(vectors):
        raise ValueError("document/vector count mismatch")

    # ── 第 1 步：全量维度校验（不静默跳过）──
    for vector in vectors:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(
                f"embedding has {len(vector)} dimensions; expected {settings.embedding_dimensions}. "
                "Refusing to truncate or pad vectors."
            )

    # ── 第 2 步：只保留该向量化的文档（直查类型不写 Milvus）──
    pairs = [(doc, vector) for doc, vector in zip(documents, vectors) if should_vectorize(doc.source_type)]
    eligible_documents = [item[0] for item in pairs]
    eligible_vectors = [item[1] for item in pairs]

    if not eligible_documents:
        return True

    store = _milvus_store()
    try:
        store.upsert_documents(eligible_documents, eligible_vectors)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        # 包一层统一异常：调用方只需捕获 MilvusUnavailableError
        raise MilvusUnavailableError("failed to write documents to Milvus") from exc
    return True


def delete_milvus_documents(
    *,
    repo_id: uuid.UUID | str | None = None,
    source_types: list[str] | None = None,
    source_id: uuid.UUID | str | None = None,
    document_ids: list[uuid.UUID | str] | None = None,
    strict: bool = True,
) -> bool:
    """从 Milvus 删除。

    【调用链】_milvus_store() → MilvusVectorStore.delete_documents() → 本函数
    【被谁调用】replace_documents() / delete_document_group() / replace_documents_by_metadata()

    strict 的两种语义：
        True  → Milvus 不可用就抛异常（删除操作必须成功，否则会留下脏数据）
        False → 静默降级（启动清理这类"尽力而为"的场景用）

    【为什么删除默认要 strict】
        如果删 PG 成功、删 Milvus 失败，就会留下"孤儿向量"：
        它还在被检索召回，但回 PG 查不到正文 → 引用里出现空内容。
        这种脏数据很难发现，所以默认宁可报错。
    """
    try:
        store = _milvus_store()
    except MilvusUnavailableError:
        if strict:
            raise
        return False

    try:
        store.delete_documents(repo_id=repo_id, source_types=source_types, source_id=source_id, document_ids=document_ids)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        if not strict:
            # 非严格模式：记日志继续（尽力而为）
            logger.warning("从 Milvus 删除文档失败：%s", exc)
            return False
        raise MilvusUnavailableError("failed to delete documents from Milvus") from exc
    return True


def search_milvus_documents(
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 20,
    query_embedding: list[float] | None = None,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """向量检索入口（可直接传 query_embedding 复用已算好的向量）。

    【调用链】embed_text()（未传向量时）→ MilvusVectorStore.search() → 本函数
    【被谁调用】services/rag/retrieval.py（混合检索的向量分支）

    【为什么允许外部传 query_embedding】
        混合检索里，向量分支和关键词分支要并行跑。
        如果这里自己算向量，就浪费了一次已经算好的结果
        （上层通常已经算过一次，用于别的用途）。
        —— 这是"避免重复计算"的常见优化点。
    """
    store = _milvus_store()
    try:
        # 没传就现算（注意：这里用的是 embed_text，即 is_query=False！
        # 严格来说检索应该用 embed_knowledge_query()。上游 retrieval.py 会自己传向量，
        # 所以这个分支主要给"临时调试"用。）
        vector = query_embedding or embed_text(query, settings.embedding_dimensions)
        return store.search(
            repo_id=repo_id,
            query_embedding=vector,
            source_type=source_type,
            source_types=source_types,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to search documents in Milvus") from exc


# ════════════════════════════════════════════════════════════════
# 双写封装：PG + Milvus 一起写（业务代码主要调这几个）
# ════════════════════════════════════════════════════════════════

def add_document(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    source_id: str | uuid.UUID | None,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    commit: bool = True,
) -> Document:
    """写入单条文档（PostgreSQL + Milvus 双写）。

    【调用链】追这个函数会进入：
        1. should_vectorize()        → source_policy，判断要不要向量化
        2. embed_text()              → embeddings，算向量（标题+正文）
        3. Document(...) + db.add    → PG 插入
        4. db.flush()                → 拿到 doc.id（Milvus 的 pk 要用）
        5. upsert_milvus_documents() → 写 Milvus

    【被谁调用】services/memory_hub.approve_memory_candidate()
               （记忆候选被批准后晋升为 knowledge 文档）

    【数据流向】
        入：db、repo_id、source_type、source_id、title、content、metadata、commit
        出：Document（已 flush，commit=True 时已提交）

    【★ 硬性业务规则：源码不进 RAG】
        source_type == "code_file" 直接抛异常。
        这是架构边界的最后一道防线 —— 即使有人绕过 source_policy，
        这里也会拦住。
    """
    # 【硬性业务规则】源码不进 RAG —— 在这里直接拦住
    if source_type == "code_file":
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")

    vectorized = should_vectorize(source_type)
    # 注意：向量化的是"标题\n正文"（和 document_processing 保持一致）
    vectors = [embed_text(f"{title}\n{content}", settings.embedding_dimensions)] if vectorized else []

    # 只有向量化的文档才记 embedding 元数据（用于判断"这条向量是哪个模型产的"）
    embedding_metadata = (
        {
            "embedding_provider": embedding_provider_name(),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        }
        if vectorized
        else {}
    )

    doc = Document(
        repo_id=_as_uuid(repo_id),
        source_type=source_type,
        source_id=_as_uuid(source_id),
        title=title[:500],          # PG 的 title 是 String(500)
        content=content,            # 完整正文（不截断，这是事实源）
        meta={
            # 三层合并：调用方元数据 → 策略标记 → 向量元数据
            **(metadata or {}),
            **policy_metadata(source_type),
            **embedding_metadata,
        },
    )
    db.add(doc)
    db.flush()          # 先拿到 doc.id，Milvus 的 pk 要用

    # 只有"该向量化 + Milvus 开着"才写向量
    if vectorized and settings.milvus_enabled:
        upsert_milvus_documents([doc], vectors)

    if commit:
        db.commit()
        db.refresh(doc)     # 刷新，拿到数据库生成的默认值
    return doc


def add_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    """批量写入文档（支持传入已算好的向量，避免重复计算）。

    【调用链】追这个函数会进入：
        1. should_vectorize()   → 找出哪些条目需要向量
        2. embed_texts()        → 现算向量（未传入 vectors 时）
        3. Document(...) × N    → 批量构造 ORM 对象
        4. db.add_all + flush   → 批量插入 PG
        5. upsert_milvus_documents() → 批量写 Milvus

    【被谁调用】
        services/rag/document_processing.py（上传文件，传入已算好的 vectors）
        services/rag/indexing.py（Issue/PR/CI 建索引）
        replace_documents() / replace_documents_by_metadata()

    【数据流向】
        入：db、repo_id、payloads（N 条）、vectors（可选，已算好的向量）
        出：Document 列表（已 flush；commit=True 时已提交）

    四种组合：
        vectors 传入 + sync=True   → 直接用传入的向量写 Milvus（最快）
        vectors 不传 + generate    → 现算向量再写
        generate=False             → 只写 PG，不算向量（纯关键词模式）
        sync=False                 → 只写 PG，不碰 Milvus
    """
    # ── 第 1 步：过滤空内容（空内容进库只会污染检索）──
    clean_payloads = [payload for payload in payloads if payload.content.strip()]

    # ── 第 2 步：源码硬拦截（批量版）──
    if any(payload.source_type == "code_file" for payload in clean_payloads):
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")

    # ── 第 3 步：如果传了向量，条数必须对得上 ──
    if vectors is not None and len(vectors) != len(clean_payloads):
        raise ValueError("embedding count does not match document payload count")

    # 与 clean_payloads 等长的槽位，只有需要向量化的位置会被填上
    embeddings: list[list[float] | None] = [None] * len(clean_payloads)

    # 只对"该向量化"的条目算向量，索引记下来后面用
    vector_indexes = [
        index for index, payload in enumerate(clean_payloads) if should_vectorize(payload.source_type)
    ]

    if vectors is not None:
        # 走"传入向量"路径：按位置填槽
        for index in vector_indexes:
            embeddings[index] = vectors[index]
    elif vector_indexes and generate_embeddings:
        # 走"现算"路径：只算需要的那几条（不是全部）
        generated = embed_texts(
            [f"{clean_payloads[index].title}\n{clean_payloads[index].content}" for index in vector_indexes],
            settings.embedding_dimensions,
        )
        if len(generated) != len(vector_indexes):
            raise ValueError("embedding count does not match vectorized document payload count")
        for index, embedding in zip(vector_indexes, generated):
            embeddings[index] = embedding

    # 要同步 Milvus 就必须所有向量化条目都有向量，否则会写进去一半
    if sync_vector_store and any(embeddings[index] is None for index in vector_indexes):
        raise ValueError("RAG documents require embeddings before Milvus synchronization")

    # ── 第 4 步：批量构造 ORM 对象 ──
    docs: list[Document] = []
    for payload, embedding in zip(clean_payloads, embeddings):
        vectorized = embedding is not None
        embedding_metadata = (
            {
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                # 调用方可以覆盖（比如 document_processing 传的是仓库级配置）
                **(embedding_info or {}),
            }
            if vectorized
            else {}
        )
        docs.append(
            Document(
                repo_id=_as_uuid(repo_id),
                source_type=payload.source_type,
                source_id=_as_uuid(payload.source_id),
                title=payload.title[:500],
                content=payload.content,
                meta={
                    **(payload.metadata or {}),
                    **policy_metadata(payload.source_type),
                    **embedding_metadata,
                },
            )
        )

    if not docs:
        return []

    # ── 第 5 步：写 PG（flush 拿 id），再写 Milvus ──
    db.add_all(docs)
    db.flush()

    if sync_vector_store and settings.milvus_enabled:
        vector_documents = [docs[index] for index in vector_indexes]
        vector_values = [embeddings[index] for index in vector_indexes]
        upsert_milvus_documents(
            vector_documents,
            [vector for vector in vector_values if vector is not None],
        )

    if commit:
        db.commit()
        for doc in docs:
            db.refresh(doc)
    return docs


def replace_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    source_types: list[str],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    """按来源类型整组替换（先删该仓库这些类型的全部文档，再写入新的）。

    【调用链】追这个函数会进入：
        1. db.query(Document).delete()    → 删 PG
        2. delete_milvus_documents()      → 删 Milvus
        3. add_documents()                → 写新数据（PG + Milvus）
    【被谁调用】services/rag/indexing.py（重建某类索引）

    【为什么整组替换】同步是"全量快照"语义：
        GitHub 上删掉的 Issue，本地也要消失。增量更新无法感知删除。

    【代价】
        如果这次拉取失败（payloads 为空），会把原有数据删光。
        所以调用方必须保证"拿到完整数据后再调这个函数"。
    """
    repo_uuid = _as_uuid(repo_id)

    # ── 第 1 步：删 PG（synchronize_session=False 表示不更新内存里的对象，
    #              因为后面不会再访问它们，省一次同步开销）──
    db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type.in_(source_types)).delete(
        synchronize_session=False
    )
    db.flush()

    # ── 第 2 步：删 Milvus（必须同步删，否则留下孤儿向量）──
    if sync_vector_store and settings.milvus_enabled:
        delete_milvus_documents(repo_id=repo_uuid, source_types=source_types)

    # ── 第 3 步：写新数据（commit=False，由本函数统一提交）──
    docs = add_documents(
        db,
        repo_uuid,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )

    if commit:
        db.commit()
    return docs


def replace_documents_by_metadata(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    metadata_filters: dict[str, Any],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
) -> list[Document]:
    """按元数据精确替换（只删符合过滤条件的，其他保留）。

    【调用链】should_vectorize() → delete_milvus_documents() → add_documents() → 本函数
    【被谁调用】services/rag/indexing.py（局部更新某个源对象）

    用于"某个文档重新上传"这种局部更新场景。

    【为什么要先全查再在 Python 里过滤】
        meta 字段是 JSON 类型，PostgreSQL 虽然支持 JSON 查询，
        但 SQLAlchemy 的跨数据库兼容写法很麻烦，而且 meta 里的 key 是动态的。
        数据量不大时（单个仓库的某类文档通常几十到几百条），
        在应用层过滤更简单可靠。
    """
    repo_uuid = _as_uuid(repo_id)

    rows = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type == source_type).all()

    # 在应用层做元数据匹配（JSON 字段无法用 SQL 精确比较）
    # all(...) 表示"所有过滤条件都要满足"
    document_ids = [
        row.id
        for row in rows
        if all((row.meta or {}).get(key) == value for key, value in metadata_filters.items())
    ]

    if document_ids:
        db.query(Document).filter(Document.id.in_(document_ids)).delete(synchronize_session=False)
        # 只有该向量化的类型才需要删 Milvus
        if should_vectorize(source_type) and settings.milvus_enabled:
            delete_milvus_documents(repo_id=repo_uuid, document_ids=document_ids)

    db.flush()

    # 注意：这里没有传 vectors，所以 add_documents 会现算向量
    docs = add_documents(db, repo_uuid, payloads, commit=False)

    if commit:
        db.commit()
    return docs


def delete_document_group(
    db: Session,
    repo_id: str | uuid.UUID,
    source_id: str | uuid.UUID,
    source_type: str | None = None,
    *,
    commit: bool = True,
) -> int:
    """删除某个源对象的全部切片（PG + Milvus 同步删）。

    【调用链】should_vectorize() → delete_milvus_documents() → 本函数
    【被谁调用】services/rag/document_processing.py（重传文件前清旧数据）

    【为什么叫 group】一个源对象（如一个上传文件）会被切成多个 chunk，
        删的时候必须整组删，否则会留下孤儿向量。

    【数据流向】
        入：repo_id、source_id（源对象 ID，不是切片 ID）、可选 source_type
        出：删掉的 PG 行数

    【返回值的意义】
        0 表示"本来就没有"（首次上传的正常情况），不是错误。
    """
    repo_uuid = _as_uuid(repo_id)
    source_uuid = _as_uuid(source_id)

    query = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_id == source_uuid)
    if source_type:
        query = query.filter(Document.source_type == source_type)

    document_ids = [row.id for row in query.all()]      # 先记下 ID，删完就查不到了
    count = query.delete(synchronize_session=False)

    # 只有该向量化的类型才需要删 Milvus
    # （source_type 为 None 时无法判断，保守起见还是删一次 —— 删不存在的不会报错）
    if settings.milvus_enabled and (source_type is None or should_vectorize(source_type)):
        delete_milvus_documents(repo_id=repo_uuid, source_id=source_uuid, document_ids=document_ids)

    if commit:
        db.commit()
    return count
