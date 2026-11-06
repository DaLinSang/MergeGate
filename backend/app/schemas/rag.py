"""
RAG 接口契约
============

【业务定位】知识库配置、文档上传、检索问答的格式。对应数据流 3（入库）和流 4 后半段（检索）。
【数据流向】frontend/app/rag/* → api/routes/rag.py → services/rag/{knowledge_base,qa}.py
【核心规则】embedding 三件套绑定（改了就作废索引）；答案必须带 sources；generation_mode 必须诚实。
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class KnowledgeBaseConfigUpdate(BaseModel):
    """知识库配置的部分更新（所有字段可选）。"""

    embedding_provider: str | None = None                                   # 向量化服务商
    embedding_model: str | None = None                                      # 模型标识
    embedding_dimensions: int | None = Field(default=None, ge=64, le=4096)  # 向量维度（须与模型一致）
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None  # 检索方式
    rerank_enabled: bool | None = None                                      # 是否启用重排
    rerank_provider: str | None = None                                      # 重排服务商
    rerank_model: str | None = None                                         # 重排模型
    top_k: int | None = Field(default=None, ge=1, le=20)                    # 给模型几条证据
    score_threshold_enabled: bool | None = None                             # 是否启用阈值过滤
    score_threshold: float | None = Field(default=None, ge=0, le=1)         # 分数阈值
    chunk_size: int | None = Field(default=None, ge=200, le=4000)           # 切片大小
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)          # 切片重叠


class KnowledgeBaseResponse(BaseModel):
    """知识库详情。"""

    id: UUID                                          # 仓库 ID
    name: str                                         # 仓库名
    full_name: str                                    # owner/name
    description: str = ""                             # 描述
    provider: str = "github"                          # 平台类型
    document_count: int = 0                           # 文档数
    chunk_count: int = 0                              # 切片数（真正的检索单位）
    character_count: int = 0                          # 总字符数
    config: dict[str, Any] = Field(default_factory=dict)    # 知识库配置
    runtime: dict[str, Any] = Field(default_factory=dict)   # 运行时状态
    created_at: datetime | None = None                # 创建时间
    updated_at: datetime | None = None                # 更新时间


class KnowledgeDocumentResponse(BaseModel):
    """上传文档的处理状态。status: queued→parsing→splitting→embedding→completed/failed。"""

    id: UUID                                    # 文档 ID
    repo_id: UUID                               # 所属仓库
    name: str                                   # 原始文件名
    source_type: str                            # 来源类型
    status: str                                 # 处理状态
    content_type: str | None = None             # MIME 类型
    file_size: int = 0                          # 文件字节数
    character_count: int = 0                    # 解析出的字符数
    chunk_count: int = 0                        # 切成多少块
    error_message: str | None = None            # 失败原因
    created_at: datetime | None = None          # 上传时间
    updated_at: datetime | None = None          # 状态更新时间
    completed_at: datetime | None = None        # 完成时间


class KnowledgeChunkResponse(BaseModel):
    """单个切片。"""

    id: UUID                                    # 切片 ID
    position: int                               # 第几块（从 0 开始）
    title: str                                  # 块标题（通常来自章节）
    content: str                                # 块正文
    character_count: int                        # 字符数
    enabled: bool = True                        # 是否参与检索


class RAGQueryRequest(BaseModel):
    """检索/问答请求。"""

    question: str = Field(min_length=1, max_length=4000)                # 问题
    top_k: int = Field(default=5, ge=1, le=10)                          # 召回几条
    source_types: list[str] = Field(default_factory=list, max_length=20)  # 限定来源类型
    metadata_filters: dict[str, Any] = Field(default_factory=dict)      # 元数据过滤


class RAGSourceResponse(BaseModel):
    """一条引用来源（回答里的 [1] [2]）。"""

    citation: int                                   # 引用编号
    document_id: str                                # 来自哪条 document
    source_id: str                                  # 源对象 ID
    source_type: str                                # 源类型
    title: str                                      # 证据标题
    snippet: str                                    # 证据原文片段
    score: float                                    # 融合后最终分数 0~1
    metadata: dict[str, Any] = Field(default_factory=dict)   # 额外元数据
    retrieval: dict[str, Any] = Field(default_factory=dict)  # 检索分数明细


class RAGSearchResponse(BaseModel):
    """纯检索结果（不生成答案）。"""

    query: str                                              # 查询词
    results: list[RAGSourceResponse]                        # 召回结果
    result_count: int                                       # 召回条数
    took_ms: int                                            # 耗时（毫秒）
    retrieval_config: dict[str, Any] = Field(default_factory=dict)  # 参数快照


class RetrievalTestRequest(BaseModel):
    """召回测试请求（临时覆盖参数，不改全局配置）。"""

    query: str = Field(min_length=1, max_length=250)                        # 测试查询词
    top_k: int | None = Field(default=None, ge=1, le=20)                    # 覆盖 top_k
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None # 覆盖检索方式
    rerank_enabled: bool | None = None                                      # 覆盖是否重排
    score_threshold_enabled: bool | None = None                             # 覆盖是否启用阈值
    score_threshold: float | None = Field(default=None, ge=0, le=1)         # 覆盖阈值
    metadata_filters: dict[str, Any] = Field(default_factory=dict)          # 元数据过滤


class RetrievalTestResponse(BaseModel):
    """召回测试结果。"""

    id: UUID                                               # 测试记录 ID
    query: str                                             # 查询词
    results: list[RAGSourceResponse]                       # 召回结果
    result_count: int                                      # 召回条数
    took_ms: int                                           # 耗时
    retrieval_config: dict[str, Any] = Field(default_factory=dict)  # 参数快照
    created_at: datetime | None = None                     # 测试时间


class RetrievalTestHistoryResponse(BaseModel):
    """历史测试记录（不含完整结果，省流量）。"""

    id: UUID                                # 记录 ID
    query: str                              # 查询词
    result_count: int                       # 召回条数
    took_ms: int                            # 耗时
    created_at: datetime | None = None      # 测试时间


class ModelCatalogResponse(BaseModel):
    """可用模型清单（配置面板下拉项）。"""

    embedding_models: list[dict[str, Any]]  # 可用 embedding 模型
    rerank_models: list[dict[str, Any]]     # 可用重排模型
    deepseek: dict[str, Any]                # DeepSeek 可用模型


class RAGAnswerResponse(BaseModel):
    """RAG 问答响应。generation_mode: llm / extractive（摘录降级）/ no_evidence（拒答）。"""

    question: str                                                      # 问题
    answer: str                                                        # 答案（带引用编号）
    sources: list[RAGSourceResponse]                                   # 引用来源
    retrieval_count: int                                               # 过滤后剩余条数
    generation_mode: Literal["llm", "extractive", "no_evidence"]        # 生成方式
    model: str | None = None                                           # 实际用的模型
    answer_gate: dict[str, Any] = Field(default_factory=dict)          # 准入判定详情
    retrieval_config: dict[str, Any] = Field(default_factory=dict)     # 检索参数
    took_ms: int                                                       # 总耗时


class RAGStatusResponse(BaseModel):
    """知识库运行状态（诊断用）。"""

    repo_id: UUID                                              # 哪个仓库
    document_count: int                                        # 文档总数
    source_count: int                                          # 来源总数
    source_types: dict[str, int] = Field(default_factory=dict)  # 各类型各多少条
    embedding: dict[str, Any] = Field(default_factory=dict)     # embedding 状态
    vector_store: dict[str, Any] = Field(default_factory=dict)  # Milvus 状态
    generation: dict[str, Any] = Field(default_factory=dict)    # LLM 配置状态
    supported_files: list[str] = Field(default_factory=list)    # 支持的文件后缀


class RAGReindexResponse(BaseModel):
    """重建索引结果。换模型/升级版本/数据损坏时需要重建。"""

    repo_id: UUID                       # 哪个仓库
    indexed_documents: int              # 重建了多少篇
    embedding_provider: str             # 用的服务商
    embedding_model: str                # 用的模型
