"""
知识库与记忆契约
==============

【业务定位】项目记忆体系：记忆笔记、团队成员、知识条目、知识图谱、召回事件、记忆候选。
    注意区分：schemas/rag.py 是"文档问答库"，本文件是"项目记忆库"。
【数据流向】frontend 记忆面板 → api/routes/knowledge.py → documents / memory_candidates 等表
【核心规则】记忆不自动生效：Agent 提炼的"事实"先进候选，人工 approve 才进 RAG（防幻觉污染）。
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryNoteRequest(BaseModel):
    """手工添加记忆笔记。"""

    title: str = ""                 # 笔记标题
    content: str = ""               # 笔记内容


class TeamMemberRequest(BaseModel):
    """添加/更新团队成员（用于推荐 Issue 负责人）。"""

    name: str                                   # 姓名
    role: str = ""                              # 角色
    strengths: str = ""                         # 擅长领域
    tech_stack: str = Field(default="", alias="techStack")  # 技术栈（前端传驼峰）

    model_config = {"populate_by_name": True}


class TeamMemberResponse(BaseModel):
    """团队成员详情。"""

    id: UUID                                    # 成员 ID
    name: str                                   # 姓名
    role: str = ""                              # 角色
    strengths: str = ""                         # 擅长领域
    tech_stack: str = Field(default="", alias="techStack")  # 技术栈
    created_at: datetime | None = None          # 创建时间

    model_config = {"populate_by_name": True}


class KnowledgeItemResponse(BaseModel):
    """一条知识条目（同一来源的多个 chunk 聚合展示）。"""

    id: UUID                                                    # 条目 ID
    document_ids: list[UUID]                                    # 包含的所有 chunk ID
    source_type: str                                            # 来源类型
    title: str                                                  # 标题
    content_preview: str                                        # 内容预览
    metadata: dict[str, Any] = Field(default_factory=dict)      # 元数据
    chunk_count: int                                            # 切片数
    created_at: datetime | None = None                          # 创建时间


class KnowledgeUploadResponse(BaseModel):
    """上传知识文件的响应（同步处理）。"""

    id: UUID                        # 文档记录 ID
    filename: str                   # 文件名
    chunks: int                     # 切成多少块
    indexed: bool                   # 是否已索引进向量库
    duplicate: bool = False         # 是否重复文件（按 SHA-256 跳过）


class KnowledgeGraphNodeResponse(BaseModel):
    """知识图谱的节点。"""

    id: str                                                 # 节点 ID（如 "issue:xxx"）
    type: str                                               # 节点类型
    title: str                                              # 显示标题
    subtitle: str = ""                                      # 副标题
    metadata: dict[str, Any] = Field(default_factory=dict)  # 元数据
    highlighted: bool = False                               # 是否高亮（搜索命中）


class KnowledgeGraphEdgeResponse(BaseModel):
    """知识图谱的边。"""

    id: str                                                 # 边 ID
    from_type: str                                          # 起点类型
    from_id: str                                            # 起点 ID
    to_type: str                                            # 终点类型
    to_id: str                                              # 终点 ID
    relation: str                                           # 关系：references / authored_by / relates_to
    confidence: float = 1.0                                 # 置信度（推断的边 < 1）
    source: str = "explicit"                                # explicit（显式）/ inferred（推断）
    metadata: dict[str, Any] = Field(default_factory=dict)  # 元数据


class KnowledgeGraphResponse(BaseModel):
    """知识图谱查询结果。"""

    repo_id: UUID                                           # 哪个仓库
    center_type: str | None = None                          # 中心节点类型（空=查全图）
    center_id: str | None = None                            # 中心节点 ID
    depth: int = 1                                          # 向外扩展几层
    nodes: list[KnowledgeGraphNodeResponse]                 # 节点列表
    edges: list[KnowledgeGraphEdgeResponse]                 # 边列表
    stats: dict[str, Any] = Field(default_factory=dict)     # 统计信息


class KnowledgeGraphRebuildResponse(BaseModel):
    """重建知识图谱的结果。"""

    repo_id: UUID                   # 哪个仓库
    edges: int                      # 边数
    nodes: int                      # 节点数


class RecallEventResponse(BaseModel):
    """一次召回事件（排查"为什么没找到"的第一手资料）。"""

    id: UUID                                                          # 事件 ID
    repo_id: UUID | None = None                                       # 哪个仓库
    conversation_id: UUID | None = None                               # 哪个会话
    tool_name: str                                                    # 调的工具
    query: str = ""                                                   # 查询词
    scope: str = "project"                                            # project / conversation / session
    mode: str = "hybrid"                                              # hybrid / vector / keyword
    status: str = "success"                                           # success / empty / error
    result_count: int = 0                                             # 召回条数
    results: list[dict[str, Any]] = Field(default_factory=list)       # 结果详情
    citations: list[dict[str, Any]] = Field(default_factory=list)     # 引用
    metadata: dict[str, Any] = Field(default_factory=dict)            # 元数据
    created_at: datetime | None = None                                # 发生时间


class MemorySearchResponse(BaseModel):
    """记忆检索结果。"""

    query: str                                  # 查询词
    results: list[dict[str, Any]]               # 命中的记忆条目
    result_count: int                           # 命中条数


class MemoryStatusResponse(BaseModel):
    """记忆系统整体状态（六个维度，哪个是 0 就说明那环没跑通）。"""

    repo_id: UUID                                               # 哪个仓库
    index: dict[str, Any] = Field(default_factory=dict)          # 项目索引状态
    documents: dict[str, Any] = Field(default_factory=dict)      # 文档统计
    evidence: dict[str, Any] = Field(default_factory=dict)       # 证据库统计
    recall: dict[str, Any] = Field(default_factory=dict)         # 召回统计
    candidates: dict[str, Any] = Field(default_factory=dict)     # 候选统计
    vector: dict[str, Any] = Field(default_factory=dict)         # 向量库状态


class MemoryCandidateResponse(BaseModel):
    """记忆候选（待人工审核）。status: pending / approved / rejected。"""

    id: UUID                                                    # 候选 ID
    repo_id: UUID | None = None                                 # 哪个仓库
    conversation_id: UUID | None = None                         # 哪个会话
    session_id: UUID | None = None                              # 哪一轮会话
    kind: str                                                   # fact / decision / preference
    title: str                                                  # 标题
    content: str                                                # 内容
    status: str                                                 # 审核状态
    source: str                                                 # session_sealer（自动）/ manual（手工）
    metadata: dict[str, Any] = Field(default_factory=dict)      # 元数据
    created_at: datetime | None = None                          # 创建时间
    reviewed_at: datetime | None = None                         # 审核时间


class MemoryCandidateActionResponse(BaseModel):
    """审核记忆候选的结果。"""

    id: UUID                                # 候选 ID
    status: str                             # approved / rejected
    document_id: UUID | None = None         # 批准后生成的 Document ID（进了 RAG）
