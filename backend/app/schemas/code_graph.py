"""
代码图谱契约
==========

【业务定位】代码符号与关系查询的返回格式。对应数据流 2（仓库代码 → 代码图谱）。
【数据流向】services/code_analysis.py（AST 解析写入） → api/routes/code_graph.py → 前端
【核心规则】代码不进 RAG 向量库，但可以建符号索引（符号是结构化的，不需要语义召回）。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CodeSymbolResponse(BaseModel):
    """一个代码符号（函数 / 类 / 方法）。"""

    id: UUID                                # 符号 ID
    repo_id: UUID                           # 所属仓库
    path: str                               # 文件路径
    name: str                               # 符号名
    kind: str                               # function / class / method
    language: str                           # 编程语言
    start_line: int | None = None           # 起始行
    end_line: int | None = None             # 结束行
    branch: str | None = None               # 所在分支
    commit_sha: str | None = None           # 所在提交
    pr_id: UUID | None = None               # 属于哪个 PR
    pr_number: int | None = None            # PR 编号
    metadata: dict = Field(default_factory=dict, alias="meta")  # 元数据（列名 metadata，属性 meta）
    created_at: datetime                    # 创建时间

    model_config = {"from_attributes": True, "populate_by_name": True}


class CodeGraphSearchResponse(BaseModel):
    """代码图谱搜索结果。"""

    query: str                          # 搜索词
    symbols: list[CodeSymbolResponse]   # 命中的符号
    relations: list[dict]               # 命中的关系
    documents: list[dict]               # 命中的相关文档
