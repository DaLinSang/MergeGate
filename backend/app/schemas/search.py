"""
全局检索契约
==========

【业务定位】跨来源的语义检索入口。与 RAG 问答的区别：只返回检索结果，不生成答案。
【数据流向】前端搜索框 → api/routes/search.py → services/rag/retrieval.py
【核心规则】所有过滤条件可选 —— 不填是全库检索，填了就缩小范围。最薄的一层路由。
"""

from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """检索请求。多个过滤条件可组合。"""

    repo_id: UUID                   # 在哪个仓库内检索（必填，保证隔离）
    query: str                      # 查询词
    source_type: str | None = None  # 限定来源类型
    label: str | None = None        # 限定标签
    path: str | None = None         # 限定文件路径前缀
    module: str | None = None       # 限定模块
    state: str | None = None        # 限定状态（如 open）
    start_date: str | None = None   # 起始日期
    end_date: str | None = None     # 结束日期
    limit: int = Field(default=5, ge=1, le=20)  # 返回条数


class SearchResponse(BaseModel):
    """检索结果。用 dict 因为不同来源字段结构不同。"""

    results: list[dict]             # 命中的结果列表
