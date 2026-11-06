"""
工作区契约
========

【业务定位】把多个仓库归成一组，用于跨仓库汇总报告（如前端/后端两个仓库看整体情况）。
【数据流向】frontend/app/workspaces/page.tsx → api/routes/workspaces.py → workspaces 表
【核心规则】repo_ids 用 JSON 存而不是关联表 —— 只是"一组 ID"，不需要反向查询，建表是过度设计。
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WorkspaceCreateRequest(BaseModel):
    """创建工作区的请求。"""

    name: str                                           # 工作区名称
    description: str | None = None                      # 描述
    repo_ids: list[UUID] = Field(default_factory=list)  # 包含哪些仓库


class WorkspaceResponse(BaseModel):
    """工作区详情。"""

    id: UUID                                        # 工作区 ID
    name: str                                       # 名称
    description: str | None = None                  # 描述
    repo_ids: list[str] = Field(default_factory=list)  # 包含的仓库 ID（JSON 数组存字符串）
    created_at: datetime                            # 创建时间
    updated_at: datetime | None = None              # 更新时间

    model_config = {"from_attributes": True}


class MultiRepoReportRequest(BaseModel):
    """跨仓库汇总报告的请求。"""

    workspace_id: UUID | None = None                # 用哪个工作区
    repo_ids: list[UUID] = Field(default_factory=list)  # 或直接指定仓库（优先级更高）
    start_date: date                                # 起始日期
    end_date: date                                  # 结束日期


class MultiRepoReportResponse(BaseModel):
    """跨仓库报告结果。"""

    report_markdown: str            # 报告正文（Markdown）
    metrics: dict                   # 汇总指标（Issue 数、PR 数、合并率等）
    repo_summaries: list[dict]      # 每个仓库的单独摘要
