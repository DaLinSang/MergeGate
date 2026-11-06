"""
Issue 契约
==========

【业务定位】Issue 列表格式，以及 Issue 分析结果的包装。
【数据流向】services/repos_sync.py（写入） → api/routes/issues.py → 前端 Issue 页/右栏 Tab
【核心规则】分析结果的结构在 analysis.py 的 IssueAnalysisOutput 里，这里只包一层。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import IssueAnalysisOutput


class IssueResponse(BaseModel):
    """一条 Issue。"""

    id: UUID                            # 内部 ID
    repo_id: UUID                       # 所属仓库
    number: int                         # 用户看到的编号
    title: str                          # 标题
    body: str | None = None             # 正文
    state: str                          # open / closed
    labels: list[str] = []              # 标签
    author: str | None = None           # 作者
    assignees: list[str] = []           # 负责人
    created_at: datetime | None = None  # 创建时间
    updated_at: datetime | None = None  # 更新时间

    model_config = {"from_attributes": True}


class IssueAnalysisResponse(BaseModel):
    """Issue 分诊结果。"""

    analysis_id: UUID | str         # 分析记录 ID
    result: IssueAnalysisOutput     # 分诊结论
