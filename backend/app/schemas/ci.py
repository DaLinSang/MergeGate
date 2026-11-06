"""
CI 接口契约
==========

【业务定位】CI 运行记录的列表格式，以及 CI 排障结果的包装。
【数据流向】services/repos_sync.py（写入） → api/routes/ci.py → frontend/app/ci/page.tsx
【核心规则】排障结论的结构在 analysis.py 的 CIDebugOutput 里。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import CIDebugOutput


class WorkflowRunResponse(BaseModel):
    """一条 CI 运行记录。"""

    id: UUID                            # 内部 ID
    repo_id: UUID                       # 所属仓库
    name: str                           # workflow 名称
    status: str                         # queued / in_progress / completed
    conclusion: str | None = None       # success / failure / cancelled
    html_url: str | None = None         # GitHub 链接
    jobs: list[dict] = []               # Job 与 Step 的完整树（JSON）
    logs_text: str | None = None        # 失败日志（只对失败 Run 拉取）
    created_at: datetime | None = None  # 创建时间
    updated_at: datetime | None = None  # 更新时间

    model_config = {"from_attributes": True}


class CIAnalysisResponse(BaseModel):
    """CI 排障结果。"""

    analysis_id: UUID | str     # 分析记录 ID
    result: CIDebugOutput       # 排障结论
