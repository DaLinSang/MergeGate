"""
PR 契约
=======

【业务定位】PR 列表格式（含文件与评论），以及 PR 审查结果的包装。
【数据流向】services/repos_sync.py（写入） → api/routes/pull_requests.py → 前端 PR Tab
【核心规则】PR 是"主-子"结构；子资源同步时整组替换（rebase 后文件会消失）。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import PRReviewOutput


class PRFileResponse(BaseModel):
    """PR 改动的单个文件。"""

    id: UUID                        # 文件记录 ID
    filename: str                   # 文件路径
    status: str                     # added / modified / removed / renamed
    additions: int                  # 新增行数
    deletions: int                  # 删除行数
    patch: str | None = None        # diff 文本（大文件可能为空）

    model_config = {"from_attributes": True}


class PRReviewCommentResponse(BaseModel):
    """PR 的行级 Review 评论（不含普通 Issue 评论）。"""

    id: UUID                        # 评论 ID
    body: str | None = None         # 评论内容
    path: str | None = None         # 所在文件
    line: int | None = None         # 所在行号
    author: str | None = None       # 评论人
    created_at: datetime | None = None   # 评论时间
    updated_at: datetime | None = None   # 更新时间

    model_config = {"from_attributes": True}


class PullRequestResponse(BaseModel):
    """一条 PR（含文件与评论）。"""

    id: UUID                                # 内部 ID
    repo_id: UUID                           # 所属仓库
    number: int                             # 用户看到的编号
    title: str                              # 标题
    body: str | None = None                 # PR 说明
    state: str                              # open / closed / merged
    author: str | None = None               # 作者
    base_branch: str | None = None          # 目标分支
    head_branch: str | None = None          # 来源分支
    merged_at: datetime | None = None       # 合并时间
    created_at: datetime | None = None      # 创建时间
    updated_at: datetime | None = None      # 更新时间
    files: list[PRFileResponse] = []        # 改动的文件
    review_comments: list[PRReviewCommentResponse] = []  # 行级评论

    model_config = {"from_attributes": True}


class PRAnalysisResponse(BaseModel):
    """PR 审查结果。"""

    analysis_id: UUID | str     # 分析记录 ID
    result: PRReviewOutput      # 审查结论
