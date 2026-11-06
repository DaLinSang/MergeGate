"""
动作草稿契约
============

【业务定位】数据流 7（AI 写操作 → 草稿 → 人工确认 → GitHub）的接口格式。
【数据流向】safety_agent.py（生成） → api/routes/action_drafts.py（确认） → github/client.py（执行）
【核心规则】status 必须按 pending_confirmation → confirmed → executed 流转，没人工确认永不执行。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ActionDraftCreateRequest(BaseModel):
    """创建草稿的请求。"""

    repo_id: UUID                                       # 属于哪个仓库
    draft_type: str = Field(default="issue_comment")    # issue_comment / create_issue / close_issue / add_labels
    target_type: str = Field(default="issue")           # issue / pull_request
    target_id: UUID | None = None                       # 目标对象 ID（新建时为空）
    title: str = ""                                     # 草稿标题
    content: str                                        # 要写入 GitHub 的正文
    risk_level: str = "medium"                          # low / medium / high
    metadata: dict = Field(default_factory=dict)        # 附加元数据


class ActionDraftResponse(BaseModel):
    """草稿详情。"""

    id: UUID                                    # 草稿 ID
    repo_id: UUID                               # 属于哪个仓库
    draft_type: str                             # 草稿类型
    target_type: str                            # 目标类型
    target_id: UUID | None = None               # 目标对象 ID
    title: str                                  # 标题
    content: str                                # 正文
    risk_level: str                             # 风险等级
    status: str                                 # pending_confirmation / confirmed / executed / failed / cancelled
    execution_result: dict | None = None        # GitHub 返回（执行后才有）
    error_message: str | None = None            # 失败原因
    metadata: dict = Field(default_factory=dict, alias="meta")  # 元数据（列名是 metadata，属性叫 meta）
    created_at: datetime                        # 创建时间
    updated_at: datetime | None = None          # 更新时间

    model_config = {"from_attributes": True, "populate_by_name": True}


class ActionDraftConfirmResponse(BaseModel):
    """确认草稿的响应。"""

    draft: ActionDraftResponse      # 确认后的草稿（status 已变更）
    audit_id: UUID                  # 本次确认的审计日志 ID


class AuditLogResponse(BaseModel):
    """审计日志条目。"""

    id: UUID                                # 日志 ID
    user_id: UUID | None = None             # 操作人（空=系统）
    repo_id: UUID | None = None             # 关联仓库
    action: str                             # 动作名，如 draft.confirm
    target_type: str | None = None          # 目标类型
    target_id: str | None = None            # 目标 ID
    status: str                             # success / failed
    request_json: dict                      # 请求快照
    result_json: dict                       # 结果快照
    created_at: datetime                    # 操作时间

    model_config = {"from_attributes": True}
