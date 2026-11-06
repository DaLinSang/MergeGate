"""
反馈契约
========

【业务定位】用户对回答的点赞/点踩，是质量闭环的人工信号。
【数据流向】feedback-controls.tsx → api/routes/feedback.py → chat_feedback 表 → 飞书通知
【核心规则】一条消息只能评一次；点赞不填原因，点踩必须选原因（原因才是改进依据）。
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 评价：有用 / 没用
FeedbackRating = Literal["helpful", "unhelpful"]

# 点踩原因：答案不准 / 答非所问 / 缺上下文 / 引用不可靠 / 工具失败 / 其他
FeedbackReason = Literal[
    "inaccurate",
    "not_relevant",
    "missing_context",
    "unreliable_citation",
    "tool_error",
    "other",
]

# 人工复核状态：待处理 / 复核中 / 已解决 / 已忽略
ReviewStatus = Literal["open", "in_review", "resolved", "dismissed"]


class ChatFeedbackCreate(BaseModel):
    """提交反馈的请求。"""

    repo_id: UUID                                   # 属于哪个仓库
    conversation_id: UUID                           # 属于哪个会话
    assistant_message_id: UUID                      # 针对哪条消息（表上有 unique 约束）
    rating: FeedbackRating                          # 有用 / 没用
    reason: FeedbackReason | None = None            # 点踩原因（点赞时必须为空）
    comment: str | None = Field(default=None, max_length=1000)  # 文字补充

    @model_validator(mode="after")
    def validate_reason(self) -> "ChatFeedbackCreate":
        """跨字段校验：点赞时不能填点踩原因。"""
        if self.rating == "helpful" and self.reason is not None:
            raise ValueError("有帮助反馈不需要填写负反馈原因")
        return self


class ChatFeedbackReview(BaseModel):
    """人工复核反馈。"""

    review_status: ReviewStatus                                     # 复核后的状态
    review_note: str | None = Field(default=None, max_length=2000)  # 复核备注


class ChatFeedbackResponse(BaseModel):
    """反馈详情。"""

    id: UUID                                # 反馈 ID
    repo_id: UUID                           # 所属仓库
    conversation_id: UUID                   # 所属会话
    assistant_message_id: UUID              # 针对的消息
    run_id: UUID | None = None              # 关联的运行记录（可查 trace）
    rating: FeedbackRating                  # 有用 / 没用
    reason: FeedbackReason | None = None    # 点踩原因
    comment: str | None = None              # 文字补充
    review_status: ReviewStatus             # 复核状态
    review_note: str | None = None          # 复核备注
    notification_status: str                # 通知状态：pending / sent / failed
    notification_error: str | None = None   # 推送失败原因
    notified_at: datetime | None = None     # 推送时间
    reviewed_at: datetime | None = None     # 复核时间
    created_at: datetime                    # 创建时间
    updated_at: datetime                    # 更新时间

    model_config = ConfigDict(from_attributes=True)
