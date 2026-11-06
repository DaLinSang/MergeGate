"""
会话契约
========

【业务定位】对话会话的创建与展示。业务主线第一步"定位"就靠会话完成。
【数据流向】前端左侧栏 → api/routes/conversations.py → conversations 表 → chat.py（每轮对话定位）
【核心规则】用户必须先选中"哪个仓库 + 哪个会话"，后续的「这个」「刚才那个」才有指代对象。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ConversationCreateRequest(BaseModel):
    """新建会话的请求。"""

    title: str | None = None        # 会话标题（空则自动生成）


class ConversationResponse(BaseModel):
    """会话详情。"""

    id: UUID                            # 会话 ID
    repo_id: UUID                       # 所属仓库
    title: str                          # 标题
    status: str = "active"              # active / archived
    message_count: int = 0              # 消息条数
    created_at: datetime | None = None  # 创建时间
    updated_at: datetime | None = None  # 最后更新时间（按活跃度排序用）

    model_config = {"from_attributes": True}
