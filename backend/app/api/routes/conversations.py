"""
会话管理路由
============

【路由清单】
    GET    /{repo_id}/conversations                        列出该仓库的所有会话
    POST   /{repo_id}/conversations                        新建会话
    DELETE /{repo_id}/conversations/{conversation_id}      删除会话（并返回替代会话）

【数据流向】
    前端左侧会话列表
      → 本文件
      → services/conversations.py::list_conversations / create_conversation / delete_conversation
      → chat_conversations 表（+ 级联清理 chat_messages 等）

【要点】
    1. 删除接口不只是删，还要返回"替代会话"和最新列表 ——
       否则前端删掉当前打开的会话后会白屏（不知道该切到哪）。
    2. 删除里包含级联清理（11 张表），顺序在 services/conversations.py 里维护。
    3. list 接口也 commit()：list_conversations 内部可能顺手创建了默认会话。
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.conversations import ConversationCreateRequest, ConversationResponse
from app.services.conversations import create_conversation, delete_conversation, list_conversations

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


@router.get("/{repo_id}/conversations", response_model=list[ConversationResponse])
async def get_conversations(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Any]:
    """列出会话。若一个都没有，service 层会自动建一个默认会话。"""
    _repo_or_404(db, repo_id)
    conversations = list_conversations(db, repo_id)
    # 上面可能创建了默认会话，所以要 commit（不 commit 前端刷新后又变空）
    db.commit()
    return conversations


@router.post("/{repo_id}/conversations", response_model=ConversationResponse)
async def add_conversation(
    repo_id: uuid.UUID,
    payload: ConversationCreateRequest | None = None,
    db: Session = Depends(get_db),
) -> Any:
    """新建会话。payload 可省略，标题由后端兜底成"新会话"。"""
    _repo_or_404(db, repo_id)
    conversation = create_conversation(db, repo_id, payload.title if payload else None)
    db.commit()
    # commit 后对象会过期，refresh 一下才能拿到数据库生成的 id / 时间戳
    db.refresh(conversation)
    return conversation


@router.delete("/{repo_id}/conversations/{conversation_id}")
async def remove_conversation(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """删除会话，同时返回替代会话 + 最新列表，让前端能平滑切换。"""
    _repo_or_404(db, repo_id)
    # delete_conversation 内部会级联清理消息 / 运行记录 / 记忆候选等，
    # 并保证至少留下一个会话，返回它作为 replacement
    replacement = delete_conversation(db, repo_id, conversation_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    # 这里手工拼 dict 而不是用 response_model：删除响应要同时带"替代会话 id"和"完整列表"，
    # 没有对应的 schema 类。
    return {
        "deleted": str(conversation_id),
        "active_conversation_id": str(replacement.id),
        "conversations": [
            {
                "id": str(item.id),
                "repo_id": str(item.repo_id),
                "title": item.title,
                "status": item.status,
                "message_count": item.message_count,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in conversations
        ],
    }
