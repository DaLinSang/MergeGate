"""
项目记忆索引路由
================

【路由清单】
    GET  /{repo_id}/memory-index/state     查当前索引状态（文件数 / 上次扫描时间 / 是否在 snooze）
    POST /{repo_id}/memory-index/scan      触发一次扫描（后台任务，立即返回 202）
    POST /{repo_id}/memory-index/snooze    静默 N 天，别再提醒我扫描

【数据流向】
    前端"项目记忆"卡片
      → 本文件
      → services/project_indexing.py
          scan_project_index_by_id()  → 扫仓库文档 → 写 documents 表 → 重建知识图谱
      → 后台任务（BackgroundTasks，不阻塞响应）

【要点】
    scan 是"先改状态再派任务"：先 mark_project_index_building() 写 DB，
    前端立刻能轮询到 building；再 add_task 真正去扫。
    顺序反了的话，前端可能连续几次都看到旧的 idle 状态，以为没触发。
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.project_index import (
    ProjectIndexScanResponse,
    ProjectIndexSnoozeRequest,
    ProjectIndexSnoozeResponse,
    ProjectIndexStateResponse,
)
from app.services.project_indexing import (
    get_project_index_state,
    mark_project_index_building,
    scan_project_index_by_id,
    snooze_project_index,
)

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """三个接口都要先校验仓库存在，抽出来避免重复。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


@router.get("/{repo_id}/memory-index/state", response_model=ProjectIndexStateResponse)
async def get_memory_index_state(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """查索引状态，前端据此决定显示"去扫描 / 扫描中 / 已完成"。"""
    repo = _repo_or_404(db, repo_id)
    return get_project_index_state(db, repo)


@router.post("/{repo_id}/memory-index/scan", response_model=ProjectIndexScanResponse, status_code=202)
async def start_memory_index_scan(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ProjectIndexScanResponse:
    """触发扫描。202 表示"已受理，还没做完"。"""
    repo = _repo_or_404(db, repo_id)
    # 先落状态再派任务：保证前端下一次轮询能看到 building
    mark_project_index_building(db, repo)
    # 后台任务自己开 Session（scan_project_index_by_id 按 id 查），
    # 不复用当前请求的 db —— 请求结束 Session 就关了
    background_tasks.add_task(scan_project_index_by_id, repo.id)
    return ProjectIndexScanResponse(repo_id=repo.id, started=True, status="building")


@router.post("/{repo_id}/memory-index/snooze", response_model=ProjectIndexSnoozeResponse)
async def snooze_memory_index(
    repo_id: uuid.UUID,
    payload: ProjectIndexSnoozeRequest | None = None,
    db: Session = Depends(get_db),
) -> ProjectIndexSnoozeResponse:
    """静默索引提醒。payload 可以不传，默认 7 天。"""
    repo = _repo_or_404(db, repo_id)
    row = snooze_project_index(db, repo, days=(payload.days if payload else 7))
    return ProjectIndexSnoozeResponse(repo_id=repo.id, snoozed_until=row.snoozed_until)
