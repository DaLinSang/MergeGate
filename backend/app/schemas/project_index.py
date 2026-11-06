"""
项目索引状态契约
==============

【业务定位】"项目文档扫描到哪一步了"。接新仓库后要扫 README/设计文档并索引进 RAG，需要看进度。
【数据流向】services/project_indexing.py → api/routes/project_index.py → 前端进度条
【核心规则】status 状态机：missing → building → ready / failed；可"稍后提醒"暂停自动扫描。
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectIndexStateResponse(BaseModel):
    """项目索引的当前状态。"""

    repo_id: UUID                                       # 哪个仓库
    status: str                                         # missing / building / ready / failed
    fingerprint: str = ""                               # 文档集合指纹（判断是否需要重扫）
    docs_indexed: int = 0                               # 已索引文档数
    docs_total: int = 0                                 # 待索引总数（算百分比用）
    error_message: str | None = None                    # 失败原因
    summary_json: dict[str, Any] | None = None          # 扫描结果摘要
    snoozed_until: datetime | None = None               # 在此之前不自动扫描
    last_scan_at: datetime | None = None                # 上次扫描时间


class ProjectIndexScanResponse(BaseModel):
    """手动触发扫描的响应。"""

    repo_id: UUID                       # 哪个仓库
    started: bool                       # 是否已排入队列
    status: str = "building"            # 启动后的状态


class ProjectIndexSnoozeRequest(BaseModel):
    """"稍后提醒"的请求。"""

    days: int = Field(default=7, ge=1, le=30)   # 延后几天（1~30）


class ProjectIndexSnoozeResponse(BaseModel):
    """"稍后提醒"的响应。"""

    repo_id: UUID                       # 哪个仓库
    snoozed_until: datetime             # 新的恢复扫描时间
