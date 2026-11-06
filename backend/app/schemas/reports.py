"""
周报契约
========

【业务定位】按时间区间生成研发周报。
【数据流向】frontend/app/reports/page.tsx → api/routes/reports.py → report_agent.py → documents 表
【核心规则】纯模板生成，不调 LLM —— 内容是"统计+罗列"，不需要推理，调 LLM 只会引入幻觉。
"""

from datetime import date
from uuid import UUID

from pydantic import BaseModel


class WeeklyReportRequest(BaseModel):
    """生成周报的请求。"""

    repo_id: UUID       # 针对哪个仓库
    start_date: date    # 统计起始日期（含）
    end_date: date      # 统计结束日期（含）


class WeeklyReportResponse(BaseModel):
    """周报结果。"""

    report_markdown: str    # 周报正文（Markdown，前端用 react-markdown 渲染）
