"""
结构化输出契约（转出口）
=======================

【业务定位】
    把 schemas/analysis.py 里的四个 Pydantic 模型集中在这里重新导出，
    让 services/llm/ 这一层有"自己的入口"，不用直接依赖 schemas 的路径。

    【为什么要有这个文件】
        分层习惯：services/llm/ 应该只暴露"LLM 相关的东西"。
        调用方写 `from app.services.llm.structured_outputs import SafetyOutput`
        比写 `from app.schemas.analysis import SafetyOutput` 更能表达
        "我用的是 LLM 的结构化输出契约"。

【数据流向】
    上游：app/schemas/analysis.py（真正的定义在这里）
    下游：services/agents/*（各 Agent 用这些模型约束输出）
          services/llm/client.py 的 chat_json()（配合 response_format 用）

【核心规则】
    这里【只做转出】，不定义任何新模型 —— 定义必须留在 schemas/analysis.py，
    否则会出现"两个地方都能定义契约"的混乱。

【本文件函数地图】
    无函数。__all__ 列出 5 个可导出模型：
        IssueAnalysisOutput   Issue 分析的输出
        PRReviewOutput        PR 审查的输出（含 PRReviewFinding 列表）
        CIDebugOutput         CI 排障的输出
        SafetyOutput          安全审查的输出
        PRReviewFinding       PR 单条发现（PRReviewOutput 的子结构）
"""

from app.schemas.analysis import CIDebugOutput, IssueAnalysisOutput, PRReviewFinding, PRReviewOutput, SafetyOutput

__all__ = [
    "CIDebugOutput",
    "IssueAnalysisOutput",
    "PRReviewFinding",
    "PRReviewOutput",
    "SafetyOutput",
]
