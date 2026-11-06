"""
分析结果契约（结构化输出）
========================

【业务定位】AI 必须按固定格式填表，而不是自由发挥 —— 否则前端没法渲染、评测没法比对。
【数据流向】services/agents/*.py（生成） → api/routes/{issues,pull_requests,ci}.py（返回前端）
【核心规则】用 Literal 锁死枚举；每个结论都要求配依据字段；每个 Output 都带 confidence。
"""

from typing import Literal

from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    """疑似重复的 Issue。"""

    issue_id: str                       # 那条 Issue 的内部 ID
    title: str                          # 标题，供用户判断
    score: float = Field(ge=0, le=1)    # 相似度 0~1


class IssueEvidence(BaseModel):
    """一条支撑证据。"""

    source_type: str                            # issue / pull_request / document 等
    title: str                                  # 证据标题
    snippet: str                                # 证据原文片段
    reference: str | None = None                 # 引用位置（文件/编号/页码）
    confidence: float | None = Field(default=None, ge=0, le=1)  # 该证据可信度


class IssueDrafts(BaseModel):
    """可选草稿（不直接发布）。"""

    clarification_comment: str | None = None    # 追问评论草稿
    task_breakdown: str | None = None           # 任务拆解草稿


class IssueAnalysisOutput(BaseModel):
    """Issue 分诊 Agent 的输出：该不该做、多急、谁来处理。"""

    summary: str                                                      # 一句话总结
    conclusion: Literal["进入开发", "等待澄清", "关闭", "拆分", "合并到已有 Issue"]  # 处置结论
    conclusion_reason: str                                            # 结论依据
    category: Literal["Bug", "Feature", "Question", "Documentation", "Refactor", "Test", "Ops"]  # 分类
    priority: Literal["P0", "P1", "P2", "P3"]                          # 优先级
    complexity: Literal["S", "M", "L", "XL"]                           # 复杂度
    suggested_owner: str                                              # 建议负责人
    owner_reason: str                                                 # 为什么是他
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)  # 疑似重复
    evidence: list[IssueEvidence] = Field(default_factory=list)       # 支撑证据
    checklist: list[str] = Field(default_factory=list)                # 检查清单
    action_items: list[str] = Field(default_factory=list)             # 行动项
    drafts: IssueDrafts = Field(default_factory=IssueDrafts)          # 草稿
    confidence: float = Field(ge=0, le=1)                             # 整体置信度


class PRReviewFinding(BaseModel):
    """PR 审查中的单条问题。"""

    severity: Literal["P1", "P2", "P3"]    # 严重程度
    title: str                              # 问题标题
    evidence: str                           # 依据（文件+行号+代码）
    required_action: str                    # 必须做什么
    blocking: bool = False                  # 是否阻塞合并


class PRReviewOutput(BaseModel):
    """PR 审查 Agent 的输出：有什么风险、能不能合。"""

    summary: str                                                          # 整体摘要
    plan: list[str] = []                                                  # 计划审查哪些方面
    executed_steps: list[str] = []                                        # 实际执行的步骤
    merge_recommendation: Literal["建议合入", "修改后合入", "暂缓", "拒绝"] = "暂缓"  # 合并建议
    recommendation_reason: str = ""                                       # 建议依据
    review_findings: list[PRReviewFinding] = Field(default_factory=list)  # 结构化问题列表
    key_changes: list[str] = []                                           # 关键变更
    risk_points: list[str] = []                                           # 风险点
    blocking_issues: list[str] = []                                       # 阻塞项
    review_checklist: list[str] = []                                      # 人工复核清单
    test_suggestions: list[str] = []                                      # 建议补的测试
    files_need_attention: list[str] = []                                  # 需重点看的文件
    review_comments: list[str] = []                                       # 可贴到 GitHub 的评论草稿
    confidence: float = Field(ge=0, le=1)                                 # 整体置信度


class CIDebugOutput(BaseModel):
    """CI 排障 Agent 的输出：为什么失败、怎么修。"""

    failure_summary: str                                                    # 失败总结
    failure_type: Literal["test", "build", "lint", "dependency", "permission", "environment", "unknown"]  # 失败类型
    plan: list[str] = []                                                    # 计划排查步骤
    executed_steps: list[str] = []                                          # 实际执行步骤
    first_error: str | None = None                                          # 日志中第一个错误（通常是根因）
    root_cause: str = ""                                                    # 推断的根因
    possible_causes: list[str] = []                                         # 其他可能原因
    fix_steps: list[str] = []                                               # 修复步骤
    debug_steps: list[str] = []                                             # 排查步骤
    related_files: list[str] = []                                           # 相关文件
    is_merge_blocking: bool = True                                          # 是否阻塞合并
    blocking_reason: str = ""                                               # 阻塞原因
    confidence: float = Field(ge=0, le=1)                                   # 整体置信度


class SafetyOutput(BaseModel):
    """安全 Agent 的输出：能不能让 AI 直接做。"""

    requires_confirmation: bool             # 是否需要人工确认
    risk_level: Literal["low", "medium", "high"]  # 风险等级
    reason: str                             # 判定依据
    draft: str | None = None                # 生成的草稿内容
