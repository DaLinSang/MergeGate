"""
评测契约
========

【业务定位】评测的输入格式：用例怎么定义、跑哪些指标、结果怎么返回。改完 prompt 靠它判断有没有变差。
【数据流向】frontend/app/evals/page.tsx → api/routes/evaluations.py → services/evals/*
【核心规则】异步执行：提交立即返回 eval_id，前端轮询。一次评测要跑多轮真实 ChatAgent。
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RagEvalCaseRequest(BaseModel):
    """一条评测用例：问什么、期望什么。"""

    id: str = ""                                                        # 用例 ID
    question: str                                                       # 评测问题
    reference: str                                                      # 参考答案（供 Judge 比对）
    expected_source_ids: list[str] = Field(default_factory=list)        # 期望命中的文档（算 Recall）
    source_type: str | None = None                                      # 限定来源类型
    required_tools: list[str] = Field(default_factory=list)             # 期望调用的工具
    expected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # 期望的调用序列（算 F1）
    kind: str = "positive"                                              # positive（应答）/ negative（应拒答）
    must_include: list[str] = Field(default_factory=list)               # 必须包含的关键词（硬规则）
    must_not_include: list[str] = Field(default_factory=list)           # 必须不含的词（测幻觉）
    answer_regex: str | None = None                                     # 答案必须匹配的正则
    source_file: str | None = None                                      # 引用的文档文件名


class EvalRunRequest(BaseModel):
    """发起一次评测。"""

    name: str = "basic_eval"                                            # 评测名称
    repo_id: UUID | None = None                                         # 针对哪个仓库
    eval_types: list[str] = Field(                                      # 跑哪几类评测
        default_factory=lambda: ["issue_classification", "rag", "pr_summary", "tool_calls"]
    )
    run_judge: bool = True                                              # 是否启用 LLM 判官
    top_k: int = Field(default=5, ge=1, le=20)                          # 召回条数（须与历史一致才能对比）
    suite_id: str | None = None                                         # 用哪个固定评测集
    cases: list[RagEvalCaseRequest] | None = None                       # 或内联传入用例


class EvalIssueDraftRequest(BaseModel):
    """把评测失败用例转成 Issue 草稿。"""

    title: str | None = None                                            # 草稿标题（空则自动生成）


class EvalRunResponse(BaseModel):
    """评测提交/查询的响应。status: queued / running / passed / failed / error / no_cases。"""

    eval_id: str                    # 评测任务 ID
    status: str                     # 状态
    result: dict                    # 结果详情（运行中只有 progress）
