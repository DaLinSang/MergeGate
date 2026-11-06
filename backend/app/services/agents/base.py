"""
Agent 基类
=========

【业务定位】所有 Agent 的统一契约：必须有 name / description / available_tools，必须实现 run()。
【数据流向】被 services/agents/ 下 12 个 Agent 继承；由 chat.py / workflow_orchestrator.py 统一调用。
【核心规则】不实现 run() 的类无法实例化 —— 保证 13 个 Agent 的调用方式完全一致。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Agent 抽象基类。"""

    name: str                       # Agent 标识，如 "pr_review_agent"
    description: str                # 一句话说明干什么（会展示给用户）
    available_tools: list[str]      # 这个 Agent 能用的工具名列表

    @abstractmethod
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """执行 Agent。子类必须实现。

        参数：
            input_data → 任务输入（如 {"pr_id": "..."}）
            context    → 运行上下文（含 repo / db / 团队信息）
        返回：
            统一结构 {"route", "answer", "tool_calls", "citations"}
        """
        raise NotImplementedError

    def validate_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """输出校验钩子。默认直接放行，子类可覆盖。"""
        return output

    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """统一错误处理：把异常转成标准结构，避免上层各写一套。"""
        return {
            "error": str(error),
            "agent": self.name,
            "context": {"repo_id": str(context.get("repo_id")) if context.get("repo_id") else None},
        }
