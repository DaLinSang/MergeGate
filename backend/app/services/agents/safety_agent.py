"""
安全审查 Agent
=============

【业务定位】
    判断"这个操作要不要人工确认" —— 数据流 7（AI 写操作 → 草稿 → 确认 → GitHub）
    的第一道闸门。

    【为什么需要它】
        AI 想关 Issue、加标签、发评论时，不能直接改 GitHub。
        必须先生成草稿，等人点"确认"。
        这个 Agent 就是回答"这个动作属于哪一类、要不要拦"。

【数据流向】
    上游（谁调它）：
        services/agents/workflow_orchestrator.py  工作流里的 safety_review 任务
        services/agents/chat_agent.py             工具调用前的风险判定
    下游（它写谁）：
        无 —— 纯规则判断，不碰数据库、不调模型

【核心规则】
    ★ 白名单思路反过来用：只要动作里出现"写/发/改"类关键词，就要求确认。
      宁可多拦（用户体验略差），也不能漏拦（可能造成真实损害）。

【本文件函数地图】
    SafetyAgent.run()   ★ 唯一方法：关键词匹配 → 判定风险等级 → 返回契约
"""

from typing import Any

from app.schemas.analysis import SafetyOutput
from app.services.agents.base import BaseAgent


class SafetyAgent(BaseAgent):
    """动作风险判定 Agent（纯规则，不调 LLM）。

    【为什么不用 LLM 判断】
        1. 太慢 —— 每次写操作前都要等一次模型调用
        2. 不可靠 —— 模型可能被"这个操作很安全"这类描述说服
        规则匹配虽然笨，但确定性 100%，且不可能被提示注入绕过。
    """

    name = "safety_agent"
    description = "检查某个操作是否需要人工确认。"
    available_tools = ["classify_action_risk"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """判断一个动作是否需要人工确认。

        【调用链】无 —— 纯关键词匹配
        【被谁调用】workflow_orchestrator（safety_review 任务）、chat_agent（工具调用前）

        【数据流向】
            入：input_data.action（动作名，如 "close_issue"）、input_data.draft（可选，草稿内容）
            出：SafetyOutput 的 dict（requires_confirmation / risk_level / reason / draft）

        【关键词表的两个作用】
            英文（comment / create_issue / label / close_issue / write / post / send）
            + 中文（评论 / 创建 / 新建 / 发给 / 发送 / 改标签 / 关闭 / 写回）
            中英都列，是因为动作名可能来自代码常量（英文）或用户自然语言（中文）。

        【为什么 risk_level 只有 medium / low 两档】
            当前实现里"需要确认"就是 medium，"不需要"就是 low。
            high 预留给"批量删除 / 改仓库设置"这类高危操作（还没实现）。
        """
        # 统一小写，避免 "Close_Issue" 这种大小写差异导致漏判
        action = input_data.get("action", "").lower()

        # 只要命中任一"写操作"关键词，就要求人工确认
        risky = any(
            token in action
            for token in [
                "comment",
                "create_issue",
                "label",
                "close_issue",
                "send_report",
                "write",
                "post",
                "send",
                "评论",
                "创建",
                "新建",
                "发给",
                "发送",
                "改标签",
                "关闭",
                "写回",
            ]
        )

        return SafetyOutput(
            requires_confirmation=risky,
            risk_level="medium" if risky else "low",
            # reason 是给用户看的：说清"为什么需要你确认"
            reason="MVP 阶段所有外部写操作只生成草稿。" if risky else "只读分析操作不需要人工确认。",
            # draft 原样带回（有就带上，没有就是 None）
            draft=input_data.get("draft"),
        ).model_dump()
