"""
推荐提问 Agent
============

【业务定位】
    根据当前仓库的真实状态（Issue / PR / CI），生成"用户接下来可能想问什么"的推荐提问。
    纯规则生成，不调 LLM —— 因为要快（每次打开对话页都要算）且要稳定（不能每次不一样）。

    【为什么推荐提问有价值】
        用户第一次打开一个仓库，不知道能问什么。
        给出 4~5 条"针对当前状态"的建议，能立刻把工具用起来。
        比"输入框里放个 placeholder：试试问我"有效得多。

【数据流向】
    上游（谁调它）：
        api/routes/chat.py（对话页加载时）
    下游（它调谁）：
        无 —— 只用 input_data 里传进来的数据，不查库、不调模型

【核心规则】
    1. **有数据就针对性推荐**：失败 CI → 排障类；open PR → 审查类；open Issue → 分派类。
    2. **没数据就推荐"上手类"**：怎么同步、怎么看结构、怎么建记忆。
    3. **按时间倒序取最近的**（最多 5 个），保证推荐和当前状态相关。
    4. 去重 + 排除用户已用过的（exclude）。

【本文件函数地图】
    run()                  ★ 主流程：构建候选 → 过滤 → 取前 N
    _build_candidates()    ★ 按数据状态拼候选（本文件的主体）
    _dedupe()              去重保序
    _normalize()           归一化（去空白 + 小写，用于去重比较）
    _short()               截断标题（22 字符）
    _sort_key()            按时间排序的 key
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.services.agents.base import BaseAgent


class RecommendationAgent(BaseAgent):
    """推荐提问 Agent（纯规则，不调 LLM）。"""

    name = "recommendation_agent"
    description = "根据当前仓库的 Issue、PR 和 CI 状态推荐上下文相关的聊天提问。"
    available_tools = ["repository_snapshot"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """生成推荐提问。

        【调用链】_build_candidates() / _normalize() → 本函数
        【被谁调用】api/routes/chat.py

        【数据流向】
            入：input_data.{issues, pull_requests, workflow_runs, limit, exclude}
            出：{agent, suggestions, source_counts}

        【limit 为什么夹在 4~5】
            少于 4 条看起来太单薄（用户以为没数据）；
            多于 5 条会让侧边栏拥挤。4~5 是 UI 上的经验值。

        【★ 第二段兜底逻辑（if not selected and excluded）】
            如果候选全被 exclude 排除了，就忽略 exclude 再选一遍。
            为什么：宁可给几条"用户用过的"，也不要给空白 ——
            空白区域比重复推荐更让人困惑。

        【source_counts 的用途】
            让前端能显示"基于 12 个 Issue / 5 个 PR / 3 次 CI 生成"，
            用户就知道推荐不是瞎编的。
        """
        limit = min(max(int(input_data.get("limit") or 5), 4), 5)
        # 用户已经用过的提问（归一化后比较）
        excluded = {self._normalize(item) for item in input_data.get("exclude") or []}
        candidates = self._build_candidates(input_data)

        # ── 第一轮：排除已用过的 ──
        selected: list[str] = []
        seen = set(excluded)
        for prompt in candidates:
            normalized = self._normalize(prompt)
            if not normalized or normalized in seen:
                continue
            selected.append(prompt)
            seen.add(normalized)
            if len(selected) >= limit:
                break

        # ── 第二轮兜底：全被排除时忽略 exclude ──
        if not selected and excluded:
            for prompt in candidates:
                normalized = self._normalize(prompt)
                if not normalized:
                    continue
                selected.append(prompt)
                if len(selected) >= limit:
                    break

        return {
            "agent": self.name,
            "suggestions": selected,
            "source_counts": {
                "issues": len(input_data.get("issues") or []),
                "pull_requests": len(input_data.get("pull_requests") or []),
                "workflow_runs": len(input_data.get("workflow_runs") or []),
            },
        }

    def _build_candidates(self, input_data: dict[str, Any]) -> list[str]:
        """按数据状态拼候选提问（★ 本文件的主体）。

        【调用链】_sort_key() / _short() / _dedupe() → 本函数
        【被谁调用】run()

        【数据流向】
            入：input_data.{issues, pull_requests, workflow_runs}
            出：候选提问列表（已去重）

        【★ 三段式结构，优先级从高到低】
            ① 空仓库 → 7 条"上手类"（怎么同步、怎么看结构、怎么建记忆）
            ② 有数据 → 按"最紧急"排序：
               失败 CI（最紧急）→ open PR → open Issue → 通用分析
            ③ 最后追加 16 条"通用类"（周报、概览、风险…）兜底

        【为什么失败 CI 排第一】
            它是唯一"正在流血"的问题。用户打开仓库最该先看的就是"为什么构建挂了"。

        【为什么每个实体给 4 条提问】
            4 条覆盖 4 个不同角度（定位 / 整理 / 关联 / checklist），
            用户总能挑到一条符合意图的。

        【为什么每个列表只取前 5 个实体】
            5 个实体 × 4 条 = 20 条候选，已经足够。
            取全部（比如 100 个 Issue）会生成 400 条，纯属浪费。

        【为什么最后要 _dedupe】
            不同分支会产出重复的提问（比如"总结这个 PR 的风险"在 open_prs 和 prs 两段都有）。
        """
        # 按时间倒序（最近的更相关）
        issues = sorted(input_data.get("issues") or [], key=self._sort_key, reverse=True)
        prs = sorted(input_data.get("pull_requests") or [], key=self._sort_key, reverse=True)
        runs = sorted(input_data.get("workflow_runs") or [], key=self._sort_key, reverse=True)

        # ── 情况 ①：空仓库 → 上手类引导 ──
        if not issues and not prs and not runs:
            return [
                "概览这个新仓库的当前状态",
                "查看这个项目的代码结构",
                "我该如何同步这个仓库的 Issue、PR 和 CI",
                "帮我建立项目记忆和团队分工",
                "生成初始化项目检查清单",
                "这个仓库接下来应该先配置什么",
                "帮我规划第一次项目同步和分析",
            ]

        # 分三类（只看"进行中"和"失败"的，已关闭的没推荐价值）
        open_issues = [item for item in issues if str(item.get("state") or "").lower() == "open"]
        open_prs = [item for item in prs if str(item.get("state") or "").lower() == "open"]
        failed_runs = [item for item in runs if str(item.get("conclusion") or "").lower() == "failure"]

        candidates: list[str] = []

        # ── ② 失败 CI（最紧急）→ 排障类 ──
        for run in failed_runs[:5]:
            name = self._short(run.get("name") or "最近 CI")
            candidates.extend(
                [
                    f"定位 {name} 失败原因",
                    f"把 {name} 的失败日志整理成修复步骤",
                    f"分析 {name} 失败是否和最近 PR 有关",
                    f"给 {name} 生成排查 checklist",
                ]
            )

        # 没有失败但有 CI → 健康状况类
        if not failed_runs and runs:
            latest = self._short(runs[0].get("name") or "最近 CI")
            candidates.extend(
                [
                    f"检查 {latest} 的 CI 状态",
                    "最近 CI 有没有潜在风险",
                    "总结最近 workflow 的健康状况",
                    "如果 CI 失败，下一步应该先看哪里",
                ]
            )

        # ── ③ open PR → 审查类 ──
        for pr in open_prs[:5]:
            number = pr.get("number")
            label = f"PR #{number}" if number else "最新 PR"
            title = self._short(pr.get("title") or "")
            suffix = f"：{title}" if title else ""
            candidates.extend(
                [
                    f"总结 {label}{suffix} 的风险",
                    f"检查 {label} 需要补哪些测试",
                    f"给 {label} 生成合并前 checklist",
                    f"{label} 是否会影响 CI 或发布",
                ]
            )

        # ── ④ open Issue → 分派类 ──
        for issue in open_issues[:5]:
            number = issue.get("number")
            label = f"Issue #{number}" if number else "最新 Issue"
            title = self._short(issue.get("title") or "")
            suffix = f"：{title}" if title else ""
            candidates.extend(
                [
                    f"{label}{suffix} 应该分给谁",
                    f"分析 {label} 的优先级和复杂度",
                    f"给 {label} 写一条澄清评论草稿",
                    f"把 {label} 拆成可执行任务",
                ]
            )

        # ── ⑤ 按"有什么数据"补充跨实体类提问 ──
        if failed_runs:
            candidates.extend(
                [
                    "最近 CI 为什么失败",
                    # 有 PR 才推荐"串起来分析"（否则没东西可串）
                    "把失败 CI 和相关 PR 串起来分析" if prs else "把失败 CI 整理成排查步骤",
                ]
            )
        if prs:
            candidates.extend(
                [
                    "总结这个 PR 的风险",
                    "按风险排序当前打开的 PR",
                    "哪些 PR 适合今天合并",
                    "生成一份合并前风险检查清单",
                ]
            )
        if issues:
            candidates.extend(
                [
                    "这个 Issue 应该分给谁",
                    "按优先级整理当前打开的 Issue",
                    "哪些 Issue 需要补充复现信息",
                    "根据团队成员能力推荐 Issue 负责人",
                    "分析打开时间最久的问题",
                ]
            )
        # 同时有 Issue 和 PR 才能做"站会摘要"（单一类型信息不够）
        if issues and prs:
            candidates.extend(["把当前 Issue 和 PR 做成站会摘要", "找出最可能拖慢交付的事项"])

        # ── ⑥ 通用类兜底（16 条，覆盖各种使用场景）──
        candidates.extend(
            [
                "生成本周研发周报",
                "概览当前仓库状态",
                "找出最值得优先处理的 3 件事",
                "生成一份今日工程状态摘要",
                "梳理当前项目的主要阻塞点",
                "看一下这个项目的代码结构",
                "搜索当前代码里的测试配置",
                "检查最近变更可能带来的风险",
                "帮我制定下一步排查计划",
                "哪些任务需要先找负责人确认",
                "生成一份发给团队的进展说明",
                "检查是否有需要安全确认的写操作",
                "当前仓库有哪些知识库内容可用",
                "本周研发周报应该包含哪些重点",
                "总结当前项目的质量风险",
                "列出今天可以快速推进的事项",
            ]
        )
        return self._dedupe(candidates)

    def _dedupe(self, values: list[str]) -> list[str]:
        """去重保序（归一化后比较）。

        【调用链】被 _build_candidates() 调用

        归一化后比较：忽略大小写和多余空格。
        保留第一次出现的原始写法（用户看到的版本）。
        """
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize(value)
            if not normalized or normalized in seen:
                continue
            result.append(value)
            seen.add(normalized)
        return result

    def _normalize(self, value: Any) -> str:
        """归一化：去首尾空白 + 压缩连续空白 + 转小写。

        【调用链】被 run() / _dedupe() 调用

        用于"判断两条提问是不是同一条"。
        """
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _short(self, value: Any, max_length: int = 22) -> str:
        """压缩空白并截断到 22 字符（用于把标题嵌进提问里）。

        【调用链】被 _build_candidates() 调用

        【为什么是 22】
            提问本身还有前缀（"定位 xxx 失败原因"）。
            标题太长会让整条提问超出一行，看起来很难读。
        """
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip() + "..."

    def _sort_key(self, item: dict[str, Any]) -> float:
        """取时间戳作为排序 key（用于按"最近"排序）。

        【调用链】被 _build_candidates() 调用

        【四级回退】updated_at > created_at > merged_at > closed_at
            不同实体有不同字段（PR 有 merged_at，Issue 有 closed_at）。
            取第一个有值的。

        【为什么要处理两种类型（datetime / str）】
            input_data 可能来自数据库（datetime 对象），
            也可能来自 JSON 序列化后再传进来（ISO 字符串）。

        【为什么失败返回 0.0】
            没有时间的排最后，但不会报错。
        """
        for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
            value = item.get(key)
            if isinstance(value, datetime):
                return value.timestamp()
            if isinstance(value, str) and value:
                try:
                    # GitHub 的 "Z" 要换成 "+00:00" 才能被 fromisoformat 解析
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
        return 0.0
