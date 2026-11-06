"""
Issue 分析 Agent（分诊 / 优先级 / 负责人 / 结论）
==============================================

【业务定位】
    回答"这个 Issue 该怎么办"：归类、定优先级、估复杂度、推荐负责人、给结论和 checklist。
    这是数据流"定位 → 取证 → 判断 → 沉淀"里"判断"这一步的最典型实现。

    【和 PR 审查 Agent 同构的"双轨制"】
        先用规则算一份（_fallback_analysis），再让 LLM 分析，逐字段合并。
        规则保证"事实不编"，LLM 补充"表达和推理"。

【数据流向】
    上游（谁调它）：
        services/agents/workflow_orchestrator.py   工作流里的 issue_analysis 任务
        services/evals/runner.py                   评测（注入 NoopLLM 走纯规则）
    下游（它调谁）：
        llm.client.chat_json()              调模型（用 ISSUE_ANALYSIS_PROMPT）
        schemas.analysis.IssueAnalysisOutput 输出契约校验

【核心规则】
    1. ★ **suggested_owner 必须来自真实团队成员**（不允许编造人名）。
    2. ★ **conclusion 只有五个合法值**（进入开发/等待澄清/关闭/拆分/合并到已有 Issue）。
    3. 证据必须能追溯到输入上下文（evidence 里每条都有 reference）。
    4. confidence 由"描述长度 + 证据数 + 重复项 + 是否定到人"决定（0.35~0.86）。

【本文件函数地图】

    ┌─ 主流程 ────────────────────────────────────────
    │  run()                     ★ 规则兜底 → LLM → 合并 → 校验
    │  _fallback_analysis()      ★ 纯规则分析（七个维度）
    │
    ├─ 分类三兄弟 ───────────────────────────────────
    │  _classify_category()      七类（Bug/Doc/Test/Ops/Refactor/Feature/Question）
    │  _classify_priority()      P0/P1/P2/P3
    │  _estimate_complexity()    S/M/L/XL
    │
    ├─ 负责人推荐（最复杂的一块）──────────────────────
    │  _suggest_owner()          ★ 三级回退：已有 assignee → 团队匹配 → 待分配
    │      ├ _suggest_owner_from_team()  ★ 画像匹配打分（关键词 + 领域规则）
    │      ├ _infer_area()               按关键词推断方向（backend/frontend/ci/docs）
    │      ├ _owner_signal_text()        拼"用于匹配的信号文本"
    │      ├ _profile_terms()            从画像抽词
    │      └ _owner_domain_rules()       ★ 四条领域规则（桌面化/后端/前端/CI）
    │
    ├─ 结论与产出 ───────────────────────────────────
    │  _choose_conclusion()      ★ 五级判定
    │      ├ _looks_like_multi_scope()  判断是否"范围过大"
    │      └ _needs_clarification()     ★ 判断是否缺关键信息（按类型不同标准）
    │  _build_checklist()        按结论给 4~5 步 checklist
    │  _build_drafts()           生成澄清评论草稿 / 任务拆解草稿
    │
    ├─ 证据收集 ─────────────────────────────────────
    │  _collect_evidence()       五类证据（Issue 本身/相似/代码/文档/会话/团队）
    │  _dedupe_evidence()        去重（按四元组）
    │  _evidence_from_context()  上下文 → 证据
    │  _team_evidence()          团队成员 → 证据
    │
    └─ 合并与工具 ───────────────────────────────────
       _merge_with_fallback()    ★ LLM + 规则合并（含负责人白名单校验）
       _is_allowed_owner()       负责人是否在真实团队里
       _should_prefer_fallback_owner() 泛称 vs 具体人
       _estimate_confidence() / _clip() / _validated_duplicates() / _build_summary()
"""

import re
from typing import Any

from app.schemas.analysis import IssueAnalysisOutput, IssueDrafts
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import ISSUE_ANALYSIS_PROMPT


class IssueAgent(BaseAgent):
    """Issue 分诊 Agent（规则兜底 + LLM 增强）。

    【为什么规则部分这么长（400 多行）】
        Issue 分诊的很多判断有明确的业务标准：
            Bug 类必须有复现步骤或影响范围，否则"等待澄清"
            XL 复杂度或包含多个独立目标 → "拆分"
            相似度 ≥ 0.86 → "合并到已有 Issue"
        这些标准写成规则比让模型判断更稳定、更可解释。
        模型的价值在于"读懂正文的言外之意"，规则负责"守住底线"。
    """

    name = "issue_analyst_agent"
    description = "分析 GitHub Issue 的分诊结论、优先级、复杂度、负责人、证据和下一步动作。"
    available_tools = ["get_issue_detail", "search_similar_issues", "search_related_code", "search_project_docs", "read_thread_context"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """分析一个 Issue。

        【调用链】追这个函数会进入：
            1. _fallback_analysis()    → 纯规则分析（永远成功）
            2. llm.chat_json()         → LLM 分析（可能返回空）
            3. _merge_with_fallback()  → 合并（含负责人白名单校验）
            4. IssueAnalysisOutput.model_validate → 契约校验
        【被谁调用】workflow_orchestrator（issue_analysis 任务）、evals/runner

        【数据流向】
            入：input_data（Issue 的 number/title/body/labels/state/assignees）、
                context（duplicate_candidates / code_references / project_documents /
                         conversation_evidence / team_members）
            出：IssueAnalysisOutput 的 dict

        【★ 三层降级】
            LLM 空 → fallback
            LLM 校验失败 → fallback（静默）
            LLM 正常 → 合并结果
        """
        fallback = self._fallback_analysis(input_data, context)
        llm_output = await self.llm.chat_json(ISSUE_ANALYSIS_PROMPT, {"issue": input_data, "context": context})
        if llm_output:
            try:
                normalized = self._merge_with_fallback(llm_output, fallback, input_data, context)
                return IssueAnalysisOutput.model_validate(normalized).model_dump()
            except Exception:
                pass
        return fallback

    def _fallback_analysis(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """纯规则分析（★ 七个维度依次算）。

        【调用链】追这个函数会进入：
            1. _classify_category() / _classify_priority() / _estimate_complexity()
            2. _suggest_owner()          → 负责人推荐
            3. _validated_duplicates()   → 校验重复项
            4. _choose_conclusion()      → 结论（依赖前面的分类和重复项）
            5. _build_checklist() / _build_drafts() → 产出
            6. _collect_evidence() / _estimate_confidence() → 证据与置信度
        【被谁调用】run()

        【数据流向】
            入：input_data、context
            出：IssueAnalysisOutput 的 dict

        【★ 顺序有依赖，不能随便调换】
            conclusion 依赖 category / complexity / duplicates。
            所以必须先算分类和重复项，再定结论。
        """
        # 标题 + 正文拼起来（关键词匹配用，统一小写）
        text = f"{input_data.get('title', '')} {input_data.get('body', '')}".lower()
        body = str(input_data.get("body") or "")
        # 标签也转小写（GitHub 的标签大小写不统一）
        labels = [label.lower() for label in input_data.get("labels", [])]

        # ── 分类三兄弟 ──
        category = self._classify_category(text, labels)
        priority = self._classify_priority(text, labels)
        complexity = self._estimate_complexity(text, labels)

        # ── 负责人（最复杂的一块）──
        owner, owner_reason = self._suggest_owner(input_data, text, context)

        # ── 重复项校验（只保留有 id + title 的）──
        duplicates = self._validated_duplicates(context.get("duplicate_candidates", []))

        # ── 结论（依赖上面的结果）──
        conclusion, conclusion_reason = self._choose_conclusion(input_data, text, labels, category, complexity, duplicates)

        # ── 产出 ──
        checklist = self._build_checklist(conclusion, category)
        drafts = self._build_drafts(input_data, conclusion, checklist)
        evidence = self._collect_evidence(input_data, context)
        confidence = self._estimate_confidence(body, evidence, duplicates, owner)

        return IssueAnalysisOutput(
            summary=self._build_summary(input_data, conclusion),
            conclusion=conclusion,
            conclusion_reason=conclusion_reason,
            category=category,
            priority=priority,
            complexity=complexity,
            suggested_owner=owner,
            owner_reason=owner_reason,
            duplicate_candidates=duplicates,
            evidence=evidence,
            checklist=checklist,
            # action_items 和 checklist 是同一份内容（schema 里两个字段都要）
            action_items=checklist,
            drafts=drafts,
            confidence=confidence,
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any], input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """把 LLM 结果合并进规则结果（★ 六步修正）。

        【调用链】_dedupe_evidence() / _is_allowed_owner() / _should_prefer_fallback_owner() → 本函数
        【被谁调用】run()

        【数据流向】
            入：output（LLM）、fallback（规则）、input_data、context
            出：合并后的 dict

        【★ 六步修正，每一步都在防一类具体问题】
            ① checklist / action_items 互相同步
               模型可能只给其中一个。两个字段语义相同，必须一致。
            ② evidence 空就用 fallback + 去重
               证据是"可追溯性"的基础，不能空。
            ③ duplicate_candidates 空就用 fallback
            ④ drafts 不是 dict 就用 fallback（模型可能返回字符串）
            ⑤ ★ 负责人白名单校验
               模型可能编一个人名（"张伟"）。不在团队列表里就退回 fallback。
               这是"防编造负责人"这条规则的代码层保障。
            ⑥ ★ 泛称 vs 具体人
               模型给了"backend"（方向泛称），规则给了"张三"（具体人）→ 用规则。
               因为"分给 backend"等于没分。
            ⑦ category 修正：模型说 Question，但规则识别出 Bug → 信规则
               （Question 是"兜底类别"，模型容易往这里躲）
        """
        merged = {**fallback, **output}

        # ① checklist / action_items 互相同步
        if not merged.get("checklist") and merged.get("action_items"):
            merged["checklist"] = merged["action_items"]
        if not merged.get("action_items") and merged.get("checklist"):
            merged["action_items"] = merged["checklist"]

        # ② evidence：空则兜底，然后统一去重
        if not merged.get("evidence"):
            merged["evidence"] = fallback["evidence"]
        merged["evidence"] = self._dedupe_evidence(merged.get("evidence"))

        # ③ 重复项兜底
        if not merged.get("duplicate_candidates"):
            merged["duplicate_candidates"] = fallback["duplicate_candidates"]

        # ④ drafts 类型兜底
        if not isinstance(merged.get("drafts"), dict):
            merged["drafts"] = fallback["drafts"]

        # ⑤ ★ 负责人白名单校验（防编造人名）
        if not self._is_allowed_owner(str(merged.get("suggested_owner") or ""), input_data, context):
            merged["suggested_owner"] = fallback["suggested_owner"]
            merged["owner_reason"] = fallback["owner_reason"]

        # ⑥ ★ 泛称 vs 具体人
        if self._should_prefer_fallback_owner(str(merged.get("suggested_owner") or ""), str(fallback.get("suggested_owner") or "")):
            merged["suggested_owner"] = fallback["suggested_owner"]
            merged["owner_reason"] = fallback["owner_reason"]

        # ⑦ category 修正：Question 是兜底类别，模型容易往这躲
        if merged.get("category") == "Question" and fallback.get("category") in {"Bug", "Feature", "Documentation", "Ops", "Refactor", "Test"}:
            merged["category"] = fallback["category"]
        return merged

    def _should_prefer_fallback_owner(self, current_owner: str, fallback_owner: str) -> bool:
        """判断是否该用规则的负责人覆盖模型的（★ 泛称 vs 具体人）。

        【调用链】被 _merge_with_fallback() 调用

        【规则】
            模型给的是泛称（backend / frontend / ci / docs / 待分配 / 空）
            且规则给的是具体人名
            → 用规则的

        【为什么】
            "建议分给 backend" 等于没建议 —— 用户还是要自己找人。
            规则如果能定位到具体人（"张三"），就该用具体人。
        """
        generic_owners = {"", "待分配", "unassigned", "backend", "frontend", "ci", "docs"}
        return fallback_owner not in generic_owners and current_owner.lower() in generic_owners

    def _is_allowed_owner(self, owner: str, input_data: dict[str, Any], context: dict[str, Any]) -> bool:
        """校验负责人是否"真实存在"（★ 防编造人名）。

        【调用链】被 _merge_with_fallback() 调用

        【白名单来源两个】
            ① input_data.assignees（Issue 当前的指派人）
            ② context.team_members 里每个成员的 name

        【为什么"待分配"/"unassigned"/空 直接放行】
            这些不是"人名"，是"还没定"的占位符，不需要校验。

        【★ 为什么必须有这个校验】
            LLM 很容易"顺手编一个名字"（尤其是中文名）。
            编出来的名字会让用户以为"系统认识这个人" —— 严重的信任问题。
        """
        normalized_owner = owner.strip().lower()
        # 占位符放行
        if normalized_owner in {"", "待分配", "unassigned"}:
            return True

        # 白名单：Issue 现有指派人
        allowed = {str(item).strip().lower() for item in input_data.get("assignees") or [] if str(item).strip()}

        # 白名单：团队成员
        members = context.get("team_members") or []
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict) and str(member.get("name") or "").strip():
                    allowed.add(str(member.get("name")).strip().lower())

        return normalized_owner in allowed

    def _build_summary(self, input_data: dict[str, Any], conclusion: str) -> str:
        """一句话摘要（"Issue #42 xxx 的建议结论是：进入开发。"）。

        【调用链】被 _fallback_analysis() 调用

        有 number 就带编号（更精确），没有就用 "Issue" 兜底。
        """
        number = input_data.get("number")
        title = str(input_data.get("title") or "Untitled Issue").strip()
        prefix = f"Issue #{number}" if number else "Issue"
        return f"{prefix} {title} 的建议结论是：{conclusion}。"

    def _classify_category(self, text: str, labels: list[str]) -> str:
        """按关键词判断 Issue 类别（★ 有序规则）。

        【调用链】被 _fallback_analysis() 调用

        【数据流向】
            入：text（标题+正文小写）、labels
            出：Bug / Documentation / Test / Ops / Refactor / Feature / Question

        【★ 顺序很重要：从"最具体"到"最宽泛"】
            Bug 的关键词最独特（error/500/crash）
            Feature 的关键词最泛（feature/enhancement/需求）
            Question 放最后（它本质是"以上都不是"的兜底）

            如果把 Feature 放前面，"修复 error 的新功能需求"会被判成 Feature，
            而它其实是 Bug。

        【★ 为什么标签拼在最前面】
            `" ".join(labels) + " " + text`
            标签是"人工标注的强信号"，应该优先于正文关键词。
            放在前面虽然不影响 any() 的判断（它只看"有没有"），
            但语义上表达"标签优先"的意图。

        【兜底返回 Feature】
            实在识别不出就假设是功能需求 —— 这是最常见的 Issue 类型，
            而且 Feature 的 checklist 最通用。
        """
        joined = " ".join(labels) + " " + text
        rules = [
            ("Bug", ["bug", "error", "fail", "500", "crash", "exception", "broken", "regression"]),
            ("Documentation", ["doc", "documentation", "readme", "guide", "文档", "说明"]),
            ("Test", ["test", "pytest", "coverage", "单测", "测试"]),
            ("Ops", ["ops", "ci", "workflow", "deploy", "docker", "infra", "环境", "权限"]),
            ("Refactor", ["refactor", "cleanup", "重构", "technical debt"]),
            ("Feature", ["feature", "enhancement", "proposal", "需求", "新增"]),
            ("Question", ["question", "how to", "why does", "咨询", "疑问"]),
        ]
        for category, tokens in rules:
            if any(token in joined for token in tokens):
                return category
        return "Feature"

    def _classify_priority(self, text: str, labels: list[str]) -> str:
        """判断优先级 P0/P1/P2/P3（★ 从高到低判定）。

        【调用链】被 _fallback_analysis() 调用

        【★ 顺序：P0 → P1 → P3 → 默认 P2】
            为什么 P3 在 P2 之前判：因为 P3 有明确关键词（typo/minor），
            而 P2 是"默认档"。先判明确的，剩下的才落默认。

            P0 和 P1 必须在前面：它们是"紧急"信号，
            一旦命中就不能被后面的 P3 关键词（比如正文里恰好有 "minor"）覆盖。

        【默认 P2 而不是 P3】
            P3 意味着"可以慢慢来"。默认给 P2 更保守 ——
            让用户自己降级，比让紧急问题被埋没好。
        """
        joined = " ".join(labels) + " " + text
        if any(token in joined for token in ["p0", "sev0", "critical", "security", "data loss", "数据丢失", "全量不可用"]):
            return "P0"
        if any(token in joined for token in ["p1", "sev1", "crash", "cannot login", "500", "生产", "阻塞", "blocker"]):
            return "P1"
        if any(token in joined for token in ["p3", "minor", "polish", "typo", "文案", "低优"]):
            return "P3"
        return "P2"

    def _estimate_complexity(self, text: str, labels: list[str]) -> str:
        """估算复杂度 S/M/L/XL（★ 先关键词，再按长度）。

        【调用链】被 _fallback_analysis() 调用

        【四级】
            XL  架构级（architecture/migration/rewrite/跨模块）
            L   大改动（large/complex/integration/schema/多模块）
            S   小改动（短文本 + typo/copy/doc）
            M   默认；但如果文本超长（>1400 字符）就升 L

        【★ 为什么 S 要加"长度 < 280"的条件】
            typo 一定是很短的 Issue。如果正文写了 2000 字还说是 typo，那不合理。
            所以 S 需要两个条件同时满足：短 + 关键词。

        【★ 为什么长文本默认升 L】
            描述越长，说明问题越复杂（涉及多个场景/多个模块）。
            1400 字符是经验阈值（约 700 汉字）。
        """
        joined = " ".join(labels) + " " + text
        if any(token in joined for token in ["xl", "architecture", "migration", "rewrite", "跨模块", "架构"]):
            return "XL"
        if any(token in joined for token in ["large", "complex", "integration", "schema", "多模块"]):
            return "L"
        if len(text) < 280 and any(token in joined for token in ["typo", "copy", "文案", "doc"]):
            return "S"
        return "M" if len(text) < 1400 else "L"

    def _suggest_owner(self, input_data: dict[str, Any], issue_text: str, context: dict[str, Any]) -> tuple[str, str]:
        """推荐负责人（★ 三级回退）。

        【调用链】_owner_signal_text() / _suggest_owner_from_team() / _infer_area() → 本函数
        【被谁调用】_fallback_analysis()

        【数据流向】
            入：input_data、issue_text、context
            出：(负责人, 理由)

        【★ 三级回退，每级都给出可解释的理由】
            ① 已有 assignee → 直接沿用（最可靠，不打乱现状）
            ② 团队画像匹配 → 给出匹配分和命中词
            ③ 匹配不上 → "待分配" + 说明缺什么

        【★ 为什么"待分配"的理由要写得这么细】
            四种不同情况给四句不同的话：
                没团队成员 + 能识别方向 → "这个 Issue 更像 backend 方向，请先添加对应成员"
                没团队成员               → "请先添加开发成员"
                有团队但没匹配上方向      → "未找到与 xxx 方向匹配的成员"
                有团队但完全没信号        → "匹配度不足，需人工确认"
            用户看到理由就知道"下一步该做什么"，而不是只看到"待分配"三个字。
        """
        # ── ① 已有 assignee → 直接沿用 ──
        assignees = input_data.get("assignees") or []
        if assignees:
            owner = str(assignees[0])
            return owner, f"Issue 已分配给 {owner}，优先沿用现有负责人。"

        # ── ② 团队画像匹配 ──
        signal_text = self._owner_signal_text(issue_text, context)
        team_owner, team_reason = self._suggest_owner_from_team(issue_text, signal_text, context)
        if team_owner:
            return team_owner, team_reason or "团队画像与 Issue 关键词匹配。"

        # ── ③ 匹配不上 → 待分配（四种理由）──
        area = self._infer_area(signal_text)
        members = context.get("team_members") or []
        if not isinstance(members, list) or not members:
            if area != "待分配":
                return "待分配", f"当前项目还没有添加团队成员；这个 Issue 更像 {area} 方向，请先在团队面板添加对应开发成员后再分配。"
            return "待分配", "当前项目还没有添加团队成员，请先在团队面板添加开发成员后再分配。"
        if area != "待分配":
            return "待分配", f"未找到与 {area} 方向匹配的团队成员，请先补充团队成员画像后再分配。"
        return "待分配", "当前 Issue 与团队画像、代码/文档证据的匹配度不足，需要人工确认负责人。"

    def _suggest_owner_from_team(self, issue_text: str, signal_text: str, context: dict[str, Any]) -> tuple[str | None, str | None]:
        """按团队画像给每个成员打分，取最高分（★ 两种打分方式）。

        【调用链】_profile_terms() / _owner_domain_rules() → 本函数
        【被谁调用】_suggest_owner()

        【数据流向】
            入：issue_text、signal_text（含代码/文档/会话信号）、context
            出：(最佳成员名, 理由)；都不匹配返回 (None, None)

        【★ 两种打分，权重差 8 倍】
            ① 画像词命中：每个命中词 +1
               把成员的 role/strengths/techStack 拆成词，看有多少词出现在信号文本里。
            ② 领域规则命中：每次 +8
               比如 Issue 里有 "electron" 且成员画像里有 "desktop/javascript" →
               这条规则直接给 8 分，压过零散词匹配。

            为什么领域规则权重这么高：
                "electron" 和 "desktop" 在语义上强相关，但字面不同。
                靠词匹配永远命不中，必须用"领域 → 双向关键词表"来表达这种关联。

        【★ 为什么要求 score > best_score（严格大于）】
            同分时保留【先出现】的成员。
            团队列表通常按添加顺序，保留前面的更符合直觉（也保证确定性）。

        【★ 理由只取前 4 个】
            "匹配词：api、database、token、auth、jwt、postgres…" 太长没人看。
            4 个足够说明"为什么是他"。
        """
        members = context.get("team_members") or []
        if not isinstance(members, list):
            return None, None

        best_name = None
        best_score = 0
        best_reasons: list[str] = []

        for member in members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            # 把画像字段拼成一段文本（兼容 techStack / tech_stack 两种命名）
            profile = " ".join(
                str(member.get(field) or "").lower()
                for field in ["role", "strengths", "techStack", "tech_stack"]
            )

            score = 0
            reasons: list[str] = []

            # ① 画像词命中（每个 +1）
            for term in self._profile_terms(profile):
                if len(term) >= 2 and term in signal_text:
                    score += 1
                    reasons.append(term)

            # ② 领域规则命中（每次 +8）
            for domain, issue_terms, profile_terms in self._owner_domain_rules():
                if any(term in signal_text for term in issue_terms) and any(term in profile for term in profile_terms):
                    score += 8
                    reasons.append(domain)

            if name and score > best_score:
                best_name = name
                best_score = score
                best_reasons = reasons

        if best_name and best_score > 0:
            # 去重保序，取前 4 个
            reason_text = "、".join(dict.fromkeys(best_reasons[:4])) or "团队画像"
            return best_name, f"团队成员画像与 Issue/代码/文档信号匹配：{reason_text}，匹配分 {best_score}。"
        return None, None

    def _infer_area(self, issue_text: str) -> str:
        """推断 Issue 属于哪个方向（backend / frontend / ci / docs / 待分配）。

        【调用链】被 _suggest_owner() 调用

        【★ 顺序有影响】
            backend 在前：它的关键词（api/database/token/auth）比较独特。
            frontend 有 "page"/"按钮" 这类词，但也可能出现在后端 Issue 里
            （"给 API 加分页 page"）。
            所以先判更独特的。

        【兜底 "待分配"】
            识别不出方向时返回"待分配"，让上层给出"请补充成员画像"的提示。
        """
        area_keywords = {
            "backend": ["api", "database", "token", "server", "fastapi", "postgres", "auth", "jwt"],
            "frontend": ["ui", "page", "button", "react", "next", "css", "layout", "browser", "页面", "按钮", "electron", "desktop", "桌面化", "桌面端"],
            "ci": ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy"],
            "docs": ["readme", "doc", "文档", "说明"],
        }
        for area, keywords in area_keywords.items():
            if any(keyword in issue_text for keyword in keywords):
                return area
        return "待分配"

    def _owner_signal_text(self, issue_text: str, context: dict[str, Any]) -> str:
        """拼"用于匹配负责人的信号文本"（★ 比 Issue 正文信息更丰富）。

        【调用链】被 _suggest_owner() 调用

        【数据流向】
            入：issue_text、context
            出：一段小写文本

        【★ 为什么要拼上下文，而不只用 Issue 正文】
            用户写 Issue 常常很简短（"登录挂了"）。
            但 context 里有代码引用、项目文档、历史会话 ——
            这些能补充"这块代码是谁写的、属于哪个模块"。
            只用正文，负责人匹配率会很低。

        【★ 为什么要做"桌面化 → electron desktop"的语义扩展】
            "桌面化"是中文说法，"electron" 是技术实现。
            成员画像里通常写技术栈（electron/react），不会写"桌面化"。
            不加这个扩展，"桌面化"类的 Issue 永远匹配不到前端成员。
            这是"领域同义词"的手工补充 —— 简单但有效。
        """
        parts = [issue_text]

        # 从三个上下文来源补充信号（每个最多取 6 条）
        for key in ["code_references", "project_documents", "conversation_evidence"]:
            items = context.get(key)
            if not isinstance(items, list):
                continue
            for item in items[:6]:
                if isinstance(item, dict):
                    parts.extend([str(item.get("title") or ""), str(item.get("snippet") or ""), str(item.get("content") or "")])

        text = " ".join(parts).lower()

        # ★ 语义扩展：中文业务词 → 技术栈词
        if any(term in text for term in ["桌面化", "桌面端", "桌面应用", "electron"]):
            text += " electron desktop windows macos javascript node frontend"
        return text

    def _profile_terms(self, profile: str) -> list[str]:
        """从成员画像里抽词（★ 先去掉"虚词"再切）。

        【调用链】被 _suggest_owner_from_team() 调用

        【为什么先 replace 掉"擅长/熟悉/技术栈"】
            这些词在每个人的画像里都有，匹配上它们等于没匹配。
            直接删掉比"加入停用词表"更简单。
        """
        normalized = profile.replace("擅长", " ").replace("熟悉", " ").replace("技术栈", " ")
        return re.findall(r"[a-zA-Z0-9_+#.\u4e00-\u9fff]+", normalized.lower())

    def _owner_domain_rules(self) -> list[tuple[str, list[str], list[str]]]:
        """四条领域规则：(领域名, Issue 侧关键词, 画像侧关键词)。

        【调用链】被 _suggest_owner_from_team() 调用

        【数据流向】
            出：[(domain, issue_terms, profile_terms)] 列表

        【★ 这是"跨语言语义匹配"的核心】
            同一条规则里两侧关键词【字面不同】：
                Issue 写"桌面化"，画像写"electron/desktop"
                Issue 写"后端/API"，画像写"fastapi/postgres"
            靠这种"双向词表"才能跨越"业务说法"和"技术栈"的鸿沟。

        【为什么只有四条】
            覆盖当前团队的主要方向。加太多会互相干扰
            （一个 Issue 同时命中两条规则时，谁加分都不对）。
        """
        return [
            (
                "桌面化/electron",
                ["桌面化", "桌面端", "桌面应用", "electron", "desktop", "windows", "macos", "native app"],
                ["electron", "desktop", "桌面", "windows", "macos", "tauri", "javascript", "js", "node", "frontend", "前端"],
            ),
            (
                "后端/API",
                ["api", "database", "token", "server", "fastapi", "postgres", "auth", "jwt"],
                ["api", "database", "server", "fastapi", "postgres", "auth", "jwt", "后端"],
            ),
            (
                "前端/UI",
                ["ui", "page", "button", "react", "next", "css", "layout", "browser", "页面", "按钮"],
                ["ui", "react", "next", "css", "browser", "frontend", "前端"],
            ),
            (
                "CI/工程化",
                ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy"],
                ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy", "工程化"],
            ),
        ]

    def _validated_duplicates(self, candidates: Any) -> list[dict[str, Any]]:
        """校验"重复 Issue 候选"（★ 只保留字段完整的）。

        【调用链】被 _fallback_analysis() / _collect_evidence() 调用

        【数据流向】
            入：candidates（可能不是列表、可能是脏数据）
            出：[{issue_id, title, score}]，最多 5 条

        【★ 三条清洗规则】
            ① 不是 list → 返回空（防上游传错类型）
            ② 缺 issue_id 或 title → 丢掉（没 ID 无法追溯，没标题无法展示）
            ③ score 转 float 失败 → 用 0.0；并夹到 0~1
               为什么要夹：上游可能算出 1.5 或 -0.2（归一化 bug），
               不夹会让"相似度 150%"这种荒谬数据流到前端。
        """
        if not isinstance(candidates, list):
            return []
        rows = []
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            issue_id = str(item.get("issue_id") or item.get("source_id") or "").strip()
            title = str(item.get("title") or "").strip()
            try:
                score = max(0.0, min(1.0, float(item.get("score") or 0)))
            except (TypeError, ValueError):
                score = 0.0
            if issue_id and title:
                rows.append({"issue_id": issue_id, "title": title, "score": score})
        return rows

    def _choose_conclusion(
        self,
        input_data: dict[str, Any],
        text: str,
        labels: list[str],
        category: str,
        complexity: str,
        duplicates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """决定 Issue 结论（★ 五级判定，从上到下命中即返回）。

        【调用链】_looks_like_multi_scope() / _needs_clarification() → 本函数
        【被谁调用】_fallback_analysis()

        【数据流向】
            入：Issue 元信息、text、labels、category、complexity、duplicates
            出：(结论, 理由)

        【★ 五级顺序，从"最确定"到"最宽松"】
            ① 已关闭 或 带无效标签     → 关闭
            ② 相似度 ≥ 0.86           → 合并到已有 Issue
            ③ 复杂度 XL 或 范围过大    → 拆分
            ④ 缺关键信息               → 等待澄清
            ⑤ 都不满足                 → 进入开发

        【★ 为什么"合并"在"拆分"之前】
            如果这个 Issue 就是别人提过的那个，那根本不需要拆 ——
            先合并，避免制造重复工作。

        【★ 0.86 这个阈值怎么来的】
            0.8 太松（容易误合并，把不同问题当成重复）；
            0.9 太严（几乎不会触发）。
            0.86 是"高置信重复"的经验线，和 LLM 的语义相似度分布吻合。

        【★ 为什么"等待澄清"排在"拆分"之后】
            拆分的前提是"知道要做什么，只是太大"。
            如果连要做什么都不清楚（缺信息），应该先澄清，再谈拆分。
        """
        state = str(input_data.get("state") or "").lower()
        joined = " ".join(labels) + " " + text

        # ① 关闭
        if state == "closed" or any(token in joined for token in ["invalid", "wontfix", "won't fix", "not planned", "无效", "不处理"]):
            return "关闭", "Issue 已关闭或带有无效/不处理信号，建议归档并保留关闭说明。"

        # ② 合并（相似度 ≥ 0.86）
        if any(float(item.get("score") or 0) >= 0.86 for item in duplicates):
            return "合并到已有 Issue", "相似历史 Issue 的匹配度较高，建议合并上下文，避免重复推进。"

        # ③ 拆分
        if complexity in {"XL"} or self._looks_like_multi_scope(text):
            return "拆分", "Issue 范围偏大或包含多个独立目标，拆分后更容易排期和验收。"

        # ④ 等待澄清
        if self._needs_clarification(text, category):
            return "等待澄清", "当前描述缺少影响范围、复现/验收条件或期望结果，直接进入开发风险较高。"

        # ⑤ 进入开发
        return "进入开发", "Issue 信息基本可执行，优先确认验收口径后进入开发。"

    def _looks_like_multi_scope(self, text: str) -> bool:
        """判断 Issue 是否"包含多个独立目标"（★ 两种信号）。

        【调用链】被 _choose_conclusion() 调用

        【两种信号（或关系）】
            ① 列表项多：`\n-` + `\n*` + `、` + `;` 出现 ≥ 5 次
               说明作者列了一堆要做的事
            ② 宽泛词：同时/以及/多个/all of/migration/architecture/跨模块
               这些词暗示"这不是一件事"

        【为什么阈值是 5】
            1~4 个列表项很常见（就是正常的分点说明）。
            5 个以上才开始像"任务清单"而不是"问题描述"。
        """
        separators = text.count("\n-") + text.count("\n*") + text.count("、") + text.count(";")
        broad_terms = ["同时", "以及", "并且", "多个", "all of", "migration", "architecture", "跨模块"]
        return separators >= 5 or any(term in text for term in broad_terms)

    def _needs_clarification(self, text: str, category: str) -> bool:
        """判断 Issue 是否"缺关键信息，需要澄清"（★ 按类型不同标准）。

        【调用链】被 _choose_conclusion() 调用

        【数据流向】
            入：text、category
            出：bool

        【★ 三类判断标准，各不相同】
            通用：文本 < 80 字符 → 一定缺信息（连描述都没写清）
            Bug：必须有"复现步骤" 或 "影响范围" 之一
                 （缺了就无法定位，也没法判断严重程度）
            Feature：必须有"验收标准" 或 "价值说明"，且文本 < 220 字符
                 （需求可以短，但必须说清"做到什么算完成"）

        【★ 为什么 Bug 用 or（有任一即可），Feature 用 and（都要有）】
            Bug 只要给了复现步骤，即使没说影响范围，也能开工。
            Feature 光说"我要这个功能"不够，得说清"什么算做完"。

        【其他类型（Documentation/Test/Ops/Refactor）→ 不要求澄清】
            这几类通常目标明确（"补个文档""加个测试"），不需要复杂澄清。
        """
        # 通用：太短
        if len(text.strip()) < 80:
            return True

        if category == "Bug":
            has_repro = any(token in text for token in ["reproduce", "steps", "复现", "步骤", "expected", "actual", "期望", "实际"])
            has_impact = any(token in text for token in ["impact", "影响", "scope", "范围", "blocking", "阻塞"])
            return not (has_repro or has_impact)

        if category == "Feature":
            has_acceptance = any(token in text for token in ["acceptance", "验收", "criteria", "why", "value", "价值", "场景"])
            return not has_acceptance and len(text.strip()) < 220

        return False

    def _collect_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
        """收集证据（★ 五类，按可信度排序）。

        【调用链】_validated_duplicates() / _evidence_from_context() / _team_evidence() / _dedupe_evidence() → 本函数
        【被谁调用】_fallback_analysis()

        【数据流向】
            入：input_data、context
            出：证据列表（最多 10 条，已去重）

        【★ 五类证据及 confidence】
            ① Issue 本身          1.0     （原文，最可信）
            ② 相似 Issue          = 相似度
            ③ 代码引用 / 文档 / 会话 = 上游 score
            ④ 团队成员             None    （只是"谁能做"，不是"事实依据"）

        【为什么 Issue 本身要放第一条】
            所有结论都必须能回到"原文说了什么"。
            把它放第一条，前端展示时用户先看到"依据是什么"。

        【为什么最后截断到 10 条】
            证据太多会稀释重点。10 条足够支撑一个分诊结论。
        """
        # ① Issue 本身（confidence 1.0）
        evidence = [
            {
                "source_type": "issue",
                "title": f"Issue #{input_data.get('number')}: {input_data.get('title')}",
                "snippet": self._clip("\n".join(filter(None, [str(input_data.get("title") or ""), str(input_data.get("body") or "")])), 320),
                "reference": str(input_data.get("id") or input_data.get("number") or ""),
                "confidence": 1.0,
            }
        ]

        # ② 相似 Issue（最多 3 条，confidence = 相似度）
        for duplicate in self._validated_duplicates(context.get("duplicate_candidates", []))[:3]:
            evidence.append(
                {
                    "source_type": "similar_issue",
                    "title": duplicate["title"],
                    "snippet": f"相似度 {duplicate['score']:.2f}，可作为重复/历史处理经验参考。",
                    "reference": duplicate["issue_id"],
                    "confidence": duplicate["score"],
                }
            )

        # ③ 代码 / 文档 / 会话
        for key, source_type in [
            ("code_references", "code"),
            ("project_documents", "document"),
            ("conversation_evidence", "conversation"),
        ]:
            evidence.extend(self._evidence_from_context(context.get(key), source_type))

        # ④ 团队成员
        evidence.extend(self._team_evidence(context.get("team_members")))

        # 去重 + 截断
        return self._dedupe_evidence(evidence)[:10]

    def _dedupe_evidence(self, items: Any) -> list[dict[str, Any]]:
        """证据去重（★ 四元组作为 key）。

        【调用链】被 _collect_evidence() / _merge_with_fallback() 调用

        【数据流入出】
            入：items（可能不是列表）
            出：去重后的列表

        【★ 为什么用四元组 (source_type, reference, title, snippet[:180])】
            单看 source_type 太粗（同类型有不同证据）
            单看 snippet 太长（占内存）
            取 snippet 前 180 字符 —— 足够区分不同证据，又不会太长。

        【为什么 snippet 为空就丢掉】
            没有内容的证据无法展示、无法核对 —— 是无效证据。
        """
        if not isinstance(items, list):
            return []
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in items:
            if not isinstance(item, dict) or not item.get("snippet"):
                continue
            key = (
                str(item.get("source_type") or ""),
                str(item.get("reference") or ""),
                str(item.get("title") or ""),
                str(item.get("snippet") or "")[:180],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
        return rows

    def _evidence_from_context(self, items: Any, default_source_type: str) -> list[dict[str, Any]]:
        """把上下文条目转成证据（★ 字段名容错）。

        【调用链】被 _collect_evidence() 调用

        【★ 三处字段名容错】
            title     可能是 title / path / source_id
            snippet   可能是 snippet / content
            reference 可能是 reference / source_id / id
            不同来源（代码/文档/会话）的字段命名不一样，全试一遍。

        【★ confidence 为 None 的情况】
            上游没给 score → confidence = None（表示"未评估"）。
            为什么不用 0：0 会被理解成"完全不可信"，而 None 是"不知道"。
        """
        if not isinstance(items, list):
            return []
        rows = []
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("path") or item.get("source_id") or default_source_type).strip()
            snippet = self._clip(str(item.get("snippet") or item.get("content") or ""), 320)
            # 没内容就不算证据
            if not snippet:
                continue
            score = item.get("score")
            try:
                confidence = max(0.0, min(1.0, float(score))) if score is not None else None
            except (TypeError, ValueError):
                confidence = None
            rows.append(
                {
                    "source_type": str(item.get("source_type") or default_source_type),
                    "title": title,
                    "snippet": snippet,
                    "reference": str(item.get("reference") or item.get("source_id") or item.get("id") or ""),
                    "confidence": confidence,
                }
            )
        return rows

    def _team_evidence(self, members: Any) -> list[dict[str, Any]]:
        """把团队成员转成证据（最多 3 人）。

        【调用链】被 _collect_evidence() 调用

        【★ confidence 固定 None】
            团队成员不是"事实依据"，而是"分配依据"。
            给 1.0 会误导（好像"这个人在 Issue 里被提到"）。

        【★ 画像为空时的兜底文案】
            "团队成员资料可作为负责人分配依据。" ——
            至少说明这条证据的用途，不留空。
        """
        if not isinstance(members, list) or not members:
            return []
        rows = []
        for member in members[:3]:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            # 用 "；" 拼接画像（比空格更易读）
            profile = "；".join(
                str(member.get(field) or "").strip()
                for field in ["role", "strengths", "techStack", "tech_stack"]
                if str(member.get(field) or "").strip()
            )
            rows.append(
                {
                    "source_type": "team_member",
                    "title": f"团队成员：{name}",
                    "snippet": profile or "团队成员资料可作为负责人分配依据。",
                    "reference": name,
                    "confidence": None,
                }
            )
        return rows

    def _build_checklist(self, conclusion: str, category: str) -> list[str]:
        """按结论生成 checklist（★ 每种结论一套，4~5 步）。

        【调用链】被 _fallback_analysis() 调用

        【★ 五套 checklist，内容完全不同】
            等待澄清      → 教用户"该补什么信息"
            合并到已有     → 教用户"怎么合并"
            拆分          → 教用户"怎么拆"
            关闭          → 教用户"关闭前要做什么"
            进入开发      → 标准开发流程（Bug 类额外插一步"先写失败测试"）

        【★ 为什么"进入开发"的 Bug 类要插"先写复现测试"】
            这是 TDD 的核心实践：先写一个能复现 bug 的失败测试，
            再修到测试通过。这样能保证"修好了"且"不会回归"。
            插在 index 2 的位置（定位代码之后、开始实现之前）。
        """
        if conclusion == "等待澄清":
            return [
                "请补充问题背景、影响范围和用户场景。",
                "Bug 类问题补充复现步骤、期望结果、实际结果和日志/截图。",
                "Feature 类需求补充验收标准和非目标范围。",
                "澄清后重新运行 Issue Agent 分析。",
            ]
        if conclusion == "合并到已有 Issue":
            return [
                "确认相似 Issue 是否覆盖当前诉求。",
                "把当前 Issue 的新增信息整理为评论追加到已有 Issue。",
                "为当前 Issue 标记 duplicate/关联关系。",
                "经人工确认后关闭当前重复 Issue。",
            ]
        if conclusion == "拆分":
            return [
                "按用户价值、模块边界或交付顺序拆成 2-5 个子任务。",
                "为每个子任务补充验收标准、依赖关系和建议负责人。",
                "确认最小可交付范围，优先排第一个可独立验证的子任务。",
                "把拆分说明作为评论回填到原 Issue。",
            ]
        if conclusion == "关闭":
            return [
                "确认关闭原因是无效、重复、已完成还是不计划处理。",
                "补充一条简短关闭评论，说明依据和替代路径。",
                "经人工确认后更新标签或关闭 Issue。",
            ]

        # 默认：进入开发
        baseline = [
            "确认 Issue 验收标准和不做范围。",
            "定位相关代码/文档位置并补充设计备注。",
            "创建开发分支并实现最小修复/功能增量。",
            "补充自动化测试或手工验证步骤。",
            "提交 PR 并在描述中关联该 Issue。",
        ]
        if category == "Bug":
            # TDD：先写能复现的失败测试
            baseline.insert(2, "先写一个能复现问题的失败测试或最小复现脚本。")
        return baseline

    def _build_drafts(self, input_data: dict[str, Any], conclusion: str, checklist: list[str]) -> IssueDrafts:
        """生成动作草稿（★ 只是草稿，必须人工确认才发出去）。

        【调用链】被 _fallback_analysis() 调用

        【数据流向】
            入：input_data、conclusion、checklist
            出：IssueDrafts（clarification_comment / task_breakdown）

        【三种情况产出不同草稿】
            等待澄清      → 澄清评论（列出要补的信息）
            拆分 / 进入开发 → 任务拆解（把 checklist 前 5 步转成列表）
            合并到已有     → 合并说明

        【★ 为什么"拆分"和"进入开发"共用 task_breakdown】
            两者的下一步都是"具体做什么"，只是拆分还需要额外拆子任务。
            checklist 已经包含了这两类动作，直接复用。

        【★ 为什么只取 checklist[:5]】
            草稿是给用户"贴到 Issue 里"的。
            5 条是"足够具体 + 不冗长"的平衡点。
        """
        number = input_data.get("number")
        title = str(input_data.get("title") or "该 Issue")
        clarification = None
        task_breakdown = None

        if conclusion == "等待澄清":
            clarification = (
                f"针对 Issue #{number} {title}，目前信息还不足以直接进入开发。"
                "请补充：影响范围、复现步骤/使用场景、期望结果、实际结果，以及是否有日志、截图或相关版本信息。"
            )

        if conclusion in {"拆分", "进入开发"}:
            task_breakdown = "建议下一步拆解：\n" + "\n".join(f"- {item}" for item in checklist[:5])

        if conclusion == "合并到已有 Issue":
            task_breakdown = (
                f"建议将 Issue #{number} 的新增信息合并到相似历史 Issue，并在当前 Issue 留下 duplicate 说明后再关闭。"
            )

        return IssueDrafts(clarification_comment=clarification, task_breakdown=task_breakdown)

    def _estimate_confidence(self, body: str, evidence: list[dict[str, Any]], duplicates: list[dict[str, Any]], owner: str) -> float:
        """按"信息充足度"算置信度（0.35~0.86）。

        【调用链】被 _fallback_analysis() 调用

        【加分表】
            基础              0.45
            正文 ≥ 120 字符   +0.12   ← 描述够详细
            证据 ≥ 3 条       +0.14   ← 有交叉印证
            有重复项          +0.08   ← 有历史可参考
            定到了具体人      +0.08   ← 责任明确

        【★ 为什么"有重复项"也加分】
            不是"重复是好事"，而是"找到了重复说明检索有效"。
            这提高的是"结论可信度"，不是"Issue 质量"。

        【为什么上限 0.86】
            纯规则分析无法理解正文的言外之意，不可能 100% 确定。
        """
        score = 0.45
        if len(body.strip()) >= 120:
            score += 0.12
        if len(evidence) >= 3:
            score += 0.14
        if duplicates:
            score += 0.08
        if owner != "待分配":
            score += 0.08
        return round(max(0.35, min(0.86, score)), 2)

    def _clip(self, text: str, limit: int) -> str:
        """按字符截断（证据片段用）。

        【调用链】被 _collect_evidence() / _evidence_from_context() 调用

        limit-1 给省略号留位置。用 "…"（单字符）比 "..." 省 2 个字符。
        """
        text = (text or "").strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
