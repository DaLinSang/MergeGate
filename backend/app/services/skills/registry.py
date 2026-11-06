"""
技能注册表（SKILL.md 加载 + 激活 + 资源校验）
==========================================

【业务定位】
    "技能"是一份 SKILL.md（带 frontmatter），描述"某类任务该怎么做、能用哪些工具"。
    本文件负责：加载所有技能、校验完整性、按消息激活技能、加载技能引用的资源。

    【技能解决了什么问题】
        Agent 的系统提示词是固定的。但不同任务需要不同的"操作手册"：
            分析 Issue 要"先看复现步骤，再看相似 Issue"
            审查 PR 要"先计划，再执行，再给结论"
        把这些手册做成技能，按需激活 —— 提示词就不会膨胀成几万字。

    【★ 技能的工具白名单是权限边界，不是建议】
        SkillManifest.tools 列出这个技能允许用哪些工具。
        chat_agent 会强校验：激活了技能后，调用白名单外的工具会被拦。
        所以技能不只是"提示"，还是"沙箱"。

【数据流向】
    上游（谁调它）：
        services/agents/chat_agent.py   activate() 激活技能 + 校验工具白名单
        api/routes/skills.py            技能列表
        services/evals/runner.py        检查技能注册完整性
    下游（它读谁）：
        backend/skills/*/SKILL.md       技能定义文件
        backend/skills/*/references/…   技能引用的资源文件

【核心规则】
    1. **name 和 entrypoint 必须全局唯一**（否则映射关系有歧义）。
    2. **资源必须留在技能目录内**（防目录穿越），且单个不超过 64 KiB。
    3. **技能激活有三种方式**：explicit（用户指定）> entrypoint（专用 Agent）> automatic（关键词匹配）。
    4. 加载失败要明确报错（`SkillRegistryError`），不能静默跳过 ——
       少加载一个技能会导致工具白名单缺失，进而"某些工具莫名不能用"。

【本文件函数地图】

    ┌─ 数据结构 ──────────────────────────────────────
    │  SkillManifest      SKILL.md 的 frontmatter（元数据）
    │  SkillDefinition    运行时定义（清单 + 正文）
    │  SkillActivation    ★ 一次激活的快照（含资源内容）
    │
    ├─ 注册表 ────────────────────────────────────────
    │  SkillRegistry      ★ 核心类
    │      ├ activate()              ★ 三种激活方式
    │      ├ activate_entrypoint()   按工具名激活
    │      ├ _activation()           构造激活快照
    │      ├ _explicit_skills()      解析"用户指定"的技能
    │      ├ _activation_score()     ★ 关键词匹配打分
    │      ├ _load_referenced_resources() ★ 加载并校验资源
    │      ├ _load_skills()          扫描 */SKILL.md
    │      └ _validate_unique_keys() 唯一性校验
    │
    ├─ 解析 ──────────────────────────────────────────
    │  _load_skill()          读一个 SKILL.md
    │  _split_frontmatter()   ★ 拆 frontmatter 和正文
    │  _parse_frontmatter()   简易 YAML 解析（不依赖 pyyaml）
    │  _parse_value()         值类型推断（列表/布尔/去引号）
    │
    └─ 单例 ──────────────────────────────────────────
       get_skill_registry()   @lru_cache(maxsize=1)
"""

import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillRegistryError(ValueError):
    """技能注册表异常（清单非法 / 资源越界 / 名称重复）。

    【为什么继承 ValueError】
        这些都是"配置写错了"，属于值错误。
        上层可以用 except ValueError 统一捕获。
    """
    pass


class SkillManifest(BaseModel):
    """SKILL.md 的 frontmatter（技能元数据）。

    【字段说明】
        name             技能 ID（唯一，如 "issue-triage"）
        title            显示名
        version          版本号（改了要升，便于追溯"当时用的是哪版"）
        description      一句话说明
        category         分类（general / engineering / …）
        entrypoint       ★ 对应的工具名（如 "issue_agent.analyze_issue"）
        tools            ★ 允许使用的工具白名单（权限边界）
        input_modes      接受的输入形态（text / diff / logs…）
        triggers         ★ 自动激活的关键词
        workflow_steps   执行步骤（会渲染进 prompt）
        output_contract  输出契约（freeform / JSON schema 名）
        safety_level     安全级别（read_only / write_draft / …）
    """

    name: str
    title: str
    version: str
    description: str
    category: str = "general"
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str = "freeform"
    safety_level: str = "read_only"


class SkillDefinition(BaseModel):
    """运行时技能定义（清单 + 文件路径 + 正文）。"""

    manifest: SkillManifest
    path: str
    instructions: str

    def trace_metadata(self) -> dict[str, Any]:
        """给工具调用轨迹打标记（用于 trace 展示"这次用了哪个技能"）。

        【调用链】被 annotate_tool_call() / trace_metadata_by_tool() 调用

        只带 4 个字段（不带 instructions）—— 轨迹里存全文太占空间。
        """
        return {
            "skill_name": self.manifest.name,
            "skill_title": self.manifest.title,
            "skill_version": self.manifest.version,
            "skill_steps": self.manifest.workflow_steps,
        }

    def prompt_summary(self) -> str:
        """渲染成一行"技能清单"（注入 system prompt 用）。

        【调用链】被 prompt_context() 调用

        【为什么压成一行】
            技能可能有十几个。每个写一段会把 prompt 撑爆。
            一行包含：名字 + 版本 + 入口 + 描述 + 步骤 + 安全级别 —— 够模型判断"要不要用"。
        """
        steps = ", ".join(self.manifest.workflow_steps) if self.manifest.workflow_steps else "在 SKILL.md 中定义"
        return (
            f"- {self.manifest.name} v{self.manifest.version} -> {self.manifest.entrypoint}: "
            f"{self.manifest.description} 步骤：{steps}。安全级别：{self.manifest.safety_level}。"
        )


class SkillActivation(BaseModel):
    """一次技能激活的快照（★ 会真正注入 prompt）。

    【为什么要做成"快照"而不是只存引用】
        激活后要往 prompt 里塞内容（正文 + 资源）。
        如果只存 name，后续每次用都要重新读文件（慢，且文件可能被改）。
        做成快照 = 这一轮对话内技能内容固定，可复现。

    【instruction_digest 的用途】
        技能正文的哈希（前 16 位）。
        改了 SKILL.md 但没升版本号时，digest 会变 —— 能发现"内容漂移"。
    """

    skill_name: str
    skill_title: str
    skill_version: str
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str
    safety_level: str
    activation_mode: str        # explicit / entrypoint / automatic
    activation_reason: str      # 人话说明"为什么激活它"
    instructions: str           # SKILL.md 正文
    instruction_digest: str     # 正文哈希前 16 位
    resources: list[dict[str, Any]] = Field(default_factory=list)  # 引用的资源（含内容）

    def trace_metadata(self) -> dict[str, Any]:
        """给工具轨迹打标记（含激活方式和原因）。

        【调用链】被 SkillRegistry.annotate_tool_call() 调用

        比 SkillDefinition.trace_metadata 多了三个字段：
            activation_mode / activation_reason / instruction_digest
        这样能回答"这次为什么用了这个技能、用的是哪版内容"。
        """
        return {
            "skill_name": self.skill_name,
            "skill_title": self.skill_title,
            "skill_version": self.skill_version,
            "skill_steps": self.workflow_steps,
            "skill_activation_mode": self.activation_mode,
            "skill_activation_reason": self.activation_reason,
            "skill_instruction_digest": self.instruction_digest,
        }

    def prompt_block(self) -> str:
        """渲染成注入 prompt 的文本块（★ 技能的实际生效形式）。

        【调用链】被 chat_agent 调用（拼进 system prompt）

        【结构】
            ## 标题 (name v版本)
            激活方式：xxx；原因：xxx
            允许工具：xxx
            输出契约：xxx；安全级别：xxx

            <SKILL.md 正文>

            按需加载的 Skill 资源：
            ### 资源：references/xxx.md
            <资源内容>

        【为什么"允许工具"要写进 prompt】
            让模型知道"我只能用这几个工具"，减少它调用不存在工具的概率。
            真正的拦截在 chat_agent（白名单校验），这里只是"提前告知"。

        【为什么资源内容要拼进来而不是让模型自己去读】
            模型没有文件读取能力（除非给工具）。
            把资源直接塞进 prompt 是最可靠的方式 ——
            代价是占 token，所以资源限制在 64 KiB 以内、最多 8 个。
        """
        resource_text = ""
        if self.resources:
            sections = []
            for resource in self.resources:
                content = str(resource.get("content") or "").strip()
                # 非文本资源 content 是空串（只记 size/digest），跳过
                if content:
                    sections.append(f"### 资源：{resource['path']}\n{content}")
            if sections:
                resource_text = "\n\n按需加载的 Skill 资源：\n\n" + "\n\n".join(sections)
        return (
            f"## {self.skill_title} ({self.skill_name} v{self.skill_version})\n"
            f"激活方式：{self.activation_mode}；原因：{self.activation_reason}\n"
            f"允许工具：{', '.join(self.tools) or '无'}\n"
            f"输出契约：{self.output_contract}；安全级别：{self.safety_level}\n\n"
            f"{self.instructions.strip()}"
            f"{resource_text}"
        )


class SkillRegistry:
    """技能注册表（★ 加载 / 查询 / 激活）。

    【三个索引，各管一种查询】
        _by_name        按技能名查（用户显式指定时用）
        _by_entrypoint  按工具名查（专用 Agent 激活时用）—— 最常用
        _skills         全量列表（列表页、自动匹配时用）
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._skills = self._load_skills()
        # 先校验唯一性，再建索引 —— 否则重复 key 会静默覆盖
        self._validate_unique_keys()
        self._by_name = {skill.manifest.name: skill for skill in self._skills}
        self._by_entrypoint = {skill.manifest.entrypoint: skill for skill in self._skills}

    def list_skills(self) -> list[SkillDefinition]:
        """列出所有技能（按 name 排序，保证前端展示稳定）。"""
        return sorted(self._skills, key=lambda skill: skill.manifest.name)

    def get_skill(self, name: str) -> SkillDefinition | None:
        """按技能名取。"""
        return self._by_name.get(name)

    def skill_for_entrypoint(self, tool_name: str) -> SkillDefinition | None:
        """按工具名取技能（专用 Agent 执行时用这个）。

        【调用链】被 activate_entrypoint() / annotate_tool_call() / tool_description_suffix() 调用
        """
        return self._by_entrypoint.get(tool_name)

    def trace_metadata_by_tool(self) -> dict[str, dict[str, Any]]:
        """所有"工具名 → 技能元数据"的映射（批量给轨迹用）。"""
        return {tool_name: skill.trace_metadata() for tool_name, skill in self._by_entrypoint.items()}

    def annotate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """给一条工具调用打上"属于哪个技能"的标记。

        【调用链】被 chat_agent 调用（记录工具轨迹时）

        【数据流向】
            入：tool_call（含 tool_name）
            出：新 dict（原字段 + 技能元数据）

        【注意 return dict(tool_call)】
            找不到技能时返回【副本】而不是原对象 ——
            避免调用方修改返回值时意外改了原数据。
        """
        tool_name = str(tool_call.get("tool_name") or "")
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return dict(tool_call)
        # 技能元数据覆盖同名键（工具轨迹里的 skill_* 字段以技能为准）
        return {**tool_call, **skill.trace_metadata()}

    def prompt_context(self) -> str:
        """渲染"技能清单"（注入 system prompt 的开头部分）。

        【调用链】被 chat_agent 调用

        没有技能时返回一句说明（而不是空串）——
        空串会让模型以为"这一段被截断了"，反而困惑。
        """
        skills = self.list_skills()
        if not skills:
            return "当前没有注册 DevFlow 技能。"
        return "\n".join(skill.prompt_summary() for skill in skills)

    def tool_description_suffix(self, tool_name: str) -> str:
        """给工具描述追加"注册技能"信息（让模型知道这个工具属于哪个技能）。

        【调用链】被 tool_description_suffixes() 调用
        """
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return ""
        return (
            f" 注册技能：{skill.manifest.title} v{skill.manifest.version}。"
            f"{skill.manifest.description}"
        )

    def tool_description_suffixes(self) -> dict[str, str]:
        """批量版（一次拿所有工具的追加描述）。"""
        return {tool_name: self.tool_description_suffix(tool_name) for tool_name in self._by_entrypoint}

    def activate(
        self,
        message: str,
        *,
        requested_name: str | None = None,
        max_auto_skills: int = 3,
    ) -> list[SkillActivation]:
        """激活技能（★ 三种方式，优先级从高到低）。

        【调用链】_explicit_skills() / _activation_score() / _activation() → 本函数
        【被谁调用】services/agents/chat_agent.py（每轮对话开始）

        【数据流向】
            入：message（用户消息）、requested_name（用户显式指定的技能名）、max_auto_skills
            出：SkillActivation 列表（可能为空）

        【★ 三种方式】
            ① explicit    用户明确说了"用 xxx 技能" → 只激活它（不自动补）
            ② automatic   关键词匹配 → 最多 max_auto_skills 个（默认 3）
            （③ entrypoint 是另一个方法 activate_entrypoint，专用 Agent 走那条）

        【★ 为什么 explicit 命中就不再做自动匹配】
            用户明确指定了 A，系统又自动加上 B、C —— 会稀释 A 的指令。
            尊重用户意图：说了一个就只用一个。

        【★ 自动匹配的排序】
            分数降序 → 名字升序（同分时按名字，保证确定性）。
            只取 score > 0 的（0 分说明一个关键词都没命中）。

        【★ max_auto_skills=3 的意义】
            激活太多技能会让 prompt 里塞进几份操作手册，互相干扰。
            3 个是"覆盖足够 + 不互相污染"的经验值。
        """
        # ── ① 显式指定 ──
        explicit = self._explicit_skills(message, requested_name)
        if explicit:
            return [self._activation(skill, "explicit", "用户明确指定 Skill") for skill in explicit]

        # ── ② 自动匹配 ──
        scored = [
            (self._activation_score(message, skill), skill)
            for skill in self._skills
        ]
        # 只留命中过的
        selected = [
            (score, skill)
            for score, skill in scored
            if score > 0
        ]
        # 分数降序 → 名字升序
        selected.sort(key=lambda item: (-item[0], item[1].manifest.name))
        return [
            self._activation(skill, "automatic", f"消息命中 Skill 触发条件，匹配分 {score}")
            for score, skill in selected[:max_auto_skills]
        ]

    def activate_entrypoint(self, tool_name: str, *, reason: str = "专用 Agent 执行入口") -> SkillActivation | None:
        """按工具名激活（专用 Agent 执行时用）。

        【调用链】skill_for_entrypoint() / _activation() → 本函数
        【被谁调用】chat_agent（决定走专用 Agent 时）

        【和 activate() 的区别】
            这个不需要"匹配"，因为工具名是确定的。
            走这条路时，技能是"必然激活"的（不是靠关键词猜）。
        """
        skill = self.skill_for_entrypoint(tool_name)
        if skill is None:
            return None
        return self._activation(skill, "entrypoint", reason)

    def _activation(self, skill: SkillDefinition, mode: str, reason: str) -> SkillActivation:
        """构造激活快照（★ 会加载资源）。

        【调用链】_load_referenced_resources() → 本函数
        【被谁调用】activate() / activate_entrypoint()

        【instruction_digest 为什么取前 16 位】
            完整 sha256 是 64 个十六进制字符，太长。
            16 位（64 bit）碰撞概率极低，够用于"发现内容变了"。
        """
        instructions = skill.instructions.strip()
        return SkillActivation(
            skill_name=skill.manifest.name,
            skill_title=skill.manifest.title,
            skill_version=skill.manifest.version,
            entrypoint=skill.manifest.entrypoint,
            tools=skill.manifest.tools,
            workflow_steps=skill.manifest.workflow_steps,
            output_contract=skill.manifest.output_contract,
            safety_level=skill.manifest.safety_level,
            activation_mode=mode,
            activation_reason=reason,
            instructions=instructions,
            instruction_digest=hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16],
            resources=self._load_referenced_resources(skill),
        )

    def _explicit_skills(self, message: str, requested_name: str | None) -> list[SkillDefinition]:
        """解析"用户明确指定的技能"（★ 三种识别路径）。

        【调用链】_skill_by_name_or_title() → 本函数
        【被谁调用】activate()

        【数据流向】
            入：message、requested_name
            出：技能列表（空表示"用户没指定"）

        【★ 三条识别路径，顺序不能乱】
            ① requested_name 有值（API 参数直接传了技能名）→ 最可靠，优先
               找不到就抛异常（用户明确要的东西不能静默忽略）
            ② 消息里出现 "skill"/"技能" + 技能名/标题 → 精确匹配
            ③ 正则 "用/使用/调用 xxx 技能" → 提取名字
               找不到也抛异常（用户说要用了，不能说"没听懂"就跳过）

        【★ 为什么要先判 "skill"/"技能" 才做 ②】
            否则每轮对话都要遍历所有技能做子串匹配（性能浪费），
            而且容易误判（用户说"这个 issue 的技能栈"可能被当成指定技能）。

        【★ 为什么 ② 和 ③ 一个返回一个抛异常】
            ② 是"技能名完整出现在消息里" —— 匹配到了就返回，没匹配到说明只是闲聊；
            ③ 是"用户用了'用 xxx 技能'的句式" —— 句式明确表示意图，
               这时找不到就必须报错（否则用户以为生效了，实际没有）。
        """
        # ── 路径 ①：API 参数显式指定 ──
        if requested_name:
            skill = self._skill_by_name_or_title(requested_name)
            if skill is None:
                raise SkillRegistryError(f"未找到用户指定的 Skill：{requested_name}")
            return [skill]

        lowered = message.lower()
        # 没提到 skill / 技能 → 不是显式指定
        if "skill" not in lowered and "技能" not in message:
            return []

        # ── 路径 ②：技能名/标题完整出现在消息里 ──
        selected = [
            skill
            for skill in self._skills
            if skill.manifest.name.lower() in lowered or skill.manifest.title.lower() in lowered
        ]
        if selected:
            return selected

        # ── 路径 ③：句式匹配"用 xxx 技能" ──
        match = re.search(
            r"(?:use|using|用|使用|调用|启用)\s*([A-Za-z][A-Za-z0-9_-]+)\s*(?:skill|技能)",
            message,
            flags=re.IGNORECASE,
        )
        if match:
            # 句式明确但找不到 → 必须报错
            raise SkillRegistryError(f"未找到用户指定的 Skill：{match.group(1)}")
        return []

    def _skill_by_name_or_title(self, value: str) -> SkillDefinition | None:
        """按名字或标题查技能（大小写不敏感）。

        【调用链】被 _explicit_skills() 调用

        用 next(..., None) 而不是下标查找 —— 找不到返回 None，不抛 StopIteration。
        """
        normalized = value.strip().lower()
        return next(
            (
                skill
                for skill in self._skills
                if skill.manifest.name.lower() == normalized or skill.manifest.title.lower() == normalized
            ),
            None,
        )

    def _activation_score(self, message: str, skill: SkillDefinition) -> int:
        """按 triggers 关键词给技能打分（★ 两种匹配方式）。

        【调用链】被 activate() 调用

        【数据流向】
            入：message、skill
            出：分数（0 表示没命中）

        【★ 两种匹配方式，按 trigger 的形态自动选】
            ① trigger 是"纯 ASCII 标识符"（如 pytest、ci、pr-42）：
               用词边界正则 `(?<![a-z0-9_])xxx(?![a-z0-9_])`
               为什么：避免 "ci" 匹配到 "specific"、"social" 里的子串。
            ② trigger 含中文或其他字符（如 "报错"、"合并"）：
               直接用子串包含（中文没有词边界概念）。

        【★ 分数 = min(trigger 长度, 8)】
            长 trigger 更具体（"pull request" 比 "pr" 更明确），给更高分。
            但封顶 8，防止超长 trigger 一把拉满。
            至少 1 分（命中就至少 1）。

        【为什么不按命中次数累加】
            同一个 trigger 在消息里出现 3 次不代表"更相关"。
            按"命中了几个不同的 trigger"累加才有意义（这里就是循环累加各 trigger 的分）。
        """
        lowered = message.lower()
        score = 0
        for trigger in skill.manifest.triggers:
            normalized = trigger.strip().lower()
            if not normalized:
                continue
            if re.fullmatch(r"[a-z0-9_+#.-]+", normalized):
                # 纯标识符 → 词边界匹配（防子串误命中）
                matched = re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", lowered)
            else:
                # 含中文 → 子串匹配
                matched = normalized in lowered
            if matched:
                score += max(1, min(len(normalized), 8))
        return score

    def _load_referenced_resources(self, skill: SkillDefinition) -> list[dict[str, Any]]:
        """加载 SKILL.md 里用反引号引用的资源文件（★ 五道校验）。

        【调用链】被 _activation() 调用

        【数据流向】
            入：skill
            出：[{path, content, size, digest}] 列表

        【怎么找资源引用】
            正则 `` `((?:references|templates|scripts|assets)/[^`]+)` ``
            只认这四个目录下的反引号路径 ——
            这样 SKILL.md 里写 `references/checklist.md` 就会被自动加载。

        【★ 五道校验，任何一道不过就抛异常】
            ① 数量 ≤ 8（防止一个技能塞几十个资源把 prompt 撑爆）
            ② 路径必须留在技能目录内（`relative_to` 检查，防 ../../ 穿越）
            ③ 文件必须存在
            ④ 单个 ≤ 64 KiB（防止大文件占满上下文）
            ⑤ 只读文本类后缀的内容（二进制只记 size/digest）

        【★ 为什么校验失败要抛异常而不是跳过】
            资源缺失意味着技能"操作手册不完整"。
            静默跳过会让模型按残缺的手册执行 —— 比报错更危险。

        【★ 为什么二进制资源 content 是空串】
            prompt 里塞不了二进制。但还是要记录 size 和 digest ——
            这样至少能知道"这个技能依赖了哪些二进制资源、有没有变"。
        """
        # 用 set 去重（同一个资源可能被引用多次），sorted 保证顺序稳定
        references = sorted(
            {
                match.group(1).strip()
                for match in re.finditer(
                    r"`((?:references|templates|scripts|assets)/[^`]+)`",
                    skill.instructions,
                )
            }
        )
        # 校验 ①
        if len(references) > 8:
            raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 8 个")

        skill_root = Path(skill.path).resolve().parent
        resources: list[dict[str, Any]] = []
        # 只有这些后缀才读内容（其余当二进制，只记元数据）
        text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py", ".sh", ".ps1"}

        for relative in references:
            resource_path = (skill_root / relative).resolve()

            # 校验 ②：必须留在技能目录内
            try:
                resource_path.relative_to(skill_root)
            except ValueError as exc:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用了目录外资源：{relative}") from exc

            # 校验 ③：必须存在
            if not resource_path.is_file():
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源不存在：{relative}")

            # 校验 ④：大小上限
            if resource_path.stat().st_size > 64 * 1024:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 64 KiB：{relative}")

            # 校验 ⑤：只读文本内容
            content = resource_path.read_text(encoding="utf-8") if resource_path.suffix.lower() in text_suffixes else ""

            resources.append(
                {
                    # 统一成 POSIX 路径（Windows 的反斜杠会破坏展示）
                    "path": relative.replace("\\", "/"),
                    "content": content,
                    "size": resource_path.stat().st_size,
                    "digest": hashlib.sha256(resource_path.read_bytes()).hexdigest()[:16],
                }
            )
        return resources

    def _load_skills(self) -> list[SkillDefinition]:
        """扫描 root 下所有 `*/SKILL.md` 并加载。

        【调用链】_load_skill() → 本函数
        【被谁调用】__init__()

        【为什么 glob("*/SKILL.md") 而不是 rglob】
            只支持一层目录（skills/<name>/SKILL.md）。
            rglob 会把 references/ 里的 .md 也扫进来 —— 那些不是技能定义。

        【root 不存在 → 返回空列表】
            不是错误（可能还没装任何技能）。
        """
        if not self.root.exists():
            return []
        skills: list[SkillDefinition] = []
        # sorted 保证加载顺序稳定（报错时"第一个出错的是哪个"可复现）
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(_load_skill(skill_file))
        return skills

    def _validate_unique_keys(self) -> None:
        """校验 name 和 entrypoint 全局唯一（★ 必须在建索引之前）。

        【调用链】被 __init__() 调用

        【★ 为什么这一步不能省】
            重复的 key 在建 dict 时会【静默覆盖】——
            `_by_entrypoint["x"]` 只剩最后一个技能，前面的永久丢失。
            而且不会报错，只会表现为"某些技能莫名不生效"。
            所以必须提前显式校验。

        【两个维度都要查】
            name 重复 → 用户按名字激活时不知道该用哪个
            entrypoint 重复 → 按工具名激活时同上
        """
        seen_names: set[str] = set()
        seen_entrypoints: set[str] = set()
        for skill in self._skills:
            if skill.manifest.name in seen_names:
                raise SkillRegistryError(f"Skill 名称重复：{skill.manifest.name}")
            if skill.manifest.entrypoint in seen_entrypoints:
                raise SkillRegistryError(f"Skill entrypoint 重复：{skill.manifest.entrypoint}")
            seen_names.add(skill.manifest.name)
            seen_entrypoints.add(skill.manifest.entrypoint)


# ════════════════════════════════════════════════════════════════
# SKILL.md 解析（不依赖 pyyaml，手写简易 frontmatter 解析）
# ════════════════════════════════════════════════════════════════

def _load_skill(path: Path) -> SkillDefinition:
    """读一个 SKILL.md，拆成 manifest + instructions。

    【调用链】_split_frontmatter() → SkillManifest.model_validate() → 本函数
    【被谁调用】SkillRegistry._load_skills()

    【为什么用 Pydantic 校验 manifest】
        frontmatter 是手写的，容易缺字段/写错类型。
        model_validate 会给出精确的错误（哪个字段、什么类型不对），
        比后面用到时才 AttributeError 好得多。
    """
    text = path.read_text(encoding="utf-8")
    manifest_data, instructions = _split_frontmatter(text, path)
    manifest = SkillManifest.model_validate(manifest_data)
    return SkillDefinition(manifest=manifest, path=str(path), instructions=instructions)


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    """把 SKILL.md 拆成 (frontmatter dict, 正文)。

    【调用链】_parse_frontmatter() → 本函数
    【被谁调用】_load_skill()

    【数据流向】
        入：text（文件内容）、path（仅用于报错）
        出：(frontmatter 解析结果, 正文)

    【格式要求】
        第一行必须是 `---`
        找到下一个 `---` 作为结束
        中间是 frontmatter，之后是正文

    【为什么要先统一换行符】
        Windows 的 \r\n 会让 `line.strip() == "---"` 判断失效（行尾有 \r）。
        统一成 \n 后才可靠。

    【两种错误】
        第一行不是 ---        → "必须以 frontmatter 开头"
        找不到结束的 ---      → "缺少 frontmatter 结束标记"

    【body_start = index + 1 的原因】
        index 是结束标记 `---` 的行号（0-based，从 lines[1:] 枚举时已 +1 对齐）。
        +1 跳过结束标记本身，正文从下一行开始。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillRegistryError(f"{path} 必须以 frontmatter 开头")

    frontmatter_lines: list[str] = []
    body_start = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = index + 1
            break
        frontmatter_lines.append(line)
    if body_start is None:
        raise SkillRegistryError(f"{path} 缺少 frontmatter 结束标记")

    return _parse_frontmatter(frontmatter_lines), "\n".join(lines[body_start:]).strip()


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    """解析简易 YAML（只支持 key: value 和 key:\\n- item 两种形态）。

    【调用链】_parse_value() → 本函数
    【被谁调用】_split_frontmatter()

    【为什么不直接用 pyyaml】
        技能文件的 frontmatter 结构极简（没有嵌套、没有锚点、没有多行字符串）。
        手写 30 行就能覆盖，还少一个依赖。
        （注意：project_indexing.py 的配置用了 pyyaml —— 那个配置结构更复杂）

    【★ 状态机逻辑（current_list_key）】
        遇到 `key:`（值空）   → 进入"列表模式"，current_list_key = key
        遇到 `- item`         → 追加到 current_list_key 的列表
        遇到 `key: value`     → 退出列表模式（current_list_key = None）

        这个状态机是必要的：YAML 用缩进表达层级，而这里只看行首字符，
        必须靠"记住上一个空值 key"来判断 `- item` 属于谁。

    【三处错误检查】
        ① `- item` 前面没有空值 key → "列表项缺少父级 key"
        ② key 已存在但不是列表 → 类型冲突
        ③ 行里没有冒号 → 不是合法的 key: value

    【注释行和空行直接跳过】
        支持 `#` 注释，方便在技能文件里写说明。
    """
    data: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        # 跳过空行和注释
        if not stripped or stripped.startswith("#"):
            continue

        # 列表项
        if stripped.startswith("- "):
            if current_list_key is None:
                raise SkillRegistryError("列表项缺少父级 key")
            data.setdefault(current_list_key, [])
            if not isinstance(data[current_list_key], list):
                raise SkillRegistryError(f"{current_list_key} 不是列表")
            data[current_list_key].append(_parse_value(stripped[2:]))
            continue

        # 非 key: value 行
        if ":" not in raw_line:
            raise SkillRegistryError(f"frontmatter 行无效：{raw_line}")

        # 只 split 一次：value 里可能还有冒号（如 URL）
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillRegistryError("frontmatter key 不能为空")

        if value == "":
            # 空值 → 后面可能跟列表项
            data[key] = []
            current_list_key = key
        else:
            data[key] = _parse_value(value)
            current_list_key = None
    return data


def _parse_value(value: str) -> Any:
    """把字符串值推断成合适的 Python 类型（★ 四步）。

    【调用链】被 _parse_frontmatter() 调用

    【数据流向】
        入：value（原始字符串）
        出：list / bool / str

    【四步推断】
        ① `[a, b, c]` 形态 → 尝试 ast.literal_eval
           失败就手动按逗号切（容错：有些写法不是合法 Python 字面量）
        ② "true"/"false" → 布尔
        ③ 被引号包裹 → 去掉引号（保留原样，不再推断）
        ④ 其他 → 原字符串

    【为什么用 ast.literal_eval 而不是 eval】
        literal_eval 只解析字面量（字符串/数字/列表/字典），不执行代码。
        eval 会执行任意代码 —— 用在这里就是 RCE 漏洞。

    【为什么 ① 还要手动兜底切分】
        `[a, b]` 在 Python 里 a、b 是未定义变量，literal_eval 会抛 NameError。
        但 YAML 里这种写法很常见（不加引号的字符串列表）。
        所以失败时手动切分。
    """
    value = value.strip()
    if not value:
        return ""

    # ① 列表形态
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # 兜底：手动按逗号切，去掉每项的引号
            parsed = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else value

    # ② 布尔
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # ③ 去引号
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # ④ 原样
    return value


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    """获取技能注册表单例（★ 进程内只加载一次）。

    【调用链】SkillRegistry() → 本函数
    【被谁调用】services/agents/chat_agent.py、api/routes/skills.py、evals/runner.py

    【为什么用 lru_cache】
        加载技能要读文件 + 解析 + 校验资源（可能几十个文件）。
        每轮对话都做一遍太浪费。缓存后只做一次。

    【★ 代价：改了 SKILL.md 必须重启进程】
        maxsize=1 意味着永远不会重新加载。
        开发时改了技能文件看不到效果 —— 这是刻意的取舍（性能优先）。

    【root 路径怎么算】
        parents[2]：skills/registry.py → skills → services → app
        再拼 "skills" → app/skills/
        （技能文件在 backend/app/skills/<name>/SKILL.md）
    """
    root = Path(__file__).resolve().parents[2] / "skills"
    return SkillRegistry(root)
