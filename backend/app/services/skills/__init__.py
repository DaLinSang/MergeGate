"""
Skill 子系统（包入口）
=====================

【业务定位】
    把 skills/registry.py 里的核心类型和单例工厂集中导出，
    让调用方只依赖 `app.services.skills`，不依赖具体文件路径。

【数据流向】
    上游：services/skills/registry.py（真正的实现）
    下游：services/agents/chat_agent.py（加载技能、校验工具白名单）
          api/routes/skills.py（技能列表接口）

【核心规则】
    只做转出，不在这里定义任何逻辑 —— 避免"两处都能改"的混乱。

【本文件函数地图】
    无函数。__all__ 列出 6 个可导出对象：
        SkillManifest      SKILL.md 解析出的清单（name / description / tools 白名单）
        SkillDefinition    运行时技能定义（清单 + 正文 + 资源）
        SkillActivation    一次技能激活的记录（哪个技能、为什么激活）
        SkillRegistry      注册表（加载 / 查找 / 校验）
        SkillRegistryError 注册表异常（清单非法 / 工具越权）
        get_skill_registry 单例工厂（进程内只加载一次）
"""

from app.services.skills.registry import SkillActivation, SkillDefinition, SkillManifest, SkillRegistry, SkillRegistryError, get_skill_registry

__all__ = ["SkillActivation", "SkillDefinition", "SkillManifest", "SkillRegistry", "SkillRegistryError", "get_skill_registry"]
