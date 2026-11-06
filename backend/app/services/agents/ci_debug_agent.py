"""
CI 排障 Agent
============

【业务定位】
    分析一次失败的 CI 运行：找到首个错误、判断失败类型、给出修复步骤、判断是否阻塞合并。

    【核心设计：规则兜底 + LLM 增强的"双轨制"】
        先用纯规则算一份完整结果（_fallback_debug），
        再让 LLM 分析一次；LLM 的结果会和规则结果【逐字段合并】，
        哪个字段 LLM 没给，就用规则的值。
        最后用 CIDebugOutput 校验（不合法就退回纯规则结果）。

        为什么这么做：
            纯 LLM → 会编造日志里没有的错误（幻觉）
            纯规则 → 表达死板，但绝不会编
            合并   → 用规则的"事实"兜住 LLM 的"表达"

【数据流向】
    上游（谁调它）：
        services/agents/workflow_orchestrator.py   工作流里的 ci_debug 任务
        services/evals/runner.py                   评测
    下游（它调谁）：
        llm.client.chat_json()    调模型（用 CI_DEBUG_PROMPT）
        schemas.analysis.CIDebugOutput  输出契约校验

【核心规则】
    1. ★ first_error 必须来自日志原文（不允许模型编）。
    2. failure_type 分七类（permission/environment/dependency/lint/build/test/unknown）。
    3. confidence 由"证据充足度"决定（0.36~0.86），不是拍脑袋。
    4. is_merge_blocking 由 conclusion 决定（失败就阻塞）。

【本文件函数地图】
    run()                      ★ 主流程：规则兜底 → LLM → 合并 → 校验
    _fallback_debug()          ★ 纯规则分析（六步）
        ├ _classify_failure()       ★ 失败分类（有序规则）
        ├ _extract_first_error()    ★ 抽首个错误块（带上下文行）
        ├ _extract_related_files()  抽相关文件
        ├ _infer_root_cause()       推断根因（按类型）
        ├ _possible_causes()        可能原因（按类型）
        ├ _fix_steps()              修复步骤（按类型）
        ├ _is_merge_blocking()      是否阻塞合并
        └ _estimate_confidence()    置信度（按证据加分）
    _merge_with_fallback()     ★ LLM 结果与规则结果的合并策略
"""

import re
from typing import Any

from app.schemas.analysis import CIDebugOutput
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import CI_DEBUG_PROMPT


class CIDebugAgent(BaseAgent):
    """CI 排障 Agent（规则兜底 + LLM 增强）。

    【为什么不用纯 LLM】
        CI 日志动辄几万行，模型读不全。
        而"找第一个 error 行"用正则又快又准 —— 这类确定性工作不该交给模型。
        模型的价值在于"解释为什么"和"给出修复建议"，那才是它擅长的。
    """

    name = "ci_debug_agent"
    description = "分析 workflow 失败日志，并提出排查步骤。"
    available_tools = ["get_workflow_run", "get_workflow_logs", "extract_failed_log_sections"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """分析一次失败的 CI 运行。

        【调用链】追这个函数会进入：
            1. _fallback_debug()            → 纯规则算一份完整结果（永远成功）
            2. llm.chat_json()              → LLM 分析（可能返回空）
            3. _merge_with_fallback()       → 逐字段合并
            4. CIDebugOutput.model_validate → 契约校验
        【被谁调用】workflow_orchestrator（ci_debug 任务）、evals/runner

        【数据流向】
            入：input_data（workflow_run 的字段：name/status/conclusion/jobs/logs_text）、
                context（recent_prs / code_references）
            出：CIDebugOutput 的 dict

        【★ 三层降级，保证永远有结果】
            ① LLM 返回空 → 用 fallback
            ② LLM 返回了但校验失败（字段类型不对/缺必填）→ 用 fallback
            ③ LLM 正常 → 用合并结果
            注意 except Exception: pass —— 校验失败不报错，静默降级。
            因为"排障结果差一点"比"整个任务失败"好。
        """
        # ── 步骤 1：规则先算一份（保证有兜底）──
        fallback = self._fallback_debug(input_data, context)

        # ── 步骤 2：LLM 分析 ──
        llm_output = await self.llm.chat_json(CI_DEBUG_PROMPT, {"workflow_run": input_data, "context": context})

        # ── 步骤 3：有结果就合并 + 校验 ──
        if llm_output:
            try:
                return CIDebugOutput.model_validate(self._merge_with_fallback(llm_output, fallback)).model_dump()
            except Exception:
                # 校验失败 → 静默降级到规则结果
                pass
        return fallback

    def _fallback_debug(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """纯规则分析（不调 LLM，★ 六步）。

        【调用链】追这个函数会进入：
            1. _classify_failure()        → 判断失败类型
            2. _extract_first_error()     → 抽首个错误块
            3. _extract_related_files()   → 抽相关文件
            4. _infer_root_cause()        → 推断根因
            5. _possible_causes() / _fix_steps() → 按类型给建议
            6. _is_merge_blocking() / _estimate_confidence() → 结论与置信度
        【被谁调用】run()

        【数据流向】
            入：input_data、context
            出：CIDebugOutput 的 dict

        【为什么 plan 和 executed_steps 都写死】
            它们描述的是"这个 Agent 的工作流程"，不是分析结果。
            写死有两个好处：
                ① 用户能看到"它检查了哪些维度"（可解释）
                ② 模型知道"应该按这个流程思考"（prompt 里也要求了）
        """
        # logs 用 lower 版本做关键词匹配；raw_logs 保留原文（要展示给用户）
        logs = (input_data.get("logs_text") or "").lower()
        raw_logs = str(input_data.get("logs_text") or "")

        # ── 找出失败的 job 和 step ──
        failed_jobs = [
            job
            for job in input_data.get("jobs", [])
            if isinstance(job, dict) and job.get("conclusion") == "failure"
        ]
        failed_steps = [
            step.get("name")
            for job in failed_jobs
            for step in job.get("steps", [])
            if isinstance(step, dict) and step.get("conclusion") == "failure" and step.get("name")
        ]

        # ── 六步分析 ──
        failure_type = self._classify_failure(logs)
        first_error = self._extract_first_error(raw_logs)
        related_files = self._extract_related_files(raw_logs, context)

        failed_job_names = [str(job.get("name")) for job in failed_jobs if job.get("name")]
        # 提示串（只列前 3 个，避免刷屏）
        job_hint = f"失败 job：{', '.join(failed_job_names[:3])}。" if failed_job_names else ""
        step_hint = f"失败 step：{', '.join(failed_steps[:3])}。" if failed_steps else ""

        root_cause = self._infer_root_cause(failure_type, first_error, failed_job_names, failed_steps)
        fix_steps = self._fix_steps(failure_type, related_files)
        is_merge_blocking = self._is_merge_blocking(input_data, failure_type)

        blocking_reason = (
            "当前 workflow run 结论为 failure，属于合并前需要恢复的质量门禁。"
            if is_merge_blocking
            else "当前 workflow run 没有失败结论，不阻塞合并。"
        )

        # 排障计划（固定六步，对应下面 executed_steps 的六条）
        plan = [
            "找出失败 job 和失败 step。",
            "提取首个关键错误块。",
            "判断失败类型：依赖、测试、构建、权限、环境变量或未知。",
            "搜索相关代码、测试文件和 workflow 配置。",
            "关联最近 PR 或变更模块。",
            "输出根因、修复步骤和是否阻塞合并。",
        ]
        # 已执行的步骤（每一条都带具体数字，让用户知道"真的查了"）
        executed_steps = [
            f"已识别失败 job {len(failed_jobs)} 个：{', '.join(failed_job_names[:3]) or '未提供 job 明细'}。",
            f"已识别失败 step {len(failed_steps)} 个：{', '.join(failed_steps[:3]) or '未提供 step 明细'}。",
            f"已提取首个关键错误块：{first_error or '日志中未找到明确 error 块'}。",
            f"已分类失败类型为 {failure_type}，关联文件 {len(related_files)} 个。",
            f"已结合最近 PR 信号 {len(context.get('recent_prs') or [])} 条和代码/配置线索 {len(context.get('code_references') or [])} 条。",
            "已生成根因、修复步骤和阻塞判断。",
        ]

        return CIDebugOutput(
            failure_summary=input_data.get("name", "CI run") + f" 执行失败。{job_hint}{step_hint}需要结合日志定位首个错误块。",
            failure_type=failure_type,
            plan=plan,
            executed_steps=executed_steps,
            first_error=first_error,
            root_cause=root_cause,
            possible_causes=self._possible_causes(failure_type),
            fix_steps=fix_steps,
            debug_steps=["查看失败 job 的首个错误块", "本地复现对应命令", "检查最近 PR 修改的相关文件", "修复后重新运行失败 job 和完整 workflow"],
            related_files=related_files,
            is_merge_blocking=is_merge_blocking,
            blocking_reason=blocking_reason,
            confidence=self._estimate_confidence(first_error, failed_jobs, related_files, context),
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        """把 LLM 结果合并进规则结果（★ 三层合并策略）。

        【调用链】被 run() 调用

        【数据流向】
            入：output（LLM 返回）、fallback（规则结果）
            出：合并后的 dict

        【三层策略，从"整体"到"局部"】
            ① 基础合并 {**fallback, **output}
               LLM 给了什么就用什么，没给的保留 fallback

            ② 空值兜底（两类字段分别处理）
               列表字段（plan/executed_steps/...）：空列表 → 用 fallback
               文本字段（failure_summary/root_cause/...）：空串 → 用 fallback
               为什么要分两类：列表判 `not merged.get(key)`，文本判 `.strip()`。

            ③ ★ 语义兜底（两处特殊规则）
               failure_type == "unknown" 但 fallback 有具体分类 → 用 fallback
                   （模型说"不知道"，但规则明明识别出了 test/build，以规则为准）
               first_error 为空 → 用 fallback
                   （first_error 必须来自日志原文，绝不允许模型留空或编造）
        """
        # ① 基础合并
        merged = {**fallback, **output}

        # ② 列表字段：空就用 fallback
        for key in ["plan", "executed_steps", "possible_causes", "fix_steps", "debug_steps", "related_files"]:
            if not merged.get(key):
                merged[key] = fallback[key]

        # ② 文本字段：空串/纯空白就用 fallback
        for key in ["failure_summary", "failure_type", "root_cause", "blocking_reason"]:
            if not str(merged.get(key) or "").strip():
                merged[key] = fallback[key]

        # ③ 语义兜底：模型说 unknown，但规则识别出来了 → 信规则
        if merged.get("failure_type") == "unknown" and fallback.get("failure_type") != "unknown":
            merged["failure_type"] = fallback["failure_type"]

        # ③ 语义兜底：first_error 必须来自日志原文
        if not merged.get("first_error"):
            merged["first_error"] = fallback["first_error"]
        return merged

    def _classify_failure(self, logs: str) -> str:
        """按关键词判断失败类型（★ 有序规则，命中即返回）。

        【调用链】被 _fallback_debug() 调用

        【数据流向】
            入：logs（已转小写）
            出：permission / environment / dependency / lint / build / test / unknown

        【★ 顺序为什么不能乱：从"最具体"到"最宽泛"】
            permission   权限（关键词最独特，几乎不会误判）
            environment  环境变量
            dependency   依赖
            lint         代码风格
            build        构建
            test         测试（关键词最泛，"测试"两个字到处都是）

            如果把 test 放前面，一份"测试因权限失败"的日志会被判成 test，
            而真正原因是 permission —— 修复方向完全错。

        【为什么 lint 在 build 之前】
            很多项目的 lint 是 build 的前置步骤，日志里会同时出现两者。
            lint 更具体（eslint/ruff/flake8 这些工具名），优先匹配更准。

        【为什么最后两行都是 return "unknown"】
            原实现的写法（保持不动）：识别不出具体类型就是 unknown。
            第一个 if 只是显式表达"日志里有 error 字样但分不出类"。
        """
        ordered_rules = [
            ("permission", ["permission", "denied", "forbidden", "eacces", "权限"]),
            ("environment", ["missing env", "environment variable", "secret", "not set", "环境变量"]),
            ("dependency", ["npm err", "pip install", "dependency", "module not found", "no matching distribution", "cannot find module", "依赖"]),
            ("lint", ["eslint", "lint", "ruff", "flake8", "biome"]),
            ("build", ["build failed", "compilation", "webpack", "vite", "next build", "tsc", "type error", "构建"]),
            ("test", ["pytest", "assert", "test failed", "tests failed", "failed tests", "断言", "测试"]),
        ]
        for failure_type, tokens in ordered_rules:
            if any(token in logs for token in tokens):
                return failure_type
        # 有错误但分不出类 → 仍然是 unknown（需要人工看日志）
        if "failed" in logs or "error" in logs:
            return "unknown"
        return "unknown"

    def _extract_first_error(self, logs: str) -> str | None:
        """抽"首个错误块"（带前后几行上下文）。

        【调用链】被 _fallback_debug() 调用

        【数据流向】
            入：logs（原始文本，保留大小写）
            出：错误块文本（≤900 字符）；没有日志返回 None

        【★ 为什么取前后几行而不是只取那一行】
            错误信息常常是"一行结果 + 几行上下文"：
                ERROR: ModuleNotFoundError: No module named 'foo'
                  File "app/main.py", line 12, in <module>
                    import foo
            只给一行，看不出是哪个文件哪一行出的问题。

        【窗口：前 2 行、后 5 行】
            前面少取（通常是无关的进度输出），后面多取（通常有调用栈）。

        【为什么截断到 900 字符】
            有些 traceback 极长。900 字符足够看到"错误类型 + 首个位置"。

        【兜底：找不到关键词就返回前 6 行】
            至少让用户看到日志开头（通常是"正在执行 xxx 命令"）。
        """
        if not logs.strip():
            return None

        lines = [line.strip() for line in logs.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            lowered = line.lower()
            if any(token in lowered for token in ["error", "failed", "failure", "exception", "traceback", "assertionerror", "npm err", "fatal"]):
                # 前 2 行、后 5 行的窗口
                start = max(0, index - 2)
                end = min(len(lines), index + 5)
                return "\n".join(lines[start:end])[:900]
        # 没有任何错误关键词 → 返回前 6 行
        return "\n".join(lines[:6])[:900]

    def _extract_related_files(self, logs: str, context: dict[str, Any]) -> list[str]:
        """从日志和上下文里抽出相关文件路径。

        【调用链】被 _fallback_debug() 调用

        【数据来源两处】
            ① 日志里出现的、带已知扩展名的路径（正则匹配）
            ② context.code_references 里的引用标题（含 "." 的当文件）

        【清理步骤】
            strip("`'\"()[]{}:,;")  去掉包裹的标点和引号
            replace("\\\\", "/")      统一成 POSIX 路径
            去重保序，最多 8 个

        【为什么最多 8 个】
            排障建议里列 30 个文件没人看。8 个足够指向"问题集中在哪"。
        """
        # ① 从日志抽（只认代码/配置类扩展名）
        candidates = re.findall(r"[\w./\\-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|toml|ini|cfg|md)", logs)

        # ② 从上下文抽
        for item in context.get("code_references") or []:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("reference") or "").strip()
                # 含 "." 才当文件（否则可能是 Issue 标题）
                if "." in title:
                    candidates.append(title)

        # 清理 + 去重
        normalized = []
        for item in candidates:
            cleaned = item.strip("`'\"()[]{}:,;").replace("\\", "/")
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized[:8]

    def _infer_root_cause(self, failure_type: str, first_error: str | None, failed_job_names: list[str], failed_steps: list[str]) -> str:
        """按失败类型推断根因（★ 每种类型一句专门的话）。

        【调用链】被 _fallback_debug() 调用

        【为什么用模板而不是让模型自由发挥】
            根因是"最容易被幻觉污染"的字段。
            模板保证了"说的方向一定对"（比如 test 类型必然指向断言/用例），
            只是细节不如模型丰富。

        【scope 的构造】
            job 名 + step 名的前 2 个，用逗号连起来。
            用 filter(None, ...) 过滤掉空串，避免出现 "（, ）" 这种怪样子。
        """
        scope = ", ".join(filter(None, [", ".join(failed_job_names[:2]), ", ".join(failed_steps[:2])]))

        if failure_type == "test":
            return f"失败集中在测试执行阶段{f'（{scope}）' if scope else ''}，首个错误指向断言、用例或被测行为不一致。"
        if failure_type == "build":
            return f"失败集中在构建/类型检查阶段{f'（{scope}）' if scope else ''}，首个错误通常来自编译、类型或打包配置问题。"
        if failure_type == "dependency":
            return "日志显示依赖安装或模块解析失败，优先检查 lockfile、包版本和 CI 缓存。"
        if failure_type == "permission":
            return "日志显示权限不足或访问被拒绝，优先检查 token、文件权限和 workflow 权限配置。"
        if failure_type == "environment":
            return "日志显示环境变量或 secret 缺失，优先检查 workflow env 配置和仓库 secrets。"
        if failure_type == "lint":
            return "日志显示 lint/格式检查失败，优先定位对应文件的规则违规并本地运行 lint。"
        # unknown：老实说"信息不够"，并指出最可靠的入口
        return f"日志缺少足够明确的分类信号，首个错误块是当前最可靠的排查入口：{first_error[:160] if first_error else '未提供日志'}"

    def _possible_causes(self, failure_type: str) -> list[str]:
        """按类型列出可能原因（每种 3 条）。

        【调用链】被 _fallback_debug() 调用

        注意 unknown 也有条目 —— "信息不足"本身就是一种原因，
        并给出"下一步该看什么"。
        """
        mapping = {
            "test": ["测试断言不一致", "最近变更改变了接口行为", "测试数据或环境初始化不完整"],
            "build": ["类型错误或编译错误", "构建配置与代码变更不匹配", "生成产物或路径引用缺失"],
            "lint": ["格式或静态规则违规", "新增代码未经过本地 lint", "配置升级导致规则变化"],
            "dependency": ["依赖版本变化", "lockfile 未同步", "CI 缓存或 registry 配置异常"],
            "permission": ["workflow 权限不足", "token scope 不足", "脚本访问受限目录"],
            "environment": ["环境变量缺失", "secret 未配置", "CI 与本地默认配置不一致"],
            "unknown": ["首个错误块信息不足", "上游命令吞掉了真实错误", "需要查看完整 job 日志"],
        }
        return mapping[failure_type]

    def _fix_steps(self, failure_type: str, related_files: list[str]) -> list[str]:
        """按类型给出修复步骤（每种 3 步）。

        【调用链】被 _fallback_debug() 调用

        【focus 的作用】
            如果识别到了相关文件，就把前 3 个拼进第一步（"重点检查 xxx"），
            让建议更具体、可执行。

        【为什么每条都从"本地复现"开始】
            CI 上试错成本高（每次几分钟）。先本地复现再改，效率高得多。
        """
        focus = f"，重点检查 {', '.join(related_files[:3])}" if related_files else ""
        mapping = {
            "test": [f"本地运行失败测试命令并复现{focus}", "对照最近 PR 修改确认预期行为", "修正代码或测试数据后补充回归用例"],
            "build": [f"本地运行构建/类型检查命令{focus}", "修复首个编译或类型错误", "确认生成路径、导入路径和配置文件同步更新"],
            "lint": [f"本地运行 lint/format 命令{focus}", "修复首个规则违规", "必要时调整规则并说明原因"],
            "dependency": ["重新安装依赖并确认 lockfile 更新", "检查 package/requirements 与 CI 版本是否一致", "清理 CI 缓存后重跑"],
            "permission": ["检查 workflow permissions 配置", "确认 token/secrets 的 scope", "避免脚本写入受限目录或使用受限 API"],
            "environment": ["补齐 workflow env 或 repository secrets", "为本地和 CI 提供一致默认配置", "在日志中避免泄露 secret 明文"],
            "unknown": ["展开完整 job 日志", "找到最早失败命令的 stderr", "必要时在 workflow 中增加诊断输出后重跑"],
        }
        return mapping[failure_type]

    def _is_merge_blocking(self, input_data: dict[str, Any], failure_type: str) -> bool:
        """判断这次 CI 失败是否阻塞合并。

        【调用链】被 _fallback_debug() 调用

        【两个条件（或关系）】
            ① conclusion == "failure"  → 明确失败，当然阻塞
            ② status == "completed" 且类型不是 unknown
               → 已完成 + 能分类，说明确实有实质失败，阻塞

        【为什么 ② 要排除 unknown】
            分不出类型的失败可能是环境抖动（runner 被回收、网络超时）。
            这种情况让用户先人工看一眼，不要直接卡住合并。
        """
        conclusion = str(input_data.get("conclusion") or "").lower()
        status = str(input_data.get("status") or "").lower()
        return conclusion == "failure" or (status == "completed" and failure_type != "unknown")

    def _estimate_confidence(self, first_error: str | None, failed_jobs: list[dict[str, Any]], related_files: list[str], context: dict[str, Any]) -> float:
        """按"证据充足度"算置信度（0.36~0.86）。

        【调用链】被 _fallback_debug() 调用

        【加分表】
            基础            0.42
            有首个错误块    +0.16   ← 最重要（没错误块等于没定位）
            有失败 job      +0.12
            有相关文件      +0.08
            有最近 PR 信号  +0.05
            有代码引用      +0.05

        【为什么上限只有 0.86 而不是 1.0】
            纯规则分析不可能 100% 确定根因。
            留出余量是诚实的表现 —— 也提醒用户"还要人工确认"。

        【为什么下限 0.36】
            即使什么证据都没有，也有一点基础判断（日志说失败了）。
            但不能低于 0.36，否则看起来像"完全瞎猜"。
        """
        score = 0.42
        if first_error:
            score += 0.16
        if failed_jobs:
            score += 0.12
        if related_files:
            score += 0.08
        if context.get("recent_prs"):
            score += 0.05
        if context.get("code_references"):
            score += 0.05
        return round(max(0.36, min(0.86, score)), 2)
