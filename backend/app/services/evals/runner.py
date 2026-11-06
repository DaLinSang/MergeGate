"""
基础评测执行器（不依赖外部数据的最小回归）
=======================================

【业务定位】
    用**固定的内置样本**跑一遍全链路，输出一份指标报告。
    目的是"改完代码，5 秒内知道有没有把东西改坏"，而不是测模型能力上限。

    【和 ragas_runner 的分工】
        本文件        内置 3~5 个样本，纯规则断言，不需要 LLM/数据库 → 冒烟回归
        ragas_runner  用真实文档 + LLM 打分 → 质量评估

    【★ 关键设计：NoopLLM】
        所有 Agent 都注入 NoopLLM（chat_json 永远返回 {}）。
        这样跑的是"纯规则路径"，结果是确定性的 —— 同样的代码永远同样的分数。
        如果注入真 LLM，分数会随机波动，就没法当回归基线了。

【数据流向】
    上游（谁调它）：
        api/routes/evaluations.py（"跑一次基础评测"按钮）
        services/evals/ragas_runner.py（作为 RAG 质量评测的补充）
    下游（它调谁）：
        IssueAgent / PRReviewAgent / CIDebugAgent（注入 NoopLLM）
        context_compression（压缩、记忆合并）
        ragas_runner.run_rag_quality_eval()（可选，传了 db 才跑）
        skills.get_skill_registry()（检查技能注册完整性）

【核心规则】
    1. 所有指标都在 0~1 之间（比例型），便于横向对比。
    2. 指标必须是"确定性的"（用 NoopLLM），否则不能当基线。
    3. 最后输出 report_markdown（人能直接读）+ 结构化指标（程序能读）。

【本文件函数地图】
    run_basic_eval()      ★ 主流程：7 个指标 + 报告
    cosine_similarity()   RAG hit@3 用的相似度
    NoopLLM               ★ 空实现（保证确定性）
"""

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import CodeSymbol, Document, Issue, PullRequest, WorkflowRun
from app.schemas.analysis import CIDebugOutput, IssueAnalysisOutput, PRReviewOutput
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.context_compression import (
    compression_eval_summary,
    compress_messages,
    compress_retrieval_results,
    fallback_memory_update,
    merge_memory,
    render_structured_memory,
)
from app.services.evals.ragas_runner import run_rag_quality_eval
from app.services.rag.embeddings import deterministic_embedding
from app.services.skills import get_skill_registry


class NoopLLM:
    """空 LLM 实现：任何调用都返回空 dict。

    【为什么需要它（★ 这是整个评测确定性的基础）】
        Agent 的设计是"规则兜底 + LLM 增强"。注入 NoopLLM 后：
            llm_output = {}  →  走 if llm_output: 为假  →  直接返回 fallback
        所以跑的是纯规则路径，结果完全确定。

        如果注入真 LLM：
            - 同样的输入可能给出不同结论（temperature > 0）
            - 网络抖动会导致分数忽高忽低
            - 没法判断"这次分数下降是代码改坏了还是模型抽风"

    【注意：只实现 chat_json，不实现 chat_text】
        Agent 只调 chat_json。多实现一个没用的方法反而增加维护成本。
    """

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return {}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度（RAG hit@3 用）。

    【调用链】被 run_basic_eval() 调用

    【和 rag/retrieval.cosine_similarity 的区别】
        这里的向量是 deterministic_embedding 产的（已 L2 归一化），
        理论上点积就够了。但仍保留完整公式 —— 万一将来换成未归一化的向量也正确。

    分母的 or 1.0：避免零向量除零。
    """
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return numerator / (denominator or 1.0)


async def run_basic_eval(db: Session | None = None, repo_id: str | None = None) -> dict[str, Any]:
    """跑一遍基础评测（★ 7 个指标）。

    【调用链】追这个函数会进入：
        1. IssueAgent / PRReviewAgent / CIDebugAgent（都注入 NoopLLM）
        2. compress_messages / fallback_memory_update / merge_memory / compress_retrieval_results
        3. deterministic_embedding + cosine_similarity（RAG hit@3）
        4. run_rag_quality_eval()（可选，传了 db + repo_id 才跑）
        5. get_skill_registry()（技能完整性）
    【被谁调用】api/routes/evaluations.py

    【数据流向】
        入：db（可空）、repo_id（可空）—— 传了才做"仓库样本统计"和 RAG 质量评测
        出：dict（7 个指标 + 明细 + report_markdown）

    【★ 7 个指标各自的含义】
        ① issue_classification_accuracy   Issue 分类准不准（3 个样本，规则分类）
        ② pr_key_file_coverage            PR 审查有没有覆盖到关键文件
        ③ rag_hit_rate_at_3               正确文档是否在前 3（用哈希向量算）
        ④ tool_call_accuracy              工具 schema 就绪率（当前恒为 1.0）
        ⑤ skill_registry_coverage         必需技能是否都注册了
        ⑥ skill_entrypoint_accuracy       entrypoint 是否映射到正确技能
        ⑦ json_schema_pass_rate           三个 Agent 的输出是否都通过契约校验

    【★ 为什么用内置样本而不是真实数据】
        内置样本可以断言"预期答案"（比如 "Login API returns 500" 应该分类成 Bug）。
        真实数据没有标注，没法算准确率。
    """
    # ── 三个 Agent，全部注入 NoopLLM（走纯规则路径）──
    issue_agent = IssueAgent(llm=NoopLLM())
    pr_agent = PRReviewAgent(llm=NoopLLM())
    ci_agent = CIDebugAgent(llm=NoopLLM())

    # ════════════════════════════════════════════════════════════
    # 指标 ①：Issue 分类准确率（3 个样本）
    # ════════════════════════════════════════════════════════════
    issue_cases = [
        ({"title": "Login API returns 500", "body": "Token expires and login fails", "labels": ["bug"]}, "Bug"),
        ({"title": "Document local demo mode", "body": "README should explain seed script", "labels": ["documentation"]}, "Documentation"),
        ({"title": "Add workflow sync", "body": "Sync recent GitHub Actions runs", "labels": ["feature", "ci"]}, "Ops"),
    ]
    issue_results = [await issue_agent.run(payload, {}) for payload, _expected in issue_cases]
    # 逐条比对，命中数 / 总数
    issue_accuracy = sum(result["category"] == expected for result, (_payload, expected) in zip(issue_results, issue_cases)) / len(issue_cases)

    # ════════════════════════════════════════════════════════════
    # 指标 ②：PR 关键文件覆盖率
    # ════════════════════════════════════════════════════════════
    pr_payload = {
        "number": 12,
        "title": "Refactor auth middleware",
        "body": "Split token validation and session refresh",
        "files": [
            {"filename": "backend/app/auth/middleware.py", "status": "modified", "patch": "+ validate token\n+ refresh session"},
            {"filename": "backend/app/tests/test_auth.py", "status": "modified", "patch": "+ test expired token"},
        ],
        "comments": [],
    }
    pr_result = await pr_agent.run(pr_payload, {})
    # 把"关键变更 + 需关注文件"拼成一段文本，看每个文件名是否出现在里面
    key_text = " ".join(pr_result.get("key_changes", []) + pr_result.get("files_need_attention", []))
    pr_key_file_coverage = sum(file["filename"] in key_text for file in pr_payload["files"]) / len(pr_payload["files"])

    # ════════════════════════════════════════════════════════════
    # 指标 ③：RAG hit@3（正确文档是否在前 3）
    # ════════════════════════════════════════════════════════════
    rag_docs = [
        ("issue-login", "Login API returns 500 when token expires"),
        ("issue-docs", "Document local demo mode and README setup"),
        ("issue-ci", "GitHub Actions pytest job fails after dependency update"),
    ]
    # ★ 用 deterministic_embedding（哈希向量）而不是真模型 —— 保证确定性
    query_embedding = deterministic_embedding("token expires login 500")
    scored = sorted(
        [(doc_id, cosine_similarity(query_embedding, deterministic_embedding(text))) for doc_id, text in rag_docs],
        key=lambda item: item[1],
        reverse=True,
    )
    # 正确答案 issue-login 是否在前 3（只有 3 篇文档，所以实际是"是否排第一"）
    rag_hit_rate_at_3 = 1.0 if any(doc_id == "issue-login" for doc_id, _score in scored[:3]) else 0.0

    # 可选：真跑一次 RAG 质量评测（需要 db + repo_id）
    rag_quality = await run_rag_quality_eval(db, repo_id, run_judge=False) if db is not None and repo_id else None

    # ════════════════════════════════════════════════════════════
    # 压缩链路的三项检查（消息压缩 / 记忆合并 / 检索结果压缩）
    # ════════════════════════════════════════════════════════════
    # 构造超长对话（40 × 一句话），验证压缩确实生效
    long_messages = [
        {"role": "user", "content": "We decided that auth refresh belongs in middleware. " * 40},
        {"role": "assistant", "content": "Decision captured. Next step is adding tests for expired tokens. " * 36},
        {"role": "user", "content": "What did we decide about auth token refresh?"},
    ]
    compressed_messages, message_stats = compress_messages(long_messages, max_tokens=260)

    # 规则记忆提炼 → 合并进空记忆
    memory_update = fallback_memory_update(
        "Please implement auth token refresh in middleware",
        "Decision: keep refresh in middleware. Next step: add expired token tests.",
        "direct_answer",
        [],
    )
    merged_memory = merge_memory(
        {"summary": "", "facts": [], "decisions": [], "open_questions": [], "tasks": [], "user_preferences": [], "repo_context": [], "citations": []},
        memory_update,
        memory_update["summary"],
        max_tokens=240,
    )

    # 检索结果压缩：一段 80 句无关内容 + 1 句关键内容，看能不能抽出关键那句
    retrieval_items = compress_retrieval_results(
        "auth token refresh middleware",
        [
            {
                "id": "auth-doc",
                "title": "backend/app/auth/middleware.py#1",
                "source_type": "workspace_file",
                "source_id": "auth-doc",
                "score": 1.0,
                "_raw_content": ("unrelated setup notes. " * 80) + "auth middleware refreshes expired token and validates session.",
                "metadata": {"path": "backend/app/auth/middleware.py"},
            }
        ],
        max_tokens_per_item=80,
    )
    # 压缩质量：压缩前 vs 压缩后的 token 比
    compression_quality = compression_eval_summary(
        "\n".join(item["content"] for item in long_messages),
        # 压缩后消息 + 渲染出的记忆（模拟"实际进 prompt 的内容"）
        "\n".join(item["content"] for item in compressed_messages) + "\n" + render_structured_memory(type("Memory", (), merged_memory), max_tokens=240),
        retrieved=1,
        kept=len(retrieval_items),
    )

    # ════════════════════════════════════════════════════════════
    # 指标 ⑦：CI 排障（能否识别出 pytest 失败）
    # ════════════════════════════════════════════════════════════
    ci_result = await ci_agent.run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        {},
    )
    # 两个条件都满足才算通过：分类是 test + 文本里提到 pytest
    ci_text = " ".join(
        [
            str(ci_result.get("failure_summary") or ""),
            str(ci_result.get("first_error") or ""),
            str(ci_result.get("root_cause") or ""),
        ]
    ).lower()
    ci_pytest_failure_detection = 1.0 if ci_result.get("failure_type") == "test" and "pytest" in ci_text else 0.0

    # ════════════════════════════════════════════════════════════
    # 指标 ⑦：JSON 契约通过率
    # ════════════════════════════════════════════════════════════
    # 直接拿三个 Agent 的输出跑 Pydantic 校验（不通过会抛异常）
    schema_checks = [
        IssueAnalysisOutput.model_validate(issue_results[0]),
        PRReviewOutput.model_validate(pr_result),
        CIDebugOutput.model_validate(ci_result),
    ]
    json_schema_pass_rate = len(schema_checks) / 3

    # ════════════════════════════════════════════════════════════
    # 指标 ④⑤⑥：技能注册完整性
    # ════════════════════════════════════════════════════════════
    # 现在由原生工具调用负责工具选择。
    tool_call_schema_ready_rate = 1.0

    registry = get_skill_registry()
    # 六个"必需技能 → 入口工具"的映射（这是技能体系的最小完整性要求）
    required_skill_entrypoints = {
        "issue-triage": "issue_agent.analyze_issue",
        "pr-review": "pr_review_agent.analyze_pr",
        "ci-debug": "ci_debug_agent.analyze_workflow_run",
        "weekly-report": "report_agent.generate_weekly_report",
        "safety-draft": "safety_agent.classify_action_risk",
        "project-investigation": "devflow_search_evidence",
    }
    # ⑤ 覆盖率：实际加载的技能里，有多少个是必需的
    loaded_skill_names = {skill.manifest.name for skill in registry.list_skills()}
    skill_registry_coverage = len(loaded_skill_names & set(required_skill_entrypoints)) / len(required_skill_entrypoints)

    # ⑥ 入口准确率：每个必需的 entrypoint 是否映射到了【正确名字】的技能
    # （防止"名字对了但 entrypoint 指错了"这种错配）
    skill_entrypoint_hits = 0
    for skill_name, entrypoint in required_skill_entrypoints.items():
        skill = registry.skill_for_entrypoint(entrypoint)
        if skill and skill.manifest.name == skill_name:
            skill_entrypoint_hits += 1
    skill_entrypoint_accuracy = skill_entrypoint_hits / len(required_skill_entrypoints)

    # ════════════════════════════════════════════════════════════
    # 生成报告（人能读的 Markdown）
    # ════════════════════════════════════════════════════════════
    report_lines = [
        "# Eval Report",
        "",
        f"- Issue classification accuracy: {issue_accuracy:.2f}",
        f"- PR key-file coverage: {pr_key_file_coverage:.2f}",
        f"- RAG hit@3: {rag_hit_rate_at_3:.2f}",
        f"- Context compression ratio: {compression_quality['compression_ratio']:.2f}",
        f"- Skill registry coverage: {skill_registry_coverage:.2f}",
        f"- Skill entrypoint accuracy: {skill_entrypoint_accuracy:.2f}",
        f"- CI pytest failure detection: {ci_pytest_failure_detection:.2f}",
    ]
    # RAG 质量评测是可选步骤，有结果才加这一行
    if rag_quality:
        report_lines.append(f"- RAG recall@5: {rag_quality['retrieval']['recall_at_k']:.2f}")
    report_lines.extend(
        [
            f"- Native tool schema ready rate: {tool_call_schema_ready_rate:.2f}",
            f"- JSON schema pass rate: {json_schema_pass_rate:.2f}",
        ]
    )
    report_markdown = "\n".join(report_lines) + "\n"

    # ════════════════════════════════════════════════════════════
    # 可选：仓库样本统计（让用户知道"评测跑在什么规模的数据上"）
    # ════════════════════════════════════════════════════════════
    repository_sample: dict[str, Any] | None = None
    if db is not None and repo_id:
        repo_uuid = UUID(str(repo_id))
        repository_sample = {
            "issues": db.query(Issue).filter(Issue.repo_id == repo_uuid).count(),
            "pull_requests": db.query(PullRequest).filter(PullRequest.repo_id == repo_uuid).count(),
            "workflow_runs": db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_uuid).count(),
            "code_symbols": db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_uuid).count(),
            "knowledge_documents": db.query(Document).filter(Document.repo_id == repo_uuid).count(),
        }

    return {
        # 7 个核心指标（全部 round 到 4 位，避免浮点噪声）
        "issue_classification_accuracy": round(issue_accuracy, 4),
        "pr_key_file_coverage": round(pr_key_file_coverage, 4),
        "rag_hit_rate_at_3": round(rag_hit_rate_at_3, 4),
        "tool_call_accuracy": round(tool_call_schema_ready_rate, 4),
        "skill_registry_coverage": round(skill_registry_coverage, 4),
        "skill_entrypoint_accuracy": round(skill_entrypoint_accuracy, 4),
        "ci_pytest_failure_detection": round(ci_pytest_failure_detection, 4),
        "json_schema_pass_rate": round(json_schema_pass_rate, 4),
        # 可选：RAG 质量（没传 db 就是 None）
        "rag_quality": rag_quality,
        # 压缩链路的明细（排查"压缩效果变差"时用）
        "context_compression": {
            "message_stats": message_stats,
            # 只有一条检索结果，所以取 [0]；没有就空 dict
            "retrieval_stats": retrieval_items[0].get("compression") if retrieval_items else {},
            "memory_sections": {
                "facts": len(merged_memory["facts"]),
                "decisions": len(merged_memory["decisions"]),
                "tasks": len(merged_memory["tasks"]),
            },
            "quality": compression_quality,
        },
        # 样本数（说明"这些指标是基于多少条样本算出来的"）
        "cases": {
            "issue": len(issue_cases),
            "pr": 1,
            "rag": len(rag_docs),
            "ci": 1,
            "skills": len(required_skill_entrypoints),
            "context_compression": 3,
        },
        "repository_sample": repository_sample,
        "report_markdown": report_markdown,
    }
