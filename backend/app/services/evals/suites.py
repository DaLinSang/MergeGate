"""
评测集加载（从 JSON 文件读固定回归集）
===================================

【业务定位】
    RAG 评测的"题库管理"：从 docs/datas/evals/*.json 读固定评测集，
    校验它引用的源文档是否已经索引，并转成 RagEvalCase 列表。

    【为什么要有"固定评测集"】
        临时出题（自动冒烟）只能验证"链路能跑"，
        但没法回答"改完 prompt 之后效果变好了还是变差了"。
        固定题集 + 固定答案 = 可对比的基线。

【数据流向】
    上游（谁调它）：
        api/routes/evaluations.py      评测集列表 / 加载
        services/evals/runner.py       跑评测时加载
    下游（它调谁）：
        services/evals/ragas_runner.RagEvalCase   评测用例的数据结构
        documents 表（校验 source_file 是否已索引）

【核心规则】
    1. ★ **ready 校验**：评测集里写的 source_file 必须在 documents 表里能找到。
       找不到就是"题目问的东西还没入库"，跑出来的分数没有意义。
    2. **sha256 记录题库版本**：题库改了要能看出来（否则两次评测结果不可比）。
    3. **两种来源**：
        fixed     人工维护的 JSON 文件（可作为发布基线）
        auto_smoke 从当前文档临时出题（只验证链路，不作基线）

【本文件函数地图】
    list_eval_suites()          ★ 列出可用评测集（含 ready 状态）
        ├ _read_suite()             读文件 + 校验格式 + 算 sha256
        ├ _suite_applies_to_repo()  按 repository_names 过滤
        └ _suite_metadata()         ★ 组装元数据 + 检查 missing_sources
    load_eval_suite()           ★ 加载具体评测集（转成 RagEvalCase）
        └ _matching_documents()     按文件名匹配已索引的文档
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document, Repository
from app.services.evals.ragas_runner import RagEvalCase


# 默认题库目录：项目根下的 docs/datas/evals/
# parents[4] 的算法：evals/suites.py → evals → services → app → backend → 项目根
DEFAULT_SUITE_DIR = Path(__file__).resolve().parents[4] / "docs" / "datas" / "evals"

# 自动冒烟测试的固定 ID（不是文件，是虚拟题库）
AUTO_SMOKE_SUITE_ID = "auto-document-smoke"


@dataclass(frozen=True)
class LoadedEvalSuite:
    """加载完成的评测集（元数据 + 用例列表）。

    【为什么 frozen】
        加载后不该被改动。评测要可复现，用例被中途改了就没法解释结果。
    """

    metadata: dict[str, Any]
    cases: list[RagEvalCase]


def _read_suite(path: Path) -> tuple[dict[str, Any], str]:
    """读一个评测集 JSON 文件，校验基本格式，返回 (内容, sha256)。

    【调用链】被 list_eval_suites() / load_eval_suite() 调用

    【数据流向】
        入：path（JSON 文件路径）
        出：(payload dict, sha256 十六进制)

    【校验三件事】
        ① 必须是 JSON 对象（不能是数组）
        ② 必须有 name
        ③ 必须有非空 cases 数组

    【★ 为什么 sha256 要算原始字节（raw）而不是解析后再序列化】
        解析后重新序列化会改变空格/键顺序，算出来的哈希不稳定。
        直接对文件原始字节哈希，才能准确反映"文件有没有变"。

    【为什么错误信息带文件名】
        题库可能有十几个。不带文件名，用户不知道是哪个文件坏了。
    """
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"评测集 {path.name} 必须是 JSON 对象")

    name = str(payload.get("name") or "").strip()
    cases = payload.get("cases")
    if not name or not isinstance(cases, list) or not cases:
        raise ValueError(f"评测集 {path.name} 缺少 name 或 cases")

    return payload, hashlib.sha256(raw).hexdigest()


def _matching_documents(db: Session, repo_id: Any, source_file: str) -> list[Document]:
    """找出"文件名匹配 source_file"的所有已索引文档切片。

    【调用链】被 _suite_metadata() / load_eval_suite() 调用

    【数据流向】
        入：repo_id、source_file（如 "report.pdf" 或 "docs/setup.md"）
        出：Document 列表（同一文件的多个切片）

    【★ 为什么要匹配三个字段 + 只比文件名】
        文档入库时，文件名可能落在不同地方：
            metadata.filename     上传时的原始文件名
            metadata.display_title 展示标题
            metadata.source_file  某些来源会带
        而且可能是完整路径（docs/a/b.md）或只有文件名（b.md）。
        所以统一取 Path(...).name 只比"文件名部分"，
        并用 casefold()（比 lower() 更彻底的 Unicode 大小写折叠）。

    【代价】
        只比文件名意味着不同目录下的同名文件会一起命中。
        对评测来说可以接受（题库里写的通常是唯一文件名）。
    """
    expected = Path(source_file).name.casefold()
    documents = (
        db.query(Document)
        .filter(Document.repo_id == repo_id)
        .order_by(Document.created_at.asc())
        .all()
    )

    matches: list[Document] = []
    for document in documents:
        metadata = document.meta or {}
        # 三个可能的字段都试一遍（取 basename + casefold）
        candidates = {
            Path(str(metadata.get("filename") or "")).name.casefold(),
            Path(str(metadata.get("display_title") or "")).name.casefold(),
            Path(str(metadata.get("source_file") or "")).name.casefold(),
        }
        if expected in candidates:
            matches.append(document)
    return matches


def _suite_applies_to_repo(payload: dict[str, Any], repo: Repository) -> bool:
    """判断评测集是否适用于这个仓库。

    【调用链】被 list_eval_suites() / load_eval_suite() 调用

    【规则】
        repository_names 不存在或为空 → 适用于所有仓库（通用题库）
        否则 → 仓库的 name 或 full_name 必须在列表里

    【为什么 name 和 full_name 都认】
        题库里可能写 "devflow-ai"（只写仓库名），也可能写 "owner/devflow-ai"（全名）。
        两个都比一遍，减少"明明适用却说没找到"的困惑。
    """
    names = payload.get("repository_names")
    if not isinstance(names, list) or not names:
        return True
    accepted = {str(item).casefold() for item in names if str(item).strip()}
    return repo.name.casefold() in accepted or repo.full_name.casefold() in accepted


def _suite_metadata(
    db: Session,
    repo: Repository,
    path: Path,
    payload: dict[str, Any],
    sha256: str,
) -> dict[str, Any]:
    """组装评测集元数据，并检查它引用的源文档是否已索引（★ ready 判定）。

    【调用链】_matching_documents() → 本函数
    【被谁调用】list_eval_suites() / load_eval_suite()

    【数据流向】
        入：db、repo、path、payload、sha256
        出：metadata dict（含 ready / missing_sources）

    【★ ready 为什么重要】
        题库里的题是基于某份文档出的。如果那份文档还没索引，
        模型必然答不出来 —— 分数低是"环境问题"不是"模型问题"。
        提前标出 ready=false + 缺哪些文件，用户就知道该先传什么。

    【missing_sources 为什么要去重】
        10 道题可能都基于同一份 report.pdf。不去重会列出 10 次。
    """
    missing_sources: list[str] = []
    for item in payload["cases"]:
        source_file = str(item.get("source_file") or "").strip()
        # 有 source_file 但库里找不到 → 记为缺失
        if source_file and not _matching_documents(db, repo.id, source_file):
            missing_sources.append(source_file)
    # 去重保序
    missing_sources = list(dict.fromkeys(missing_sources))

    return {
        "id": str(payload["name"]),
        "name": str(payload["name"]),
        "label": str(payload.get("label") or payload["name"]),
        "description": str(payload.get("description") or "人工维护的固定回归评测集"),
        "schema_version": int(payload.get("schema_version") or 1),
        "case_count": len(payload["cases"]),
        # 题库版本指纹：改了题库，sha256 就变，两次评测结果不可直接对比
        "sha256": sha256,
        "source": "fixed",
        "filename": path.name,
        # ★ 没有缺失源文档才算"可用"
        "ready": not missing_sources,
        "missing_sources": missing_sources,
        "recommended": bool(payload.get("recommended", False)),
    }


def list_eval_suites(
    db: Session,
    repo_id: Any,
    *,
    suite_dir: Path = DEFAULT_SUITE_DIR,
) -> list[dict[str, Any]]:
    """列出所有可用的评测集（★ 评测页面的数据源）。

    【调用链】_read_suite() / _suite_applies_to_repo() / _suite_metadata() → 本函数
    【被谁调用】api/routes/evaluations.py

    【数据流向】
        入：db、repo_id、suite_dir（可覆盖，测试用）
        出：metadata 列表 + 最后追加一条"自动冒烟"

    【排序规则（三段）】
        ① recommended 优先（推荐的题库排前面）
        ② ready 优先（能跑的排前面）
        ③ label 字母序（保证稳定）
        —— 用 not 取反是因为 Python 升序排 False(0) 在 True(1) 之前。

    【★ 为什么"自动冒烟"是虚拟题库（不在文件系统里）】
        它是"从当前仓库文档临时出题"，题目是运行时生成的，没有固定文件。
        所以 case_count 和 sha256 都是 None。

    【★ 为什么自动冒烟明确标注"不能作为发布基线"】
        题目临时生成 → 每次跑的内容可能不同 → 分数不可比。
        不标清楚的话，用户会拿它当"质量指标"，这是误导。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")

    suites: list[dict[str, Any]] = []
    # 目录不存在 → 只返回自动冒烟（不报错，因为可能还没建题库）
    if suite_dir.is_dir():
        for path in sorted(suite_dir.glob("*.json")):
            payload, sha256 = _read_suite(path)
            if _suite_applies_to_repo(payload, repo):
                suites.append(_suite_metadata(db, repo, path, payload, sha256))

    # 三段排序
    suites.sort(key=lambda item: (not item["recommended"], not item["ready"], item["label"]))

    # 追加虚拟题库（永远可用）
    suites.append(
        {
            "id": AUTO_SMOKE_SUITE_ID,
            "name": AUTO_SMOKE_SUITE_ID,
            "label": "自动文档冒烟测试",
            "description": "从当前仓库文档临时出题，只验证链路能运行，不能作为发布基线。",
            "schema_version": 1,
            "case_count": None,
            "sha256": None,
            "source": "auto_smoke",
            "filename": None,
            "ready": True,
            "missing_sources": [],
            "recommended": False,
        }
    )
    return suites


def load_eval_suite(
    db: Session,
    repo_id: Any,
    suite_id: str,
    *,
    suite_dir: Path = DEFAULT_SUITE_DIR,
) -> LoadedEvalSuite:
    """加载指定评测集，转成可执行的 RagEvalCase 列表（★ 评测执行的起点）。

    【调用链】追这个函数会进入：
        1. _read_suite()            → 读文件
        2. _suite_applies_to_repo() → 校验适用性
        3. _suite_metadata()        → 校验 ready（内部调 _matching_documents）
        4. _matching_documents()    → 把 source_file 映射成 expected_source_ids
    【被谁调用】services/evals/runner.py / api/routes/evaluations.py

    【数据流向】
        入：db、repo_id、suite_id、suite_dir
        出：LoadedEvalSuite（metadata + cases）

    【★ 自动冒烟特殊处理：直接返回空 cases】
        它的题目是运行时生成的（在 ragas_runner 里），这里只给元数据。
        调用方看到 cases 为空 + source="auto_smoke"，就知道该走另一条路径。

    【★ 为什么 ready=false 要抛异常（而不是返回空）】
        跑一个"源文档都不在库里"的题库，结果必然全错。
        抛异常让用户立刻知道"先去传文档"，而不是拿到一堆 0 分。

    【★ expected_source_ids 为什么同时放 document.id 和 source_id】
        评测要判断"检索到的证据是不是来自预期文档"。
        RAG 检索返回的可能是 document_id（切片级）或 source_id（源对象级）。
        两个都放进去，匹配时任意一个命中就算对 —— 兼容两种粒度。

    【★ expected_source_ids 为什么用 dict.fromkeys 去重】
        同一文件的多个切片会产生多个 ID。
        去重后列表更短，匹配时也更快。
    """
    # ── 自动冒烟：虚拟题库，只给元数据 ──
    if suite_id == AUTO_SMOKE_SUITE_ID:
        return LoadedEvalSuite(
            metadata={
                "id": AUTO_SMOKE_SUITE_ID,
                "name": AUTO_SMOKE_SUITE_ID,
                "label": "自动文档冒烟测试",
                "description": "从当前仓库文档临时出题，只验证链路能运行，不能作为发布基线。",
                "source": "auto_smoke",
                "sha256": None,
                "case_count": None,
            },
            cases=[],
        )

    repo = db.get(Repository, repo_id)
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")

    # ── 遍历目录找匹配的题库 ──
    # 注意这个写法：`for ... in ... if ... else []` —— 目录不存在时迭代空列表
    for path in sorted(suite_dir.glob("*.json")) if suite_dir.is_dir() else []:
        payload, sha256 = _read_suite(path)
        # 名字不匹配或不适用这个仓库 → 跳过
        if str(payload["name"]) != suite_id or not _suite_applies_to_repo(payload, repo):
            continue

        metadata = _suite_metadata(db, repo, path, payload, sha256)

        # ★ 源文档缺失 → 拒绝加载（跑出来一定是错的）
        if not metadata["ready"]:
            missing = "、".join(metadata["missing_sources"])
            raise ValueError(f"评测集 {suite_id} 缺少已索引来源：{missing}")

        # ── 逐题转成 RagEvalCase ──
        cases: list[RagEvalCase] = []
        for index, item in enumerate(payload["cases"], start=1):
            case_id = str(item.get("id") or f"case-{index}").strip()
            question = str(item.get("question") or "").strip()
            reference = str(item.get("reference") or "").strip()
            # 题干和参考答案都必填（缺了没法评测）
            if not question or not reference:
                raise ValueError(f"评测集 {suite_id} 的 {case_id} 缺少 question 或 reference")

            source_file = str(item.get("source_file") or "").strip()
            documents = _matching_documents(db, repo.id, source_file) if source_file else []
            # 把"预期来源"展开成 ID 列表（document.id + source_id 都要）
            expected_source_ids = list(
                dict.fromkeys(
                    str(value)
                    for document in documents
                    for value in (document.id, document.source_id)
                    if value
                )
            )

            cases.append(
                RagEvalCase(
                    id=case_id,
                    question=question,
                    reference=reference,
                    expected_source_ids=expected_source_ids,
                    source_type=str(item.get("source_type") or "knowledge_file"),
                    # 工具断言：必须调用哪些工具（验证 Agent 的工具选择能力）
                    required_tools=[str(value) for value in item.get("required_tools") or []],
                    expected_tool_calls=[dict(value) for value in item.get("expected_tool_calls") or []],
                    # 题型：positive（该答出来）/ 其他（如 negative 该拒答）
                    kind=str(item.get("kind") or "positive"),
                    # 内容断言：必须包含 / 必须不包含
                    must_include=[str(value) for value in item.get("must_include") or []],
                    must_not_include=[str(value) for value in item.get("must_not_include") or []],
                    answer_regex=str(item.get("answer_regex") or "").strip() or None,
                    source_file=source_file or None,
                )
            )

        # 实际用例数可能和文件里写的不同（理论上一样，这里以实际为准）
        metadata["case_count"] = len(cases)
        return LoadedEvalSuite(metadata=metadata, cases=cases)

    raise ValueError(f"未找到评测集：{suite_id}")
