"""
知识图谱（把研发实体连成一张图）
=============================

【业务定位】
    把 Issue / PR / CI / 文档 / 记忆 / 代码文件 连成一张图，回答"谁和谁有关"：
        "这个 PR 修了哪些 Issue"     → resolves 边
        "这个 PR 改了哪些文件"       → touches_file 边
        "这次 CI 失败跟哪个 PR 有关" → failed_in_ci 边
        "这条 AI 回答引用了什么证据" → used_as_evidence 边

    【为什么需要图，而不是只用向量检索】
        向量检索回答"内容相似"，图回答"结构关联"。
        用户问"这个 PR 关联哪些 Issue"，向量检索可能召不回（措辞不同），
        但图里有一条明确的 resolves 边 —— 直接查边就行，100% 准确。

【数据流向】
    上游（谁调它）：
        project_indexing.scan_project_index()  扫描完成后重建图谱
        api/routes/knowledge.py                图谱查询接口
    下游（它读谁 / 写谁）：
        读：issues / pull_requests / pr_files / workflow_runs / documents
            thread_memories / conversation_memories / agent_runs
        写：knowledge_graph_edges 表

【核心规则】
    1. **整组替换**：重建时先删该仓库所有边，再全量生成（图是派生数据，可重建）。
    2. **边的置信度分两档**：
       1.0 / 0.9  → 显式关系（assigned_to、resolves、cites）
       0.75 / 0.8 → 推断关系（正文里提到 #123、提到某个文件路径）
       为什么区分：推断会误判（"#123"可能只是举例），前端可以按置信度标色。
    3. **节点数上限 72**（GRAPH_NODE_LIMIT）：图太大前端渲染不动，也不好看。

【本文件函数地图】

    ┌─ 对外入口 ──────────────────────────────────────
    │  rebuild_knowledge_graph()   ★ 重建（删全部 → 生成 → 落库）
    │  knowledge_graph_payload()   ★ 查询（子图 + 统计）
    │
    ├─ 边的生成（最复杂的部分）────────────────────────
    │  _generate_edges()          ★ 六类来源依次加边
    │      ├ add_edge()（内部闭包）  去重 + 置信度取最大
    │      ├ _add_text_reference_edges()  ★ 从正文里抽 #123 / 文件路径
    │      ├ _closing_issue_refs()   "fixes #123" → resolves
    │      ├ _issue_refs()           "#123" → mentions
    │      └ _pr_refs()              "PR #45" → mentions
    │
    ├─ 节点的收集 ───────────────────────────────────
    │  _collect_nodes()           ★ 七类节点
    │      ├ _document_groups()     把多切片文档合并成一个节点
    │      ├ _decision_nodes()      从记忆里抽"决策"节点
    │      └ _fallback_node()       边里引用了但没收集到的节点（占位）
    │
    ├─ 子图选取 ─────────────────────────────────────
    │  _select_subgraph()         ★ 有 center 用 BFS，没 center 按优先级排
    │      └ _node_priority()        节点类型优先级
    │
    └─ 工具 ─────────────────────────────────────────
       normalize_node_type()      类型别名归一
       _team_members_by_name()    人名 → 节点 ID
       _code_path_to_node_id()    文件路径 → 节点 ID
       _extract_citation_like_values()  递归找引用
       _node_key() / _edge_id() / _path_node_id() / _clip()
"""

import json
import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    ConversationMemory,
    Document,
    Issue,
    KnowledgeGraphEdge,
    PRFile,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)

# 单次返回的节点上限（前端渲染能力 + 可读性）
GRAPH_NODE_LIMIT = 72

# ── 三个正则：从自由文本里抽引用 ──
# Issue 引用：#123
# 前面不能是字母（避免匹配 "abc#123" 这种）
ISSUE_REF_RE = re.compile(r"(?<![A-Za-z])#(\d+)\b")

# 关闭型引用：fixes #123 / closed #45 / resolves #7
# 这类是【强关系】（PR 明确说要修这个 Issue），比普通 mentions 重要
CLOSING_ISSUE_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+(?:issue\s+)?#(\d+)\b",
    re.IGNORECASE,
)

# PR 引用：PR #45 / pull request 45
PR_REF_RE = re.compile(r"\b(?:PR|pull request)\s*#?(\d+)\b", re.IGNORECASE)

# ── 节点类型别名表 ──
# 为什么要别名：同一个概念在不同地方叫法不同（PR / pull_request；ci / ci_run）。
# 归一化后图里只有一套类型，前端渲染和查询都简单。
NODE_TYPE_ALIASES = {
    "pr": "pull_request",
    "pull_request": "pull_request",
    "workflow_run": "ci_run",
    "ci": "ci_run",
    "ci_run": "ci_run",
    "memory": "memory_note",
    "memory_note": "memory_note",
    "knowledge_file": "document",
    "weekly_report": "document",
    "project_doc": "document",
    "project_manifest": "document",
    "project_overview": "document",
    "document": "document",
    "code_file": "code_file",
    "team_member": "team_member",
    "issue": "issue",
    "session": "session_decision",
    "session_decision": "session_decision",
}

# 这些 source_type 在图上统一显示成 "document" 节点（都是"资料"）
DOCUMENT_NODE_TYPES = {"knowledge_file", "weekly_report", "project_doc", "project_manifest", "project_overview"}


@dataclass
class DocumentGroup:
    """把"同一份文档的多个切片"合并成一个节点。

    【为什么需要合并】
        documents 表里一份文档被切成 N 块（N 行）。
        如果每块都当节点，图里会出现 20 个"README" —— 没法看。
        合并后一个文档 = 一个节点，content 取前 4 块拼接（够表达大意）。
    """

    source_type: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: Any = None


def normalize_node_type(value: str | None) -> str:
    """把节点类型别名归一成标准名。

    【调用链】被几乎所有函数调用（_node_key / _generate_edges / _collect_nodes …）

    【为什么未知类型原样返回】
        可能是新加的类型还没登记别名。原样返回至少能用，不会丢节点。
    """
    key = (value or "").strip()
    return NODE_TYPE_ALIASES.get(key, key)


# ════════════════════════════════════════════════════════════════
# 对外入口
# ════════════════════════════════════════════════════════════════

def rebuild_knowledge_graph(db: Session, repo: Repository) -> dict[str, int]:
    """重建一个仓库的整张图（★ 写入口）。

    【调用链】_generate_edges() / _collect_nodes() → 本函数
    【被谁调用】project_indexing.scan_project_index()（扫描完成后）

    【数据流向】
        入：db、repo
        出：{"edges": 边数, "nodes": 节点数}

    【★ 为什么是"先删后建"】
        图是纯派生数据（从 Issue/PR/文档算出来的）。
        增量更新要处理"删掉的 Issue 对应的边"，复杂且容易漏。
        全量重建几十毫秒就够（几百条边），简单可靠。

    【★ 为什么 db.commit() 在中间】
        边要落库后，_collect_nodes 才能从 knowledge_graph_edges 表读到
        "被引用但没收集到的节点"（fallback 节点）。
        所以必须先 commit 再统计节点数。
    """
    # ── 步骤 1：删该仓库所有旧边 ──
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)

    # ── 步骤 2：生成新边 ──
    edges = _generate_edges(db, repo.id)
    db.add_all(
        KnowledgeGraphEdge(
            repo_id=repo.id,
            from_type=edge["from_type"],
            from_id=edge["from_id"],
            to_type=edge["to_type"],
            to_id=edge["to_id"],
            relation=edge["relation"],
            confidence=edge["confidence"],
            source=edge["source"],
            meta=edge["metadata"],
        )
        for edge in edges
    )
    db.commit()

    # ── 步骤 3：统计节点（此时边已落库，能读到 fallback 节点）──
    nodes = _collect_nodes(db, repo.id)
    return {"edges": len(edges), "nodes": len(nodes)}


def knowledge_graph_payload(
    db: Session,
    repo: Repository,
    *,
    center_type: str | None = None,
    center_id: str | None = None,
    depth: int = 1,
) -> dict[str, Any]:
    """查图谱（★ 读入口，给前端渲染）。

    【调用链】追这个函数会进入：
        1. rebuild_knowledge_graph()  → 图是空的就重建（懒加载）
        2. _collect_nodes()           → 收集所有节点
        3. _select_subgraph()         → 选出要展示的子图（≤72 个）
        4. _edge_payload()            → 边转 dict
    【被谁调用】api/routes/knowledge.py（GET /knowledge/graph）

    【数据流向】
        入：repo、center_type/center_id（以谁为中心）、depth（展开几层，1~2）
        出：{repo_id, center_type, center_id, depth, nodes, edges, stats}

    【★ 两种展示模式】
        ① 有 center（用户点开某个 Issue）：以它为中心 BFS 展开 depth 层
        ② 没 center（打开图谱页）：按"是否有边 + 类型优先级"排序取前 72 个
        —— 见 _select_subgraph() 的详细说明。

    【★ 为什么"边数为 0 就重建"】
        首次访问、或上次重建失败时图是空的。
        懒加载比"每次启动都重建"更省资源。

    【★ used_as_evidence 的特殊处理】
        这条边表示"某条 AI 回答用了这个证据"。
        被用到的节点要标 highlighted —— 前端高亮，让用户看到"AI 真的引用了它"。
    """
    # ── 步骤 0：懒加载（图为空就重建）──
    edge_count = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).count()
    if edge_count == 0:
        rebuild_knowledge_graph(db, repo)

    # ── 步骤 1：收集节点 + 边 ──
    nodes_by_key = _collect_nodes(db, repo.id)
    edge_rows = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).all()
    edges = [_edge_payload(row) for row in edge_rows]

    # ── 步骤 2：如果有 center，确保它一定在节点表里 ──
    center_key = None
    resolved_center_type = normalize_node_type(center_type)
    if resolved_center_type and center_id:
        center_key = _node_key(resolved_center_type, center_id)
        # setdefault：节点已存在就用它，不存在就造一个占位节点
        # （比如用户直接输了一个还没被收集的 ID）
        nodes_by_key.setdefault(
            center_key,
            _fallback_node(resolved_center_type, center_id, highlighted=True),
        )

    # ── 步骤 3：选子图（depth 夹在 1~2，太深图会爆炸）──
    selected_keys, truncated = _select_subgraph(nodes_by_key, edges, center_key, max(1, min(depth, 2)))

    # ── 步骤 4：只保留两端都在子图里的边 ──
    selected_edges = [
        edge for edge in edges if _node_key(edge["from_type"], edge["from_id"]) in selected_keys and _node_key(edge["to_type"], edge["to_id"]) in selected_keys
    ]

    # ── 步骤 5：被 AI 当作证据用过的节点 → 高亮 ──
    for edge in selected_edges:
        if edge["relation"] == "used_as_evidence":
            nodes_by_key[_node_key(edge["to_type"], edge["to_id"])]["highlighted"] = True

    return {
        "repo_id": repo.id,
        "center_type": resolved_center_type or None,
        "center_id": center_id,
        "depth": depth,
        "nodes": [nodes_by_key[key] for key in selected_keys if key in nodes_by_key],
        "edges": selected_edges,
        "stats": {
            "total_nodes": len(nodes_by_key),
            "total_edges": len(edges),
            "visible_nodes": len(selected_keys),
            "visible_edges": len(selected_edges),
            "truncated": truncated,
        },
    }


# ════════════════════════════════════════════════════════════════
# 边的生成（本文件最复杂的部分）
# ════════════════════════════════════════════════════════════════

def _generate_edges(db: Session, repo_id: uuid.UUID) -> list[dict[str, Any]]:
    """生成一个仓库的所有边（★ 六类来源）。

    【调用链】追这个函数会进入：
        1. _document_groups()             → 文档分组
        2. _team_members_by_name()        → 人名 → 节点 ID
        3. _code_path_to_node_id()        → 文件路径 → 节点 ID
        4. _add_text_reference_edges()    → 从正文抽引用（Issue/PR/文件）
        5. _decision_nodes()              → 记忆里的"决策"
        6. _extract_citation_like_values()→ 从 tool_calls 里递归找引用
    【被谁调用】rebuild_knowledge_graph()

    【数据流向】
        入：db、repo_id
        出：边列表（已去重）

    【六类边，按加边顺序】
        ① Issue  → 团队成员（assigned_to）          显式，confidence 1.0
        ② Issue  → 正文提到的 Issue/PR/文件（mentions）  推断，0.75~0.8
        ③ PR     → Issue（resolves / mentions）
                   PR → 作者（mentions）
        ④ PR     → 改动文件（touches_file）          显式，1.0
        ⑤ CI     → PR / Issue（failed_in_ci）        推断，0.85 / 0.75
        ⑥ 文档/记忆 → 提到的实体（cites / mentions / documents）
           决策    → 引用的证据（cites）
           运行记录 → 用过的证据（used_as_evidence）

    【★ 为什么用 dict 去重而不是 list】
        同一条边可能被多次发现（比如正文里提了两次 #123）。
        用 (from_type, from_id, to_type, to_id, relation) 做 key 天然去重，
        而且能顺便合并 metadata 和取最大置信度。

    【★ add_edge 内部闭包的三个保护】
        ① from_id / to_id 为空 → 跳过（没有 ID 无法建边）
        ② 自环（from == to）→ 跳过（"Issue 提到自己"没意义）
        ③ 已存在 → 只更新置信度（取 max）和合并 metadata，不重复加

    【★ CI 只处理失败的】
        成功的 CI 日志没有"排查价值"，加边只会让图变噪。
    """
    # ── 一次性把要用的数据都查出来（避免循环里查库）──
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    prs = db.query(PullRequest).filter(PullRequest.repo_id == repo_id).all()
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    # PRFile 没有 repo_id，要 join PullRequest 才能按仓库过滤
    pr_files = db.query(PRFile).join(PullRequest, PRFile.pr_id == PullRequest.id).filter(PullRequest.repo_id == repo_id).all()
    docs = _document_groups(db, repo_id)

    # ── 建立几个查找表（number → 对象 / 名字 → ID / 路径 → ID）──
    issues_by_number = {issue.number: issue for issue in issues}
    prs_by_number = {pr.number: pr for pr in prs}
    team_by_name = _team_members_by_name(docs)
    code_path_to_id = _code_path_to_node_id(pr_files)

    # 边的容器：key = (from_type, from_id, to_type, to_id, relation)
    edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def add_edge(
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        relation: str,
        *,
        confidence: float = 1.0,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """加一条边（去重 + 置信度取最大）。

        【调用链】被 _generate_edges() 内部各处调用（也是传给 _add_text_reference_edges 的回调）
        """
        from_type = normalize_node_type(from_type)
        to_type = normalize_node_type(to_type)
        # 保护 ①：缺 ID 或自环 → 跳过
        if not from_id or not to_id or (from_type == to_type and from_id == to_id):
            return
        key = (from_type, str(from_id), to_type, str(to_id), relation)
        existing = edges.get(key)
        if existing:
            # 保护 ③：已存在 → 取最大置信度 + 合并 metadata
            existing["confidence"] = max(existing["confidence"], confidence)
            existing["metadata"].update(metadata or {})
            return
        edges[key] = {
            "id": _edge_id(from_type, str(from_id), to_type, str(to_id), relation),
            "from_type": from_type,
            "from_id": str(from_id),
            "to_type": to_type,
            "to_id": str(to_id),
            "relation": relation,
            "confidence": confidence,
            "source": source,
            "metadata": metadata or {},
        }

    # ── ① Issue 的边 ──
    for issue in issues:
        # Issue → 指派人（显式，最可信）
        for assignee in issue.assignees or []:
            member_id = team_by_name.get(str(assignee).strip().lower())
            if member_id:
                add_edge("issue", str(issue.id), "team_member", member_id, "assigned_to", metadata={"assignee": assignee})
        # Issue 正文里提到的 Issue / PR / 文件
        _add_text_reference_edges(
            add_edge,
            "issue",
            str(issue.id),
            f"{issue.title}\n{issue.body or ''}",
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue="mentions",
            relation_for_pr="mentions",
            relation_for_code="mentions",
        )

    # ── ② PR 的边 ──
    for pr in prs:
        # 把分支名也拼进文本：分支名常含 Issue 号（如 "fix/123-timeout"）
        text = f"{pr.title}\n{pr.body or ''}\n{pr.head_branch or ''}\n{pr.base_branch or ''}"

        # 先取"关闭型引用"（fixes #123），这类是强关系
        closing_refs = _closing_issue_refs(text)
        for number in closing_refs:
            issue = issues_by_number.get(number)
            if issue:
                add_edge("pull_request", str(pr.id), "issue", str(issue.id), "resolves", metadata={"issue_number": number})

        # 剩下的普通引用 → mentions（排除已经算作 resolves 的）
        for number in _issue_refs(text) - closing_refs:
            issue = issues_by_number.get(number)
            if issue:
                add_edge("pull_request", str(pr.id), "issue", str(issue.id), "mentions", metadata={"issue_number": number})

        # PR → 作者
        author_member = team_by_name.get(str(pr.author or "").strip().lower())
        if author_member:
            add_edge("pull_request", str(pr.id), "team_member", author_member, "mentions", metadata={"author": pr.author})

    # ── ③ PR → 改动文件（显式，1.0）──
    for pr_file in pr_files:
        code_id = code_path_to_id.get(pr_file.filename) or _path_node_id(pr_file.filename)
        add_edge("pull_request", str(pr_file.pr_id), "code_file", code_id, "touches_file", metadata={"path": pr_file.filename, "status": pr_file.status})

    # ── ④ CI 的边（只处理失败的）──
    for run in runs:
        if (run.conclusion or "").lower() != "failure":
            continue
        # 日志里可能提到 PR 号或 Issue 号
        text = f"{run.name}\n{run.logs_text or ''}"
        for number in _pr_refs(text):
            pr = prs_by_number.get(number)
            if pr:
                # 日志里提到 PR → 置信度 0.85（日志常有噪声，不给满分）
                add_edge("ci_run", str(run.id), "pull_request", str(pr.id), "failed_in_ci", confidence=0.85, metadata={"pr_number": number})
        for number in _issue_refs(text):
            issue = issues_by_number.get(number)
            if issue:
                add_edge("ci_run", str(run.id), "issue", str(issue.id), "failed_in_ci", confidence=0.75, metadata={"issue_number": number})

    # ── ⑤ 文档 / 记忆的边 ──
    for doc in docs:
        # 团队成员不是"资料"，跳过（它只作为边的目标出现）
        if doc.source_type == "team_member":
            continue
        from_type = "memory_note" if doc.source_type == "memory_note" else "document"
        # 记忆用 cites（"这条记忆引用了什么"），普通文档用 mentions
        relation = "cites" if doc.source_type == "memory_note" else "mentions"
        # 把 metadata 也拼进文本 —— 文件路径常存在 metadata 里
        text = f"{doc.title}\n{doc.content}\n{json.dumps(doc.metadata, ensure_ascii=False, default=str)}"
        _add_text_reference_edges(
            add_edge,
            from_type,
            doc.source_id,
            text,
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue=relation,
            relation_for_pr=relation,
            relation_for_code="cites" if doc.source_type == "memory_note" else "documents",
        )

    # ── ⑥ 决策节点的边（从记忆里抽的 decisions）──
    for decision in _decision_nodes(db, repo_id):
        text = str(decision["metadata"].get("content") or decision["title"])
        # 决策正文里提到的实体
        _add_text_reference_edges(
            add_edge,
            "session_decision",
            decision["id"],
            text,
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue="cites",
            relation_for_pr="cites",
            relation_for_code="cites",
        )
        # 记忆里结构化存的 citations（比正文推断更可信，0.9）
        for citation in decision["metadata"].get("citations") or []:
            target_type = normalize_node_type(str(citation.get("type") or citation.get("source_type") or ""))
            target_id = str(citation.get("id") or citation.get("source_id") or "")
            if target_type and target_id:
                add_edge("session_decision", decision["id"], target_type, target_id, "cites", confidence=0.9, source="memory_citation")

    # ── ⑦ 运行记录 → 用过的证据（最近 25 次）──
    # 这条边让前端能高亮"AI 实际引用了哪些资料"
    for run in db.query(AgentRun).filter(AgentRun.repo_id == repo_id).order_by(AgentRun.created_at.desc()).limit(25):
        source_id = f"agent_run:{run.id}"
        for citation in _extract_citation_like_values(run.tool_calls or []):
            target_type = normalize_node_type(str(citation.get("type") or citation.get("source_type") or ""))
            target_id = str(citation.get("id") or citation.get("source_id") or "")
            if target_type and target_id:
                add_edge("session_decision", source_id, target_type, target_id, "used_as_evidence", confidence=0.75, source="tool_call_trace")

    return list(edges.values())


def _add_text_reference_edges(
    add_edge: Any,
    from_type: str,
    from_id: str,
    text: str,
    issues_by_number: dict[int, Issue],
    prs_by_number: dict[int, PullRequest],
    code_path_to_id: dict[str, str],
    *,
    relation_for_issue: str,
    relation_for_pr: str,
    relation_for_code: str,
) -> None:
    """从一段自由文本里抽引用，加三条边（Issue / PR / 文件）。

    【调用链】_issue_refs() / _pr_refs() → 本函数
    【被谁调用】_generate_edges()（三处：Issue / 文档 / 决策）

    【数据流向】
        入：add_edge 回调、起点、文本、三个查找表、三种关系名
        出：无（通过 add_edge 副作用加边）

    【为什么关系名要作为参数传】
        同一段文本，不同来源的语义不同：
            Issue 提到 #123  → mentions
            记忆提到 #123    → cites（"我引用了它"）
            决策提到 #123    → cites
        所以关系名不能写死。

    【★ 为什么文件匹配要求 len(path) >= 6】
        太短的路径（如 "a.py"）几乎会在任何文本里误命中。
        至少 6 个字符（如 "app/x.py"）才有区分度。

    【置信度分级】
        Issue 引用 0.75（编号容易撞车）
        PR 引用    0.8 （"PR #45"这种写法比较明确）
        文件引用   0.8 （完整路径命中，比较可信）
    """
    # Issue 引用
    for number in _issue_refs(text):
        issue = issues_by_number.get(number)
        if issue:
            add_edge(from_type, from_id, "issue", str(issue.id), relation_for_issue, confidence=0.75, metadata={"issue_number": number})

    # PR 引用
    for number in _pr_refs(text):
        pr = prs_by_number.get(number)
        if pr:
            add_edge(from_type, from_id, "pull_request", str(pr.id), relation_for_pr, confidence=0.8, metadata={"pr_number": number})

    # 文件路径引用（遍历所有 PR 改过的文件，看文本里有没有提到）
    for path, code_id in code_path_to_id.items():
        if len(path) >= 6 and path in text:
            add_edge(from_type, from_id, "code_file", code_id, relation_for_code, confidence=0.8, metadata={"path": path})


# ════════════════════════════════════════════════════════════════
# 节点的收集
# ════════════════════════════════════════════════════════════════

def _collect_nodes(db: Session, repo_id: uuid.UUID) -> dict[str, dict[str, Any]]:
    """收集一个仓库的所有节点（★ 七类）。

    【调用链】_document_groups() / _decision_nodes() / _fallback_node() → 本函数
    【被谁调用】rebuild_knowledge_graph() / knowledge_graph_payload()

    【数据流向】
        入：db、repo_id
        出：{node_key: node_dict}

    【七类节点】
        issue / pull_request / ci_run        —— 协作事实
        team_member / memory_note / document —— 人和资料
        code_file                            —— 代码文件
        session_decision                     —— 决策 + AI 回答

    【★ 最后一步：给边里引用到的节点补占位】
        边可能指向一个"没被收集到"的节点（比如引用了别的仓库的 Issue）。
        不补的话，前端渲染边时会找不到两端节点而报错。
        用 _fallback_node 造一个"只有 ID 没有详情"的占位节点。
    """
    nodes: dict[str, dict[str, Any]] = {}

    def put(node_type: str, node_id: str, title: str, subtitle: str = "", metadata: dict[str, Any] | None = None, highlighted: bool = False) -> None:
        """加一个节点（已存在则合并 metadata + 合并 highlighted）。"""
        normalized = normalize_node_type(node_type)
        key = _node_key(normalized, str(node_id))
        if key in nodes:
            # 已存在：合并（不要覆盖，因为前面的可能有更多信息）
            nodes[key]["metadata"].update(metadata or {})
            nodes[key]["highlighted"] = nodes[key]["highlighted"] or highlighted
            return
        nodes[key] = {
            "id": str(node_id),
            "type": normalized,
            "title": title or str(node_id),
            "subtitle": subtitle or "",
            "metadata": metadata or {},
            "highlighted": highlighted,
        }

    # ── ① Issue ──
    for issue in db.query(Issue).filter(Issue.repo_id == repo_id).all():
        put("issue", str(issue.id), f"#{issue.number} {issue.title}", issue.state, {"number": issue.number, "labels": issue.labels or [], "assignees": issue.assignees or []})

    # ── ② PR（subtitle 优先显示 merged，比 state 更有信息量）──
    for pr in db.query(PullRequest).filter(PullRequest.repo_id == repo_id).all():
        subtitle = "merged" if pr.merged_at else pr.state
        put("pull_request", str(pr.id), f"#{pr.number} {pr.title}", subtitle, {"number": pr.number, "author": pr.author, "head_branch": pr.head_branch})

    # ── ③ CI ──
    for run in db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all():
        put("ci_run", str(run.id), run.name, run.conclusion or run.status, {"status": run.status, "conclusion": run.conclusion, "html_url": run.html_url})

    # ── ④ 文档 / 记忆 / 团队成员（从 DocumentGroup 里分）──
    docs = _document_groups(db, repo_id)
    for doc in docs:
        if doc.source_type == "team_member":
            put("team_member", doc.source_id, doc.metadata.get("name") or doc.title, doc.metadata.get("role") or "", doc.metadata)
        elif doc.source_type == "memory_note":
            put("memory_note", doc.source_id, doc.title, "memory note", doc.metadata)
        elif doc.source_type in DOCUMENT_NODE_TYPES:
            # metadata | {...} 是字典合并运算符（3.9+），把 source_type 塞进去
            put("document", doc.source_id, doc.metadata.get("display_title") or doc.title, doc.source_type, doc.metadata | {"source_type": doc.source_type})

    # ── ⑤ 代码文件（从 PR 改动的文件里来）──
    for pr_file in db.query(PRFile).join(PullRequest, PRFile.pr_id == PullRequest.id).filter(PullRequest.repo_id == repo_id).all():
        put("code_file", _path_node_id(pr_file.filename), pr_file.filename, "PR touched file", {"path": pr_file.filename, "status": pr_file.status})

    # ── ⑥ 决策节点 ──
    for decision in _decision_nodes(db, repo_id):
        put("session_decision", decision["id"], decision["title"], decision["subtitle"], decision["metadata"])

    # ── ⑦ AI 回答（也归到 session_decision 类型）──
    for run in db.query(AgentRun).filter(AgentRun.repo_id == repo_id).order_by(AgentRun.created_at.desc()).limit(25):
        put("session_decision", f"agent_run:{run.id}", f"AI answer: {run.route}", "tool evidence path", {"route": run.route, "created_at": run.created_at.isoformat() if run.created_at else None})

    # ── ⑧ 给边里引用到的节点补占位（避免前端渲染找不到节点）──
    for edge in db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo_id).all():
        from_key = _node_key(edge.from_type, edge.from_id)
        to_key = _node_key(edge.to_type, edge.to_id)
        nodes.setdefault(from_key, _fallback_node(edge.from_type, edge.from_id))
        nodes.setdefault(to_key, _fallback_node(edge.to_type, edge.to_id))

    return nodes


# ════════════════════════════════════════════════════════════════
# 子图选取
# ════════════════════════════════════════════════════════════════

def _select_subgraph(
    nodes_by_key: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    center_key: str | None,
    depth: int,
) -> tuple[list[str], bool]:
    """选出要展示的节点（≤ GRAPH_NODE_LIMIT 个）。

    【调用链】_node_priority() → 本函数
    【被谁调用】knowledge_graph_payload()

    【数据流向】
        入：所有节点、所有边、center_key（可空）、depth
        出：(选中的 key 列表, 是否被截断)

    【★ 两种模式】
        ① 有 center（用户点开某个节点）→ BFS 展开
           从 center 出发，逐层扩展邻居，最多 depth 层。
           邻居排序用 title —— 保证每次渲染顺序一致（不会跳来跳去）。

        ② 没 center（打开图谱总览）→ 按优先级排序
           排序键三段：
              0/1        有边的优先（孤立节点没意义）
              类型优先级  issue > PR > CI > 人 > 记忆 > 决策 > 文档 > 文件
              标题        保证稳定
           取前 72 个。

    【★ 为什么 BFS 里 seen 要在入队时标记（而不是出队时）】
        出队时标记会导致同一个节点被多次入队（重复），队列膨胀。
        入队即标记是标准写法。

    【★ truncated 的两种含义】
        模式 ①：队列还没空 → 说明还有更多邻居没展示
        模式 ②：节点总数 > 72 → 说明被截断了
    """
    # ── 模式 ①：以某个节点为中心 BFS ──
    if center_key and center_key in nodes_by_key:
        # 建邻接表（无向图：两个方向都连）
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            a = _node_key(edge["from_type"], edge["from_id"])
            b = _node_key(edge["to_type"], edge["to_id"])
            adjacency[a].add(b)
            adjacency[b].add(a)

        selected: list[str] = []
        seen = {center_key}
        # 队列元素：(节点 key, 距中心的层数)
        queue: deque[tuple[str, int]] = deque([(center_key, 0)])

        while queue and len(selected) < GRAPH_NODE_LIMIT:
            key, distance = queue.popleft()
            selected.append(key)
            # 到了深度上限 → 不再扩展（但仍然把它自己加进 selected）
            if distance >= depth:
                continue
            # 邻居按 title 排序，保证渲染顺序稳定
            for nxt in sorted(adjacency.get(key, []), key=lambda item: nodes_by_key.get(item, {}).get("title", item)):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, distance + 1))
        return selected, bool(queue)

    # ── 模式 ②：总览，按优先级排序 ──
    touched: set[str] = set()
    for edge in edges:
        touched.add(_node_key(edge["from_type"], edge["from_id"]))
        touched.add(_node_key(edge["to_type"], edge["to_id"]))

    ranked = sorted(
        nodes_by_key,
        key=lambda key: (0 if key in touched else 1, _node_priority(nodes_by_key[key]["type"]), nodes_by_key[key]["title"]),
    )
    selected = ranked[:GRAPH_NODE_LIMIT]
    return selected, len(ranked) > GRAPH_NODE_LIMIT


# ════════════════════════════════════════════════════════════════
# 文档分组与决策节点
# ════════════════════════════════════════════════════════════════

def _document_groups(db: Session, repo_id: uuid.UUID) -> list[DocumentGroup]:
    """把 documents 表按 (source_type, source_id) 分组成"一个文档一个节点"。

    【调用链】被 _generate_edges() / _collect_nodes() 调用

    【数据流向】
        入：db、repo_id
        出：DocumentGroup 列表

    【三个处理细节】
        ① 按 chunk_index 排序：保证 content 拼接顺序正确
           （content 取前 4 块 —— 够表达大意，又不至于太长）
        ② title 优先用 metadata.display_title：那是"人写的标题"，
           documents.title 是"标题 + 章节名"，不适合当节点标题
        ③ created_at 取第一块的时间（代表整份文档的时间）
    """
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for doc in db.query(Document).filter(Document.repo_id == repo_id).order_by(Document.created_at.asc()).all():
        groups[(doc.source_type, str(doc.source_id))].append(doc)

    output: list[DocumentGroup] = []
    for (source_type, source_id), docs in groups.items():
        # 按切片序号排序（保证内容顺序）
        docs.sort(key=lambda item: ((item.meta or {}).get("chunk_index", 0), item.created_at))
        first = docs[0]
        metadata = dict(first.meta or {})
        title = str(metadata.get("display_title") or first.title)
        # 只取前 4 块拼内容（够表达大意）
        content = "\n".join(doc.content or "" for doc in docs[:4])
        output.append(DocumentGroup(source_type, source_id, title, content, metadata, first.created_at))
    return output


def _decision_nodes(db: Session, repo_id: uuid.UUID) -> list[dict[str, Any]]:
    """从记忆里抽出"决策"节点。

    【调用链】被 _generate_edges() / _collect_nodes() 调用

    【数据流向】
        入：db、repo_id
        出：[{id, title, subtitle, metadata}] 列表

    【为什么决策要单独成节点】
        记忆里的 decisions 是"团队做过的判断"（如"决定用 PostgreSQL"）。
        它们和 Issue/PR 有引用关系（"因为这个 Issue 我们才这么定"）。
        单独成节点，图上就能看到"决策 ← 依据"这条链。

    【ID 格式：{thread|conversation}:{memory_id}:decision:{index}】
        为什么要带 index：一条记忆里可能有 5 个决策，要能区分。
        为什么带前缀：会话级和仓库级记忆的 UUID 可能重复，前缀能避免撞车。
    """
    output: list[dict[str, Any]] = []
    thread_memories = db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo_id).all()
    conversation_memories = db.query(ConversationMemory).filter(ConversationMemory.repo_id == repo_id).all()

    for prefix, memory in [("thread", item) for item in thread_memories] + [("conversation", item) for item in conversation_memories]:
        for index, decision in enumerate(memory.decisions or []):
            text = str(decision).strip()
            if not text:
                continue
            output.append(
                {
                    "id": f"{prefix}:{memory.id}:decision:{index}",
                    # 标题截断到 72（图上的节点标签不能太长）
                    "title": _clip(text, 72),
                    "subtitle": "session decision",
                    "metadata": {
                        "content": text,
                        "memory_id": str(memory.id),
                        "citations": memory.citations or [],
                        "sessions_incorporated": memory.sessions_incorporated,
                    },
                }
            )
    return output


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _team_members_by_name(docs: list[DocumentGroup]) -> dict[str, str]:
    """建立"人名 → 节点 ID"的查找表。

    【调用链】被 _generate_edges() 调用

    【为什么同时索引 name 和 title】
        团队成员的 source_type 是 team_member，name 在 metadata 里。
        但不同来源可能只填了 title。两个都索引，匹配成功率更高。
        都转小写 —— Issue 的 assignee 大小写不一定规范。
    """
    output: dict[str, str] = {}
    for doc in docs:
        if doc.source_type != "team_member":
            continue
        for value in [doc.metadata.get("name"), doc.title]:
            text = str(value or "").strip().lower()
            if text:
                output[text] = doc.source_id
    return output


def _code_path_to_node_id(pr_files: list[PRFile]) -> dict[str, str]:
    """建立"文件路径 → 代码节点 ID"的查找表。

    【调用链】被 _generate_edges() 调用

    用 setdefault：同一个文件可能被多个 PR 改过，只保留第一个映射（值都一样）。
    """
    output: dict[str, str] = {}
    for pr_file in pr_files:
        output.setdefault(pr_file.filename, _path_node_id(pr_file.filename))
    return output


def _extract_citation_like_values(value: Any) -> list[dict[str, Any]]:
    """递归地从任意嵌套结构里找出"像引用"的字典。

    【调用链】被 _generate_edges() 调用（处理 tool_calls）

    【什么算"像引用"】
        同时有 (source_type 或 type) 和 (source_id 或 id) 的字典。

    【为什么用递归】
        tool_calls 的结构是嵌套的：{tool_name, arguments: {...}, result: {...}}，
        引用可能藏在任意深度的 result 里。
        递归遍历最省事，而且结构变化时不用改代码。
    """
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if ("source_type" in value or "type" in value) and ("source_id" in value or "id" in value):
            found.append(value)
        for child in value.values():
            found.extend(_extract_citation_like_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_extract_citation_like_values(child))
    return found


def _edge_payload(row: KnowledgeGraphEdge) -> dict[str, Any]:
    """ORM 边 → dict（给前端）。

    【调用链】被 knowledge_graph_payload() 调用

    注意 metadata 字段：ORM 属性叫 meta（metadata 是 SQLAlchemy 保留字），
    但对外字段名是 metadata。
    """
    return {
        "id": str(row.id),
        "from_type": row.from_type,
        "from_id": row.from_id,
        "to_type": row.to_type,
        "to_id": row.to_id,
        "relation": row.relation,
        "confidence": row.confidence,
        "source": row.source,
        "metadata": row.meta or {},
    }


def _fallback_node(node_type: str, node_id: str, highlighted: bool = False) -> dict[str, Any]:
    """造一个"占位节点"（只有 ID，没有详情）。

    【调用链】被 _collect_nodes() / knowledge_graph_payload() 调用

    【什么时候用】
        边指向一个没被收集到的节点（跨仓库引用、已删除的对象）。
        不补占位的话，前端渲染边时找不到两端节点会报错。

    subtitle 写 "referenced item" —— 让用户知道"这个节点只有引用，没有详情"。
    """
    return {
        "id": str(node_id),
        "type": normalize_node_type(node_type),
        "title": str(node_id),
        "subtitle": "referenced item",
        "metadata": {},
        "highlighted": highlighted,
    }


def _node_key(node_type: str, node_id: str) -> str:
    """节点唯一键："{type}:{id}"。

    【调用链】被几乎所有函数调用（这是图的"主键"）

    用字符串拼接而不是 tuple：dict 的 key 要能序列化（前端/日志里能看到）。
    """
    return f"{normalize_node_type(node_type)}:{node_id}"


def _edge_id(from_type: str, from_id: str, to_type: str, to_id: str, relation: str) -> str:
    """边的稳定 ID（uuid5，同样输入 → 同样 ID）。

    【调用链】被 add_edge() 调用

    【为什么用 uuid5 而不是 uuid4】
        重建图谱时，同一条边应该得到同一个 ID。
        这样前端缓存的节点位置不会因为重建而失效。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"devflow-graph:{from_type}:{from_id}:{relation}:{to_type}:{to_id}"))


def _path_node_id(path: str) -> str:
    """文件路径 → 代码节点 ID。

    【调用链】被 _code_path_to_node_id() / _collect_nodes() 调用

    加 "path:" 前缀：避免文件路径和某个 UUID 撞车（虽然概率极低，但前缀更稳）。
    """
    return f"path:{path}"


def _issue_refs(text: str) -> set[int]:
    """抽文本里所有 Issue 编号（#123 → 123）。

    【调用链】被 _add_text_reference_edges() / _generate_edges() 调用

    用 set：同一个编号出现多次只算一次。
    """
    return {int(match.group(1)) for match in ISSUE_REF_RE.finditer(text or "")}


def _closing_issue_refs(text: str) -> set[int]:
    """抽"关闭型"引用（fixes #123 / closes #45 / resolves #7）。

    【调用链】被 _generate_edges() 调用

    这类是强关系：PR 明确说"我修了它"。
    要和普通 mentions 区分开（前者是 resolves，后者是 mentions）。
    """
    return {int(match.group(1)) for match in CLOSING_ISSUE_RE.finditer(text or "")}


def _pr_refs(text: str) -> set[int]:
    """抽文本里的 PR 编号（PR #45 / pull request 45）。

    【调用链】被 _add_text_reference_edges() / _generate_edges() 调用
    """
    return {int(match.group(1)) for match in PR_REF_RE.finditer(text or "")}


def _node_priority(node_type: str) -> int:
    """节点类型的展示优先级（数字越小越靠前）。

    【调用链】被 _select_subgraph() 调用

    【为什么是这个顺序】
        Issue > PR > CI  → 协作事实，用户最常看
        人 > 记忆 > 决策  → 上下文
        文档 > 文件      → 资料（数量多，放后面）

    未知类型返回 20（排最后，但不至于丢）。
    """
    order = {
        "issue": 0,
        "pull_request": 1,
        "ci_run": 2,
        "team_member": 3,
        "memory_note": 4,
        "session_decision": 5,
        "document": 6,
        "code_file": 7,
    }
    return order.get(node_type, 20)


def _clip(text: str, limit: int) -> str:
    """按字符截断（图上节点标签用）。

    【调用链】被 _decision_nodes() 调用

    limit-1 是给 "..." 留位置。
    """
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."
