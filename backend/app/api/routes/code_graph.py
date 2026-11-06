"""
代码图谱路由
============

【路由清单】
    POST /repos/{repo_id}/code-graph/rebuild        重建代码图谱（克隆 + AST 解析 + 落库）
    GET  /repos/{repo_id}/code-symbols              按路径 / PR 列符号（函数、类）
    GET  /repos/{repo_id}/code-graph/search         搜符号 + 关联关系 + 代码片段

【数据流向】
    rebuild
      → services/code_analysis.py::sync_repository_code_analysis()
      → git clone / worktree → AST 解析 → code_symbols + code_relations 表

    search
      → code_symbols 表（按 name / path 模糊匹配）
      → code_relations 表（谁调用了谁）
      → services/code_search.py::read_code_excerpt()  读真实代码片段
      → services/worktree_manager.py::pr_worktree_path()  优先读 PR 快照

【要点】
    1. 三个接口都要 repo:read 权限（除了 rebuild 也是 read —— 重建只读 GitHub 不改远端）。
    2. search 优先读 **PR worktree 快照**：用户看 PR 时，要看到的是"这个 PR 里的代码"，
       而不是主分支的代码。没有快照才退回主仓库 checkout。
    3. relations 的查询同时按 id 和 name 匹配 —— 因为有些关系（比如"调用了外部库函数"）
       只有 target_name，没有 target_symbol_id。
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.code_graph import CodeGraphSearchResponse, CodeSymbolResponse
from app.services.code_analysis import sync_repository_code_analysis
from app.services.code_search import read_code_excerpt, repository_checkout_path
from app.services.permissions import require_permission
from app.services.worktree_manager import pr_worktree_path

router = APIRouter()


@router.post("/repos/{repo_id}/code-graph/rebuild")
async def rebuild_code_graph(
    repo_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    """重建代码图谱。同步执行，仓库大时会慢（会克隆代码 + 全量 AST 解析）。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    symbol_count = await sync_repository_code_analysis(db, repo)
    return {"repo_id": str(repo.id), "indexed_symbols": symbol_count}


@router.get("/repos/{repo_id}/code-symbols", response_model=list[CodeSymbolResponse])
async def list_code_symbols(
    repo_id: UUID,
    path: str | None = None,
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[CodeSymbol]:
    """列符号。path 用 contains 做子串匹配（前端传目录前缀即可）。"""
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    if path:
        query = query.filter(CodeSymbol.path.contains(path))
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    # 按路径 + 行号排序：前端按文件分组展示时顺序稳定
    return query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()


@router.get("/repos/{repo_id}/code-graph/search", response_model=CodeGraphSearchResponse)
async def search_code_graph(
    repo_id: UUID,
    q: str = Query(default=""),
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    """搜符号：返回符号列表 + 关系边 + 代码片段（三段一起给前端画图）。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    text = q.strip()
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    # PR 维度过滤：看 PR 时只关心这个 PR 涉及的符号
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    # q 为空时返回该仓库前 N 个符号（相当于"浏览模式"）
    if text:
        query = query.filter(or_(CodeSymbol.name.contains(text), CodeSymbol.path.contains(text)))
    symbols = query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()

    # ── 关系边：找出和这些符号相关的所有边 ──
    symbol_ids = [item.id for item in symbols]
    relations = []
    if symbol_ids:
        rows = (
            db.query(CodeRelation)
            .filter(
                CodeRelation.repo_id == repo_id,
                # 这两个是"条件生效"写法：pr_id 为空时传 True，相当于不加这个条件
                CodeRelation.pr_id == pr_id if pr_id else True,
                CodeRelation.pr_number == pr_number if pr_number else True,
                # 三种命中方式：作为源、作为目标、或按名字命中（外部符号没有 id）
                or_(CodeRelation.source_symbol_id.in_(symbol_ids), CodeRelation.target_symbol_id.in_(symbol_ids), CodeRelation.target_name.in_([item.name for item in symbols])),
            )
            # 边数比点多数倍，多取一些避免图断边
            .limit(limit * 4)
            .all()
        )
        relations = [
            {
                "id": str(row.id),
                "source_symbol_id": str(row.source_symbol_id) if row.source_symbol_id else None,
                "target_symbol_id": str(row.target_symbol_id) if row.target_symbol_id else None,
                "source_name": row.source_name,
                "target_name": row.target_name,
                "relation_type": row.relation_type,
                "path": row.path,
                "branch": row.branch,
                "pr_id": str(row.pr_id) if row.pr_id else None,
                "pr_number": row.pr_number,
                "metadata": row.meta or {},
            }
            for row in rows
        ]

    # ── 代码片段：优先读 PR 快照，没有才读主仓库 ──
    documents = []
    checkout_path = repository_checkout_path(repo)
    selected_pr = None
    if pr_id:
        selected_pr = db.get(PullRequest, pr_id)
    elif pr_number:
        selected_pr = (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == repo_id, PullRequest.number == pr_number)
            .one_or_none()
        )
    if selected_pr and selected_pr.repo_id == repo_id:
        snapshot_path = pr_worktree_path(repo, selected_pr)
        if snapshot_path.exists():
            # 有 PR worktree 就用它：用户要看的是"这个 PR 的代码"
            checkout_path = snapshot_path
    if checkout_path.exists():
        for symbol in symbols[:limit]:
            # 只读 7 行：符号起始行 + 6 行，够看出签名和开头逻辑
            snippet = read_code_excerpt(checkout_path, symbol.path, symbol.start_line, line_count=7)
            if not snippet:
                # 文件在本地不存在（比如 PR 新增文件还没同步）—— 跳过，不报错
                continue
            documents.append(
                {
                    "id": f"workspace:{symbol.path}:{symbol.start_line}",
                    "title": f"{symbol.path}:{symbol.start_line}",
                    "source_id": symbol.path,
                    "path": symbol.path,
                    "symbol": symbol.name,
                    "start_line": symbol.start_line,
                    "end_line": symbol.start_line + 6,
                    "pr_id": str(selected_pr.id) if selected_pr else None,
                    "pr_number": selected_pr.number if selected_pr else None,
                    "snippet": snippet[:1200],
                }
            )
    return {"query": text, "symbols": symbols, "relations": relations, "documents": documents}
