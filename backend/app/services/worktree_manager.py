"""
PR 分支快照（git worktree 生命周期）
=================================

【业务定位】
    PR 审查要读 **PR 分支的代码**，而不是主分支。
    本文件用 `git worktree` 为每个 PR 建一个独立工作目录，
    让"实时搜索源码"看到的是 PR 里的代码。

    【为什么不用 git checkout 切分支】
        主仓库正在被其他任务用（同步、图谱分析）。切分支会影响它们。
        `git worktree` 让同一个仓库同时挂多个工作目录 ——
        互不干扰，也不用复制整个 .git。

【数据流向】
    上游（谁调它）：
        services/agents/pr_review_agent.py（审查前准备快照）
        api/routes/pull_requests.py（用户点"查看 PR 代码"）
        services/scheduler.py（定期清理过期快照）
    下游（它调谁）：
        code_analysis._sync_checkout()   拿到主仓库路径
        code_analysis._sync_code_graph() 建 PR 维度的代码图谱
        code_analysis._git_text()        取 commit sha
        写：worktrees/<repo_id>/pr-<n>-<repo>/ 目录 + .devflow-pr-snapshot.json
        删：code_symbols / code_relations（该 PR 的旧图谱）

【核心规则】
    1. ★ **同一个 PR 不能并发建快照** —— 用每 (repo, pr) 一把锁串行化。
       （两个请求同时 git worktree add 会互相破坏）
    2. **失败必须降级，不能报错**：快照建不出来就继续用主分支代码，
       返回 status="fallback" 让调用方知道"看到的是主分支"。
    3. **清理要先让 git 删，再兜底 rmtree** —— 否则主仓库会留下失效的 worktree 记录。

【本文件函数地图】

    ┌─ 主流程 ────────────────────────────────────────
    │  prepare_pr_worktree()      ★ 建/更新 PR 快照 + 重建该 PR 的代码图谱
    │  cleanup_pr_snapshots()     ★ 清理过期快照 + 删对应图谱
    │
    ├─ 路径与元数据 ─────────────────────────────────
    │  pr_worktree_path()         算出快照目录路径
    │  _snapshot_meta_path()      元数据文件路径
    │  _write_snapshot_meta()     写元数据（含更新时间，清理时要用）
    │
    ├─ 内部工具 ─────────────────────────────────────
    │  _checkout_root()           系统代码根目录
    │  _snapshot_lock()           ★ 按 key 取锁（防并发）
    │  _run_git()                 跑 git（带超时，返回 stdout）
    │  _sync_pr_snapshot_analysis()  建 PR 维度图谱
    │  _snapshot_response()       统一返回结构
"""

import json
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository
from app.services.code_analysis import _git_text, _sync_checkout, _sync_code_graph

# ── 每 (repo, pr) 一把锁 ──
# 为什么需要：两个请求同时给同一个 PR 建 worktree，git 会报"目录已存在"或产生半成品。
# 为什么用 dict + guard lock：
#   dict 本身不是线程安全的，读写要加 guard；
#   但锁的粒度必须是"每个 PR 一把"，不能用一把全局锁（会串行化所有 PR）。
_SNAPSHOT_LOCKS: dict[str, threading.Lock] = {}
_SNAPSHOT_LOCKS_GUARD = threading.Lock()


def _checkout_root() -> Path:
    """系统管理代码的根目录（同 code_analysis._checkout_root，但返回 resolve 后的绝对路径）。

    【调用链】被 pr_worktree_path() / cleanup_pr_snapshots() 调用

    【和 code_analysis 版本的区别】
        这里多了一步 resolve()。因为后面要拼 worktrees/<id> 子目录，
        路径必须规范化（避免 ../ 出现在路径中间）。
    """
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _snapshot_lock(key: str) -> threading.Lock:
    """按 key 取一把锁（没有就创建）。

    【调用链】被 prepare_pr_worktree() 调用

    【★ 为什么取锁本身也要加 guard】
        `_SNAPSHOT_LOCKS` 是普通 dict，多线程同时 get/set 会出问题
        （可能创建出两把锁，导致互斥失效）。
        所以用一个全局 guard 保护"取锁"这个动作本身。

    【为什么锁不回收】
        锁对象很轻（一个布尔状态），几千个也不占多少内存。
        回收需要判断"没人用了"，容易出错。不回收更简单安全。
    """
    with _SNAPSHOT_LOCKS_GUARD:
        lock = _SNAPSHOT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SNAPSHOT_LOCKS[key] = lock
        return lock


def _run_git(args: list[str], cwd: Path) -> str:
    """跑 git 命令，返回 stdout（失败抛 CalledProcessError）。

    【调用链】被 prepare_pr_worktree() / cleanup_pr_snapshots() 调用

    【和 code_analysis._run_git 的区别】
        这个返回 stdout（需要 commit sha 之类的输出），
        那个丢弃 stdout（fetch/pull 的输出没用）。

    【参数】
        check=True     非 0 退出码抛异常（调用方用 try 包住做降级）
        stderr=PIPE    错误信息留着（调用方会拼进 message）
        timeout=180    克隆/切分支可能慢，3 分钟上限
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    return result.stdout.strip()


def pr_worktree_path(repo: Repository, pr: PullRequest) -> Path:
    """算出某个 PR 的快照目录路径。

    【调用链】_checkout_root() → 本函数
    【被谁调用】prepare_pr_worktree()

    【目录结构】
        .data/repos/worktrees/<repo_id>/pr-<number>-<owner>__<name>/

    【为什么带 repo_id 一级目录】
        清理时可以按仓库整目录扫（cleanup_pr_snapshots 就是这么做的）。

    【为什么目录名带 PR number】
        人工排查时一眼能看出这是哪个 PR 的快照。
    """
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / "worktrees" / str(repo.id) / f"pr-{pr.number}-{safe_name}"


def _snapshot_meta_path(path: Path) -> Path:
    """元数据文件路径（放在快照目录里面）。

    【调用链】被 _write_snapshot_meta() / cleanup_pr_snapshots() 调用

    【为什么放在快照目录里而不是数据库】
        快照是"文件系统状态"，元数据和它同生共死最直观。
        目录被手动删掉，元数据也一起没了 —— 不会出现"库里有记录但目录不在"。
    """
    return path / ".devflow-pr-snapshot.json"


def _write_snapshot_meta(path: Path, meta: dict[str, Any]) -> None:
    """写快照元数据（自动补 updated_at）。

    【调用链】_snapshot_meta_path() → 本函数
    【被谁调用】prepare_pr_worktree()

    【★ updated_at 是关键字段】
        清理逻辑靠它判断"这个快照多久没用了"。
        每次 prepare 都会刷新它 —— 相当于"最近有人用过，别清"。
    """
    payload = {**meta, "updated_at": datetime.now(timezone.utc).isoformat()}
    _snapshot_meta_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_pr_snapshot_analysis(
    db: Session,
    repo: Repository,
    pr: PullRequest,
    snapshot_path: Path,
    branch: str,
    commit_sha: str | None,
) -> int:
    """为这个 PR 快照建"PR 维度"的代码图谱。

    【调用链】code_analysis._sync_code_graph() → 本函数
    【被谁调用】prepare_pr_worktree()

    【★ 关键：传 pr_id / pr_number】
        这让写进 code_symbols 的行带上 PR 标记。
        效果：主分支图谱和 PR 图谱共存，互不覆盖
        （_sync_code_graph 删除时是按 branch + pr_id 过滤的）。
        所以"查这个 PR 改了哪些符号"能直接查出来。

    【返回】写入的符号数（给前端展示"索引了多少符号"）
    """
    return _sync_code_graph(
        db,
        repo,
        snapshot_path,
        branch=branch,
        commit_sha=commit_sha,
        pr_id=pr.id,
        pr_number=pr.number,
    )


def _snapshot_response(
    *,
    status: str,
    message: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    pr: PullRequest,
    snapshot_path: Path | None = None,
    indexed_symbols: int | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    """统一返回结构（★ 五种 status 的语义都靠这个函数表达）。

    【调用链】被 prepare_pr_worktree() 调用（5 处）

    【数据流向】
        入：status、message、branch、commit_sha、pr、snapshot_path、indexed_symbols、fallback
        出：dict

    【五种 status】
        ready          快照就绪，用的是 PR 分支代码
        unsupported    本地仓库模式，不支持自动建快照
        missing_head   PR 没有 head_branch，只能用主分支
        failed         git 不可用
        fallback       建快照过程中出错，降级到主分支

    【★ checkout_kind 这个字段的意义】
        只有 ready 才是 "pr_snapshot"，其他都是 "default_checkout"。
        调用方（PR 审查 Agent）据此知道"我看到的代码是不是 PR 的"。
        如果不是，就要在结论里声明"基于主分支分析" —— 这是诚实性要求。
    """
    return {
        "status": status,
        "message": message,
        "snapshot_name": "PR 分支快照",
        "branch": branch,
        "commit_sha": commit_sha,
        "pr_id": str(pr.id),
        "pr_number": pr.number,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "indexed_symbols": indexed_symbols,
        "fallback": fallback,
        "checkout_kind": "pr_snapshot" if status == "ready" else "default_checkout",
    }


def prepare_pr_worktree(db: Session, repo: Repository, pr: PullRequest) -> dict[str, Any]:
    """准备 PR 分支快照，并刷新它的代码图谱（★ 主流程）。

    【调用链】追这个函数会进入：
        1. _snapshot_lock()                  → 拿该 PR 的锁（防并发）
        2. decrypt_token()                   → 解密 GitHub token
        3. code_analysis._sync_checkout()    → 确保主仓库存在
        4. _run_git(fetch/checkout/worktree)→ 建/更新快照
        5. code_analysis._git_text()         → 取 commit sha
        6. _sync_pr_snapshot_analysis()      → 建 PR 维度图谱
        7. _write_snapshot_meta()            → 写元数据
    【被谁调用】services/agents/pr_review_agent.py、api/routes/pull_requests.py

    【数据流向】
        入：db、repo、pr
        出：_snapshot_response 的 dict

    Prepare a PR branch snapshot for live search and refresh its code graph.

    【★ 为什么整个函数体都在 with lock 里】
        git worktree add 不是原子操作。两个并发请求会：
            ① 都判断"目录不存在" → 都去 add → 一个失败
            ② 或者产生半成品目录（.git 文件写了一半）
        加锁后同一 PR 串行，问题消失。

    【★ 四道前置检查，全部"降级不报错"】
        ① 本地模式 → unsupported（用户自己的目录不该被我们动）
        ② 没有 head_branch → missing_head（没有分支就无从建快照）
        ③ 没有 git → failed
        ④ 其他异常 → fallback（用完主分支代码继续）

        为什么都不抛异常：PR 审查的核心价值是"给建议"，
        不是"必须有 PR 分支代码"。降级后照样能审（只是基于主分支）。

    【★ 两种 git 路径（已有目录 vs 新建）】
        已有目录：
            fetch origin <ref>            拉最新
            checkout -B <local> FETCH_HEAD 强制切到该提交
            —— 复用目录，快（不用重新 checkout 全部文件）

        新建：
            worktree add --force -B <local> <target> FETCH_HEAD
            —— --force 允许目标目录已存在（虽然前面判过，但防竞态）
            —— -B 会重置分支（避免"分支已存在"报错）

    【★ 为什么本地分支名要加 devflow/ 前缀】
        避免和用户的真实分支重名。
        `devflow/pr-42` 一看就是系统建的临时分支。
    """
    # ── 步骤 1：拿锁（同一个 PR 串行）──
    lock_key = f"{repo.id}:{pr.id}"
    lock = _snapshot_lock(lock_key)
    with lock:
        # ── 前置检查 ①：本地模式不支持（不动用户的目录）──
        if (repo.checkout_mode or "managed") == "local":
            return _snapshot_response(status="unsupported", message="本地仓库模式不自动创建或删除 PR 分支快照。", pr=pr, fallback=repo.local_path)

        # ── 前置检查 ②：PR 没有源分支 ──
        if not pr.head_branch:
            return _snapshot_response(status="missing_head", message="PR 缺少 head branch，继续使用默认分支代码。", pr=pr)

        # ── 前置检查 ③：没有 git ──
        if not shutil.which("git"):
            return _snapshot_response(status="failed", message="未找到 git，可继续使用默认分支代码。", pr=pr)

        token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
        target = pr_worktree_path(repo, pr)

        try:
            # ── 步骤 2：确保主仓库存在（没有就克隆）──
            base_checkout = _sync_checkout(repo, token)
            target.parent.mkdir(parents=True, exist_ok=True)

            ref = pr.head_branch
            local_branch = f"devflow/pr-{pr.number}"

            # ── 步骤 3：拉 PR 分支的最新提交 ──
            _run_git(["fetch", "origin", ref], cwd=base_checkout)

            if target.exists():
                # 快照已存在 → 更新到最新提交
                _run_git(["fetch", "origin", ref], cwd=target)
                _run_git(["checkout", "-B", local_branch, "FETCH_HEAD"], cwd=target)
            else:
                # 新建 worktree（从主仓库分出）
                _run_git(["worktree", "add", "--force", "-B", local_branch, str(target), "FETCH_HEAD"], cwd=base_checkout)

            # ── 步骤 4：取 commit sha（记录"这次快照是哪个提交"）──
            commit_sha = _git_text(["rev-parse", "HEAD"], target)

            # ── 步骤 5：建 PR 维度的代码图谱 ──
            indexed_symbols = _sync_pr_snapshot_analysis(db, repo, pr, target, ref, commit_sha)

            # ── 步骤 6：写元数据（updated_at 供清理判断）──
            _write_snapshot_meta(
                target,
                {
                    "repo_id": str(repo.id),
                    "pr_id": str(pr.id),
                    "pr_number": pr.number,
                    "branch": ref,
                    "commit_sha": commit_sha,
                    "indexed_symbols": indexed_symbols,
                    "snapshot_kind": "pr_snapshot",
                },
            )
            db.commit()
            return _snapshot_response(
                status="ready",
                branch=ref,
                commit_sha=commit_sha,
                pr=pr,
                snapshot_path=target,
                indexed_symbols=indexed_symbols,
            )
        except Exception as exc:
            # ── 任何失败 → 回滚 + 降级（不抛异常）──
            db.rollback()
            return _snapshot_response(
                status="fallback",
                message=f"PR 分支快照准备失败，继续使用默认分支代码：{exc}",
                branch=repo.default_branch or "main",
                pr=pr,
            )


def cleanup_pr_snapshots(db: Session, repo: Repository, *, max_age_hours: int = 168) -> dict[str, Any]:
    """清理过期的 PR 快照（默认 7 天没用就删），并删掉对应的图谱。

    【调用链】_snapshot_meta_path() / _run_git() / _checkout_root() → 本函数
    【被谁调用】services/scheduler.py（定期清理）、api/routes/pull_requests.py（手动清理）

    【数据流向】
        入：db、repo、max_age_hours（默认 168 小时 = 7 天）
        出：{"removed": 删了几个, "paths": 删了哪些目录}

    【★ 判断"过期"的两级回退】
        ① 元数据里的 updated_at（每次 prepare 都刷新）
        ② 元数据损坏/不存在 → 用目录的 mtime
        为什么要有 ②：老快照可能没有元数据文件（早期版本没写），
        不兜底就永远清不掉。

    【★ 为什么先 `git worktree remove` 再 rmtree】
        直接 rmtree 会留下问题：主仓库的 .git/worktrees/ 里还有这个 worktree 的记录，
        下次 `git worktree list` 会显示一个失效路径。
        先让 git 自己删，它会同步清理记录。

    【★ 为什么 rmtree 还要 ignore_errors=True 兜底】
        git 删失败（比如目录被占用）时，至少把目录强删掉。
        宁可留一条失效的 worktree 记录，也不要留下几百 MB 的垃圾目录。

    【★ 为什么要删 code_symbols / code_relations】
        快照删了，但图谱还在库里。
        不删的话，检索会命中"已经不存在的 PR 的符号"，给出错误的位置信息。
    """
    root = _checkout_root() / "worktrees" / str(repo.id)
    # 目录不存在 → 没有快照可清
    if not root.exists():
        return {"removed": 0, "paths": []}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed_paths: list[str] = []
    removed_pr_ids: set[uuid.UUID] = set()

    for path in root.iterdir():
        if not path.is_dir():
            continue

        # ── 读元数据（拿 updated_at 和 pr_id）──
        meta_path = _snapshot_meta_path(path)
        updated_at = None
        snapshot_pr_id: uuid.UUID | None = None
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                updated_raw = metadata.get("updated_at")
                updated_at = datetime.fromisoformat(updated_raw) if updated_raw else None
                if metadata.get("pr_id"):
                    snapshot_pr_id = uuid.UUID(str(metadata["pr_id"]))
            except (OSError, ValueError, json.JSONDecodeError):
                # 元数据坏了 → 后面用目录 mtime 兜底
                updated_at = None

        # ── 兜底：用目录修改时间 ──
        if updated_at is None:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        # 还没过期 → 跳过
        if updated_at >= cutoff:
            continue

        # ── 删除目录（先让 git 删，失败再 rmtree）──
        try:
            # Remove from Git first so the main checkout does not keep stale worktree records.
            # 主仓库目录名格式：<repo_id>-<owner>__<name>，用 glob 找到它
            main_checkout = next((item for item in _checkout_root().glob(f"{repo.id}-*") if item.is_dir()), None)
            if main_checkout:
                _run_git(["worktree", "remove", "--force", str(path)], cwd=main_checkout)
            elif path.exists():
                shutil.rmtree(path)
        except Exception:
            # git 删失败 → 强删目录（宁可留失效记录，也不留垃圾）
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

        removed_paths.append(str(path))
        if snapshot_pr_id:
            removed_pr_ids.add(snapshot_pr_id)

    # ── 删掉这些 PR 的代码图谱 ──
    if removed_pr_ids:
        db.query(CodeRelation).filter(
            CodeRelation.repo_id == repo.id,
            CodeRelation.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
        db.query(CodeSymbol).filter(
            CodeSymbol.repo_id == repo.id,
            CodeSymbol.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
    db.commit()
    return {"removed": len(removed_paths), "paths": removed_paths}
