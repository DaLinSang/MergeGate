"""
仓库代码同步与代码图谱分析
=========================

【业务定位】
    数据流 2（仓库代码 → 代码图谱）的**全部实现**，包含两件事：
        1. 把代码弄到本地（克隆 / 拉取 / 直接用用户的本地目录）
        2. 扫描代码，抽出"符号"（函数/类）和"关系"（import/call），写进 code_symbols / code_relations

    【为什么源码不进 RAG，而要走代码图谱】
        见 rag/source_policy.py 末尾的论证。一句话：
            源码有精确结构（文件+行号），用 AST/正则直接定位比"语义猜"准得多，
            而且代码频繁修改，向量化容易过期。
        所以本项目是两条路：
            源码     → 本文件（代码图谱 + 实时文件读取）
            文档/Issue → RAG 向量库

【数据流向】
    上游（谁调它）：
        api/routes/repos.py（连接仓库后触发）
        services/scheduler.py（定时同步）
        project_indexing.py（复用 _repo_checkout_path / _stored_checkout_path 取路径）
    下游（它写谁）：
        code_symbols 表    ← _sync_code_graph()
        code_relations 表  ← _sync_code_graph()
        documents 表       ← purge_legacy_code_documents()（清理历史遗留的 code_file）

【核心规则】
    1. ★ **token 绝不能出现在日志/错误信息里** —— 所以它被塞进 clone URL 的 userinfo 部分，
       且所有 git 命令的 stderr 都要小心处理（见 _clone_url_with_token 的注释）。
    2. **整组替换**：每次同步先删该分支的旧符号/旧关系，再全量重建。
       （代码是"全量快照"语义，增量更新会留下已删除的函数）
    3. **只扫文本类文件**：二进制/超大文件直接跳过，避免读进内存。
    4. 同步失败不抛异常（后台任务），只记 warning。

【本文件函数地图】

    ┌─ 一、代码怎么到本地（git 相关）────────────────────
    │  _checkout_root()            系统管理目录（.data/repos）
    │  _repo_checkout_path()       按仓库算出目标目录
    │  _stored_checkout_path()     用户配置的本地路径
    │  _clone_url()                取/拼 clone 地址
    │  _clone_url_with_token()     ★ 把 token 注入 URL（安全关键）
    │  _run_git()                  跑 git 命令（统一超时与错误处理）
    │  _is_git_work_tree()         判断目录是不是 git 工作树
    │  _sync_checkout()            ★ 三种模式：用本地 / 更新已有 / 全新克隆
    │
    ├─ 二、文件筛选与读取 ─────────────────────────────
    │  _is_text_candidate()        是不是"可能是文本"的文件
    │  _read_text()                读文件（大小/二进制/多编码）
    │  _language_for()             从后缀推语言
    │  _git_text()                 跑 git 取一段输出（取分支名/commit）
    │  _line_for_offset()          字符偏移 → 行号
    │
    ├─ 三、符号与关系抽取 ─────────────────────────────
    │  _symbol_matches()           ★ 按语言抽符号（Python / TS 系）
    │  _sync_code_graph()          ★ 两遍扫描：先建符号，再建关系
    │
    └─ 四、对外入口 ───────────────────────────────────
       purge_legacy_code_documents()   清理历史遗留的 code_file 文档
       sync_repository_code_analysis()     ★ 主流程（async）
       sync_repository_code_analysis_by_id()  后台任务入口
"""

import asyncio
import logging
import os
import shutil
import subprocess
import uuid
import re
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, Document, Repository
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# 扫描时跳过的目录（生成物/依赖，没有分析价值且体积大）
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# 认为是"文本"的扩展名（能安全 decode 成字符串）
TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".md",
    ".mdx",
    ".txt",
    ".dockerfile",
}

# 没有扩展名但肯定是文本的文件
TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
}

# ── 三个正则：符号 / 调用 / import ──
# Python 符号：行首的 def 或 class（MULTILINE 让 ^ 匹配每行开头）
PY_SYMBOL_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

# TS/JS 符号：允许 export / async 前缀，覆盖 function/class/interface/type/const/let
TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)

# 调用：任何 "标识符(" 都算（会有大量误报，后面用黑名单过滤）
CALL_RE = re.compile(r"\b([A-Za-z_][$A-Za-z0-9_]*)\s*\(")

# import：兼容 Python 的 from X import / import X，以及 JS 的 import ... from "x"
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.\/-]+)\s+import|import\s+([\w.\/-]+)|import\s+.*?\s+from\s+['\"]([^'\"]+)['\"])", re.MULTILINE)


# ════════════════════════════════════════════════════════════════
# 一、代码怎么到本地
# ════════════════════════════════════════════════════════════════

def _checkout_root() -> Path:
    """系统管理代码的根目录（默认 .data/repos）。

    【调用链】被 _repo_checkout_path() 调用

    【为什么要把相对路径转绝对】
        settings.repo_checkout_dir 默认是相对路径（.data/repos）。
        相对谁？取决于进程启动目录 —— 这很不可靠。
        统一转成基于 cwd 的绝对路径，行为才确定。

    【为什么 mkdir 在这里做】
        调用方（克隆）一定需要这个目录存在。放在这里保证"拿到根目录就能用"。
    """
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_checkout_path(repo: Repository) -> Path:
    """算出某个仓库的系统管理目录。

    【调用链】_checkout_root() → 本函数
    【被谁调用】_sync_checkout() / project_indexing._repo_root()

    【目录名格式：{repo.id}-{owner}__{name}】
        为什么要带 id：同一个 full_name 可能被删了重新连接（id 不同），
        带上 id 就不会互相覆盖。
        为什么要把 "/" 换成 "__"：斜杠在路径里是分隔符，不能直接用。
    """
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / f"{repo.id}-{safe_name}"


def _stored_checkout_path(repo: Repository) -> Path | None:
    """取用户配置的本地路径（repo.local_path），没有则返回 None。

    【调用链】被 _sync_checkout() / project_indexing._repo_root() 调用

    【为什么要 expandvars + expanduser】
        用户可能写 "~/code/my-repo" 或 "$HOME/code/repo"。
        不展开的话 Path("~/...") 会被当成字面量目录名（含波浪号），永远不存在。

    【为什么要 resolve】
        解析掉 ../ 和符号链接，得到规范路径。后面判断"是否在仓库内"依赖它。
    """
    raw_path = getattr(repo, "local_path", None)
    if not raw_path:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _clone_url(repo: Repository) -> str:
    """取 clone 地址（没配就按 GitHub 默认规则拼）。

    【调用链】被 _clone_url_with_token() / sync_repository_code_analysis() 调用
    """
    if repo.clone_url:
        return repo.clone_url
    return f"https://github.com/{repo.owner}/{repo.name}.git"


def _clone_url_with_token(url: str, token: str | None) -> str:
    """把 token 注入 clone URL 的 userinfo 部分（★ 安全关键函数）。

    【调用链】被 _sync_checkout() 调用

    【数据流向】
        入：url（如 https://github.com/o/r.git）、token
        出：https://x-access-token:TOKEN@github.com/o/r.git（有 token 时）

    【为什么不用 git 的 credential helper / --config】
        clone 命令里直接带 URL 是最简单可靠的方式，且不需要写临时文件
        （写临时文件有泄露风险，忘了删就留在磁盘上）。

    【三道保护，一个都不能少】
        ① token 为空 → 原样返回（公开仓库不需要认证）
        ② scheme 不是 http/https → 原样返回
           （SSH 地址 git@github.com:o/r.git 不能用这种方式注入 token）
        ③ netloc 里已经有 "@" → 原样返回
           （URL 里已经带了凭证，再拼会变成两个 @，URL 非法）

    【★ 为什么用 quote(token, safe="")]
        token 里可能含特殊字符（@ : / 等）。不转义的话会把 URL 结构弄坏。
        safe="" 表示"全部转义"，最保险。

    【★ 为什么用户名固定写 x-access-token】
        这是 GitHub 对"用 token 代替密码"的约定写法。
        写别的名字（如 git）在部分场景下会认证失败。
    """
    if not token:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or "@" in parsed.netloc:
        return url
    safe_token = quote(token, safe="")
    return urlunparse(parsed._replace(netloc=f"x-access-token:{safe_token}@{parsed.netloc}"))


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    """跑一条 git 命令（失败抛 CalledProcessError）。

    【调用链】被 _sync_checkout() 调用（fetch / checkout / pull / clone）

    【参数设计】
        check=True              → 非 0 退出码抛异常（不静默失败）
        stdout=DEVNULL          → git 的正常输出没用，丢掉（避免刷日志）
        stderr=PIPE             → 错误信息要留着（调用方会读来打日志）
        text=True               → 返回字符串而不是 bytes
        timeout=180             → 克隆大仓库可能慢，但 3 分钟足够；超时防止卡死

    【★ 注意：stderr 里可能含带 token 的 URL】
        git 报错时经常把完整命令（含 URL）打出来。
        所以调用方在记录日志时只能记 exc.stderr，且不该把它暴露给前端。
    """
    subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )


def _is_git_work_tree(path: Path) -> bool:
    """判断一个目录是不是 git 工作树。

    【调用链】被 _sync_checkout() 调用

    【为什么不用 "检查 .git 目录存在"】
        子模块、worktree、bare 仓库的情况各不相同。
        `git rev-parse --is-inside-work-tree` 是官方判断方式，最准。

    【三重前置检查，避免无谓的子进程开销】
        ① 路径不存在
        ② 不是目录
        ③ 系统里没有 git
    """
    if not path.exists() or not path.is_dir() or not shutil.which("git"):
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(path),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _sync_checkout(repo: Repository, token: str | None) -> Path:
    """把代码同步到本地，返回可用的目录路径（★ 三种模式）。

    【调用链】追这个函数会进入：
        _stored_checkout_path() → 判断是否用户本地目录
        _is_git_work_tree()     → 校验本地目录
        _clone_url_with_token() → 拼认证 URL
        _run_git()              → 执行 fetch / checkout / pull / clone
    【被谁调用】sync_repository_code_analysis()（经 asyncio.to_thread 包成异步）

    【数据流向】
        入：repo、token（已解密的明文）
        出：Path（本地可读的代码目录）

    【三种模式（按优先级）】
        ① local 模式（checkout_mode="local"）：用户指定的目录，只校验不克隆
           —— 适合"我本地已经有代码，别给我复制一份"
        ② 目录已存在且是 git 工作树：fetch + checkout + pull（增量更新）
           —— 失败只 warning，继续用缓存（★ 见下方说明）
        ③ 全新：git clone --depth 1（浅克隆，只取最新一次提交）

    【★ 为什么增量更新失败不抛异常】
        "代码旧一点"比"整个同步失败"好得多。
        网络抖动、force push 导致 ff-only 失败都很常见，
        这时用本地已有代码继续建图谱，用户至少还能用。

    【★ 为什么 clone 用 --depth 1】
        全量历史可能几百 MB，而我们只需要"当前代码长什么样"。
        浅克隆只取最新提交，快 10 倍以上。

    【★ 为什么已有目录要 rmtree】
        目录存在但不是 git 工作树 → 说明是上次失败留下的残骸。
        不删掉的话 clone 会因为"目录非空"失败。
        但如果 stored_path 是用户配置的目录且非空 → 拒绝删（保护用户数据）。
    """
    stored_path = _stored_checkout_path(repo)
    checkout_mode = getattr(repo, "checkout_mode", None) or "managed"

    # ── 模式 ①：用户本地目录（只校验，不动它）──
    if stored_path and checkout_mode == "local":
        if not _is_git_work_tree(stored_path):
            raise RuntimeError(f"已配置的本地仓库路径不是 git 工作树：{stored_path}")
        return stored_path

    target = stored_path or _repo_checkout_path(repo)
    clone_url = _clone_url_with_token(_clone_url(repo), token)
    branch = repo.default_branch or "main"

    if not shutil.which("git"):
        raise RuntimeError("未找到 git 可执行文件，无法克隆仓库代码")

    # ── 模式 ②：已有 git 工作树 → 增量更新 ──
    if (target / ".git").exists() or _is_git_work_tree(target):
        try:
            _run_git(["fetch", "--depth", "1", "origin", branch], cwd=target)
            _run_git(["checkout", branch], cwd=target)
            # --ff-only：只允许快进合并。如果本地有分叉就失败（不强行覆盖）
            _run_git(["pull", "--ff-only", "origin", branch], cwd=target)
        except subprocess.CalledProcessError as exc:
            # ★ 失败不抛：用缓存代码继续（旧数据比没数据好）
            # 注意这里只记 stderr，且 stderr 可能含 token —— 只进日志，不返回给用户
            logger.warning("更新现有 checkout %s 失败，将使用缓存文件：%s", target, (exc.stderr or exc).strip())
        return target

    # ── 模式 ③：全新克隆 ──
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # ★ 保护用户数据：用户配置的目录且非空 → 拒绝删
        if stored_path and any(target.iterdir()):
            raise RuntimeError(f"Configured download target already exists and is not an empty git work tree: {target}")
        # 上次失败留下的残骸 → 删掉
        shutil.rmtree(target)

    clone_args = ["clone", "--depth", "1"]
    if branch:
        clone_args.extend(["--branch", branch])
    clone_args.extend([clone_url, str(target)])
    _run_git(clone_args)
    return target


# ════════════════════════════════════════════════════════════════
# 二、文件筛选与读取
# ════════════════════════════════════════════════════════════════

def _is_text_candidate(path: Path) -> bool:
    """判断文件是否"可能是文本"（值得读进来分析）。

    【调用链】被 _sync_code_graph() 调用

    【两道判断】
        ① 文件名在白名单里（Dockerfile / Makefile 这些没扩展名）
        ② 扩展名在白名单里，或者名字以 dockerfile 结尾（Dockerfile.prod）
    """
    if path.name in TEXT_FILENAMES:
        return True
    suffix = path.suffix.lower()
    return suffix in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 300_000) -> str | None:
    """读文本文件（同 project_indexing._read_text 的逻辑，上限更小）。

    【调用链】被 _sync_code_graph() 调用

    【为什么这里的上限是 300KB 而不是 400KB】
        代码文件通常比文档小。单文件 300KB 已经很夸张（约 8000 行），
        再大的文件多半是生成代码，分析它没意义。

    【四道检查】OSError / 空 / 超大 / 前 4KB 有 \\x00（二进制）
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _language_for(path: Path) -> str:
    """从后缀推断语言名（就用后缀，如 "py" / "ts"）。

    【调用链】被 _symbol_matches() 调用

    没有后缀时用文件名（如 "Dockerfile"）—— 至少能区分开。
    """
    suffix = path.suffix.lower().lstrip(".")
    return suffix or path.name


def _git_text(args: list[str], cwd: Path) -> str | None:
    """跑 git 命令并返回 stdout（失败返回 None，不抛异常）。

    【调用链】被 sync_repository_code_analysis() 调用（取分支名 / commit sha）

    【为什么这里不抛异常】
        这两个值都是"锦上添花"：
        分支名取不到可以用 repo.default_branch 兜底；
        commit sha 取不到就是 None（图谱少一个版本标记而已）。
        不该因为取不到就让整个同步失败。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _line_for_offset(text: str, offset: int) -> int:
    """字符偏移 → 行号（1-based）。

    【调用链】被 _symbol_matches() / _sync_code_graph() 调用

    数 offset 之前有几个换行，再 +1（因为行号从 1 开始）。
    """
    return text.count("\n", 0, offset) + 1


# ════════════════════════════════════════════════════════════════
# 三、符号与关系抽取
# ════════════════════════════════════════════════════════════════

def _symbol_matches(path: Path, text: str) -> list[dict[str, object]]:
    """按语言抽取符号（函数 / 类 / 接口）。

    【调用链】_language_for() / _line_for_offset() → 本函数
    【被谁调用】_sync_code_graph()

    【数据流向】
        入：path（判断语言）、text（源码）
        出：[{name, kind, language, start_line}] 列表

    【只支持 Python 和 TS/JS 的原因】
        这两种用正则能抽得比较准。
        其他语言（Java/Go/Rust）在 rag/chunking.py 的 _generic_symbol() 里有更完整的正则表，
        但本文件只做"图谱"用的最小实现 —— 覆盖主力语言即可。

    【★ 为什么不用 AST】
        Python 有 ast 模块，理论上更准。但这里要同时处理多种语言，
        统一用正则实现更简单，且"抽出符号名+行号"这个精度要求不高。
        （rag/chunking.py 里的 chunk_code_text 才用 AST，因为那里需要精确的代码块边界）

    【TS 的 kind 判定为什么看声明文本】
        正则把 function/class/interface/type/const 都匹配进同一个分组了，
        所以要回看 match.group(0)（完整声明文本）来判断具体是哪种。
    """
    language = _language_for(path)
    matches = []

    if path.suffix.lower() == ".py":
        for match in PY_SYMBOL_RE.finditer(text):
            # group(1) 是 def/class，group(2) 是名字
            matches.append({"name": match.group(2), "kind": match.group(1), "language": language, "start_line": _line_for_offset(text, match.start())})

    elif path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        for match in TS_SYMBOL_RE.finditer(text):
            declaration = match.group(0)
            # 从完整声明里判断具体类型
            kind = "class" if "class " in declaration else "interface" if "interface " in declaration else "type" if "type " in declaration else "function"
            matches.append({"name": match.group(1), "kind": kind, "language": language, "start_line": _line_for_offset(text, match.start())})

    return matches


def _sync_code_graph(
    db: Session,
    repo: Repository,
    checkout_path: Path,
    *,
    branch: str | None,
    commit_sha: str | None,
    pr_id: uuid.UUID | None = None,
    pr_number: int | None = None,
    max_files: int = 500,
) -> int:
    """扫描代码目录，重建符号表和关系表（★ 本文件的核心）。

    【调用链】追这个函数会进入：
        1. 删除该 (repo, branch, pr) 的旧符号/旧关系
        2. 第一遍：_is_text_candidate() + _read_text() + _symbol_matches() → 建符号
        3. db.flush() 拿符号 id
        4. 第二遍：IMPORT_RE / CALL_RE → 建关系
    【被谁调用】sync_repository_code_analysis()（经 asyncio.to_thread）

    【数据流向】
        入：db、repo、checkout_path、branch、commit_sha、pr_id/pr_number（PR 维度图谱）
        出：写入的符号数（int）

    【★ 为什么是"两遍扫描"】
        建关系时需要 target_symbol_id，但那时符号可能还没建出来。
        所以第一遍先把所有符号建好、flush 拿到 id，
        第二遍再建关系时就能按名字查到 id（by_name 字典）。

    【★ 为什么先删后建（整组替换）】
        代码是"全量快照"：删掉的函数必须从图谱里消失。
        增量更新无法感知"删除"，会留下永远不存在的符号。
        删除条件带上 branch / pr_id / pr_number —— 保证只删"这一次同步的范围"。

    【★ 为什么 max_files 默认 500】
        大仓库有上万个文件。全扫要几分钟且占大量内存。
        500 个文件足够覆盖"主要模块"，是速度与覆盖度的折中。

    【★ CALL_RE 的误报怎么处理】
        "if(" / "for(" / "print(" 这些也会被匹配成"调用"。
        所以有黑名单（if/for/while/return/print/len/str/int）。
        而且 seen_calls 去重 —— 同一个文件里调 100 次 foo() 只记一条关系。

    【关系的两端可能为空】
        source_symbol_id / target_symbol_id 允许 None：
        跨文件调用的目标可能没被抽出来（比如第三方库）。
        这时只存名字（source_name / target_name），保证关系不丢。
    """
    # ── 步骤 1：删旧数据（只删本次同步范围：同仓库 + 同分支 + 同 PR）──
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id, CodeRelation.branch == branch, CodeRelation.pr_id == pr_id, CodeRelation.pr_number == pr_number).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id, CodeSymbol.branch == branch, CodeSymbol.pr_id == pr_id, CodeSymbol.pr_number == pr_number).delete(synchronize_session=False)

    symbol_rows: list[CodeSymbol] = []
    files_seen = 0

    # ── 步骤 2：第一遍 —— 抽符号 ──
    for path in checkout_path.rglob("*"):
        if files_seen >= max_files:
            break
        if not path.is_file():
            continue
        # 路径任一段命中 SKIP_DIRS 就跳过（如 node_modules/foo/bar.js）
        if any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts):
            continue
        if not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        symbols = _symbol_matches(path, text)
        # 没有符号的文件不用记（比如纯配置）
        if not symbols:
            continue

        relative_path = path.relative_to(checkout_path).as_posix()
        files_seen += 1
        for item in symbols:
            row = CodeSymbol(
                repo_id=repo.id,
                path=relative_path,
                name=str(item["name"]),
                kind=str(item["kind"]),
                language=str(item["language"]),
                start_line=int(item["start_line"]),
                # end_line 留空：正则拿不到结束位置（AST 才能）
                end_line=None,
                branch=branch,
                commit_sha=commit_sha,
                pr_id=pr_id,
                pr_number=pr_number,
                meta={},
            )
            db.add(row)
            symbol_rows.append(row)

    # ★ flush 拿到所有符号的 id（第二遍建关系要用）
    db.flush()
    by_name = {row.name: row for row in symbol_rows}

    # ── 步骤 3：第二遍 —— 建关系（import + call）──
    for path in checkout_path.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts) or not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()

        # ① import 关系：文件 → 被导入的模块
        # 正则有三个分组（Python from X / Python import X / JS from "x"），取第一个非空
        import_targets = [next((part for part in match.groups() if part), "") for match in IMPORT_RE.finditer(text)]
        for target in [item for item in import_targets if item]:
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    source_name=relative_path,
                    target_name=target,
                    relation_type="imports",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    meta={},
                )
            )

        # ② call 关系：本文件的符号 → 被调用的符号
        local_symbols = [row for row in symbol_rows if row.path == relative_path]
        # 一个文件可能有多个符号，取第一个作为 source（简化处理）
        source = local_symbols[0] if local_symbols else None
        seen_calls: set[str] = set()
        for match in CALL_RE.finditer(text):
            target_name = match.group(1)
            # 去重 + 过滤语言关键字和内置函数
            if target_name in seen_calls or target_name in {"if", "for", "while", "return", "print", "len", "str", "int"}:
                continue
            seen_calls.add(target_name)
            target = by_name.get(target_name)
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    source_symbol_id=source.id if source else None,
                    target_symbol_id=target.id if target else None,
                    source_name=source.name if source else relative_path,
                    target_name=target_name,
                    relation_type="calls",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    # 记录调用发生的行号（便于回原文定位）
                    meta={"line": _line_for_offset(text, match.start())},
                )
            )
    return len(symbol_rows)


# ════════════════════════════════════════════════════════════════
# 四、对外入口
# ════════════════════════════════════════════════════════════════

def purge_legacy_code_documents(db: Session, repo_id: uuid.UUID | None = None) -> int:
    """清理历史遗留的 code_file 文档（源码不该在 RAG 里）。

    【调用链】被 sync_repository_code_analysis() 调用
    【下游】删 documents 表里 source_type="code_file" 的行

    【为什么需要这个函数】
        早期版本曾把源码也写进 RAG。后来改成"源码走图谱"，
        但老数据还在库里 —— 会被检索命中，返回过期的代码片段。
        所以每次同步代码时顺手清掉。

    【返回删了多少条】
        0 是正常的（没有遗留数据）。
    """
    query = db.query(Document).filter(Document.source_type == "code_file")
    if repo_id is not None:
        query = query.filter(Document.repo_id == repo_id)
    return query.delete(synchronize_session=False)


async def sync_repository_code_analysis(db: Session, repo: Repository) -> int:
    """同步代码并重建代码图谱（★ 主流程）。

    【调用链】追这个函数会进入：
        1. decrypt_token()            → 解密 GitHub token（core/security）
        2. asyncio.to_thread(_sync_checkout)   → 克隆/更新代码（阻塞操作丢线程池）
        3. _git_text()                → 取分支名 / commit sha
        4. asyncio.to_thread(_sync_code_graph) → 扫描建图谱（阻塞操作丢线程池）
        5. purge_legacy_code_documents()       → 清理历史遗留
    【被谁调用】sync_repository_code_analysis_by_id()（后台任务）

    【数据流向】
        入：db、repo
        出：符号数（int）

    【★ 为什么 git 操作要 asyncio.to_thread】
        subprocess.run 是同步阻塞的。直接 await 会卡住整个事件循环
        （其他请求全部停住）。to_thread 把它丢到线程池，事件循环继续跑。

    【★ token 只在内存里出现一次】
        解密 → 拼进 URL → 传进 _sync_checkout → 用完就没了。
        不写日志、不落库、不返回。
    """
    # 解密 token（没配就是 None，公开仓库也能克隆）
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None

    checkout_path = await asyncio.to_thread(_sync_checkout, repo, token)

    # 分支名：优先仓库配置 > 当前 git 分支 > main
    branch = repo.default_branch or _git_text(["branch", "--show-current"], checkout_path) or "main"
    commit_sha = _git_text(["rev-parse", "HEAD"], checkout_path)

    symbol_count = await asyncio.to_thread(
        _sync_code_graph,
        db,
        repo,
        checkout_path,
        branch=branch,
        commit_sha=commit_sha,
    )

    # 清掉历史遗留的 code_file 文档
    purge_legacy_code_documents(db, repo.id)

    # 补全 clone_url（首次连接时可能是空的）
    repo.clone_url = repo.clone_url or _clone_url(repo)
    db.commit()
    return symbol_count


async def sync_repository_code_analysis_by_id(repo_id: uuid.UUID) -> int | None:
    """后台任务入口：按 repo_id 同步（自己开 Session + 吞异常）。

    【调用链】sync_repository_code_analysis() → 本函数
    【被谁调用】api/routes/repos.py / services/scheduler.py 的 BackgroundTasks

    【数据流向】
        入：repo_id
        出：符号数；仓库不存在或失败返回 None

    【★ 为什么这里吞掉所有异常】
        它是后台任务，没有调用方接异常。
        抛出去只会变成 "Task exception was never retrieved" 的噪音日志。
        记一条 warning 就够（用户能在日志里看到原因）。
    """
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        try:
            return await sync_repository_code_analysis(db, repo)
        except Exception as exc:
            logger.warning("Failed to sync repository code analysis for %s: %s", repo_id, exc)
            return None
    finally:
        db.close()
