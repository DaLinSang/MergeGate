"""
源码检索（工作区实时搜索）
=======================

【业务定位】
    数据流 2 的"查询侧"：Agent 需要看源码时，**实时**在工作区里搜。
    这是"源码不进 RAG"这个架构决策的另一半 ——
    既然不预先向量化，就必须有一个又快又准的实时检索通道。

    【和 RAG 检索的分工】
        RAG 检索   → 文档 / Issue / PR / CI 日志（语义召回，可能过期）
        本文件     → 源码（精确匹配，永远是最新的）

【数据流向】
    上游（谁调它）：
        services/agents/chat_agent.py 的 workspace.search_code 工具
        api/routes/search.py          代码搜索接口
    下游（它调谁）：
        ripgrep（系统里有就用，快 10 倍）
        Python 自带的 rglob + 逐行匹配（没有 rg 时的兜底）

【核心规则】
    ★ 三道安全边界（这是"能读用户代码"的功能，必须严格）：
        ① 搜索路径必须落在仓库工作区内（防 ../ 逃逸）
        ② 不能搜索 SKIP_DIRS 里的目录（.git / node_modules）
        ③ 不能搜索密钥文件（.env / .npmrc / .pypirc）
    1. ripgrep 优先，失败自动降级到 Python 扫描。
    2. 分数归一化到 0~1（和 RAG 检索的结果格式保持一致，便于融合）。

【本文件函数地图】

    ┌─ 入口 ──────────────────────────────────────────
    │  search_repository_code()  ★ 按 Repository 搜（业务入口）
    │  search_checkout_code()    ★ 按路径搜（核心实现）
    │
    ├─ 安全边界 ─────────────────────────────────────
    │  resolve_search_root()     ★ 路径校验（防逃逸 + 防敏感目录）
    │  repository_checkout_path() 取仓库的本地目录
    │  _is_text_file()           过滤敏感/二进制文件
    │  _read_text()              读文件
    │
    ├─ 检索实现 ─────────────────────────────────────
    │  query_terms()             ★ 查询分词（含代码标识符优先）
    │  _ripgrep_matches()        ripgrep 路径（JSON 输出解析）
    │  _python_matches()         Python 兜底扫描
    │  _score_match()            单行打分
    │
    └─ 上下文 ───────────────────────────────────────
       read_code_excerpt()       读某行附近的代码（带行号）
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.db.models import Repository


# 跳过这些目录（生成物/依赖，搜它们既慢又没意义）
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

# ★ 密钥文件黑名单：绝不搜索、绝不在结果里出现
SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}

# 文本类扩展名（能安全 decode）
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

# 查询分词正则（★ 特意允许 _ . $ : / \ - 这些"代码特征字符"）
# 因为用户可能搜 "app/services/chat.py" 或 "SessionLocal.commit"，要整串保留
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$:/\\-]*|[0-9]{3,}|[\u4e00-\u9fff]{2,}")

# 停用词：这些词出现在几乎每个文件里，搜它们等于没搜
_STOP_TERMS = {
    "about",
    "after",
    "before",
    "code",
    "error",
    "failed",
    "failure",
    "file",
    "from",
    "into",
    "issue",
    "project",
    "pull",
    "request",
    "test",
    "tests",
    "that",
    "the",
    "this",
    "with",
    "代码",
    "问题",
    "失败",
    "项目",
    "相关",
    "这个",
}


# ════════════════════════════════════════════════════════════════
# 路径与安全边界
# ════════════════════════════════════════════════════════════════

def repository_checkout_path(repo: Repository) -> Path:
    """取仓库的本地代码目录。

    【调用链】code_analysis._repo_checkout_path() → 本函数
    【被谁调用】search_repository_code()

    【两级优先】
        ① repo.local_path（用户配置的本地目录）
        ② 系统管理目录（code_analysis._repo_checkout_path）

    【为什么 import 放在函数内部】
        避免循环依赖：code_analysis 可能间接依赖本文件。
        局部 import 把依赖推迟到调用时，绕开循环。
    """
    raw_path = str(getattr(repo, "local_path", None) or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    from app.services.code_analysis import _repo_checkout_path

    return _repo_checkout_path(repo).resolve()


def resolve_search_root(checkout_path: str | Path, relative_path: str | None = None) -> tuple[Path, Path]:
    """校验搜索路径（★ 本文件最重要的安全函数）。

    【调用链】被 read_code_excerpt() / search_checkout_code() 调用

    【数据流向】
        入：checkout_path（仓库根）、relative_path（用户想搜的子路径）
        出：(checkout, root) —— 都已 resolve

    【★ 五道校验，任何一道不过就抛 ValueError】
        ① 仓库目录不存在 → "还没同步到本地"
        ② root 不在 checkout 内（relative_to 抛 ValueError）→ 目录穿越
           用户传 "../../etc" 就靠这条拦住
        ③ 路径里含 SKIP_DIRS 的段 → 不搜 node_modules/.git
        ④ 文件名在密钥黑名单里 → 不搜 .env
        ⑤ 路径不存在

    【为什么要处理 "./" "/" "\\" 这些输入】
        前端可能传 ".", "./", "/", "\\" 表示"搜整个仓库"。
        统一规范化成 "."，避免拼出奇怪的路径。

    【为什么校验顺序不能换】
        先做"是否在仓库内"（②），再做"是否敏感"（③④）。
        否则用户传 "../../.env" 时，先判敏感会给出"不搜敏感文件"，
        掩盖了真正的"越界"问题 —— 错误信息会误导排查。
    """
    checkout = Path(checkout_path).expanduser().resolve()
    if not checkout.exists() or not checkout.is_dir():
        raise ValueError("仓库代码还没有同步到本地。")

    # 规范化输入（"." / "./" / "/" / "\\" 都表示仓库根）
    requested = str(relative_path or ".").strip()
    requested = "." if requested in {"", ".", "./", "/", "\\"} else requested.lstrip("/\\")
    root = (checkout / requested).resolve()

    # ★ 校验 ②：必须落在仓库内（防 ../ 逃逸）
    try:
        root.relative_to(checkout)
    except ValueError as exc:
        raise ValueError("搜索路径位于仓库工作区之外。") from exc

    # 校验 ③：不能在被跳过的目录里
    if any(part in SKIP_DIRS for part in root.relative_to(checkout).parts):
        raise ValueError("搜索路径位于被跳过的目录中。")

    # 校验 ④：不能是密钥文件
    if root.name in SECRET_FILENAMES:
        raise ValueError("不会搜索敏感配置文件。")

    # 校验 ⑤：必须存在
    if not root.exists():
        raise ValueError("搜索路径不存在。")

    return checkout, root


# ════════════════════════════════════════════════════════════════
# 查询分词
# ════════════════════════════════════════════════════════════════

def query_terms(query: str, limit: int = 16) -> list[str]:
    """把查询拆成检索词，并给"代码标识符"更高优先级。

    【调用链】被 search_checkout_code() 调用

    【数据流向】
        入：query、limit（最多 16 个词）
        出：排序后的词列表（代码标识符在前）

    【打分规则（三元组：代码特征, 长度, 词）】
        代码特征 = 1 如果：
            含 _ . $ : / \\ - 或数字（如 "app/main.py"、"get_user"、"v2"）
            或者同时含小写和大写（驼峰，如 "SessionLocal"）
        排序：代码特征优先 → 长度长的优先 → 字母序（保证稳定）

    【★ 为什么要给代码标识符更高优先级】
        查询 "SessionLocal 报错" 时：
            "SessionLocal" 是精确锚点，必须保留
            "报错" 太泛，几乎每个文件都有
        如果按原顺序截断到 16 个词，可能把关键标识符挤掉。

    【★ 停用词的处理：不是直接丢弃】
        先把非停用词放 candidates，停用词放 fallback。
        如果过滤后一个都不剩（比如查询全是停用词），
        就用 fallback —— 总比返回空结果好。
    """
    candidates: list[tuple[int, int, str]] = []
    fallback_candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for position, match in enumerate(_QUERY_TOKEN_RE.finditer(query or "")):
        value = match.group(0).strip()
        normalized = value.lower()
        # 太短（1 个字符）或重复 → 跳过
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)

        # 判断是否"像代码"
        code_like = bool(
            re.search(r"[_.$:/\\\-0-9]", value)
            or (any(char.islower() for char in value) and any(char.isupper() for char in value))
        )
        candidate = (1 if code_like else 0, len(value), normalized)
        fallback_candidates.append(candidate)
        # 非停用词才进正式候选
        if normalized not in _STOP_TERMS:
            candidates.append(candidate)

    # 全是停用词 → 退回用全部（总比空好）
    candidates = candidates or fallback_candidates
    # 代码特征优先 → 长度优先 → 字母序
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in candidates[: max(1, limit)]]


def _is_text_file(path: Path) -> bool:
    """判断是否是可搜索的文本文件。

    【调用链】被 read_code_excerpt() / _python_matches() 调用

    【除了白名单，还要排除三类】
        SECRET_FILENAMES           密钥
        .log / .db / .sqlite*      日志和数据库（体积大、内容无意义）
    """
    if path.name in SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 180_000) -> str | None:
    """读文本文件（四道保护：OSError / 空 / 超大 / 二进制）。

    【调用链】被 read_code_excerpt() / _python_matches() 调用

    【和 code_analysis._read_text 的区别】
        这里用 path.read_bytes()（先全读再判大小），
        那边用 read(max_bytes+1)（避免读超大文件）。
        这里上限 180KB 比较小，全读可接受，代码更简洁。
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_code_excerpt(checkout_path: str | Path, relative_path: str, start_line: int, line_count: int = 7) -> str:
    """读某一行附近的代码（带行号），给 Agent 看上下文。

    【调用链】resolve_search_root() / _is_text_file() / _read_text() → 本函数
    【被谁调用】chat_agent 的 workspace.read_file 工具、前端"查看上下文"

    【数据流向】
        入：checkout_path、relative_path、start_line、line_count（默认 7 行）
        出：形如 "app/main.py:10-16\\n10: xxx\\n11: yyy" 的文本；失败返回 ""

    【为什么返回带行号的格式】
        模型要能引用"第 12 行"，行号必须显式写出来。
        格式和 grep -n 的输出一致，模型见过很多，容易理解。

    【边界处理】
        start 至少 1；end 不超过文件总行数；start > end 返回空
        —— 防止越界索引导致 IndexError。
    """
    checkout, path = resolve_search_root(checkout_path, relative_path)
    if not path.is_file() or not _is_text_file(path):
        return ""
    text = _read_text(path)
    if text is None:
        return ""

    lines = text.splitlines()
    start = max(int(start_line), 1)
    end = min(start + max(int(line_count), 1) - 1, len(lines))
    if start > end:
        return ""

    relative = path.relative_to(checkout).as_posix()
    return f"{relative}:{start}-{end}\n" + "\n".join(
        f"{line_number}: {lines[line_number - 1]}" for line_number in range(start, end + 1)
    )


# ════════════════════════════════════════════════════════════════
# 检索实现：ripgrep 优先，Python 兜底
# ════════════════════════════════════════════════════════════════

def _score_match(relative_path: str, line: str, terms: list[str]) -> float:
    """给"某一行"打分。

    【调用链】被 _ripgrep_matches() / _python_matches() 调用

    【三部分加权】
        matched * 4      命中【不同词】的个数（最重要：覆盖了查询的几个方面）
        occurrences      出现总次数（min(count,4) 防止某个词刷分）
        path_hits * 2    路径里出现（说明搜的是这个文件，强信号）

    【为什么 occurrences 要 cap 到 4】
        一个词出现 100 次的行（比如压缩过的 JS）会把分数拉到极高，
        但它未必是最相关的。cap 住避免这种偏置。
    """
    path_text = relative_path.lower()
    line_text = line.lower()
    matched = sum(1 for term in terms if term in line_text or term in path_text)
    occurrences = sum(min(line_text.count(term), 4) for term in terms)
    path_hits = sum(path_text.count(term) for term in terms)
    return float(matched * 4 + occurrences + path_hits * 2)


def _ripgrep_matches(checkout: Path, root: Path, terms: list[str], max_matches: int) -> list[tuple[float, str, int, str]]:
    """用 ripgrep 搜索（快，但要解析 JSON 输出）。

    【调用链】_score_match() → 本函数
    【被谁调用】search_checkout_code()（系统里有 rg 时）

    【数据流向】
        入：checkout、root、terms、max_matches
        出：[(score, relative_path, line_number, line)] —— 每个文件只留最高分那一行

    【★ 命令参数逐个说明】
        --json            结构化输出（好解析，且含行号/文件路径）
        --ignore-case     忽略大小写
        --line-number     带行号
        --max-columns 500 超长行截断（防止压缩 JS 把内存打爆）
        --glob !**/.env*   ★ 在 rg 层面就排除敏感文件（双保险）
        --glob !**/*.log  排除日志

    【★ 为什么 returncode 允许 0 和 1】
        rg 的约定：0 = 有匹配，1 = 没匹配，2+ = 出错。
        1 不是错误，只是没找到 —— 不能当异常处理。

    【★ 为什么每个文件只留最高分那一行】
        一个词在一个文件里可能出现 50 次。全返回会淹没其他文件。
        只留最相关的一行，配合 read_code_excerpt() 让 Agent 自己展开上下文。

    【路径处理的两个细节】
        ① rg 可能返回相对路径（相对于 cwd），要拼上 checkout
        ② resolve 后再 relative_to 校验 —— 防止 rg 跟随符号链接跑到仓库外
    """
    pattern = "|".join(re.escape(term) for term in terms)
    command = [
        "rg",
        "--json",
        "--ignore-case",
        "--line-number",
        "--max-columns",
        "500",
        "--glob",
        "!**/.env*",
        "--glob",
        "!**/.npmrc",
        "--glob",
        "!**/.pypirc",
        "--glob",
        "!**/*.log",
        "--glob",
        "!**/*.db",
        "--glob",
        "!**/*.sqlite*",
        pattern,
        str(root),
    ]
    result = subprocess.run(
        command,
        cwd=str(checkout),
        check=False,            # 不用 check：returncode=1 是"没匹配"，不是错误
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",       # 输出可能有非法 UTF-8（二进制混入），容错
        timeout=15,
    )
    # 0=有匹配 1=无匹配，2+ 才是真错误
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "ripgrep 搜索失败")

    best_by_file: dict[str, tuple[float, str, int, str]] = {}

    for raw_line in result.stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            # 单行解析失败不影响其他行（rg 可能混入非 JSON 的提示）
            continue
        # 只要 match 事件（还有 begin/end/summary 等）
        if event.get("type") != "match":
            continue

        data = event.get("data") or {}
        raw_path = str((data.get("path") or {}).get("text") or "")
        path = Path(raw_path)
        # rg 返回相对路径时，拼上 checkout
        if not path.is_absolute():
            path = checkout / path

        # ★ 二次校验：resolve 后必须仍在仓库内（防符号链接逃逸）
        try:
            relative = path.resolve().relative_to(checkout).as_posix()
        except ValueError:
            continue

        # ★ 二次校验：敏感目录/文件（rg 的 glob 已经挡过一次，这里再挡一次）
        if any(part in SKIP_DIRS for part in Path(relative).parts) or Path(relative).name in SECRET_FILENAMES:
            continue

        line_number = int(data.get("line_number") or 1)
        line = str((data.get("lines") or {}).get("text") or "").strip()[:320]
        score = _score_match(relative, line, terms)
        candidate = (score, relative, line_number, line)

        # 同一文件只留最高分
        current = best_by_file.get(relative)
        if current is None or candidate[0] > current[0]:
            best_by_file[relative] = candidate

        # 够多了就停（rg 输出是按文件顺序的，不是按分数）
        if len(best_by_file) >= max_matches:
            break

    return list(best_by_file.values())


def _python_matches(checkout: Path, root: Path, terms: list[str], max_files: int = 1000) -> list[tuple[float, str, int, str]]:
    """用 Python 逐文件逐行扫描（没有 ripgrep 时的兜底）。

    【调用链】_is_text_file() / _read_text() / _score_match() → 本函数
    【被谁调用】search_checkout_code()（rg 不存在或失败时）

    【数据流向】
        入：checkout、root、terms、max_files（上限 1000 个文件）
        出：[(score, relative_path, line_number, line)] —— 每个文件一条

    【为什么慢还留着】
        不是所有环境都装了 ripgrep（Windows、精简容器）。
        没有兜底的话，这些环境下"搜代码"功能直接不可用。

    【max_files=1000 的意义】
        大仓库有上万个文件，全扫要几十秒。
        1000 个文件（按 rglob 顺序）通常能覆盖主要代码，是速度的折中。
    """
    # root 可能是文件（搜单个文件）也可能是目录
    paths = [root] if root.is_file() else root.rglob("*")
    matches: list[tuple[float, str, int, str]] = []
    files_seen = 0

    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(checkout)
        # 跳过敏感/跳过目录/非文本
        if any(part in SKIP_DIRS for part in relative.parts) or not _is_text_file(path):
            continue
        files_seen += 1
        if files_seen > max_files:
            break
        text = _read_text(path)
        if text is None:
            continue

        relative_text = relative.as_posix()
        best: tuple[float, str, int, str] | None = None
        # 逐行打分，只留最高分那一行
        for line_number, line in enumerate(text.splitlines(), start=1):
            score = _score_match(relative_text, line, terms)
            if score <= 0:
                continue
            candidate = (score, relative_text, line_number, line.strip()[:320])
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best:
            matches.append(best)
    return matches


# ════════════════════════════════════════════════════════════════
# ★ 入口
# ════════════════════════════════════════════════════════════════

def search_checkout_code(
    checkout_path: str | Path,
    query: str,
    *,
    limit: int = 12,
    relative_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """在指定的代码目录里搜索（★ 核心实现）。

    【调用链】追这个函数会进入：
        1. query_terms()          → 分词
        2. resolve_search_root()  → 路径校验
        3. _ripgrep_matches() 或 _python_matches() → 实际搜索
    【被谁调用】search_repository_code() / api/routes/search.py

    【数据流向】
        入：checkout_path、query、limit、relative_path（限定子目录）、metadata
        出：统一格式的候选列表（和 RAG 检索结果同构，便于 Agent 混用）

    【★ 为什么返回格式要和 RAG 检索一致】
        都是给 Agent 的"证据"。字段一致（id/title/source_type/snippet/score/metadata）
        Agent 就不用区分"这条来自向量库还是来自工作区"。
        区别只在 metadata.retrieval_mode = "workspace"。

    【★ 分数归一化：score / max_score】
        原始分数是 4/8/12 这种绝对值，没有上界。
        除以本次最高分 → 全部落在 0~1，和 RAG 的 combined_score 可比。
        这样前端可以混排，阈值过滤也能统一。

    【limit 夹在 1~50】
        防止前端传 999999 把结果列表撑爆。

    【rg 失败为什么静默降级】
        rg 可能超时、可能被权限拦住、可能版本不兼容。
        这些都不该让"搜代码"整个失败 —— Python 扫描慢但能用。
    """
    terms = query_terms(query)
    if not terms:
        return []

    checkout, root = resolve_search_root(checkout_path, relative_path)
    capped_limit = min(max(int(limit), 1), 50)

    search_tool = "python_scan"
    if shutil.which("rg"):
        try:
            # 多取一些（×8），后面排序后只留 limit 条
            matches = _ripgrep_matches(checkout, root, terms, max_matches=max(capped_limit * 8, 80))
            search_tool = "ripgrep"
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            # rg 不可用 → 降级到 Python 扫描
            matches = _python_matches(checkout, root, terms)
    else:
        matches = _python_matches(checkout, root, terms)

    # 分数降序 → 路径 → 行号（保证结果稳定，同样的查询永远同样顺序）
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not matches:
        return []

    max_score = max(item[0] for item in matches) or 1.0

    output = []
    for score, path, line_number, line in matches[:capped_limit]:
        item_metadata = {
            "path": path,
            "line": line_number,
            # start_line / end_line 都给同一行：调用方可以用 read_code_excerpt 展开
            "start_line": line_number,
            "end_line": line_number,
            "checkout_path": str(checkout),
            # ★ 标记来源：workspace（工作区实时）vs rag（向量库）
            "retrieval_mode": "workspace",
            "search_tool": search_tool,
            **(metadata or {}),
        }
        output.append(
            {
                # id 里带行号：同一文件的不同行是不同"证据"
                "id": f"workspace:{path}:{line_number}",
                "title": f"{path}:{line_number}",
                "source_type": "workspace_file",
                "source_id": path,
                "snippet": f"{path}:{line_number}: {line}",
                # 归一化到 0~1
                "score": score / max_score,
                "metadata": item_metadata,
            }
        )
    return output


def search_repository_code(
    repo: Repository,
    query: str,
    *,
    limit: int = 12,
    relative_path: str | None = None,
    checkout_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """按 Repository 搜代码（业务入口）。

    【调用链】repository_checkout_path() / search_checkout_code() → 本函数
    【被谁调用】
        services/agents/chat_agent.py 的 workspace.search_code 工具
        api/routes/search.py

    【数据流向】
        入：repo、query、limit、relative_path、checkout_path（可覆盖）、metadata
        出：候选列表；仓库没同步/路径非法时返回 []

    【为什么路径不存在返回 [] 而不是报错】
        对 Agent 来说"搜不到"和"没同步"处理方式一样（都是"没有结果"）。
        返回 [] 让它自然走"没找到证据"的分支，不必额外处理异常。

    【为什么要捕获 ValueError】
        resolve_search_root 会为"越界/敏感目录/不存在"抛 ValueError。
        这些都是"用户输入问题"，不是系统故障 —— 返回空列表即可。
        （安全上也不会泄露"这个路径存不存在"的信息）
    """
    root = Path(checkout_path).resolve() if checkout_path else repository_checkout_path(repo)
    if not root.exists():
        return []
    try:
        return search_checkout_code(
            root,
            query,
            limit=limit,
            relative_path=relative_path,
            # 带上 repo_id，便于下游追溯"这条证据来自哪个仓库"
            metadata={"repo_id": str(repo.id), **(metadata or {})},
        )
    except ValueError:
        return []
