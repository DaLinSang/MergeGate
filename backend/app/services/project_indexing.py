"""
项目文档索引（扫描仓库 → 建 project_doc 索引）
============================================

【业务定位】
    数据流 2（仓库代码 → 代码图谱）的**入口**：扫描仓库里的文档类文件
    （README / docs/ / ADR / 依赖清单），切成块写进 RAG。

    【为什么要单独做"项目文档索引"】
        Issue/PR/CI 是"过程数据"，只能回答"发生过什么"。
        README / 架构文档 / 依赖清单是"背景知识"，能回答"这是什么项目、怎么跑"。
        用户问"这个项目怎么启动"，靠 Issue 是查不到的。

【数据流向】
    上游（谁调它）：
        api/routes/project_index.py          手动触发扫描
        services/scheduler.py                定时检查（fingerprint 变了才扫）
        app/main.py                          启动时回填
    下游（它调它 / 写谁）：
        rag/chunking.chunk_document()        切片
        rag/vector_store.replace_documents() 整组替换写入 documents 表
        chat_memory.EvidenceStore            刷新证据库（同事务）
        knowledge_graph.rebuild_knowledge_graph()  重建图谱
        project_index_states 表              ← 索引状态（前端进度条）

【核心规则】
    1. **安全第一**：绝不索引密钥文件（.env / *.pem / credentials.json）。
    2. **三层来源分级**：authoritative（README/架构文档）> soft_clue（CHANGELOG）
       > derived（依赖清单）。分级用于前端展示"这份资料有多可信"。
    3. **fingerprint 决定要不要重扫**：文件 mtime + size + 策略 的哈希。
       没变就跳过，避免每次启动都全量扫一遍。
    4. **策略可配置**：.devflow/index.yml 支持 auto（自动发现）/ allowlist（白名单）。

【本文件函数地图】

    ┌─ 配置与发现 ────────────────────────────────────
    │  load_project_index_policy()  ★ 读 .devflow/index.yml
    │      └ _normalize_patterns()      清洗 glob（防目录穿越）
    │  _discover_project_docs()     ★ os.walk 扫描 + 分类 + 策略过滤
    │      ├ _classify_file()           ★ 判断文件属于哪一类、哪一级
    │      │    └ _is_secret()          密钥文件直接排除
    │      ├ _is_inside()              防符号链接逃逸
    │      └ _policy_allows()          include/exclude 判断
    │  discover_project_docs()     对外简化版（只要候选列表）
    │  project_fingerprint()       ★ 生成"内容指纹"（决定要不要重扫）
    │
    ├─ 状态管理 ─────────────────────────────────────
    │  get_project_index_state()      读状态（含 stale 判定）
    │  mark_project_index_building()  标"构建中"
    │  snooze_project_index()         用户点"稍后提醒"
    │  is_snoozed()                   是否在暂缓期内
    │
    ├─ 内容提取 ─────────────────────────────────────
    │  _read_text()                 读文件（大小/二进制/多编码处理）
    │  _extract_title()             取第一个 # 标题
    │  _summary_from_text()         取第一段当摘要
    │  _detect_tech_stack()         按清单文件推断技术栈
    │  _safe_dirs()                 列顶层目录（给前端展示结构）
    │  _build_summary()             组装 summary_json
    │  _payloads_for_candidates()   ★ 候选文件 → 切片 → DocumentPayload
    │
    └─ 主流程 ───────────────────────────────────────
       scan_project_index()         ★ 完整扫描（状态机 + 事务 + 图谱重建）
       scan_project_index_by_id()   后台任务入口（自己开 Session）
"""

import fnmatch
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.db.models import ProjectIndexState, Repository
from app.db.session import SessionLocal
from app.services.knowledge_graph import rebuild_knowledge_graph
from app.services.rag.chunking import chunk_document
from app.services.code_analysis import _repo_checkout_path, _stored_checkout_path
from app.services.rag.vector_store import DocumentPayload, replace_documents, stable_source_id


# ════════════════════════════════════════════════════════════════
# 常量：分类规则与安全边界
# ════════════════════════════════════════════════════════════════

# 本文件负责的三类来源（整组替换时按这三类删旧数据）
PROJECT_INDEX_SOURCE_TYPES = ["project_overview", "project_doc", "project_manifest"]

# 索引状态的合法取值
PROJECT_INDEX_STATUSES = {"missing", "stale", "building", "ready", "failed"}

# 用户自定义策略的配置文件路径（相对仓库根目录）
PROJECT_INDEX_CONFIG_PATH = Path(".devflow/index.yml")

# 两种扫描模式
PROJECT_INDEX_MODES = {"auto", "allowlist"}

# ── 三层来源分级 ──
# authoritative：权威文档（README / 架构 / 贡献指南）—— 可信度最高
AUTHORITATIVE_TOP_LEVEL = {"README", "ARCHITECTURE", "CONTRIBUTING"}

# derived：派生数据（依赖清单）—— 由工具生成，可信但只反映依赖关系
MANIFEST_FILES = {
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "pnpm-workspace.yaml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
}

# soft_clue：弱线索（变更日志）—— 有信息但不代表当前状态
SOFT_CLUE_TOP_LEVEL = {"CHANGELOG", "HISTORY", "RELEASES"}

# ── 扫描时跳过的目录 ──
# 这些都是"生成物"或"依赖"，没有索引价值，而且体积巨大
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

# 例外：这些隐藏目录【要】索引（Issue 模板、PR 模板、workflow 说明都在这里）
INDEXABLE_HIDDEN_DIRS = {".github"}

# ── 密钥文件黑名单（★ 安全红线，绝不能索引）──
# 一旦这些文件进了向量库，任何能检索的人都能把密钥捞出来
SECRET_NAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc", "credentials.json"}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


# ════════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProjectDocCandidate:
    """一个"待索引的文档"候选。

    【为什么是 frozen】
        它是扫描阶段的产物，后续只读不写。
        冻住可以避免"某个环节偷偷改了候选"这类难查的 bug。
    """

    path: Path                  # 绝对路径（读文件用）
    relative_path: str          # 仓库内相对路径（写 metadata、算 source_id 用）
    source_type: str            # project_doc / project_manifest
    tier: str                   # authoritative / soft_clue / derived（可信度分级）


@dataclass(frozen=True)
class ProjectIndexPolicy:
    """索引策略（来自 .devflow/index.yml，没有配置就是全默认）。

    【两种模式】
        auto       自动发现：扫到符合条件的就索引，exclude 可以排除
        allowlist  白名单：只有 include 命中的才索引（严格模式）

    为什么要 allowlist：有些仓库的 docs/ 里混着大量无关内容
    （第三方文档、过期的设计稿），白名单能精确控制。
    """

    mode: str = "auto"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    include_manifests: bool = True
    config_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的 dict（写进 fingerprint 和 summary_json）。

        注意键名是 camelCase —— 因为要直接返回给前端。
        """
        return {
            "mode": self.mode,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "includeManifests": self.include_manifests,
            "configPath": self.config_path,
        }


@dataclass(frozen=True)
class ProjectDocDiscovery:
    """一次扫描的完整结果（不只是候选，还带"被排除了什么"）。

    【为什么要记录 excluded_paths】
        用户会问"为什么我的文档没被索引"。
        把被策略排除的路径记下来，前端能直接告诉他原因。
    """

    candidates: list[ProjectDocCandidate]
    excluded_paths: list[str]
    policy: ProjectIndexPolicy
    truncated: bool = False     # 是否因为 max_files 上限被截断


# ════════════════════════════════════════════════════════════════
# 路径与时间工具
# ════════════════════════════════════════════════════════════════

def _now() -> datetime:
    """当前 UTC 时间（统一入口，便于测试替换）。"""
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """把 naive datetime 补上 UTC 时区。

    【为什么需要】
        SQLite 存的时间没有时区，读出来是 naive；
        而 _now() 是 aware。两者直接比较会抛 TypeError。
        统一补成 aware 才能安全比较（snooze 判断就靠这个）。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _repo_root(repo: Repository) -> Path:
    """取仓库的本地代码目录。

    【调用链】_stored_checkout_path() / _repo_checkout_path() → 本函数
    【被谁调用】_safe_project_root()

    【两级优先】
        ① stored：系统管理的目录（.data/repos/xxx）—— 优先
        ② 用户指定的 local_path —— 兜底
    为什么优先 stored：那是系统自己克隆的，路径一定有效；
    用户的 local_path 可能已经不存在了。
    """
    stored = _stored_checkout_path(repo)
    if stored:
        return stored
    return _repo_checkout_path(repo)


def _safe_project_root(repo: Repository) -> Path | None:
    """取项目根目录，并确认它真实存在。

    【调用链】_repo_root() → 本函数

    【为什么返回 None 而不是抛异常】
        调用方需要区分"路径不存在"（可以给用户友好提示）和"其他错误"。
        返回 None 让判断更直白。

    【为什么要 resolve()】
        解析掉符号链接和 ../，得到规范化的绝对路径。
        后面所有"是否在仓库内"的判断都基于它。
    """
    root = _repo_root(repo)
    if not root.exists() or not root.is_dir():
        return None
    try:
        return root.resolve()
    except OSError:
        return None


def _is_secret(path: Path) -> bool:
    """判断是不是密钥文件（★ 安全红线）。

    【调用链】被 _classify_file() 调用

    【三重判断】
        ① 文件名在黑名单里（.env / .npmrc / credentials.json）
        ② 后缀在黑名单里（.key / .pem / .p12 / .pfx）
        ③ 文件名以 credentials 开头（credentials-prod.json 这类变体）

    【为什么用黑名单而不是白名单】
        文档类文件扩展名太多（.md/.rst/.txt/.adoc…），白名单会漏。
        而密钥文件的特征是"少而明确"，黑名单更合适。
    """
    name = path.name
    return name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES or name.lower().startswith("credentials")


def _read_text(path: Path, max_bytes: int = 400_000) -> str | None:
    """读文本文件，带大小/二进制/编码三重保护。

    【调用链】被 _payloads_for_candidates() 调用

    【数据流向】
        入：path、max_bytes（400KB 上限）
        出：文本内容；读不了返回 None

    【四道检查，任何一道不过就返回 None】
        ① OSError          → 文件被删/没权限
        ② 空文件           → 没内容
        ③ 超过 max_bytes   → 太大（读 max_bytes+1 是为了判断"是否超了"）
        ④ 前 4KB 有 \\x00  → 是二进制文件（改后缀的图片/压缩包）

    【编码顺序同 document_extraction】
        utf-8-sig（带 BOM）→ utf-8 → gb18030（中文旧文件）→ latin-1（永不失败）
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None
    # 空 / 超大 / 二进制 → 不索引
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _stem_upper(path: Path) -> str:
    """取文件名（不含后缀）并转大写。

    【调用链】被 _classify_file() 调用

    为什么要大写：README.md / readme.md / Readme.md 都该被识别成 README。
    """
    return path.stem.upper()


def _classify_file(path: Path, rel: str) -> tuple[str, str] | None:
    """判断一个文件属于哪一类、哪一级（★ 扫描的核心分类逻辑）。

    【调用链】_is_secret() / _stem_upper() → 本函数
    【被谁调用】_discover_project_docs()

    【数据流向】
        入：path（绝对路径）、rel（仓库内相对路径）
        出：(source_type, tier)；不需要索引则返回 None

    【分类规则】
        .md 文件：
            docs/ 下、或 README/ARCHITECTURE/CONTRIBUTING、或 ADR 开头 → authoritative
            CHANGELOG/HISTORY/RELEASES、或 .github/ISSUE_TEMPLATE/ → soft_clue
            其他 .md → 不索引（比如散落各处的临时笔记）
        依赖清单（MANIFEST_FILES）→ project_manifest / derived
        其余 → 不索引

    【为什么只索引 docs/ 和根目录的 README】
        仓库里 .md 可能成百上千（每个模块一个 README）。
        全索引会让"项目文档"这一类被稀释，检索时噪声很大。
        只留"真正的项目级文档"，精度优先。
    """
    # 密钥文件第一道拦
    if _is_secret(path):
        return None

    name = path.name
    stem = _stem_upper(path)
    rel_posix = rel.replace("\\", "/")

    if path.suffix.lower() == ".md":
        # 权威文档
        if rel_posix.startswith("docs/") or stem in AUTHORITATIVE_TOP_LEVEL or stem.startswith("ADR"):
            return "project_doc", "authoritative"
        # 弱线索
        if stem in SOFT_CLUE_TOP_LEVEL or rel_posix.startswith(".github/ISSUE_TEMPLATE/"):
            return "project_doc", "soft_clue"

    if name in MANIFEST_FILES:
        return "project_manifest", "derived"

    return None


def _is_inside(root: Path, candidate: Path) -> bool:
    """判断 candidate 是否真的在 root 里面（防符号链接逃逸）。

    【调用链】被 _discover_project_docs() 调用

    【为什么需要】
        os.walk 会跟随符号链接。如果仓库里有个链接指向 /etc，
        就会去扫系统目录 —— 既慢又可能读到不该读的文件。

    【实现】resolve 后用 relative_to 判断。抛 ValueError 说明不在里面。
    """
    try:
        candidate.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


# ════════════════════════════════════════════════════════════════
# 策略：读配置 + 模式判断
# ════════════════════════════════════════════════════════════════

def _normalize_patterns(value: Any, field_name: str) -> tuple[str, ...]:
    """清洗 glob 模式列表（★ 这里有安全校验）。

    【调用链】被 load_project_index_policy() 调用

    【数据流向】
        入：value（YAML 里的原始值）、field_name（用于报错）
        出：规范化后的模式元组（去重保序）

    【三步清洗】
        ① 反斜杠转正斜杠（Windows 路径 → POSIX glob）
        ② 去掉开头的 "./" 和 "/"（统一成相对路径）
        ③ 以 "/" 结尾的补 "**"（"docs/" 表示"docs 下所有文件"）

    【★ 安全检查：拒绝 .. 和 ../ 开头】
        否则用户能写 "../**" 把仓库外的文件也索引进来。
        这是配置注入的一种形式，必须拦。

    【为什么要去重保序】
        include 里重复写同一个模式没有意义，但顺序会影响"第一个命中"的语义。
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of repository-relative glob patterns")

    patterns: list[str] = []
    for raw_pattern in value:
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        pattern = raw_pattern.strip().replace("\\", "/")
        # 去掉 "./" 前缀（可能有多层）
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.lstrip("/")
        # ★ 目录穿越检查
        if pattern == ".." or pattern.startswith("../"):
            raise ValueError(f"{field_name} patterns must stay inside the repository")
        # "docs/" → "docs/**"
        if pattern.endswith("/"):
            pattern += "**"
        patterns.append(pattern)
    return tuple(dict.fromkeys(patterns))


def load_project_index_policy(root: Path) -> ProjectIndexPolicy:
    """读取 .devflow/index.yml，解析成策略对象。

    【调用链】_normalize_patterns() → 本函数
    【被谁调用】_discover_project_docs()

    【数据流向】
        入：root（仓库根目录）
        出：ProjectIndexPolicy（没有配置文件就返回全默认）

    【为什么所有校验失败都抛 ValueError】
        配置写错是"用户的问题"，应该立刻告诉他哪里错了，
        而不是悄悄用默认值跑（那样用户会以为配置生效了）。
        抛出的消息直接带字段名，用户能自己修。

    【版本号校验】
        version=1 是当前唯一支持的版本。将来加字段时用 version 区分，
        避免老配置被新代码误读。
    """
    root = root.resolve()
    config_path = root / PROJECT_INDEX_CONFIG_PATH

    # 没有配置文件 → 全默认（auto 模式，扫全部符合条件的）
    if not config_path.is_file():
        return ProjectIndexPolicy()

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {PROJECT_INDEX_CONFIG_PATH.as_posix()}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{PROJECT_INDEX_CONFIG_PATH.as_posix()} must contain a YAML object")

    # 版本校验
    version = raw.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported project index config version: {version}")

    project_docs = raw.get("project_docs") or {}
    if not isinstance(project_docs, dict):
        raise ValueError("project_docs must be a YAML object")

    # 模式
    mode = str(project_docs.get("mode") or "auto").strip().lower()
    if mode not in PROJECT_INDEX_MODES:
        raise ValueError(f"project_docs.mode must be one of: {', '.join(sorted(PROJECT_INDEX_MODES))}")

    include = _normalize_patterns(project_docs.get("include"), "project_docs.include")
    exclude = _normalize_patterns(project_docs.get("exclude"), "project_docs.exclude")

    # allowlist 模式必须给 include，否则什么都扫不到
    if mode == "allowlist" and not include:
        raise ValueError("project_docs.include cannot be empty when mode is allowlist")

    include_manifests = raw.get("include_manifests", True)
    if not isinstance(include_manifests, bool):
        raise ValueError("include_manifests must be true or false")

    return ProjectIndexPolicy(
        mode=mode,
        include=include,
        exclude=exclude,
        include_manifests=include_manifests,
        config_path=PROJECT_INDEX_CONFIG_PATH.as_posix(),
    )


def _matches_any(relative_path: str, patterns: tuple[str, ...]) -> bool:
    """相对路径是否命中任一 glob 模式。

    【调用链】被 _policy_allows() 调用

    用 fnmatchcase（区分大小写）而不是 fnmatch ——
    路径大小写在 Linux 上是有意义的，不能混。
    """
    return any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in patterns)


def _policy_allows(candidate: ProjectDocCandidate, policy: ProjectIndexPolicy) -> bool:
    """按策略判断某个候选是否该被索引。

    【调用链】_matches_any() → 本函数
    【被谁调用】_discover_project_docs()

    【判定顺序（★ 顺序很重要）】
        ① exclude 命中 → 直接排除（exclude 优先级最高，任何模式都盖不过）
        ② 依赖清单 → 只看 include_manifests 开关（不受 allowlist 影响）
        ③ allowlist 模式 → 必须命中 include
        ④ auto 模式 → 放行

    【为什么 exclude 优先】
        用户写 exclude 是"明确不要"，应该压过一切其他规则。
        如果 allowlist 优先，用户就没法用 include 大范围 + exclude 排除个别了。
    """
    if _matches_any(candidate.relative_path, policy.exclude):
        return False
    if candidate.source_type == "project_manifest":
        return policy.include_manifests
    if policy.mode == "allowlist":
        return _matches_any(candidate.relative_path, policy.include)
    return True


def _discover_project_docs(root: Path, max_files: int = 500) -> ProjectDocDiscovery:
    """遍历仓库，找出所有该索引的文档（★ 扫描主逻辑）。

    【调用链】追这个函数会进入：
        1. load_project_index_policy()  → 读策略
        2. _classify_file()             → 分类（内部判密钥）
        3. _policy_allows()             → 策略过滤
        4. _is_inside()                 → 防符号链接逃逸
    【被谁调用】discover_project_docs() / project_fingerprint() / scan_project_index()

    【数据流向】
        入：root（已 resolve）、max_files（上限 500）
        出：ProjectDocDiscovery（候选 + 被排除的 + 策略 + 是否截断）

    【为什么 dir_names[:] = 排序过滤（就地修改）】
        os.walk 支持通过修改 dir_names 来控制"要不要往下走"。
        必须用切片赋值（就地修改），不能重新赋值 —— 重新赋值不会影响遍历。
        这里同时做了两件事：
            ① 排序（保证多次扫描顺序一致 → fingerprint 稳定）
            ② 过滤（跳过 node_modules / .venv，以及除 .github 外的隐藏目录）

    【为什么有 max_files 上限】
        有些仓库有几千个文档。全索引会跑很久、占大量向量空间。
        500 个足够覆盖"项目级文档"，超了就标记 truncated 让用户知道。
    """
    root = root.resolve()
    policy = load_project_index_policy(root)

    results: list[ProjectDocCandidate] = []
    excluded_paths: list[str] = []
    truncated = False

    for current_root, dir_names, file_names in os.walk(root):
        # ★ 就地修改：控制遍历范围 + 排序保证稳定
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name not in SKIP_DIRS and (not name.startswith(".") or name in INDEXABLE_HIDDEN_DIRS)
        )
        for filename in sorted(file_names):
            path = Path(current_root) / filename
            # 防符号链接逃逸
            if not _is_inside(root, path):
                continue
            rel = path.relative_to(root).as_posix()
            classification = _classify_file(path, rel)
            # 不是文档类文件 → 跳过（大多数文件走这里，所以放最前面）
            if not classification:
                continue
            source_type, tier = classification
            candidate = ProjectDocCandidate(path=path, relative_path=rel, source_type=source_type, tier=tier)
            # 策略排除 → 记下来（用于前端解释"为什么没索引"）
            if not _policy_allows(candidate, policy):
                excluded_paths.append(rel)
                continue
            # 达到上限
            if len(results) >= max_files:
                truncated = True
                break
            results.append(candidate)
        if truncated:
            break

    return ProjectDocDiscovery(results, excluded_paths, policy, truncated)


def discover_project_docs(root: Path, max_files: int = 500) -> list[ProjectDocCandidate]:
    """对外简化版：只要候选列表（不要策略和被排除的信息）。

    【调用链】_discover_project_docs() → 本函数
    【被谁调用】api/routes/project_index.py（"预览会索引哪些文件"）
    """
    return _discover_project_docs(root, max_files=max_files).candidates


def project_fingerprint(repo: Repository) -> str:
    """生成"项目内容指纹"—— 决定要不要重新扫描。

    【调用链】_discover_project_docs() → 本函数
    【被谁调用】_state_payload() / mark_project_index_building() / scan_project_index()

    【数据流向】
        入：repo
        出：形如 "files:abc123..." 的字符串；路径不存在返回 ""

    【指纹包含什么】
        ① 策略本身（改了 include/exclude 也算"变了"，要重扫）
        ② 每个候选文件的：相对路径 + mtime_ns + size

    【为什么用 mtime + size 而不是内容哈希】
        内容哈希要读全部文件（几百个 × 几百 KB），太慢。
        mtime + size 已经能覆盖 99% 的"内容变了"场景，而且只要 stat 一次。
        （极端情况：改了内容但 mtime 和 size 都没变 —— 几乎不可能）

    【为什么排序（sort_keys=True + 文件已排序）】
        指纹必须稳定：同样的内容 → 同样的指纹。
        不排序的话，字典遍历顺序变化会导致指纹乱变，天天重扫。
    """
    root = _safe_project_root(repo)
    if not root:
        return ""

    hasher = hashlib.sha256()
    discovery = _discover_project_docs(root, max_files=1000)
    # ① 策略参与指纹（改配置 = 需要重扫）
    hasher.update(json.dumps(discovery.policy.as_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    # ② 每个文件的路径 + 修改时间 + 大小
    for candidate in discovery.candidates:
        try:
            stat = candidate.path.stat()
        except OSError:
            # 文件在扫描和 stat 之间被删了 → 跳过（不影响其他文件）
            continue
        hasher.update(candidate.relative_path.encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        hasher.update(str(stat.st_size).encode("ascii"))
    digest = hasher.hexdigest()
    return f"files:{digest}"


# ════════════════════════════════════════════════════════════════
# 状态管理
# ════════════════════════════════════════════════════════════════

def _get_state_row(db: Session, repo_id: uuid.UUID) -> ProjectIndexState | None:
    """取状态行（可能不存在）。"""
    return db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo_id).one_or_none()


def _get_or_create_state(db: Session, repo_id: uuid.UUID) -> ProjectIndexState:
    """取状态行，没有就建一条（status=missing）。

    【调用链】_get_state_row() → 本函数
    【被谁调用】mark_project_index_building() / snooze_project_index() / scan_project_index()
    """
    row = _get_state_row(db, repo_id)
    if row:
        return row
    row = ProjectIndexState(repo_id=repo_id, status="missing")
    db.add(row)
    db.flush()
    return row


def _state_payload(repo: Repository, row: ProjectIndexState | None, fingerprint: str | None = None) -> dict[str, Any]:
    """把状态行转成 API 返回结构（含 stale 判定）。

    【调用链】project_fingerprint() → 本函数
    【被谁调用】get_project_index_state() / scan_project_index()

    【数据流向】
        入：repo、row（可能为 None）、fingerprint（不传就现算）
        出：dict（前端进度条 + 诊断信息）

    【★ stale 判定逻辑】
        状态是 ready，但指纹和当前不一致 → 说明文件变了，索引过期了。
        这是"要不要重扫"的判断依据，也是前端提示"文档已更新，建议重建索引"的来源。

    【为什么 fingerprint 参数可以传】
        scan 流程里刚算过，传进来避免重复算（算一次要 stat 几百个文件）。
    """
    current_fingerprint = fingerprint if fingerprint is not None else project_fingerprint(repo)

    # 从未扫描过
    if row is None:
        return {
            "repo_id": repo.id,
            "status": "missing",
            "fingerprint": current_fingerprint,
            "docs_indexed": 0,
            "docs_total": 0,
            "error_message": None,
            "summary_json": None,
            "snoozed_until": None,
            "last_scan_at": None,
        }

    # 状态值可能是脏数据（历史版本写的），统一兜到 missing
    status = row.status if row.status in PROJECT_INDEX_STATUSES else "missing"
    # ★ 过期判定
    if status == "ready" and current_fingerprint and row.fingerprint != current_fingerprint:
        status = "stale"

    return {
        "repo_id": row.repo_id,
        "status": status,
        "fingerprint": row.fingerprint or current_fingerprint,
        "docs_indexed": row.docs_indexed or 0,
        "docs_total": row.docs_total or 0,
        "error_message": row.error_message,
        "summary_json": row.summary_json,
        "snoozed_until": row.snoozed_until,
        "last_scan_at": row.last_scan_at,
    }


def get_project_index_state(db: Session, repo: Repository) -> dict[str, Any]:
    """读当前索引状态（GET 接口用）。

    【调用链】_state_payload() → 本函数
    【被谁调用】api/routes/project_index.py
    """
    return _state_payload(repo, _get_state_row(db, repo.id))


def mark_project_index_building(db: Session, repo: Repository) -> ProjectIndexState:
    """标记"开始构建"（会立刻 commit，让前端能马上看到状态变化）。

    【调用链】_get_or_create_state() / project_fingerprint() → 本函数
    【被谁调用】api/routes/project_index.py（手动触发扫描前）

    【为什么这里要 commit 而不是 flush】
        前端会轮询状态。不 commit 的话轮询看到的还是旧状态，进度条不动。
    """
    row = _get_or_create_state(db, repo.id)
    row.status = "building"
    row.fingerprint = project_fingerprint(repo)
    row.error_message = None
    row.snoozed_until = None       # 用户主动扫描 → 取消暂缓
    row.updated_at = _now()
    db.commit()
    return row


def snooze_project_index(db: Session, repo: Repository, days: int = 7) -> ProjectIndexState:
    """用户点"稍后提醒" → 在 N 天内不再自动提示扫描。

    【调用链】_get_or_create_state() → 本函数
    【被谁调用】api/routes/project_index.py

    【为什么要把 status 改回 missing】
        如果当前是 failed，暂缓之后还留着 failed，前端会一直显示错误。
        改回 missing 表示"还没扫，但用户暂时不想扫"。
        已经是 ready / building 的保持原状（不要破坏已完成的状态）。
    """
    row = _get_or_create_state(db, repo.id)
    row.snoozed_until = _now() + timedelta(days=days)
    row.updated_at = _now()
    if row.status not in {"ready", "building"}:
        row.status = "missing"
    db.commit()
    db.refresh(row)
    return row


# ════════════════════════════════════════════════════════════════
# 内容提取
# ════════════════════════════════════════════════════════════════

def _extract_title(text: str, fallback: str) -> str:
    """取第一个 "# 标题" 作为文档标题。

    【调用链】被 _payloads_for_candidates() 调用

    没有一级标题就用 fallback（相对路径）——
    宁可显示 "docs/setup.md" 也不要显示空标题。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:500] or fallback
    return fallback


def _summary_from_text(text: str, limit: int = 360) -> str:
    """取文档第一段正文当摘要。

    【调用链】被 _build_summary() 的调用方（摘要展示）使用

    【为什么要过滤四类行】
        以 # 开头   → 标题（不是正文）
        以 | 开头   → 表格（摘出来读不懂）
        以 ``` 开头 → 代码块（不适合当摘要）
    只留真正的叙述段落。

    【兜底】一个合格段落都没有 → 取前 4 行硬拼（总比空着好）。
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    paragraphs = []
    for block in cleaned.split("\n\n"):
        # 块内多行合并成一行（摘要要能单行显示）
        value = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if value and not value.startswith("#") and not value.startswith("|") and not value.startswith("```"):
            paragraphs.append(value)

    summary = paragraphs[0] if paragraphs else " ".join(cleaned.splitlines()[:4])
    return summary[: limit - 3].rstrip() + "..." if len(summary) > limit else summary


def _detect_tech_stack(root: Path) -> list[str]:
    """按"清单文件是否存在"推断技术栈。

    【调用链】被 _build_summary() 调用

    【为什么用文件存在性而不是解析内容】
        只要看到 package.json 就知道是 Node 项目，不必解析它。
        简单、快、不会因为文件格式异常而失败。

    【返回的列表可能有多项】
        一个仓库可以同时有 package.json 和 pyproject.toml（前后端同仓）。
    """
    detectors = {
        "package.json": "node",
        "tsconfig.json": "typescript",
        "pyproject.toml": "python",
        "requirements.txt": "python",
        "go.mod": "go",
        "Cargo.toml": "rust",
        "pom.xml": "java",
        "build.gradle": "java",
    }
    return [tech for filename, tech in detectors.items() if (root / filename).exists()]


def _safe_dirs(root: Path, limit: int = 12) -> list[str]:
    """列出顶层目录（给前端展示项目结构）。

    【调用链】被 _build_summary() 调用

    【为什么要 limit 12 和跳过隐藏目录】
        前端只是展示"大概长什么样"。列 100 个目录没人看，
        列 .git / .venv 也没意义。

    【为什么 try/except OSError】
        iterdir 在某些情况下会抛（权限、路径失效）。这里返回空列表兜底。
    """
    dirs: list[str] = []
    try:
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if len(dirs) >= limit:
                break
            if child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            if child.is_dir():
                dirs.append(child.name)
    except OSError:
        return []
    return dirs


def _build_summary(
    repo: Repository,
    root: Path,
    candidates: list[ProjectDocCandidate],
    docs_indexed: int,
    discovery: ProjectDocDiscovery,
) -> dict[str, Any]:
    """组装 summary_json（前端"项目概览"面板的数据源）。

    【调用链】_detect_tech_stack() / _safe_dirs() → 本函数
    【被谁调用】scan_project_index()

    【数据流向】
        入：repo、root、candidates（全部候选）、docs_indexed（实际入库的切片数）、discovery
        出：dict（写进 project_index_states.summary_json）

    【为什么 docsList / excludedPaths 只留前 80】
        summary_json 是 JSON 列，存几千条会把行撑爆。
        80 条足够前端展示"索引了哪些 / 排除了哪些"，多了也没人看。

    【tierCoverage 和 sourceTypeCoverage 的区别】
        tier         按可信度分级（authoritative / soft_clue / derived）
        source_type  按数据类型（project_doc / project_manifest）
        两个维度都有用，所以都统计。
    """
    tier_coverage: dict[str, int] = {}
    source_type_coverage: dict[str, int] = {}
    docs_list: list[dict[str, str]] = []

    for item in candidates:
        tier_coverage[item.tier] = tier_coverage.get(item.tier, 0) + 1
        source_type_coverage[item.source_type] = source_type_coverage.get(item.source_type, 0) + 1
        docs_list.append({"path": item.relative_path, "tier": item.tier, "source_type": item.source_type})

    return {
        "projectName": repo.full_name,
        "rootPath": str(root),
        # sorted(set(...))：去重 + 稳定顺序（前端展示不会每次变）
        "techStack": sorted(set(_detect_tech_stack(root))),
        "dirStructure": _safe_dirs(root),
        "docsList": docs_list[:80],
        "tierCoverage": tier_coverage,
        "sourceTypeCoverage": source_type_coverage,
        "docsIndexed": docs_indexed,
        "indexPolicy": discovery.policy.as_dict(),
        "excludedFiles": len(discovery.excluded_paths),
        "excludedPaths": discovery.excluded_paths[:80],
        "discoveryTruncated": discovery.truncated,
    }


def _payloads_for_candidates(repo: Repository, root: Path, candidates: list[ProjectDocCandidate], fingerprint: str) -> list[DocumentPayload]:
    """把候选文件逐个读出来、切片、转成待入库的 payload。

    【调用链】追这个函数会进入：
        1. _read_text()              → 读文件
        2. _extract_title()          → 取标题
        3. chunk_document()          → 切片（rag/chunking）
        4. stable_source_id()        → 生成稳定 ID（vector_store）
    【被谁调用】scan_project_index()

    【数据流向】
        入：repo、root、candidates、fingerprint
        出：DocumentPayload 列表（每个切片一条）

    【三个关键设计】
        ① source_id 用 stable_source_id（uuid5）：
           同一份文档重复扫描必须得到同一个 ID，否则会不断产生重复文档。
        ② 切片参数 1800/180（比默认 1600/160 大）：
           文档类内容密度低（有大量标题和空白），块大一点更划算。
        ③ 标题带上章节名（"README · 安装"）：
           向量化时用"标题+正文"，带章节名能显著提升检索命中率。

    【为什么 metadata 要把 chunk 的所有字段都带上】
        chunk_document 会产出 section_path / page / parent_id / child_index 等，
        这些是后续"父块扩展"和"引用定位"的依据，不能丢。
        只排除 content（它已经单独作为 content 字段了）。
    """
    payloads: list[DocumentPayload] = []
    scanned_at = _now().isoformat()

    for candidate in candidates:
        text = _read_text(candidate.path)
        # 读不了或空文件 → 跳过（不产生空文档）
        if not text or not text.strip():
            continue

        # ★ 稳定 ID：同 (仓库, 类型, 相对路径) → 同一个 UUID
        source_id = stable_source_id(repo.id, candidate.source_type, candidate.relative_path)
        title = _extract_title(text, candidate.relative_path)
        chunks = chunk_document(text, candidate.relative_path, max_chars=1800, overlap=180)

        for index, chunk in enumerate(chunks):
            content = str(chunk["content"])
            section_title = chunk.get("section_title")
            payloads.append(
                DocumentPayload(
                    source_type=candidate.source_type,
                    source_id=source_id,
                    # 标题三档：有章节名用章节名 > 多块时带序号 > 直接用文档标题
                    title=f"{title} · {section_title}" if section_title else (f"{title}#{index + 1}" if len(chunks) > 1 else title),
                    content=content,
                    metadata={
                        # 把切片元数据全带上（除 content）
                        **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                        "display_title": title,
                        "path": candidate.relative_path,
                        "root_path": str(root),
                        # 可信度分级（前端可据此排序/标色）
                        "provenance_tier": candidate.tier,
                        "chunk_index": index,
                        "chunk_chars": len(content),
                        "total_chunks": len(chunks),
                        "chunk_strategy": "structure_aware_v1",
                        # 指纹 + 扫描时间：便于判断"这条索引是哪次扫的"
                        "fingerprint": fingerprint,
                        "scanned_at": scanned_at,
                    },
                )
            )
    return payloads


# ════════════════════════════════════════════════════════════════
# ★ 主流程
# ════════════════════════════════════════════════════════════════

def scan_project_index(db: Session, repo: Repository) -> dict[str, Any]:
    """扫描一个仓库的项目文档并建索引（★ 完整流程）。

    【调用链】追这个函数会进入：
        1. _get_or_create_state()        → 建/取状态行，标 building
        2. project_fingerprint()         → 算指纹
        3. _safe_project_root()          → 取本地路径
        4. _discover_project_docs()      → 扫描候选
        5. _payloads_for_candidates()    → 切片
        6. replace_documents()           → 整组替换写入
        7. _build_summary()              → 组装摘要
        8. EvidenceStore.rebuild_repo()  → 刷新证据库（同事务）
        9. rebuild_knowledge_graph()     → 重建知识图谱
    【被谁调用】
        api/routes/project_index.py（手动触发）
        scan_project_index_by_id()（后台任务）

    【数据流向】
        入：db、repo
        出：状态 payload（dict）—— 无论成功失败都返回，不抛异常

    【★ 为什么用 replace_documents(commit=False) 而不是逐条写】
        项目文档是"全量快照"语义：删掉的文件也要从库里消失。
        整组替换保证"库里 = 磁盘上"。
        而且 generate_embeddings=False / sync_vector_store=False ——
        这里只写 PostgreSQL，向量化交给统一的 reindex 流程
        （避免边扫边算向量，慢且难控内存）。

    【★ 为什么要刷新 EvidenceStore】
        EvidenceItem 是 ChatAgent 记忆工具用的"轻量检索面"。
        它也是从 documents 派生的。如果不刷新，
        被策略排除的旧文档会一直留在证据库里 —— 用户会搜到"已经不该存在的内容"。
        所以必须和主流程在同一个事务里刷新（注释里写了 "cannot survive a successful rescan"）。

    【失败处理】
        任何异常 → rollback → 标 failed + 记录原因 → 返回状态。
        不抛异常，因为调用方（route）要的是"状态"而不是"异常"。
    """
    # ── 步骤 1：标 building（先 commit，让前端看到进度）──
    row = _get_or_create_state(db, repo.id)
    fp = project_fingerprint(repo)
    row.status = "building"
    row.fingerprint = fp
    row.error_message = None
    row.snoozed_until = None
    row.updated_at = _now()
    db.commit()

    try:
        # ── 步骤 2：取本地路径 ──
        root = _safe_project_root(repo)
        if not root:
            raise RuntimeError("项目本地路径不存在，请先连接本地仓库或完成代码同步")

        # ── 步骤 3：扫描 + 切片 ──
        discovery = _discover_project_docs(root)
        candidates = discovery.candidates
        payloads = _payloads_for_candidates(repo, root, candidates, fp)

        # ── 步骤 4：插一条"项目概况"（放在最前面，检索"这是什么项目"时最先命中）──
        overview = {
            "repo": repo.full_name,
            "description": repo.description or "",
            "default_branch": repo.default_branch or "",
            "local_path": str(root),
            "indexed_files": [candidate.relative_path for candidate in candidates[:80]],
        }
        payloads.insert(
            0,
            DocumentPayload(
                source_type="project_overview",
                source_id=stable_source_id(repo.id, "project_overview", "overview"),
                title=f"项目概况：{repo.full_name}",
                content=json.dumps(overview, ensure_ascii=False, indent=2),
                metadata={"display_title": "项目概况", "fingerprint": fp, "root_path": str(root), "chunk_index": 0},
            ),
        )

        # ── 步骤 5：整组替换写入（只写 PG，不算向量）──
        docs = replace_documents(
            db,
            repo.id,
            PROJECT_INDEX_SOURCE_TYPES,
            payloads,
            commit=False,
            generate_embeddings=False,
            sync_vector_store=False,
        )

        # ── 步骤 6：更新状态为 ready ──
        summary = _build_summary(repo, root, candidates, len(docs), discovery)
        row = _get_or_create_state(db, repo.id)
        row.status = "ready"
        row.fingerprint = fp
        row.docs_indexed = len(docs)
        row.docs_total = len(candidates)
        row.summary_json = summary
        row.last_scan_at = _now()
        row.error_message = None
        row.updated_at = _now()

        # EvidenceItems are a denormalized search surface used by ChatAgent's
        # memory tools. Refresh them in the same transaction so excluded
        # project documents cannot survive a successful rescan.
        from app.services.chat_memory import EvidenceStore

        EvidenceStore(db).rebuild_repo(repo)
        db.commit()
        db.refresh(row)

        # ── 步骤 7：重建知识图谱（在 commit 之后，失败不影响索引结果）──
        rebuild_knowledge_graph(db, repo)
        return _state_payload(repo, row, fingerprint=fp)

    except Exception as exc:
        # ── 失败：回滚 → 标 failed → 返回状态（不抛异常）──
        db.rollback()
        row = _get_or_create_state(db, repo.id)
        row.status = "failed"
        row.error_message = str(exc)
        row.updated_at = _now()
        db.commit()
        db.refresh(row)
        return _state_payload(repo, row, fingerprint=fp)


async def scan_project_index_by_id(repo_id: uuid.UUID) -> None:
    """后台任务入口：按 repo_id 扫描（自己开 Session）。

    【调用链】scan_project_index() → 本函数
    【被谁调用】api/routes/project_index.py 的 BackgroundTasks

    【为什么必须自己开 Session】
        后台任务不在请求生命周期内，请求的 Session 已经关了。
        复用会抛 InvalidRequestError。
    """
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return
        scan_project_index(db, repo)
    finally:
        db.close()


def is_snoozed(state: dict[str, Any]) -> bool:
    """判断当前是否在"稍后提醒"的暂缓期内。

    【调用链】_aware() → 本函数
    【被谁调用】services/scheduler.py（决定要不要自动扫描）、api/routes/project_index.py

    【为什么用 _aware 包一下】
        数据库读出来的 snoozed_until 可能是 naive datetime，
        直接和 aware 的 _now() 比较会抛 TypeError。
    """
    until = _aware(state.get("snoozed_until"))
    return bool(until and until > _now())
