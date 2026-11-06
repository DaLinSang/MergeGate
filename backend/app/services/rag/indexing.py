"""
仓库数据入 RAG（Issue / PR / CI → 向量库）
========================================

【在 RAG 流水线中的位置】
    入库链路的**第二个入口**（第一个是用户上传文档）：

        services/repos_sync.py   把 GitHub 数据同步到 PG
             ↓
        ★ 本文件 index_repository_documents()   把 PG 里的数据转成 RAG 文档
             ↓
        vector_store.replace_documents()        整组替换写入
             ↓
        documents 表 + Milvus

    【两个入口的区别】
        上传文档 → 解析文件（PDF/DOCX）→ 切片 → 向量化
        仓库数据 → 已经结构化在 PG 里 → 直接拼文本 → 向量化
        两条路最后都汇到 vector_store，所以检索侧完全不用区分。

【业务定位】
    数据流 3 的第二个入口：把已同步的 Issue / PR / CI 日志批量写进向量库。

【数据流向】
    上游（谁调它）：
        api/routes/rag.py（手动 reindex）
        services/rag/qa.py::reindex_documents()（重建索引）
        services/repos_sync.py 之后（同步完自动建索引）
    下游（它调谁）：
        embeddings.embed_knowledge_texts()  向量化
        vector_store.replace_documents()    整组替换写入

【核心规则】
    1. 只索引"非结构化长文本"；结构化的东西（团队成员、依赖清单）直查数据库。
    2. CI 日志必须先脱敏再入库（可能含密钥）。
    3. 只索引【失败】的 CI 运行 —— 成功的日志没有检索价值，白占空间。

【本文件函数地图】
    index_repository_documents()   ★ 主流程（4 类数据 + 向量化 + 整组替换）
        ├ _payload()               构造一条待入库文档（空内容跳过）
        ├ _iso()                   datetime → ISO 字符串
        └ _ci_log_chunks()          CI 日志脱敏 + 分块
             └ sanitize_ci_log()    ★ 脱敏（6 类规则，最关键的安全函数）
"""

import re

from sqlalchemy.orm import Session, selectinload

from app.db.models import Issue, KnowledgeBaseConfig, PullRequest, Repository, WorkflowRun
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.vector_store import DocumentPayload, replace_documents

# 从仓库同步数据里索引进 RAG 的三类来源
# 注意：这里只有三类，因为 project_doc / knowledge_file / memory_note 走别的入口
SYNCED_REPO_SOURCE_TYPES = ["issue", "pull_request", "workflow_run"]

# CI 日志切片参数：每块 8000 字符，最多 6 块（= 48000 字符上限）
# 为什么 CI 日志用 8000 而不是文档的 800：
#   日志是"连续的行"，切太碎会丢上下文（错误栈前后几行才有意义）
CI_LOG_CHUNK_CHARS = 8000
CI_LOG_MAX_CHUNKS = 6


# ════════════════════════════════════════════════════════════════
# ★ 安全：CI 日志脱敏
# ════════════════════════════════════════════════════════════════

def sanitize_ci_log(text: str) -> str:
    """CI 日志脱敏 —— 入库前必须执行，否则密钥会进向量库。

    【调用链】被 _ci_log_chunks() 调用
    【被谁调用】_ci_log_chunks()（每个 CI 日志入库前必经）

    【为什么这一步不能省】
        CI 日志里经常出现环境变量、token、私钥。
        一旦进了向量库，任何能检索的人都可能把密钥捞出来。

    脱敏覆盖：
        PEM 私钥块、GitHub token、AWS Access Key、JWT、
        Authorization: Bearer 头、GitHub Actions 的 ::add-mask::、
        以及各种 *api_key* / *token* / *secret* / *password* 的 key=value 形式。

    【数据流向】
        入：原始日志
        出：脱敏后的日志（长度基本不变，只是敏感片段被替换成 [REDACTED_XXX]）

    【为什么用 [REDACTED_XXX] 而不是直接删掉】
        删掉会让日志结构变化（"token=***"变成"token="），
        模型读到会以为"这里有个空值"，反而可能误导。
        保留占位符能说明"这里原本有个 token，但被脱敏了"。

    【规则的顺序为什么不能乱】
        ① 私钥块要最先（它跨多行，可能包含后面所有模式）
        ② 具体格式的 token（ghp_ / AKIA / JWT）要在通用 key=value 之前
           —— 否则通用规则会把 "token=ghp_xxx" 只替换成 "token=[REDACTED]"，
              虽然也安全，但丢失了"这是 GitHub token"的信息
    """
    # ── 规则 1：PEM 私钥块（跨多行，必须用 DOTALL）──
    # 匹配 -----BEGIN RSA PRIVATE KEY----- ... -----END RSA PRIVATE KEY-----
    sanitized = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # ── 规则 2：GitHub token（ghp_ / gho_ / ghu_ / ghs_ / ghr_ + 20 位以上）──
    sanitized = re.sub(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED_GITHUB_TOKEN]", sanitized)

    # ── 规则 3：AWS Access Key（AKIA + 16 位大写字母数字）──
    sanitized = re.sub(r"\bAKIA[A-Z0-9]{16}\b", "[REDACTED_AWS_ACCESS_KEY]", sanitized)

    # ── 规则 4：JWT（三段 base64url，用点连接）──
    sanitized = re.sub(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        "[REDACTED_JWT]",
        sanitized,
    )

    # ── 规则 5：Authorization: Bearer xxx ──
    # 保留 "authorization: bearer" 前缀（说明这里有认证），只替换后面的值
    sanitized = re.sub(
        r"(?i)\b(authorization\s*:\s*bearer)\s+[^\s,;]+",
        r"\1 [REDACTED]",
        sanitized,
    )

    # ── 规则 6：GitHub Actions 的 ::add-mask::（本身就是在声明"这是密钥"）──
    sanitized = re.sub(r"(?i)(::add-mask::)[^\r\n]+", r"\1[REDACTED]", sanitized)

    # ── 规则 7（兜底）：各种 key=value / key: value 形式的密钥 ──
    # 覆盖 api_key / access_key / access_token / auth_token / token /
    #      secret / password / passwd / private_key / client_secret / account_key
    # 允许前后有前缀后缀（如 GITHUB_TOKEN、DB_PASSWORD、aws_secret_access_key）
    return re.sub(
        r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|"
        r"token|secret|password|passwd|private[_-]?key|client[_-]?secret|account[_-]?key)[a-z0-9_]*)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        sanitized,
    )


def _ci_log_chunks(text: str) -> tuple[list[str], bool]:
    """把 CI 日志切成固定大小块，并返回"是否被截断"。

    【调用链】sanitize_ci_log() → 本函数
    【被谁调用】index_repository_documents()

    【数据流向】
        入：原始日志
        出：(切片列表, 是否被截断)

    【★ 顺序很重要：先脱敏，再截断】
        如果先截断再脱敏，被截断处可能把 token 切成两半 ——
        半截 token 匹配不到脱敏规则，就会漏进向量库。

    【为什么返回 truncated 标记】
        截断意味着"日志不完整"，写进 metadata 让用户知道
        "这段证据只覆盖了日志的前 48000 字符"。
    """
    # ① 先脱敏（必须在截断之前）
    sanitized = sanitize_ci_log(text.strip())

    # ② 再截断到总上限
    max_chars = CI_LOG_CHUNK_CHARS * CI_LOG_MAX_CHUNKS
    bounded = sanitized[:max_chars]

    # ③ 切成固定大小块（跳过全空白的块）
    chunks = [
        bounded[start : start + CI_LOG_CHUNK_CHARS]
        for start in range(0, len(bounded), CI_LOG_CHUNK_CHARS)
        if bounded[start : start + CI_LOG_CHUNK_CHARS].strip()
    ]
    return chunks, len(sanitized) > max_chars


def _iso(value) -> str | None:
    """datetime → ISO 字符串（元数据里统一用字符串存）。

    【调用链】被 index_repository_documents() 调用

    【为什么要转】
        metadata 会序列化成 JSON 存进 Milvus / PG 的 JSON 列。
        datetime 对象不能直接 JSON 序列化，必须转字符串。
    """
    return value.isoformat() if value else None


def _payload(source_type: str, source_id, title: str, content: str, metadata: dict) -> DocumentPayload | None:
    """构造一条待入库文档。内容为空则返回 None（跳过，不写空文档）。

    【调用链】被 index_repository_documents() 调用（4 处）

    【为什么空内容要跳过】
        空文档入库后，向量是"空文本的向量"（几乎随机），
        检索时可能被召回，但正文什么都没有 —— 纯噪声。
    """
    if not content.strip():
        return None
    return DocumentPayload(
        source_type=source_type,
        source_id=source_id,
        title=title[:500],      # 标题列有长度限制
        content=content,
        metadata=metadata,
    )


# ════════════════════════════════════════════════════════════════
# ★ 主流程
# ════════════════════════════════════════════════════════════════

def index_repository_documents(
    db: Session,
    repo_id,
    *,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> int:
    """把一个仓库的 Issue / PR / 失败 CI 全部索引进 RAG。返回写入的文档数。

    【调用链】追这个函数会进入：
        1. _payload()             → 构造文档（4 类来源各一次）
        2. _iso()                 → 时间转字符串
        3. _ci_log_chunks()       → CI 日志脱敏 + 分块
              └ sanitize_ci_log()
        4. embed_knowledge_texts() → 向量化（可选）
        5. replace_documents()     → 整组替换写入（PG + Milvus）
    【被谁调用】
        api/routes/rag.py                 ← 手动重建索引
        services/rag/qa.py::reindex_documents()  ← 全库重建
        services/repos_sync.py 之后        ← 同步完自动建索引

    【数据流向】
        入：db、repo_id、两个开关
        出：写入的文档数（int）

    两个开关的用途：
        generate_embeddings=False → 只写 PostgreSQL，不算向量（测试/降级用）
        sync_vector_store=False   → 不算向量也不写 Milvus（纯关键词模式）

    【为什么要"整组替换"而不是增量更新】
        GitHub 上的 Issue 可能被删除。增量更新感知不到"删除"，
        旧数据会永远留在库里。整组替换保证"库里 = GitHub 当前状态"。
    """
    repo = db.get(Repository, repo_id)
    # 仓库不存在 → 直接返回 0（不报错，可能是并发删除）
    if repo is None:
        return 0

    payloads: list[DocumentPayload] = []

    # ════════════════════════════════════════════════════════════
    # ── ① Issue：标题 + 正文 ──
    # ════════════════════════════════════════════════════════════
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    for issue in issues:
        # 正文拼成 "标题\n正文"（标题参与检索能提升命中率，和上传文档一致）
        content = "\n".join([issue.title, issue.body or ""])
        item = _payload(
            "issue",
            issue.id,
            issue.title,
            content,
            {
                "number": issue.number,
                "labels": issue.labels or [],
                "state": issue.state,
                "author": issue.author,
                "assignees": issue.assignees or [],
                "created_at": _iso(issue.created_at),
                "updated_at": _iso(issue.updated_at),
                "closed_at": _iso(issue.closed_at),
            },
        )
        if item:
            payloads.append(item)

    # ════════════════════════════════════════════════════════════
    # ── ② PR：标题 + 说明 + 所有 Review 评论 ──
    # ════════════════════════════════════════════════════════════
    # selectinload 预加载子表，避免 N+1 查询
    # （不加的话：查 100 个 PR 会额外发 200 条 SQL 查 files 和 comments）
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    for pr in prs:
        # 评论拼成 "path:line author: body"，保留定位信息
        comments_text = "\n".join(
            f"{comment.path or ''}:{comment.line or ''} {comment.author or ''}: {comment.body or ''}"
            for comment in pr.review_comments
        )
        content = "\n".join(
            [
                pr.title,
                pr.body or "",
                "review comments:",
                comments_text,
            ]
        )
        item = _payload(
            "pull_request",
            pr.id,
            pr.title,
            content,
            {
                "number": pr.number,
                "state": pr.state,
                "author": pr.author,
                "files": [file.filename for file in pr.files],
                "base_branch": pr.base_branch,
                "head_branch": pr.head_branch,
                "created_at": _iso(pr.created_at),
                "updated_at": _iso(pr.updated_at),
                "merged_at": _iso(pr.merged_at),
            },
        )
        if item:
            payloads.append(item)

    # ════════════════════════════════════════════════════════════
    # ── ③ 失败 CI：只索引失败的，且日志要脱敏 + 分块 ──
    # ════════════════════════════════════════════════════════════
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    for run in runs:
        # 成功的 Run 日志没有检索价值，跳过
        # （成功的日志通常几万行，全索引会白白撑大向量库）
        if run.conclusion != "failure" or not (run.logs_text or "").strip():
            continue

        log_chunks, logs_truncated = _ci_log_chunks(run.logs_text or "")

        for chunk_index, log_chunk in enumerate(log_chunks):
            # 名称 + 日志片段（名称参与检索，便于"哪个 workflow 失败了"这类问题）
            content = "\n".join([run.name, log_chunk])
            item = _payload(
                "workflow_run",
                run.id,
                # 多块时给标题加序号，避免 6 个切片标题完全相同
                run.name if len(log_chunks) == 1 else f"{run.name}#{chunk_index + 1}",
                content,
                {
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "html_url": run.html_url,
                    "jobs": [job.get("name") for job in (run.jobs or []) if isinstance(job, dict)],
                    "created_at": _iso(run.created_at),
                    "updated_at": _iso(run.updated_at),
                    "chunk_index": chunk_index,
                    "total_chunks": len(log_chunks),
                    "logs_truncated": logs_truncated,
                    "secrets_redacted": True,      # 标记已脱敏，便于审计
                },
            )
            if item:
                payloads.append(item)

    # ════════════════════════════════════════════════════════════
    # ── ④ 向量化（可选）──
    # ════════════════════════════════════════════════════════════
    vectors = None
    embedding_info = None

    # 只有需要算向量时才查配置（省一次查询）
    config = (
        db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_id).one_or_none()
        if generate_embeddings
        else None
    )

    if config is not None and payloads:
        # 与 document_processing 一致：向量化"标题 + 正文"，提升命中率
        vectors = embed_knowledge_texts(
            [f"{payload.title}\n{payload.content}" for payload in payloads],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        embedding_info = {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions,
        }

    # ════════════════════════════════════════════════════════════
    # ── ⑤ 整组替换这三类来源（先删旧的再写新的，保证幂等）──
    # ════════════════════════════════════════════════════════════
    # commit=False：由上层（route 或 qa.reindex_documents）统一提交
    docs = replace_documents(
        db,
        repo_id,
        SYNCED_REPO_SOURCE_TYPES,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    return len(docs)
