"""
RAG 知识库路由
==============

【路由清单】（15 个）
    GET    /models                                      可用 embedding / LLM 模型目录
    GET    /repositories                                列所有知识库（含统计）
    GET    /repositories/{repo_id}                      单个知识库详情
    PATCH  /repositories/{repo_id}/config               改切片/向量配置（会校验契约）
    GET    /repositories/{repo_id}/documents            列已上传文档
    POST   /repositories/{repo_id}/documents            上传文档（202 + 后台流水线）
    GET    /repositories/{repo_id}/documents/{id}/chunks  看某文档切出来的块
    POST   /repositories/{repo_id}/documents/{id}/retry   重跑失败的文档
    DELETE /repositories/{repo_id}/documents/{id}       删除文档（连同向量）
    POST   /repositories/{repo_id}/retrieval-tests      跑一次检索测试（存历史）
    GET    /repositories/{repo_id}/retrieval-tests      检索测试历史
    GET    /{repo_id}/status                            RAG 运行状态（索引了多少块）
    POST   /{repo_id}/search                            只检索，不生成
    POST   /{repo_id}/ask                               检索 + 生成带引用回答
    POST   /{repo_id}/reindex                           重建索引（换 embedding 后必做）

【数据流向】
    入库：POST /documents
      → 校验大小/后缀/去重 → 存文件 → knowledge_source_documents(queued)
      → 后台 services/rag/document_processing.process_knowledge_document()
          → 解析 → 切片 → 向量化 → 双写（Milvus + PostgreSQL）

    查询：POST /search 或 /ask
      → services/rag/knowledge_base.retrieve_knowledge()  六阶段
      → /ask 再走 services/rag/qa.answer_question()（含 answer_gate + 引用编号）

【要点】
    1. 上传接口返回 202 + 后台任务：解析大 PDF 可能几十秒，不能阻塞请求。
    2. 内容去重靠 sha256：同仓库同内容只留一份（failed 状态的不算，允许重传）。
    3. 改 config 会校验两件事：overlap < chunk_size、embedding 契约（provider+model+维度）。
       契约不一致会导致新旧向量不在同一空间，检索结果直接变随机 —— 所以用 409 拦住。
    4. knowledge_base 类型的 repo 被 _repo_or_404 挡掉：它是"虚拟仓库"，
       只是用来挂知识库文档的，不能当普通仓库操作。
"""

import asyncio
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    Document,
    KnowledgeSourceDocument,
    Repository,
    RetrievalTestRun,
)
from app.db.session import get_db
from app.schemas.rag import (
    KnowledgeBaseConfigUpdate,
    KnowledgeBaseResponse,
    KnowledgeChunkResponse,
    KnowledgeDocumentResponse,
    ModelCatalogResponse,
    RAGAnswerResponse,
    RAGQueryRequest,
    RAGReindexResponse,
    RAGSearchResponse,
    RetrievalTestHistoryResponse,
    RetrievalTestRequest,
    RetrievalTestResponse,
)
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES, process_knowledge_document
from app.services.rag.embeddings import validate_embedding_contract
from app.services.rag.knowledge_base import (
    config_dict,
    get_or_create_config,
    model_catalog,
    record_retrieval_test,
    retrieve_knowledge,
    runtime_status,
)
from app.services.rag.qa import answer_question, build_sources, rag_status, reindex_documents
from app.services.rag.vector_store import delete_document_group

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库。knowledge_base 类型的虚拟仓库不算，直接 404。"""
    repo = db.get(Repository, repo_id)
    if repo is None or repo.provider == "knowledge_base":
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _safe_filename(filename: str) -> str:
    """清洗上传文件名：去掉路径、把危险字符换成下划线、去掉首尾的点。

    为什么必须清洗：文件名会拼进存储路径，用户传 "../../etc/passwd" 会写到仓库外。
    只取 Path(...).name 就把路径部分剥掉了。
    """
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    """上传目录：{upload_dir}/{repo_id}/，不存在就创建。

    upload_dir 配的是相对路径时，要拼上进程工作目录 —— 否则 uvicorn 从别的目录启动时
    文件会写到意想不到的地方。
    """
    root = Path(settings.upload_dir)    # 读取配置里的 upload_dir（上传文件的基础根目录），转成 Path 对象
    if not root.is_absolute():
        root = Path.cwd() / root    # 相对路径 → 拼上当前程序运行目录，转成绝对路径
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_chunking(chunk_size: int, overlap: int) -> None:
    """切片参数校验。overlap >= chunk_size 会导致切不完（每块都是纯重叠）。"""
    if overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="切片重叠必须小于切片长度")


def _document_response(item: KnowledgeSourceDocument) -> KnowledgeDocumentResponse:
    """文档 ORM → 响应模型（含 status / chunk_count 等进度字段）。"""
    return KnowledgeDocumentResponse(
        id=item.id,
        repo_id=item.repo_id,
        name=item.name,
        source_type=item.source_type,
        status=item.status,
        content_type=item.content_type,
        file_size=item.file_size,
        character_count=item.character_count,
        chunk_count=item.chunk_count,
        error_message=item.error_message,
        created_at=item.created_at,
        updated_at=item.updated_at,
        completed_at=item.completed_at,
    )


def _knowledge_base(db: Session, repo: Repository) -> KnowledgeBaseResponse:
    """拼知识库详情：仓库信息 + 配置 + 统计 + 运行时状态。"""
    config = get_or_create_config(db, repo.id)
    # 一次查询同时拿"文档数"和"总字符数"（coalesce 保证没文档时返回 0 而不是 None）
    document_count, character_count = (
        db.query(func.count(KnowledgeSourceDocument.id), func.coalesce(func.sum(KnowledgeSourceDocument.character_count), 0))
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .one()
    )
    # chunk_count 统计的是 documents 表（= 已经切好入库的块）
    chunk_count = db.query(func.count(Document.id)).filter(Document.repo_id == repo.id).scalar() or 0
    return KnowledgeBaseResponse(
        id=repo.id,
        name=repo.name,
        full_name=repo.full_name,
        description=repo.description or "",
        provider=repo.provider or "github",
        document_count=int(document_count or 0),
        chunk_count=int(chunk_count),
        character_count=int(character_count or 0),
        config=config_dict(config),
        runtime=runtime_status(config),
        created_at=repo.created_at,
        updated_at=repo.updated_at,
    )


@router.get("/models", response_model=ModelCatalogResponse)
async def get_model_catalog() -> ModelCatalogResponse:
    """可选模型目录。前端"知识库设置"里的下拉框数据来源。"""
    return ModelCatalogResponse.model_validate(model_catalog())


@router.get("/repositories", response_model=list[KnowledgeBaseResponse])
async def list_rag_repositories(db: Session = Depends(get_db)) -> list[KnowledgeBaseResponse]:
    """列所有知识库（排除 knowledge_base 虚拟仓库本身）。"""
    repos = (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )
    return [_knowledge_base(db, repo) for repo in repos]


@router.get("/repositories/{repo_id}", response_model=KnowledgeBaseResponse)
async def get_rag_repository(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeBaseResponse:
    """单个知识库详情。"""
    return _knowledge_base(db, _repo_or_404(db, repo_id))


@router.patch("/repositories/{repo_id}/config", response_model=KnowledgeBaseResponse)
async def update_rag_config(
    repo_id: uuid.UUID,
    payload: KnowledgeBaseConfigUpdate,
    db: Session = Depends(get_db),
) -> KnowledgeBaseResponse:
    """改知识库配置。改完 embedding 相关字段必须重新 reindex，否则新旧向量不兼容。"""
    repo = _repo_or_404(db, repo_id)
    # commit=False：下面还要做校验，失败时不想留下半成品
    config = get_or_create_config(db, repo_id, commit=False)
    # exclude_unset：只处理前端真正传了的字段（部分更新）
    changes = payload.model_dump(exclude_unset=True)

    # 校验一：切片参数（用"改后的值"校验，不是旧值）
    next_size = int(changes.get("chunk_size", config.chunk_size))
    next_overlap = int(changes.get("chunk_overlap", config.chunk_overlap))
    _validate_chunking(next_size, next_overlap)

    # 校验二：向量空间契约。不一致的话新旧向量不能比较，检索结果会变成随机
    try:
        validate_embedding_contract(
            str(changes.get("embedding_provider", config.embedding_provider)),
            str(changes.get("embedding_model", config.embedding_model)),
            int(changes.get("embedding_dimensions", config.embedding_dimensions)),
        )
    except ValueError as exc:
        # 409 Conflict：配置和现有数据冲突，不是参数格式错误（那是 400）
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    for key, value in changes.items():
        setattr(config, key, value)
    db.commit()
    db.refresh(repo)
    return _knowledge_base(db, repo)


@router.get("/repositories/{repo_id}/documents", response_model=list[KnowledgeDocumentResponse])
async def list_documents(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeDocumentResponse]:
    """列已上传文档，新的在前。前端轮询这个接口看 status 变化。"""
    _repo_or_404(db, repo_id)
    items = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo_id)
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .all()
    )
    return [_document_response(item) for item in items]


@router.post("/repositories/{repo_id}/documents", response_model=KnowledgeDocumentResponse, status_code=202)
async def upload_document(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """上传文档。202 = 已受理，真正解析在后台跑。"""
    _repo_or_404(db, repo_id)

    # 多读 1 字节：用来判断"是否超过上限"，避免把超大文件全读进内存
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="文件不能超过 12 MB")

    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        # 415 Unsupported Media Type：类型不支持，不是参数错误
        raise HTTPException(status_code=415, detail=f"暂不支持 {suffix or '无扩展名'} 文件")

    # ── 内容去重：同仓库 + 同 sha256 + 不是 failed → 直接返回已有记录 ──
    # 排除 failed 是故意的：上次解析失败的话，用户重传同一份应该重新处理
    checksum = hashlib.sha256(data).hexdigest()
    duplicate = (
        db.query(KnowledgeSourceDocument)
        .filter(
            KnowledgeSourceDocument.repo_id == repo_id,
            KnowledgeSourceDocument.checksum == checksum,
            KnowledgeSourceDocument.status != "failed",
        )
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .first()
    )
    if duplicate is not None:
        return _document_response(duplicate)

    document = KnowledgeSourceDocument(
        repo_id=repo_id,
        name=filename,
        content_type=file.content_type,
        checksum=checksum,
        file_size=len(data),
        stored_path="pending",  # 先占位，拿到 id 后才有最终路径
        status="queued",
    )
    db.add(document)    # 放进当前 session 的待变更队列，ORM 标记：这个对象要 insert。
                        # 此时没有任何数据库网络请求，数据库完全不知道这个对象 —— 对象还在 Python 内存里。
    db.flush()  # 把 INSERT 发给数据库（还没 commit）。作用：拿到数据库生成的主键 id。
                # 后面如果 rollback，这次 INSERT 也会被撤销。

    # 文件名带 id 前缀，避免同名文件互相覆盖
    stored_path = _upload_root(repo_id) / f"{document.id}-{filename}"
    try:
        # to_thread：write_bytes 是阻塞 IO，不能直接在事件循环里跑（会卡住整个服务）
        await asyncio.to_thread(stored_path.write_bytes, data)
        document.stored_path = str(stored_path)
        db.commit()
    except Exception:
        # 写文件或 commit 失败：回滚数据库 + 删掉可能已写了一半的文件
        db.rollback()
        try:
            await asyncio.to_thread(stored_path.unlink, missing_ok=True)
        except OSError:
            pass
        raise

    db.refresh(document)    # commit 之后从数据库重新查一遍这条记录，刷新内存里的 document 对象。
                            # 单独发一次 SELECT，按主键把最新数据读回来，覆盖内存中的值。

    # 后台跑完整流水线（解析 → 切片 → 向量化 → 双写）
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


@router.get("/repositories/{repo_id}/documents/{document_id}/chunks", response_model=list[KnowledgeChunkResponse])
async def list_document_chunks(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[KnowledgeChunkResponse]:
    """看某文档切出来的块。调试切片效果用（块太长/太碎一眼就能看出来）。"""
    _repo_or_404(db, repo_id)
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_id == document_id)
        .order_by(Document.created_at.asc())
        .all()
    )
    return [
        KnowledgeChunkResponse(
            id=row.id,
            # position 从 1 开始给用户看；优先用 meta 里的 chunk_index
            position=int((row.meta or {}).get("chunk_index", index)) + 1,
            title=row.title,
            content=row.content,
            character_count=len(row.content),
        )
        for index, row in enumerate(rows)
    ]


@router.post("/repositories/{repo_id}/documents/{document_id}/retry", response_model=KnowledgeDocumentResponse)
async def retry_document(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """重跑失败文档。文件还在磁盘上，直接重新走流水线。"""
    _repo_or_404(db, repo_id)
    document = db.get(KnowledgeSourceDocument, document_id)
    if document is None or document.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    # 先改状态再派任务：前端下次轮询能看到 queued
    document.status = "queued"
    document.error_message = None
    db.commit()
    db.refresh(document)
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


@router.delete("/repositories/{repo_id}/documents/{document_id}")
async def delete_document(repo_id: uuid.UUID, document_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """删除文档：清向量 + 删文件 + 删记录。"""
    _repo_or_404(db, repo_id)
    source = db.get(KnowledgeSourceDocument, document_id)
    if source is None or source.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    # 先删向量（commit=False，和下面删记录同一个事务）
    delete_document_group(db, repo_id, document_id, source_type="knowledge_file", commit=False)
    # 再删磁盘文件。文件删失败不影响数据库记录删除 —— 最多留个孤儿文件
    path = Path(source.stored_path)
    if path.exists() and path.is_file():
        path.unlink()
    db.delete(source)
    db.commit()
    return {"id": str(document_id), "deleted": True}


@router.post("/repositories/{repo_id}/retrieval-tests", response_model=RetrievalTestResponse)
async def run_retrieval_test(
    repo_id: uuid.UUID,
    payload: RetrievalTestRequest,
    db: Session = Depends(get_db),
) -> RetrievalTestResponse:
    """跑一次检索测试（调参用）。每次结果都存历史，可以对比不同参数。"""
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.query.strip(),
        top_k=payload.top_k,
        retrieval_method=payload.retrieval_method,
        rerank_enabled=payload.rerank_enabled,
        score_threshold_enabled=payload.score_threshold_enabled,
        score_threshold=payload.score_threshold,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    # trace 里存了每阶段的 input/output 数量，是排查"为什么没结果"的关键
    run = record_retrieval_test(db, repo_id, payload.query.strip(), trace, results, duration_ms)
    return RetrievalTestResponse(
        id=run.id,
        query=run.query,
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
        created_at=run.created_at,
    )


@router.get("/repositories/{repo_id}/retrieval-tests", response_model=list[RetrievalTestHistoryResponse])
async def list_retrieval_tests(
    repo_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RetrievalTestHistoryResponse]:
    """检索测试历史（只返回摘要，点进去再看详情）。"""
    _repo_or_404(db, repo_id)
    runs = (
        db.query(RetrievalTestRun)
        .filter(RetrievalTestRun.repo_id == repo_id)
        .order_by(RetrievalTestRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrievalTestHistoryResponse(
            id=run.id,
            query=run.query,
            result_count=len(run.results_json or []),
            took_ms=run.duration_ms,
            created_at=run.created_at,
        )
        for run in runs
    ]


@router.get("/{repo_id}/status")
async def get_rag_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """RAG 运行状态：索引了多少块、Milvus 是否在线、当前配置。"""
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    return {**rag_status(db, repo_id), "config": config_dict(config), "runtime": runtime_status(config)}


@router.post("/{repo_id}/search", response_model=RAGSearchResponse)
async def search_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGSearchResponse:
    """只检索不生成。返回排序后的证据片段、分数、检索配置和耗时。

    用途：调召回质量。看某个 query 到底命中了什么、分数多少。
    """
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.question.strip(),
        top_k=payload.top_k,
        source_types=payload.source_types,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    return RAGSearchResponse(
        query=payload.question.strip(),
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
    )


@router.post("/{repo_id}/ask", response_model=RAGAnswerResponse)
async def ask_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGAnswerResponse:
    """检索 + 生成带引用的回答。

    内部顺序：retrieve_knowledge → answer_gate（不够证据就拒答，不调 LLM）→ qa.answer_question。
    """
    _repo_or_404(db, repo_id)
    result = await answer_question(
        db,
        repo_id,
        payload.question,
        payload.top_k,
        payload.source_types,
        payload.metadata_filters,
    )
    # 返回最终答案、编号来源、生成模式和耗时（generation_mode 说明是 LLM 生成还是抽取式兜底）
    return RAGAnswerResponse.model_validate(result)


@router.post("/{repo_id}/reindex", response_model=RAGReindexResponse)
async def reindex_rag(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> RAGReindexResponse:
    """重建索引。换了 embedding 模型 / 切片参数之后必须调它。"""
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    count = reindex_documents(db, repo_id)
    return RAGReindexResponse(
        repo_id=repo_id,
        indexed_documents=count,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
    )
