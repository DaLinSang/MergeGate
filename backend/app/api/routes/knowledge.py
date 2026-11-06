"""
知识与记忆路由
==============

【路由清单】（15 个，按功能分四组）

  ── 知识条目（人工录入的知识）──
    GET    /{repo_id}/knowledge                        列知识条目（按 source_id 聚合成一条）
    POST   /{repo_id}/knowledge/notes                  新建记忆笔记
    POST   /{repo_id}/knowledge/upload                 上传知识文件（同步解析 + 向量化）
    DELETE /{repo_id}/knowledge/{source_id}            删知识条目（连同向量）

  ── 记忆检索与召回观测 ──
    GET    /{repo_id}/memory/search                    手动搜记忆（并记一条召回事件）
    GET    /{repo_id}/memory/recall-events             看召回历史（谁在什么时候召回了什么）
    GET    /{repo_id}/memory/status                    记忆面板总览
    GET    /{repo_id}/memory/candidates                列待审核记忆候选
    POST   /{repo_id}/memory/candidates/{id}/approve   批准候选 → 变成正式文档
    POST   /{repo_id}/memory/candidates/{id}/reject    拒绝候选

  ── 知识图谱 ──
    GET    /{repo_id}/knowledge/graph                  取子图（可指定中心节点 + 深度）
    POST   /{repo_id}/knowledge/graph/rebuild          全量重建图谱

  ── 团队画像 ──
    GET    /{repo_id}/team-members                     列成员
    POST   /{repo_id}/team-members                     新建/覆盖成员（按名字去重）
    DELETE /{repo_id}/team-members/{member_id}         删成员

【数据流向】
    知识录入（数据流 3 的另一条路：人工录入）
      POST /knowledge/notes 或 /knowledge/upload
        → chunk_document()（上传文件才切）
        → embed_texts()（批量 64 条一次）
        → vector_store.add_document(s)()  双写 Milvus + PostgreSQL
        → rebuild_knowledge_graph()       重建图谱

    记忆审核（数据流 5 的出口）
      SessionSealer 提炼候选 → memory_candidates(pending)
        → POST .../approve
        → memory_hub.approve_memory_candidate()  写 documents(source_type="memory_note")
        → EvidenceStore.rebuild_repo() + rebuild_knowledge_graph()

【要点】
    1. 列表接口按 (source_type, source_id) 聚合 —— 一条知识可能被切成 10 个 chunk，
       但用户只想看到"1 条知识"。所以先 group 再取前 2 块做预览。
    2. 上传走【同步】处理（不像 rag.py 那样后台跑）：这里文件小（12MB 上限），
       而且用户上传完就想立刻在列表里看到，同步更符合预期。
    3. 团队画像用 stable_source_id(repo, "team_member", name) 做 id ——
       同名成员重复提交是"覆盖"而不是"新增两条"。
    4. 所有写操作后都重建图谱：图谱是派生数据，不重建就会和文档不一致。
    5. 手动搜记忆也记召回事件 —— 让"记忆被谁用过"这件事可观测。
"""

import asyncio
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, Repository
from app.db.session import get_db
from app.schemas.knowledge import (
    KnowledgeGraphRebuildResponse,
    KnowledgeGraphResponse,
    KnowledgeItemResponse,
    KnowledgeUploadResponse,
    MemoryCandidateActionResponse,
    MemoryCandidateResponse,
    MemorySearchResponse,
    MemoryStatusResponse,
    MemoryNoteRequest,
    RecallEventResponse,
    TeamMemberRequest,
    TeamMemberResponse,
)
from app.services.knowledge_graph import knowledge_graph_payload, rebuild_knowledge_graph
from app.services.chat_memory import EvidenceStore
from app.services.memory_hub import (
    approve_memory_candidate,
    list_memory_candidates,
    list_recall_events,
    memory_status,
    record_recall_event,
    reject_memory_candidate,
)
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES
from app.services.rag.embeddings import embed_texts, embedding_provider_name
from app.services.rag.vector_store import DocumentPayload, add_document, add_documents, delete_document_group, stable_source_id

router = APIRouter()

# "用户知识"的白名单：这些 source_type 才会出现在知识列表里。
# 刻意不含 issue / pull_request / code_file —— 那些是同步来的，不是用户录入的。
USER_KNOWLEDGE_SOURCE_TYPES = [
    "memory_note",
    "knowledge_file",
    "team_member",
    "weekly_report",
    "project_overview",
    "project_doc",
    "project_manifest",
]
# embedding 批大小。一次 64 条：太大容易超时，太小调用次数多
EMBEDDING_BATCH_SIZE = 64


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _clip(text: str, limit: int = 360) -> str:
    """截断预览文本。先 strip + 统一换行，再按长度截。"""
    text = (text or "").strip().replace("\r\n", "\n")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _safe_filename(filename: str) -> str:
    """清洗文件名（去掉路径 + 危险字符），防目录穿越。"""
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    """上传目录 {upload_dir}/{repo_id}/，不存在则创建。"""
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_upload_text(filename: str, data: bytes) -> str:
    """解析文件为文本。失败返回空串（由调用方判断并报 400），不抛异常。"""
    try:
        return extract_document_text(filename, data)
    except Exception:
        return ""


def _embed_default_in_batches(texts: list[str]) -> list[list[float]]:
    """分批向量化。必须分批：一次传几千条会超 embedding 接口的单请求上限。"""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embed_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                dimensions=settings.embedding_dimensions,
            )
        )
    return vectors


def _group_documents(documents: list[Document]) -> list[KnowledgeItemResponse]:
    """把 chunk 列表聚合成"知识条目"列表。

    一条知识会被切成 N 个 Document（每个是一块），但用户视角是"1 条知识"。
    所以按 (source_type, source_id) 分组，取前 2 块拼成预览。
    """
    groups: dict[tuple[str, uuid.UUID], list[Document]] = {}
    for doc in documents:
        groups.setdefault((doc.source_type, doc.source_id), []).append(doc)

    items: list[KnowledgeItemResponse] = []
    for (source_type, source_id), docs in groups.items():
        # 按 chunk_index 排序，保证预览是"文档开头"而不是随机某块
        docs.sort(key=lambda item: ((item.meta or {}).get("chunk_index", 0), item.created_at))
        first = docs[0]
        # 只取前 2 块做预览：多了没必要，前端也只显示几行
        content = "\n".join(doc.content for doc in docs[:2])
        metadata = dict(first.meta or {})
        metadata.setdefault("source_type", source_type)
        items.append(
            KnowledgeItemResponse(
                id=source_id,
                # 把所有 chunk 的 id 都带上 —— 删除时要用
                document_ids=[doc.id for doc in docs],
                source_type=source_type,
                title=metadata.get("display_title") or first.title,
                content_preview=_clip(content),
                metadata=metadata,
                chunk_count=len(docs),
                created_at=first.created_at,
            )
        )
    # 按创建时间倒序。注意要处理 created_at 为 None 的情况（否则比较会报错）
    items.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items


def _team_from_document(doc: Document) -> TeamMemberResponse:
    """团队成员文档 → 响应模型。名字优先取 meta，兜底从标题剥前缀。"""
    meta = doc.meta or {}
    return TeamMemberResponse(
        id=doc.source_id,
        name=str(meta.get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", "")),
        role=str(meta.get("role") or ""),
        strengths=str(meta.get("strengths") or ""),
        techStack=str(meta.get("tech_stack") or meta.get("techStack") or ""),
        created_at=doc.created_at,
    )


@router.get("/{repo_id}/knowledge", response_model=list[KnowledgeItemResponse])
async def list_knowledge(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeItemResponse]:
    """列用户知识条目（已按 source_id 聚合，不是按 chunk）。"""
    _repo_or_404(db, repo_id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type.in_(USER_KNOWLEDGE_SOURCE_TYPES))
        .order_by(Document.created_at.desc())
        .all()
    )
    return _group_documents(docs)


@router.get("/{repo_id}/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    repo_id: uuid.UUID,
    q: str = Query(default=""),
    limit: int = Query(default=8, ge=1, le=30),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    """手动搜记忆。会记一条 recall_event —— 让"记忆被查过"可观测。"""
    repo = _repo_or_404(db, repo_id)
    query = q.strip()
    # 空查询直接返回空：不查库，也仍然记一条（表示"用户打开面板但没搜"）
    results = EvidenceStore(db).search(repo.id, None, query, limit=limit) if query else []
    record_recall_event(
        db,
        repo_id=repo.id,
        conversation_id=None,
        tool_name="manual_memory_search",
        query=query,
        results=results,
        citations=[{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results],
        scope="project",
        mode="hybrid",
        metadata={"limit": limit, "surface": "workspace_memory"},
    )
    db.commit()
    return MemorySearchResponse(query=query, results=results, result_count=len(results))


@router.get("/{repo_id}/memory/recall-events", response_model=list[RecallEventResponse])
async def get_recall_events(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RecallEventResponse]:
    """召回历史。看"Agent 什么时候、因为什么问题召回了哪些记忆"。"""
    _repo_or_404(db, repo_id)
    return [RecallEventResponse.model_validate(item) for item in list_recall_events(db, repo_id=repo_id, conversation_id=conversation_id, limit=limit)]


@router.get("/{repo_id}/memory/status", response_model=MemoryStatusResponse)
async def get_memory_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryStatusResponse:
    """记忆面板总览：各类记忆多少条、多少候选待审核。"""
    repo = _repo_or_404(db, repo_id)
    return MemoryStatusResponse.model_validate(memory_status(db, repo))


@router.get("/{repo_id}/memory/candidates", response_model=list[MemoryCandidateResponse])
async def get_memory_candidates(
    repo_id: uuid.UUID,
    status: str | None = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[MemoryCandidateResponse]:
    """列记忆候选。默认只看 pending（待人工审核的）。"""
    _repo_or_404(db, repo_id)
    return [MemoryCandidateResponse.model_validate(item) for item in list_memory_candidates(db, repo_id=repo_id, status=status, limit=limit)]


@router.post("/{repo_id}/memory/candidates/{candidate_id}/approve", response_model=MemoryCandidateActionResponse)
async def approve_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
    """批准记忆候选 → 变成正式文档（可被检索）。

    【关键】批准后要做两件刷新：
      ① EvidenceStore.rebuild_repo() —— 否则新记忆不在证据库里，检索不到
      ② rebuild_knowledge_graph()    —— 否则图谱里没有这条记忆的边
    少做任何一件，用户都会觉得"我明明批准了，怎么还是查不到"。
    """
    repo = _repo_or_404(db, repo_id)
    try:
        candidate, doc = approve_memory_candidate(db, repo, candidate_id)
    except ValueError:
        # 用 code 形式的 detail：前端按 code 做国际化，不直接显示英文
        raise HTTPException(status_code=404, detail="candidate_not_found") from None
    EvidenceStore(db).rebuild_repo(repo)
    rebuild_knowledge_graph(db, repo)
    db.commit()
    return MemoryCandidateActionResponse(id=candidate.id, status=candidate.status, document_id=doc.id if doc else None)


@router.post("/{repo_id}/memory/candidates/{candidate_id}/reject", response_model=MemoryCandidateActionResponse)
async def reject_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
    """拒绝记忆候选。只改状态，不动文档（候选本来就没入库）。"""
    repo = _repo_or_404(db, repo_id)
    try:
        candidate = reject_memory_candidate(db, repo, candidate_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="candidate_not_found") from None
    db.commit()
    return MemoryCandidateActionResponse(id=candidate.id, status=candidate.status)


@router.post("/{repo_id}/knowledge/notes", response_model=KnowledgeItemResponse)
async def create_memory_note(
    repo_id: uuid.UUID,
    payload: MemoryNoteRequest,
    db: Session = Depends(get_db),
) -> KnowledgeItemResponse:
    """新建记忆笔记。人工录入的记忆，直接入库（不需要审核）。"""
    _repo_or_404(db, repo_id)
    content = payload.content.strip()
    title = payload.title.strip() or "未命名记忆"
    if not content and not title:
        raise HTTPException(status_code=400, detail="记忆标题或内容至少需要填写一项")
    # 笔记是"凭空新建"的，没有天然 source_id，所以生成一个
    source_id = uuid.uuid4()
    doc = add_document(
        db,
        repo_id,
        "memory_note",
        source_id,
        title,
        # 内容为空时用标题兜底 —— 否则向量化出来是空向量，永远检索不到
        content or title,
        {"display_title": title, "created_by": "user"},
    )
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return _group_documents([doc])[0]


@router.post("/{repo_id}/knowledge/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge_file(
    repo_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeUploadResponse:
    """上传知识文件。同步解析 + 切片 + 向量化 + 双写（文件小，用户想立刻看到结果）。"""
    repo = _repo_or_404(db, repo_id)
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file is too large; max size is 12 MB")

    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_KNOWLEDGE_SUFFIXES))
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Supported: {supported}")

    # ── 内容去重：sha256 存在 chunk 的 meta 里 ──
    # 注意这里是"取出该仓库所有 knowledge_file 再在内存里比"，
    # 因为 checksum 在 JSON 字段里，SQL 里查不了（JSON 无索引）
    checksum = hashlib.sha256(data).hexdigest()
    existing = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type == "knowledge_file")
        .order_by(Document.created_at.asc())
        .all()
    )
    duplicate_docs = [doc for doc in existing if (doc.meta or {}).get("checksum") == checksum]
    if duplicate_docs:
        return KnowledgeUploadResponse(
            id=duplicate_docs[0].source_id,
            filename=str((duplicate_docs[0].meta or {}).get("filename") or filename),
            chunks=len(duplicate_docs),
            indexed=True,
            duplicate=True,
        )

    # 解析是 CPU 密集（PDF/DOCX），扔到线程池，别卡事件循环
    text = await asyncio.to_thread(_extract_upload_text, filename, data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this file")

    source_id = uuid.uuid4()
    stored_path = _upload_root(repo_id) / f"{source_id}-{filename}"
    # 1800 字 / 180 重叠：知识文件偏长文档，块要比默认值大一些
    chunks = await asyncio.to_thread(chunk_document, text, filename, 1800, 180)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document did not produce indexable chunks")

    # ── 组装每个 chunk 的 payload ──
    payloads = [
        DocumentPayload(
            source_type="knowledge_file",
            source_id=source_id,
            # 标题带序号，方便引用时定位"第几块"
            title=f"{filename}#{index + 1}",
            content=str(chunk["content"]),
            metadata={
                # 把 chunking 返回的额外字段（section_path / page 等）全带上
                **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                "display_title": filename,
                "filename": filename,
                "stored_path": str(stored_path),
                "content_type": file.content_type,
                "checksum": checksum,
                "file_size": len(data),
                "chunk_index": index,
                "chunk_chars": len(str(chunk["content"])),
                "total_chunks": len(chunks),
                "chunk_strategy": "structure_aware_v1",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                # 记下向量化时用的模型和维度：换模型后能看出哪些文档需要重建
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
            },
        )
        for index, chunk in enumerate(chunks)
    ]

    # 【关键】向量化的是"标题 + 正文"，不是纯正文。
    # 原因：正文可能只写"默认 30 秒"，标题是"超时设置"，带标题才能命中"超时多久"
    embedding_inputs = [f"{payload.title}\n{payload.content}" for payload in payloads]
    vectors = await asyncio.to_thread(_embed_default_in_batches, embedding_inputs)

    try:
        await asyncio.to_thread(stored_path.write_bytes, data)
        # commit=False：和重建图谱同一个事务
        docs = add_documents(db, repo_id, payloads, commit=False, vectors=vectors)
        rebuild_knowledge_graph(db, repo)
    except Exception:
        # 失败要同时回滚数据库 + 删掉已写的文件，否则留下孤儿文件
        db.rollback()
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return KnowledgeUploadResponse(id=source_id, filename=filename, chunks=len(docs), indexed=bool(docs))


@router.get("/{repo_id}/knowledge/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    repo_id: uuid.UUID,
    center_type: str | None = Query(default=None),
    center_id: str | None = Query(default=None),
    depth: int = Query(default=1, ge=1, le=2),
    db: Session = Depends(get_db),
) -> KnowledgeGraphResponse:
    """取知识图谱子图。

    depth 限制在 1~2：深度 3 以上节点数会爆炸，前端渲染不动。
    """
    repo = _repo_or_404(db, repo_id)
    return KnowledgeGraphResponse.model_validate(
        knowledge_graph_payload(db, repo, center_type=center_type, center_id=center_id, depth=depth)
    )


@router.post("/{repo_id}/knowledge/graph/rebuild", response_model=KnowledgeGraphRebuildResponse)
async def rebuild_knowledge_graph_route(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeGraphRebuildResponse:
    """全量重建图谱。数据不一致时的"重置按钮"。"""
    repo = _repo_or_404(db, repo_id)
    result = rebuild_knowledge_graph(db, repo)
    return KnowledgeGraphRebuildResponse(repo_id=repo.id, edges=result["edges"], nodes=result["nodes"])


@router.delete("/{repo_id}/knowledge/{source_id}")
async def delete_knowledge_item(
    repo_id: uuid.UUID,
    source_id: uuid.UUID,
    source_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """删知识条目（会删掉它所有 chunk 的向量 + 记录），然后重建图谱。"""
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, source_id, source_type=source_type)
    # 删完必须重建图谱，否则图谱里还留着已删节点的边
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "source_id": str(source_id), "deleted": count}


@router.get("/{repo_id}/team-members", response_model=list[TeamMemberResponse])
async def list_team_members(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TeamMemberResponse]:
    """列团队成员（Issue 分析时用来推荐负责人的依据）。"""
    _repo_or_404(db, repo_id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type == "team_member")
        .order_by(Document.created_at.asc())
        .all()
    )
    return [_team_from_document(doc) for doc in docs]


@router.post("/{repo_id}/team-members", response_model=TeamMemberResponse)
async def create_team_member(
    repo_id: uuid.UUID,
    payload: TeamMemberRequest,
    db: Session = Depends(get_db),
) -> TeamMemberResponse:
    """新建 / 覆盖团队成员。

    source_id 由 (repo_id, "team_member", 名字小写) 稳定生成 ——
    所以同一个人重复提交是"更新"，不会出现两条同名记录。
    """
    _repo_or_404(db, repo_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="团队成员姓名为必填项")
    # 拼成 key: value 文本 —— 这样整条记录可以被向量化、被检索
    content = "\n".join(
        [
            f"name: {name}",
            f"role: {payload.role.strip()}",
            f"strengths: {payload.strengths.strip()}",
            f"tech_stack: {payload.tech_stack.strip()}",
        ]
    )
    source_id = stable_source_id(repo_id, "team_member", name.lower())
    # 先删旧的（commit=False）再插新的：等价于 upsert
    delete_document_group(db, repo_id, source_id, source_type="team_member", commit=False)
    doc = add_document(
        db,
        repo_id,
        "team_member",
        source_id,
        f"团队成员：{name}",
        content,
        {
            "display_title": name,
            "name": name,
            "role": payload.role.strip(),
            "strengths": payload.strengths.strip(),
            "tech_stack": payload.tech_stack.strip(),
        },
    )
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return _team_from_document(doc)


@router.delete("/{repo_id}/team-members/{member_id}")
async def delete_team_member(repo_id: uuid.UUID, member_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """删团队成员。注意删的是文档（向量），不是 GitHub 上的任何人。"""
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, member_id, source_type="team_member")
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "member_id": str(member_id), "deleted": count}
