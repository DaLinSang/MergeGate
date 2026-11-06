"""
应用入口
========

【业务定位】
    整个 DevFlow AI 后端的"总装车间"。
    业务上它不做任何判断，只负责把零件装好、把开关打开：
    让"用户提问 → 定位 → 取证 → 判断 → 沉淀"这条主线有地方跑。

【数据流向】
    上游：无（它是进程起点，由 uvicorn 启动）
    下游：
        18 个路由模块      → 接收所有业务请求
        PostgreSQL         → 启动时建表/补列，保证数据结构就绪
        Milvus             → 启动时探测可用性，并把状态挂到 app.state
        后台定时任务        → 每 300 秒同步所有已连接仓库

【核心业务规则】
    1. 启动必须先"准备好数据地基"（建表 + 补列 + 回填），再对外服务。
    2. Milvus 挂了不能让整个系统崩 —— 返回 503 degraded，其它功能继续可用。
    3. 后台任务必须能被优雅关闭（shutdown 时 cancel）。

【技术要点】
    - 课程项目没有 Alembic，用 create_all + 手写 ALTER TABLE 做"穷人版迁移"
    - create_all 只建缺失的表，【不会】修改已有表结构 → 新增字段必须手写 ALTER

Java 类比：
    create_app()  ≈ @SpringBootApplication 主类 + WebMvcConfigurer
    路由 include  ≈ 各 @RestController 被组件扫描自动注册（这里改成显式注册）
"""

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text

# 18 个路由模块。每个模块内部定义了一个 router（APIRouter 实例）
from app.api.routes import (
    action_drafts,
    chat,
    ci,
    code_graph,
    conversations,
    evaluations,
    feedback,
    issues,
    knowledge,
    project_index,
    pull_requests,
    rag,
    repos,
    reports,
    search,
    skills,
    webhooks,
    workspaces,
)
from app.core.config import settings
from app.db.models import Base
from app.db.session import engine
from app.services.scheduler import periodic_repo_sync_loop
from app.services.rag.vector_store import MilvusUnavailableError


# ════════════════════════════════════════════════════════════════
# CORS 配置辅助函数
# ════════════════════════════════════════════════════════════════

def _cors_origins() -> list[str]:
    """把逗号分隔的 CORS_ORIGINS 配置解析成列表。

    输入："http://localhost:3000,http://127.0.0.1:3000"
    输出：["http://localhost:3000", "http://127.0.0.1:3000"]

    为什么要 strip()：
        配置里常有人写 "a, b"（逗号后带空格），不去空格会导致匹配失败。
    为什么要过滤空串：
        末尾多一个逗号会产生 ""，空字符串在 CORS 里是非法来源。
    """
    return [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]


def _cors_origin_regex() -> str | None:
    r"""开发环境额外放行"任意端口的 localhost"。

    为什么需要正则而不只用 allow_origins 列表：
        开发时前端端口可能变成 3001、3002（端口被占用时会自动换），
        每次都改 .env 太麻烦。用正则一次性放行 localhost 的所有端口。

    为什么生产环境返回 None（禁用正则）：
        正则会放行 http://localhost:任意端口。生产环境不应该允许本机来源，
        必须严格按 allow_origins 白名单来。

    正则解释：
        ^https?://                          → http 或 https
        (localhost|127\.0\.0\.1|\[::1\])    → 三种本机写法（IPv6 要转义方括号）
        (:\d+)?$                            → 可选端口，到结尾
    """
    if settings.app_env.lower() == "production":
        return None
    return r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


# ════════════════════════════════════════════════════════════════
# 应用装配
# ════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""

    # title/version 会显示在 /docs 页面顶部
    app = FastAPI(title=settings.app_name, version="0.1.0")

    # ── ① 全局异常处理：Milvus 不可用 ──
    # 设计意图：向量库挂了不应该让整个系统崩掉（500），
    # 而是返回 503 + degraded 标记，前端可以提示"向量检索暂不可用"，
    # 同时其它功能（读数据库、看 Issue）继续可用。
    @app.exception_handler(MilvusUnavailableError)
    async def milvus_unavailable_handler(_request, exc: MilvusUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "service": "milvus", "status": "degraded"},
        )

    # ── ② CORS 中间件 ──
    # 前端在 :3000，后端在 :8000，属于不同源。浏览器默认禁止跨源请求，
    # 必须由后端显式声明"允许哪些源访问"。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),        # 白名单（来自 .env）
        allow_origin_regex=_cors_origin_regex(),  # 开发环境额外放行任意 localhost 端口
        allow_credentials=True,               # 允许携带 Cookie
        allow_methods=["*"],                  # 允许所有 HTTP 方法
        allow_headers=["*"],                  # 允许所有请求头
    )

    # ── ③ 挂载 18 个路由 ──
    # prefix 是路径前缀。例如 chat.py 里写 @router.post("/stream")，
    # 最终路径就是 /api/chat/stream。
    # tags 只影响 /docs 页面的分组显示，不影响功能。
    #
    # 注意：有 4 个模块共用 /api/repos 前缀（repos / conversations / knowledge / project_index），
    # 它们靠各自内部的子路径区分，不会冲突。
    app.include_router(repos.router, prefix="/api/repos", tags=["repos"])
    app.include_router(conversations.router, prefix="/api/repos", tags=["conversations"])
    app.include_router(workspaces.router, prefix="/api/workspaces", tags=["workspaces"])
    app.include_router(issues.router, prefix="/api", tags=["issues"])
    app.include_router(pull_requests.router, prefix="/api", tags=["pull_requests"])
    app.include_router(ci.router, prefix="/api", tags=["ci"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(feedback.router, prefix="/api/chat/feedback", tags=["chat_feedback"])
    app.include_router(action_drafts.router, prefix="/api/action-drafts", tags=["action_drafts"])
    app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
    app.include_router(knowledge.router, prefix="/api/repos", tags=["knowledge"])
    app.include_router(project_index.router, prefix="/api/repos", tags=["project_index"])
    app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
    app.include_router(code_graph.router, prefix="/api", tags=["code_graph"])
    app.include_router(evaluations.router, prefix="/api/evals", tags=["evaluations"])
    app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])

    # ── ④ 健康检查 ──
    # 最轻量的接口，用于确认"进程活着"。不查数据库、不查 Milvus。
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    # ════════════════════════════════════════════════════════════
    # ⑤ 启动钩子
    # ════════════════════════════════════════════════════════════
    @app.on_event("startup")
    async def create_demo_tables() -> None:
        """应用启动时执行：建表 → 补列 → 数据回填 → 清理 → 拉起定时任务。

        这个函数名字叫 create_demo_tables，但实际干的事远不止建表。
        顺序很重要，不能随意调整。
        """

        # ── 5.1 建表 ──
        # Base.metadata 里注册了 db/models.py 的全部 32 个类。
        # create_all 只会创建【不存在】的表，已存在的表不会改动结构。
        # 课程项目默认自动建表；生产风格流程会在后续加入 Alembic 迁移。
        Base.metadata.create_all(bind=engine)

        # inspector 用于反射（读取）现有数据库的真实结构
        inspector = inspect(engine)

        # ── 5.2 补列：因为 create_all 不会修改已存在的表 ──
        # 如果模型新增了字段，而数据库里的表是旧版本，就必须手动 ALTER TABLE。
        # 这就是"穷人版数据库迁移"——正规做法是用 Alembic 管理版本化迁移脚本。
        #
        # 每个块都是同一个模式：
        #   ① 表存在吗？ ② 取现有列名集合 ③ 缺哪列就 ALTER TABLE ADD COLUMN

        # users.role（早期版本没有角色字段）
        if "users" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("users")}
            if "role" not in columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(80) DEFAULT 'owner'"))

        # repositories 的多列补齐（支持本地接入、多 provider、仓库级 token）
        if "repositories" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("repositories")}
            with engine.begin() as connection:
                if "provider" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN provider VARCHAR(80) DEFAULT 'github'"))
                if "api_base_url" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN api_base_url TEXT"))
                if "clone_url" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN clone_url TEXT"))
                if "local_path" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN local_path TEXT"))
                if "checkout_mode" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN checkout_mode VARCHAR(80) DEFAULT 'managed'"))
                if "github_token_encrypted" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN github_token_encrypted TEXT"))

        # workflow_runs.jobs（把 jobs/steps 整体存成 JSON）
        if "workflow_runs" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("workflow_runs")}
            if "jobs" not in columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE workflow_runs ADD COLUMN jobs JSON"))

        # code_symbols / code_relations 的 PR 维度字段（支持"某个 PR 改动了哪些符号"）
        for table_name in ["code_symbols", "code_relations"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                with engine.begin() as connection:
                    if "pr_id" not in columns:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN pr_id UUID"))
                    if "pr_number" not in columns:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN pr_number INTEGER"))

        # 四张表补 conversation_id（早期只有 repo 维度，没有会话维度）
        for table_name in ["chat_messages", "chat_sessions", "agent_runs", "evidence_items"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                if "conversation_id" not in columns:
                    with engine.begin() as connection:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN conversation_id UUID"))

        # 记忆表补结构化字段（facts / tasks / user_preferences / repo_context / citations）
        for table_name in ["thread_memories", "conversation_memories"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                for column_name in ["facts", "tasks", "user_preferences", "repo_context", "citations"]:
                    if column_name not in columns:
                        with engine.begin() as connection:
                            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} JSON"))

        # documents 表【删除】embedding 列。
        # 为什么是 DROP 而不是 ADD：
        #   早期版本把向量直接存在 PostgreSQL（用 pgvector 扩展），
        #   后来架构改成"向量全部交给 Milvus"，PG 里只留全文。
        #   这个 DROP 是刻意的架构清理——保证不会有人误用 PG 里的旧向量。
        if "documents" in inspector.get_table_names():
            document_columns = {column["name"] for column in inspector.get_columns("documents")}
            if "embedding" in document_columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE documents DROP COLUMN embedding"))

        # ── 5.3 数据回填与清理 ──
        # 局部 import 的原因：这些模块依赖 db.models / settings，
        # 放在文件顶部容易形成循环导入，且只有启动时才需要它们。
        from app.db.session import SessionLocal
        from app.services.conversations import backfill_all_default_conversations
        from app.services.code_analysis import purge_legacy_code_documents
        from app.services.rag.vector_store import delete_milvus_documents, milvus_runtime_status

        db = SessionLocal()
        try:
            # 老数据没有会话概念，这里为每个仓库补一个"默认会话"
            backfill_all_default_conversations(db)
            # 清掉早期把"代码文件"误写进 RAG 的脏数据（返回 True 表示确实删了东西）
            if purge_legacy_code_documents(db):
                db.commit()
        finally:
            db.close()

        # ── 5.4 记录向量库运行时状态，供 /api/rag 相关接口返回 ──
        app.state.rag_vector_status = milvus_runtime_status()

        # ── 5.5 清掉历史遗留的 code_file 向量 ──
        # strict=False 表示 Milvus 不可用时不抛异常（启动不应被向量库拖死）。
        delete_milvus_documents(source_types=["code_file"], strict=False)

        # ── 5.6 拉起后台定时同步任务 ──
        # 注意：只定义 periodic_repo_sync_loop 函数并不会让它运行，
        # 必须 create_task 才真正进入事件循环。
        # 存到 app.state 是为了关闭时能取消它。
        if settings.auto_sync_enabled:
            app.state.repo_sync_task = asyncio.create_task(periodic_repo_sync_loop())

    # ════════════════════════════════════════════════════════════
    # ⑥ 关闭钩子
    # ════════════════════════════════════════════════════════════
    @app.on_event("shutdown")
    async def stop_background_tasks() -> None:
        """应用关闭时取消后台任务，避免进程退出时留下悬挂协程。"""
        task = getattr(app.state, "repo_sync_task", None)
        if task:
            task.cancel()

    return app


# 模块级实例：uvicorn 启动命令 `uvicorn app.main:app` 找的就是这个变量
app = create_app()
