"""
全局配置模块
=============

职责：
    把 .env 文件里的环境变量，读成一个强类型的 settings 对象，供全项目使用。

在系统中的位置：
    基础设施层（core/）。几乎所有模块都会 `from app.core.config import settings`。

读取机制（重要）：
    Settings 继承 pydantic-settings 的 BaseSettings，它会自动：
        1. 到 .env 文件里找与字段同名的变量（大小写不敏感）
           LLM_API_KEY → llm_api_key
        2. 做类型转换（"8000" → 8000，"true" → True）
        3. 校验（类型不对直接启动报错，而不是运行时才炸）

    env_file=("../.env", ".env") 是【相对路径】，按顺序查找：
        先找「当前工作目录的上一级/.env」→ 即项目根目录的 .env
        再找「当前工作目录/.env」

    ⚠️ 这就是为什么启动后端时，工作目录必须是 backend/：
       工作目录 = backend/  →  ../.env 正好是项目根的 .env  ✅
       工作目录 = 项目根目录 →  ../.env 跑到项目外面去了       ❌

为什么用 settings 而不是到处 os.getenv：
    1. 有类型（IDE 能补全，拼错字段名立刻报错）
    2. 有默认值（.env 里没配也能跑）
    3. 单点管理（改配置只看这一个文件）

⚠️ 注意：
    settings 在模块导入时就实例化了（文件最后一行），
    所以【改完 .env 必须重启后端】，热重载对 .env 不生效。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。字段名 = .env 里的变量名（小写形式）。"""

    # ────────────────────────────────────────────────────────────
    # 一、应用自身
    # ────────────────────────────────────────────────────────────
    app_env: str = "development"          # 环境标识。等于 "production" 时会关闭宽松的 CORS 正则
    app_name: str = "DevFlow AI"          # 显示在 /docs 和 /health 里的名字
    api_host: str = "0.0.0.0"             # 监听地址。0.0.0.0 = 允许局域网访问
    api_port: int = 8000                  # 监听端口
    # 允许跨域的前端地址（逗号分隔）。
    # 前端跑在 :3000，后端在 :8000，浏览器默认禁止跨域，必须在这里放行。
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

    # ────────────────────────────────────────────────────────────
    # 二、主数据库（PostgreSQL）
    # ────────────────────────────────────────────────────────────
    # 连接串格式：postgresql+驱动://用户名:密码@主机:端口/库名
    # psycopg 是 PostgreSQL 的 Python 驱动（psycopg3）
    database_url: str = "postgresql+psycopg://devflow:devflow@localhost:5432/devflow"

    # ────────────────────────────────────────────────────────────
    # 三、GitHub 接入
    # ────────────────────────────────────────────────────────────
    github_token: str | None = None                              # 全局默认 token（仓库级 token 优先级更高）
    github_api_base_url: str = "https://api.github.com"           # 换成企业版时改这里

    # ────────────────────────────────────────────────────────────
    # 四、大模型（LLM）
    # ────────────────────────────────────────────────────────────
    llm_api_key: str | None = None                                # 为空时所有 Agent 降级为规则兜底
    llm_base_url: str = "https://api.deepseek.com"                # OpenAI 兼容接口地址
    llm_model: str = "deepseek-v4-pro"                            # 模型名
    rag_llm_enabled: bool = True                                  # RAG 问答是否允许调 LLM（false 则只做抽取式回答）

    # ────────────────────────────────────────────────────────────
    # 五、向量化（Embedding）
    # ────────────────────────────────────────────────────────────
    # provider 取值：
    #   qwen_local / local → 本地跑 sentence-transformers 模型（默认）
    #   openai_compatible  → 调 EMBEDDING_BASE_URL/embeddings
    #   dashscope          → 调阿里百炼
    #   deterministic      → 本地哈希兜底，仅用于演示/测试（向量无语义）
    embedding_provider: str = "qwen_local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"            # 模型标识（同时用于向量空间契约校验）
    embedding_dimensions: int = 1024                               # 向量维度，必须与模型原生维度一致
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"       # 本地模型的 HF 仓库名
    local_embedding_device: str | None = None                      # 留空=自动（Apple Silicon 上可能用 MPS）
    qwen_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dashscope_api_key: str | None = None
    dashscope_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rerank_api_key: str | None = None                              # 重排模型的凭证（与 embedding 分开配）
    rerank_base_url: str | None = None
    qwen_rerank_model: str = "qwen3-rerank"
    # 检索时给 query 加的任务指令（部分向量模型需要 instruction 才能发挥效果）
    qwen_retrieval_instruction: str = "Given a user question, retrieve passages that contain the facts needed to answer it."

    # ────────────────────────────────────────────────────────────
    # 六、向量库（Milvus）
    # ────────────────────────────────────────────────────────────
    milvus_enabled: bool = True
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    # collection 名里带 qwen1024，是为了让人一眼看出"这个集合用的是 1024 维 Qwen 向量空间"。
    # 换 embedding 模型必须换 collection，否则不同向量空间会混在一起。
    milvus_collection: str = "devflow_rag_qwen1024"
    rag_rerank_top_n: int = 24                                     # 只对前 N 条做重排（省算力）

    # ────────────────────────────────────────────────────────────
    # 七、RAGAS 评测
    # ────────────────────────────────────────────────────────────
    ragas_enabled: bool = True
    # 下面三个留空则复用 LLM_* 的配置；想用不同的判官模型时才单独填
    ragas_judge_api_key: str | None = None
    ragas_judge_base_url: str | None = None
    ragas_judge_model: str | None = None
    ragas_timeout_seconds: float = 120.0
    ragas_judge_max_tokens: int = 8192
    ragas_answer_relevancy_strictness: int = 1
    ragas_max_cases: int = 5                                       # 单次评测最多跑几个用例
    # 五项指标的最低通过门槛。低于任一门槛 → 整体判定为质量失败。
    ragas_context_precision_min: float = 0.75                      # 检索出的上下文有多少是相关的
    ragas_context_recall_min: float = 0.80                         # 该检索的有没有漏
    ragas_faithfulness_min: float = 0.90                           # 回答是否忠于证据（防幻觉）
    ragas_answer_relevancy_min: float = 0.80                       # 回答是否切题
    ragas_agent_goal_accuracy_min: float = 0.80                    # 是否达成用户目标
    ragas_tool_call_accuracy_min: float = 0.90                     # 工具调用是否准确
    ragas_tool_call_f1_min: float = 0.90                           # 工具调用的 F1

    # ────────────────────────────────────────────────────────────
    # 八、上下文压缩（Agent 对话的 token 预算）
    # ────────────────────────────────────────────────────────────
    # 核心思路：模型窗口很大（DeepSeek 1M），但成本高、延迟大。
    # 所以主动限制"每次喂给模型的输入不超过 24000 token"，超了就压缩。
    context_max_input_tokens: int = 24000                          # 输入 token 硬上限
    context_reserved_response_tokens: int = 3000                   # 给输出预留的空间
    context_dynamic_model_window_enabled: bool = True               # 是否按模型动态推算窗口
    context_model_window_tokens: int | None = None                  # 手动指定窗口（0/None = 自动）
    context_compact_reserved_output_tokens: int = 12000
    context_auto_compact_enabled: bool = True                       # 是否自动压缩
    # 四级压力阈值：越接近上限，压缩越激进
    context_warning_buffer_tokens: int = 20000                      # 超过 → 警告
    context_auto_compact_buffer_tokens: int = 13000                 # 超过 → 自动压缩
    context_aggressive_buffer_tokens: int = 8000                    # 超过 → 激进压缩
    context_manual_buffer_tokens: int = 3000                        # 超过 → 强制压缩
    context_max_compaction_failures: int = 3                        # 连续压缩失败几次就放弃
    context_microcompact_keep_recent_messages: int = 8              # 微压缩时保留最近几条
    context_compact_keep_recent_messages: int = 8                   # 完全压缩时保留最近几条
    context_compact_summary_tokens: int = 1800                      # 摘要长度上限
    # 预算分配比例（加起来应该 = 1.0）
    context_system_ratio: float = 0.12                              # 系统提示占 12%
    context_memory_ratio: float = 0.22                              # 记忆占 22%
    context_evidence_ratio: float = 0.34                            # 证据占 34%
    context_recent_ratio: float = 0.22                              # 最近对话占 22%
    context_tool_ratio: float = 0.10                                # 工具结果占 10%

    # ────────────────────────────────────────────────────────────
    # 九、多 Agent 工作流与 Prompt 缓存
    # ────────────────────────────────────────────────────────────
    workflow_task_timeout_seconds: float = 90.0                     # 单个子任务超时
    prompt_cache_enabled: bool = True                                # 启用上游的 prompt 缓存（省 token 费用）
    prompt_cache_key: str | None = "devflow-agent-v1"                # 缓存分区键
    prompt_cache_retention: str | None = None                        # 缓存保留策略

    # ────────────────────────────────────────────────────────────
    # 十、本地路径（相对 backend/ 目录）
    # ────────────────────────────────────────────────────────────
    repo_checkout_dir: str = ".data/repos"                          # 克隆的仓库代码放这里
    upload_dir: str = ".data/uploads"                               # 用户上传的原始文档放这里

    # ────────────────────────────────────────────────────────────
    # 十一、安全与运行模式
    # ────────────────────────────────────────────────────────────
    # ⚠️ 生产必须改！用于加密数据库里的 GitHub token
    token_encryption_key: str = "change-me-in-local-dev"
    demo_mode: bool = True                                           # 演示模式：跳过真实 GitHub 校验
    github_webhook_secret: str | None = None                          # Webhook 验签密钥（生产必填）
    feishu_feedback_webhook_url: str | None = None                    # 点踩时推送的飞书机器人地址
    feedback_trace_base_url: str | None = None                        # 通知里拼 trace 链接用的后端地址

    # ────────────────────────────────────────────────────────────
    # 十二、自动同步
    # ────────────────────────────────────────────────────────────
    auto_sync_enabled: bool = True                                    # 是否启动后台定时同步
    auto_sync_interval_seconds: int = 300                             # 间隔（秒），实际最小值被限制为 30
    auto_sync_limit: int = 50                                         # 每次每类资源最多拉多少条

    # ────────────────────────────────────────────────────────────
    # Pydantic 配置
    # ────────────────────────────────────────────────────────────
    # env_file：按顺序尝试读取的 .env 路径（相对工作目录）
    # env_file_encoding：.env 的编码
    # extra="ignore"：.env 里有、但类里没定义的变量直接忽略，不报错
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")


# 模块级单例：整个进程共享这一个实例（相当于 Spring 的 @Bean @Singleton）。
# 导入本模块时立刻执行一次，读取并校验 .env。
settings = Settings()
