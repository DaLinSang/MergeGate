"""
文档解析（RAG 入库第 1 步：字节 → 纯文本）
==========================================

【在 RAG 流水线中的位置】
    用户上传文件
        ↓
    ★ 本文件：extract_document_text() 解析成纯文本
        ↓
    chunking.py：切成小块
        ↓
    embeddings.py：变成向量
        ↓
    vector_store.py：写 PostgreSQL + Milvus

    【为什么解析这一步很关键】
        RAG 的效果上限由"切片质量"决定，而切片质量由"解析质量"决定。
        PDF 解析出乱码、DOCX 丢了表格结构、编码猜错变成问号 ——
        后面再好的 embedding 模型也救不回来。

【业务定位】
    数据流 3（文档入库）的第一步：把上传的文件字节变成纯文本。
    统一把各种格式"拉平"成 Markdown 风格的纯文本，让下游只需处理一种格式。

【数据流向】
    上游（谁调它）：
        services/rag/document_processing.py::extract_text()  ← 主路径（后台任务）
    下游（它调谁）：
        pypdf（PDF）、python-docx（DOCX）—— 都是局部 import，不用就不加载

【核心规则】
    1. 带一堆资源上限（防 zip 炸弹 / 超大 PDF）—— 上传接口是攻击面。
    2. 解析失败要明确报错，绝不静默返回空串。
       因为空串会被下游当成"这份文档没内容"，用户永远不知道出了什么问题。
    3. PDF 插页码标记、DOCX 转 Markdown —— 都是为了给切片器提供结构信息。

【本文件函数地图】（按调用顺序）
    入口：extract_document_text(filename, data)   ← 按扩展名分发
        ├─ .pdf  → _extract_pdf()      → 逐页提取 + 插 <!-- page: n -->
        │                                  └ _ensure_text_limit()
        ├─ .docx → _extract_docx()     → 先 zip 安全校验，再遍历 body
        │                                  ├ _paragraph_as_markdown()  段落 → Markdown
        │                                  └ _table_as_markdown()      表格 → Markdown
        └─ 其他  → decode_text()       → 多编码尝试解码
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# 安全上限（上传接口是攻击面，必须设限）
# ════════════════════════════════════════════════════════════════

MAX_DOCUMENT_BYTES = 12 * 1024 * 1024          # 单个文件最大 12MB
MAX_EXTRACTED_CHARS = 4_000_000                # 解析出的文本最多 400 万字（约 200 万 token）
MAX_PDF_PAGES = 500                            # PDF 最多 500 页
MAX_DOCX_ENTRIES = 10_000                      # DOCX 内部条目数上限（防 zip 炸弹）
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024 # DOCX 解压后最大 64MB

# 【为什么每个上限都必要】
#   MAX_DOCUMENT_BYTES       → 防止一次上传打满内存
#   MAX_EXTRACTED_CHARS      → 防止"小文件解出大文本"（PDF 里全是文字时体积小但字符多）
#   MAX_PDF_PAGES            → 500 页 PDF 解析要几十秒，会卡住后台任务
#   MAX_DOCX_ENTRIES         → zip 炸弹：一个 1KB 的文件可以解压出几十万个小文件
#   MAX_DOCX_UNCOMPRESSED_BYTES → zip 炸弹的另一种形态：单个巨大文件


def _ensure_text_limit(text: str) -> str:
    """统一检查文本长度上限，超了直接报错。

    【调用链】被 decode_text() / _extract_pdf() / _extract_docx() 调用

    【为什么不截断而是报错】
        截断会让用户以为"整份文档都入库了"，但实际只索引了前半部分。
        后面检索不到内容时，根本想不到是被截断了。
        明确报错（前端能看到"文档太大"）比静默丢数据好。
    """
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ValueError(f"extracted text exceeds the {MAX_EXTRACTED_CHARS} character limit")
    return text


def decode_text(data: bytes) -> str:
    """把纯文本字节解码成字符串（多编码依次尝试）。

    【调用链】_ensure_text_limit() → 本函数
    【被谁调用】extract_document_text()（非 PDF/DOCX 的所有格式：.txt/.md/.csv/.log/.json）

    【编码尝试顺序，以及为什么是这个顺序】
        utf-8-sig   → 先处理带 BOM 的 Windows 文件（BOM 会被当成正文的第一个字符，必须剥掉）
        utf-8       → 标准情况，绝大多数文件走这里
        gb18030     → 兼容 GBK/GB2312（中文旧文件、Windows 导出的 CSV）
        latin-1     → 永不失败的兜底（任何字节都能"解码"，虽然可能是乱码）

    【为什么先判二进制】
        图片/压缩包/可执行文件也可能被用户改后缀上传。
        这类文件前 8KB 里几乎一定有 \\x00，用它判断又快又准。
        如果不拦，后面会解出一堆乱码，白白占用向量库。

    返回空串的情况：所有编码都失败（实际到不了这里，latin-1 不会失败）。
    """
    # 前 8KB 出现 \x00 基本可以判定是二进制（图片/可执行文件等）
    if b"\x00" in data[:8192]:
        raise ValueError("binary content cannot be indexed as text")
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return _ensure_text_limit(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return ""


def _extract_pdf(data: bytes) -> str:
    """解析 PDF，每页前插入页码标记。

    【调用链】_ensure_text_limit() → 本函数
    【被谁调用】extract_document_text()（.pdf）

    【数据流向】
        入：PDF 字节
        出：形如 "<!-- page: 1 -->\\n第一页内容\\n\\n<!-- page: 2 -->\\n第二页内容"

    【为什么要插 `<!-- page: n -->`】
        后面切片时需要知道"这段来自第几页"，回答引用才能定位到页码。
        用 HTML 注释是因为它对人不可见、对模型也无害（不会干扰理解）。

    【为什么要局部 import pypdf】
        pypdf 不轻（解析库 + 一堆依赖）。不用 PDF 的场景（比如只传 .md）
        完全没必要加载它。局部 import 让启动更快、内存更省。

    【已知局限】
        扫描版 PDF（图片型）extract_text() 会返回空，本函数会跳过该页。
        本项目不做 OCR —— 所以扫描件解析出来是空文档，会被上层报"没有可索引文本"。
    """
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))

    # 页数上限：超过就拒绝（解析 500 页要几十秒，会拖住后台任务）
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES} page limit")

    pages: list[str] = []
    extracted_chars = 0

    # ── 逐页提取，每页前面加页码标记 ──
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:                                     # 空页（扫描页）直接跳过
            content = f"<!-- page: {page_number} -->\n{text}"
            # 累计字符数：+2 是后面 join 用的两个换行
            extracted_chars += len(content) + (2 if pages else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"PDF text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            pages.append(content)

    return _ensure_text_limit("\n\n".join(pages))


def _paragraph_as_markdown(paragraph) -> str:
    """把 DOCX 段落转成 Markdown（保留标题层级和列表）。

    【调用链】被 _extract_docx() 调用
    【数据流向】入：python-docx 的 Paragraph 对象；出：一行 Markdown 文本

    【为什么必须转 Markdown】
        chunking.py 的 markdown 策略是按 `#` 标题栈来生成 section_path 的。
        如果不转，所有段落都是纯文本，切片会失去章节结构 ——
        检索结果就没有"来自 3.2 节"这种精确定位。

    【样式名 → Markdown 的映射】
        Word 的 "Heading 1"  → "# 文本"     （级别夹在 1~6，因为 Markdown 最多 6 级）
        Word 的 "List Bullet" → "- 文本"
        Word 的 "List Number" → "1. 文本"
        其他                  → 原样返回
    """
    text = (paragraph.text or "").strip()
    if not text:
        return ""

    # 取样式名（getattr 链式是为了防御：style 可能为 None）
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")

    # Word 的标题样式名形如 "Heading 1"，转成 "# 文本"
    heading = re.match(r"heading\s+(\d+)", style_name, re.IGNORECASE)
    if heading:
        # 夹到 1~6：Word 可能有 "Heading 9"，但 Markdown 只到 6 级
        level = min(6, max(1, int(heading.group(1))))
        return f"{'#' * level} {text}"

    # 列表样式（注意是小写匹配，Word 的样式名可能大小写不一致）
    lowered = style_name.lower()
    if "list bullet" in lowered:
        return f"- {text}"
    if "list number" in lowered:
        # 统一写成 "1."：Markdown 渲染时会自动编号，不需要真实序号
        return f"1. {text}"

    return text


def _table_as_markdown(table) -> str:
    """把 DOCX 表格转成 Markdown 表格（补齐列宽、转义竖线）。

    【调用链】内部嵌套 render() → 本函数
    【被谁调用】_extract_docx()

    【为什么表格要保留成 Markdown】
        表格里的信息（配置项、参数说明、错误码对照）是最常被问到的内容。
        如果直接拍平成纯文本，行列关系就丢了，模型会读错。

    【两个必须处理的细节】
        ① 列宽不齐：Word 允许不同行的单元格数不一样，
           直接渲染会错位。所以先算最大列宽，把短行用空串补齐。
        ② 单元格里有 `|`：会破坏 Markdown 表格语法，必须转义成 `\\|`。
    """
    # 先把每个单元格的换行压成空格（Markdown 表格里不能有真换行）
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    # 补齐每行到相同列数（Word 允许行内单元格数不一致）
    normalized = [row + [""] * (width - len(row)) for row in rows]

    def render(row: list[str]) -> str:
        """渲染一行：转义竖线 + 用 | 包裹每个单元格。"""
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    # 结构：表头行 + 分隔行（---）+ 数据行
    return "\n".join([render(normalized[0]), render(["---"] * width), *(render(row) for row in normalized[1:])])


def _extract_docx(data: bytes) -> str:
    """解析 DOCX（按正文顺序遍历段落与表格）。

    【调用链】追这个函数会进入：
        1. zipfile 安全校验（条目数 / 解压大小 / 加密标记）
        2. _paragraph_as_markdown()  → 段落转 Markdown
        3. _table_as_markdown()      → 表格转 Markdown
        4. _ensure_text_limit()      → 总长度校验
    【被谁调用】extract_document_text()（.docx）

    【为什么先做 zip 校验再交给 python-docx】
        DOCX 本质是 zip。恶意文件可能声明很小的体积但解压后极大（zip 炸弹），
        直接把几百 MB 灌进内存，服务就挂了。
        所以在交给 python-docx 之前先检查：条目数、解压总大小、是否加密。
        —— 这一步必须在 docx.Document() 之前，否则校验就白做了。

    【为什么按 body.iterchildren() 遍历，而不是分别取 paragraphs 和 tables】
        document.paragraphs 只返回段落、document.tables 只返回表格，各自都是独立列表。
        分别遍历会丢掉"段落和表格的相对位置"（表格本来在第 3 段后面，结果被挪到最后）。
        iterchildren() 按 XML 里的真实顺序遍历，能保证顺序正确。
    """
    # ── 第 1 步：zip 层安全校验（必须在解析之前）──
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()

            # ① 条目数：防"几万个小文件"型炸弹
            if len(infos) > MAX_DOCX_ENTRIES:
                raise ValueError(f"DOCX exceeds the {MAX_DOCX_ENTRIES} entry limit")

            # ② 解压后总大小：防"单个巨大文件"型炸弹
            #    注意读的是 file_size（解压后大小），不是 compress_size
            uncompressed_size = sum(info.file_size for info in infos)
            if uncompressed_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"DOCX expanded size exceeds the {MAX_DOCX_UNCOMPRESSED_BYTES} byte limit"
                )

            # ③ 加密标记：flag_bits & 0x1 表示条目被加密
            #    加密文档 python-docx 也打不开，提前给明确错误
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("encrypted DOCX files are not supported")
    except zipfile.BadZipFile as exc:
        # 不是合法 zip → 不是真 DOCX（可能是改了后缀的 .doc 或损坏文件）
        raise ValueError("invalid DOCX archive") from exc

    # ── 第 2 步：加载 python-docx（局部 import，避免非 DOCX 场景加载）──
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(BytesIO(data))

    blocks: list[str] = []
    extracted_chars = 0

    # ── 第 3 步：按 XML 顺序遍历，保证段落和表格的相对位置不丢 ──
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            # 段落节点 → 转 Markdown
            content = _paragraph_as_markdown(Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            # 表格节点 → 转 Markdown 表格
            content = _table_as_markdown(Table(child, document))
        else:
            # 其他节点（分节符、书签等）忽略
            content = ""

        if content:
            extracted_chars += len(content) + (2 if blocks else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"DOCX text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            blocks.append(content)

    return _ensure_text_limit("\n\n".join(blocks))


def extract_document_text(filename: str, data: bytes) -> str:
    """【入口】按扩展名分发解析器。

    【调用链】_extract_pdf() / _extract_docx() / decode_text() → 本函数
    【被谁调用】services/rag/document_processing.py::extract_text()（薄封装）

    【数据流向】
        入：filename（只用扩展名）、data（原始字节）
        出：纯文本（Markdown 风格，可能带页码标记）

    【为什么按扩展名而不是 MIME 类型】
        上传时拿到的 content_type 常常不准（浏览器按后缀猜，用户也可能伪造）。
        后缀是用户最直接的选择，按它分发最符合直觉，出错时也容易解释。

    【注意 suffix 取的是最后一个点之后】
        "archive.tar.gz" → ".gz"。对本项目够用（不支持压缩包）。
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    # 其余全部按纯文本处理：.txt/.md/.markdown/.mdx/.json/.csv/.log
    # 注意：.doc（旧版 Word，非 zip 格式）也会走到这里 → 解出乱码或报"binary content"
    return decode_text(data)
