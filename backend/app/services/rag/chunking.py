"""
文档切片（Chunking）：长文档 → 可检索的小块
==========================================

【在 RAG 流水线中的位置】
    入库链路的第 2 步，也是**决定 RAG 效果上限**的一步：

        document_extraction.py  解析出纯文本
             ↓
        ★ 本文件 chunk_document()  切成小块
             ↓
        embeddings.py           每块单独向量化
             ↓
        vector_store.py         写入 PG + Milvus

    【为什么说切片决定效果上限】
        切片切错了，后面再怎么重排、再好的模型都救不回来：
        - 切太大 → 一块里混了三个主题，向量是"平均语义"，检索命中率低
        - 切太小 → 一块里只有半句话，模型看不懂
        - 切断语义 → "……所以这个参数" 这种断片，模型无法理解

【业务定位】
    数据流 3 的第二步：把长文档切成适合检索的小块。切片质量直接决定检索质量。

【数据流向】
    上游（谁调它）：
        services/rag/document_processing.py → chunk_document()（用户上传的文档）
        services/rag/indexing.py            → chunk_document()（Issue/PR/CI 日志）
        services/code_analysis.py           → chunk_code_text()（源码，走代码图谱而非 RAG）
    下游（它调谁）：
        无外部依赖 —— 纯标准库（re / ast / csv / json / hashlib）
        ★ 这是整个 rag 包里唯一"零外部依赖"的文件，所以最容易测试和理解

【核心规则】
    1. **按结构切，不按固定字数切**：Markdown 按标题、代码按函数/类、CSV 按行、JSON 按键。
    2. **父子块设计**：parent_id 标记"属于哪个大块"，child_index 标记"大块里的第几小块"。
       检索用小切片（准），回答时用 expand_parent_context 拼回大块（全）。
    3. 代码块被切断时必须补回 ``` 围栏，否则切片不是合法代码。

【本文件函数地图】（★ 这是全项目最长的文件，建议按这个顺序读）

    ┌─ 第一步：先读总入口（10 分钟建立全局观）─────────────
    │  chunk_document()            总入口：按后缀分发到 5 种策略
    │      ├ .csv          → _csv_chunks()      按行切，每块带表头
    │      ├ .md/.pdf/.docx→ _markdown_blocks() + _pack_blocks()   按标题结构切
    │      ├ .json         → _json_blocks() + _pack_blocks()       按顶层键切
    │      └ 其他          → _plain_chunks()    滑动窗口切
    │
    ├─ 第二步：理解"块"这个核心数据结构 ──────────────────
    │  _Block                    切片的基本单元（段落/标题/代码块）
    │
    ├─ 第三步：结构感知切片的算法（这是精华）────────────
    │  _pack_blocks()            ★ 把块打包成 chunk（三条打包规则）
    │      ├ _expand_block()     超长的块先拆小
    │      │    ├ _split_long_text()        按句子边界拆
    │      │    └ _split_long_code_fence()  按行拆 + 补围栏
    │      ├ _overlap_blocks()   取上一块尾部做重叠
    │      ├ _prefix_for_chunk() 算章节前缀占多少字符
    │      └ _emit_chunk()       ★ 产出一个 chunk（含 parent_id / child_index）
    │
    ├─ 第四步：三种结构解析器 ──────────────────────────
    │  _markdown_blocks()        标题栈 + 代码围栏 + 页码标记
    │  _json_blocks()            按顶层键拆
    │  _plain_blocks()           按空行拆
    │
    ├─ 第五步：代码切片（另一条链路，不进 RAG）──────────
    │  chunk_code_text()         ★ 代码专用入口
    │      ├ _python_regions()   Python 用 AST（最准）
    │      ├ _generic_regions()  其他语言用正则（近似）
    │      │    └ _generic_symbol()  各语言的正则表
    │      └ _line_windows()     单个符号还超长时的兜底
    │
    └─ 工具函数 ─────────────────────────────────────
       _validate_limits() / _normalize_text() / _stable_id() / _line_for_offset()
       _section_prefix() / _join_block_text()
       chunk_text()        兼容旧接口（只要文本）
       chunk_pr_files()    PR diff 切片（overlap=0）
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ════════════════════════════════════════════════════════════════
# 常量：按文件类型选策略
# ════════════════════════════════════════════════════════════════

# 按 Markdown 结构切片的文件类型（PDF/DOCX 解析后也是 Markdown）
# 为什么 PDF/DOCX 也算：document_extraction 已经把它们的标题转成 # 了
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".pdf", ".docx"}
# 按结构化数据切片的类型
STRUCTURED_DATA_SUFFIXES = {".json", ".csv"}


# ════════════════════════════════════════════════════════════════
# 核心数据结构：_Block
# ════════════════════════════════════════════════════════════════

@dataclass
class _Block:
    """切片的基本单元（一个段落 / 一个标题 / 一个代码块）。

    【为什么先切 block 再打包成 chunk】
        两阶段设计：先把文档拆成语义完整的块（不破坏句子），
        再把块按字数预算打包成 chunk。这样既保证语义完整，又控制大小。

        对比"直接按 800 字硬切"：
            硬切会把句子、代码块、表格从中间劈开，切出来的片段读不懂。

    【kind 的四种取值】
        paragraph   普通段落
        heading     标题（text 为空，只带章节信息，用来触发"换章就换 chunk"）
        code_fence  代码块（切分时要补围栏）
        json_value  JSON 的一个键值

    【section_path 和 section_instance 的区别】（重要，容易混）
        section_path     = ("安装", "依赖")          给人看
        section_instance = (("安装", 0), ("依赖", 0))  给程序用

        为什么要两个：文档里可能有两个"安装"章节。
        只按标题算 parent_id，这两个章节的切片会被算成同一个父块，
        检索后拼回来就串了。带出现序号才能区分。
    """

    text: str                                       # 块文本
    start_line: int | None                          # 起始行号
    end_line: int | None                            # 结束行号
    kind: str = "paragraph"                         # paragraph / heading / code_fence / json_value
    section_path: tuple[str, ...] = ()              # 章节路径，如 ("安装", "依赖")
    section_instance: tuple[tuple[str, int], ...] = ()  # 带出现序号的章节路径（区分同名章节）
    page: int | None = None                         # PDF 页码
    metadata: dict[str, Any] = field(default_factory=dict)


# 单文档切片数上限（防止一个巨大文档把库撑爆）
# 5000 块 × 800 字 ≈ 400 万字，已经远超正常文档规模
MAX_CHUNKS_PER_DOCUMENT = 5000


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _validate_limits(max_chars: int, overlap: int) -> None:
    """校验切片参数（overlap 必须小于 max_chars，否则会死循环）。

    【调用链】被 chunk_document() 调用

    【为什么 overlap >= max_chars 会死循环】
        纯文本切片的逻辑是"下一块从 end - overlap 开始"。
        如果 overlap >= max_chars，下一块的起点 <= 上一块的起点，永远推不动。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between zero and max_chars")


def _normalize_text(text: str) -> str:
    """统一换行符（Windows 的 \\r\\n 和旧 Mac 的 \\r 都转成 \\n）。

    【调用链】被 chunk_document() / chunk_code_text() 调用

    【为什么必须先做】
        后面所有行号计算、正则匹配都假设"一行以 \\n 结尾"。
        如果混着 \\r\\n，splitlines 没问题但 len() 和 offset 会算错，
        导致 start_line / end_line 偏移，引用定位就不准了。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _stable_id(*parts: object) -> str:
    """生成稳定 ID（同样输入 → 同样 ID）。

    【调用链】被 _emit_chunk() / _plain_chunks() / _csv_chunks() / chunk_code_text() 调用

    【为什么重要】重复索引同一份文档时，ID 不变才能做幂等更新，而不是产生重复数据。

    【为什么用 "\\x1f" 当分隔符】
        这是 ASCII 的"单元分隔符"（Unit Separator），
        正常文本里绝不会出现。用它拼接可以避免
        ("ab", "c") 和 ("a", "bc") 拼出同一个字符串的歧义。

    【为什么取 24 位】
        sha256 全长 64 位，24 位（96 bit）碰撞概率已经极低，
        而且短一些便于日志阅读和调试。
    """
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _line_for_offset(text: str, offset: int, first_line: int | None) -> int | None:
    """把字符偏移量换算成行号。

    【调用链】被 _split_long_text() / _overlap_blocks() / _plain_chunks() 调用

    【参数 first_line 是什么】
        这段 text 在原文档里的起始行号。
        比如某个段落从第 10 行开始，段内偏移 0 对应第 10 行。
        first_line 为 None 表示"不知道原文位置"（如 JSON 格式化后的文本），直接返回 None。

    【实现】数 offset 之前有几个 \\n，加到 first_line 上。
    """
    if first_line is None:
        return None
    return first_line + text.count("\n", 0, max(0, offset))


# ════════════════════════════════════════════════════════════════
# 超长块的拆分（两种：普通文本 / 代码块）
# ════════════════════════════════════════════════════════════════

def _split_long_text(block: _Block, max_chars: int) -> list[_Block]:
    """把一个超长段落按句子边界切成多个块。

    【调用链】_line_for_offset() → 本函数
    【被谁调用】_expand_block()（kind != code_fence 时）

    【为什么按句子切】硬按字符数切会把句子砍成两半，
        检索到的片段读起来是断的，模型理解也会受影响。

    【数据流向】
        入：一个超长的 _Block、max_chars（每块的字符预算）
        出：若干 _Block（每个都不超过 max_chars），行号已回填

    【三步走】
        ① 按句子切成 units（句号/问号/感叹号/换行）
        ② 把 units 攒成块，直到接近 max_chars
        ③ 用 find 回填每块在原文本中的行号
    """
    # 本来就够短，直接返回（最常见的情况，快速路径）
    if len(block.text) <= max_chars:
        return [block]

    # ── ① 按句号/问号/感叹号/换行切成句子 ──
    # 正则三选一：中文标点后切 / 英文标点+空白后切 / 换行处切
    units = [
        unit
        for unit in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+|\n+", block.text)
        if unit and unit.strip()
    ]
    # 一句话就超长（比如没有标点的长串、压缩过的代码）→ 只能硬切
    if len(units) <= 1:
        units = [block.text[index : index + max_chars] for index in range(0, len(block.text), max_chars)]

    # ── ② 把句子攒成块，直到接近 max_chars ──
    pieces: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        # 单个 unit 还是超长 → 先把手上攒的结算，再把这个 unit 硬切
        if len(unit) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(unit[index : index + max_chars] for index in range(0, len(unit), max_chars))
            continue
        # 试着加进来（用换行连接）
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            # 加了会超 → 结算当前块，从这个 unit 重新开始
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)

    # ── ③ 回填行号（用 find 定位每块在原文本中的位置）──
    output: list[_Block] = []
    cursor = 0
    for piece in pieces:
        offset = block.text.find(piece, cursor)
        if offset < 0:
            # find 失败（理论上不该发生）→ 用 cursor 兜底，至少保证行号单调递增
            offset = cursor
        end_offset = min(len(block.text), offset + len(piece))
        output.append(
            _Block(
                text=piece,
                start_line=_line_for_offset(block.text, offset, block.start_line),
                # 结束行取 end_offset - 1 的位置（因为 end_offset 是"开区间"）
                end_line=_line_for_offset(block.text, max(offset, end_offset - 1), block.start_line),
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        )
        cursor = end_offset
    return output


def _split_long_code_fence(block: _Block, max_chars: int) -> list[_Block]:
    """切分超长代码块 —— 每块都要重新补上 ``` 围栏。

    【调用链】_split_long_text()（退化路径）→ 本函数
    【被谁调用】_expand_block()（kind == "code_fence" 时）

    【为什么必须补围栏】```python ... ``` 是一个整体语法结构。
        如果切成两半而不补围栏，第二块就是一段没有开头的裸代码，
        模型看到会困惑，Markdown 渲染也会错乱。

    【数据流向】
        入：一个超长的代码块 _Block
        出：若干 _Block，每块形如 "```\n<几行代码>\n```"

    【算法】
        ① 剥掉原围栏，拿到纯代码行
        ② 按行累积，直到接近 body_limit（= max_chars - 围栏本身占的长度）
        ③ 每块输出时重新包上围栏
    """
    # 本来就够短 → 直接返回
    if len(block.text) <= max_chars:
        return [block]

    lines = block.text.splitlines()
    if not lines:
        return []

    # 预算太小（连围栏都放不下），没法优雅处理 → 退化成纯文本切分
    if max_chars <= 8:
        plain = _Block(
            text=block.text,
            start_line=block.start_line,
            end_line=block.end_line,
            kind="code_fragment",     # 改类型，避免 _overlap_blocks 把它当代码块
            section_path=block.section_path,
            section_instance=block.section_instance,
            page=block.page,
            metadata=dict(block.metadata),
        )
        return _split_long_text(plain, max_chars)

    # ── ① 识别原始围栏标记（``` 或 ~~~）和语言标识 ──
    original_opener = lines[0] if lines[0].lstrip().startswith(("```", "~~~")) else "```"
    marker = original_opener.lstrip()[:3]      # 只取前 3 个字符作为标记
    opener = marker                             # 开围栏（丢掉语言标识，节省字符）
    # 最后一行是不是闭合围栏
    has_closer = len(lines) > 1 and lines[-1].lstrip().startswith(marker)
    body = lines[1:-1] if has_closer else lines[1:]   # 去掉首尾围栏，只留代码
    closing = marker
    # 每块要预留围栏本身占的字节（opener + closing + 两个换行）
    wrapper_size = len(opener) + len(closing) + 2
    body_limit = max_chars - wrapper_size

    # 空代码块（只有围栏没内容）→ 原样返回一个
    if not body:
        return [
            _Block(
                text=f"{opener}\n{closing}",
                start_line=block.start_line,
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        ]

    output: list[_Block] = []
    current: list[str] = []
    current_size = 0
    # 代码正文从原块的第二行开始（第一行是围栏）
    first_body_line = (block.start_line + 1) if block.start_line is not None else None
    current_start = first_body_line

    def flush(end_line: int | None) -> None:
        """把当前累积的行打包成一块（自动补围栏）。"""
        nonlocal current, current_size, current_start
        if not current:
            return
        body_text = "\n".join(current)
        output.append(
            _Block(
                # ★ 关键：重新包上围栏
                text=f"{opener}\n{body_text}\n{closing}",
                start_line=current_start,
                end_line=end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={
                    **block.metadata,
                    # 记录"这一块来自原始围栏的哪几行"，便于回原文定位
                    "fence_start_line": block.start_line,
                    "fence_end_line": block.end_line if has_closer else None,
                },
            )
        )
        current = []
        current_size = 0

    # ── ② 按行累积 ──
    for body_index, line in enumerate(body):
        index = (first_body_line + body_index) if first_body_line is not None else None

        # 单行就超预算（压缩过的 JS / 超长 SQL）→ 这行单独硬切
        if len(line) > body_limit:
            flush((index - 1) if index is not None else None)
            for offset in range(0, len(line), body_limit):
                piece = line[offset : offset + body_limit]
                output.append(
                    _Block(
                        text=f"{opener}\n{piece}\n{closing}",
                        start_line=index,
                        end_line=index,
                        kind=block.kind,
                        section_path=block.section_path,
                        section_instance=block.section_instance,
                        page=block.page,
                        metadata={
                            **block.metadata,
                            "fence_start_line": block.start_line,
                            "fence_end_line": block.end_line if has_closer else None,
                        },
                    )
                )
            current_start = (index + 1) if index is not None else None
            continue

        # 加了这一行会不会超预算（+1 是换行符）
        line_size = len(line) + (1 if current else 0)
        if current and current_size + line_size > body_limit:
            flush((index - 1) if index is not None else None)
            current_start = index
        if not current:
            current_start = index
        current.append(line)
        current_size += line_size

    # ── ③ 结算最后一块 ──
    last_body_line = None
    if block.end_line is not None:
        # 有闭合围栏时，最后一行正文是 end_line - 1
        last_body_line = block.end_line - (1 if has_closer else 0)
    flush(last_body_line)
    return output


def _expand_block(block: _Block, max_chars: int) -> list[_Block]:
    """按块类型选择切分策略。

    【调用链】_split_long_code_fence() / _split_long_text() → 本函数
    【被谁调用】_pack_blocks()（每个块进打包前先过这一道）

    代码块走"按行切 + 补围栏"，其他走"按句子切"。
    为什么代码不能按句子切：代码里没有句号，按句号切会切出乱码。
    """
    if block.kind == "code_fence":
        return _split_long_code_fence(block, max_chars)
    return _split_long_text(block, max_chars)


def _overlap_blocks(blocks: list[_Block], overlap: int) -> list[_Block]:
    """从上一块的尾部取一段作为下一块的开头（重叠区）。

    【调用链】_line_for_offset() → 本函数
    【被谁调用】_pack_blocks()（换新 chunk 时调用）

    【为什么要重叠】语义可能正好切在边界上。
        重叠 160 字符能让下一块看到上一块的结尾，避免"半句话"。

    【为什么从句子边界开始】直接从中间切会得到"…了这"这种断片，
        所以这里往前找最近的句号/空白作为起点。

    【数据流向】
        入：上一组 blocks、overlap（重叠字符数）
        出：重叠区对应的 _Block 列表（metadata 里标了 overlap_fragment=True）

    【三级对齐策略】
        ① 往前 64 字符内找句号/分号 → 从句子开头开始
        ② 找不到句子边界 → 找最近的空白（词边界）
        ③ 都没有 → 往后找第一个边界
    """
    if overlap <= 0 or not blocks:
        return []

    selected: list[_Block] = []
    remaining = overlap

    # 从最后一块往前取（最近的上下文最相关）
    for block in reversed(blocks):
        # 跳过空块和代码块（代码块补围栏会乱，不适合重叠）
        if not block.text or block.kind == "code_fence":
            continue

        separator = 2 if selected else 0        # 块之间拼 "\n\n"，占 2 字符
        allowance = remaining - separator
        if allowance <= 0:
            break

        # 整块都放得下 → 整块带过来
        if len(block.text) <= allowance:
            selected.append(block)
            remaining -= len(block.text) + separator
            continue

        # 块比剩余预算长 → 只取尾部一段，并对齐到句子/词边界
        offset = len(block.text) - allowance
        # 往回看一小段（最多 64 字符）找边界
        lookback_start = max(0, offset - min(64, max(16, allowance * 2)))
        lookback = block.text[lookback_start:offset]
        sentence_boundaries = list(re.finditer(r"[。！？.!?；;]\s*", lookback))
        word_boundaries = list(re.finditer(r"\s+", lookback))

        if sentence_boundaries:
            # 优先句子边界：从最后一个句号之后开始
            fragment_offset = lookback_start + sentence_boundaries[-1].end()
        elif word_boundaries:
            # 其次词边界：从最后一个空白之后开始
            fragment_offset = lookback_start + word_boundaries[-1].end()
        else:
            # 没有边界可对齐 → 往后找第一个边界
            boundary_window = block.text[offset : offset + min(48, max(8, allowance // 3))]
            boundary = re.search(r"\s+|[。！？.!?；;]\s*", boundary_window)
            fragment_offset = offset + boundary.end() if boundary else offset

        selected.append(
            _Block(
                text=block.text[fragment_offset:],
                start_line=_line_for_offset(block.text, fragment_offset, block.start_line),
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={**block.metadata, "overlap_fragment": True},   # 标记：这是重叠片段
            )
        )
        remaining = 0
        break

    # 因为是倒序取的，要翻回来保持原文顺序
    return list(reversed(selected))


# ════════════════════════════════════════════════════════════════
# 章节前缀与 chunk 产出
# ════════════════════════════════════════════════════════════════

def _section_prefix(section_path: tuple[str, ...], page: int | None, limit: int | None = None) -> str:
    """生成章节前缀（把标题层级 + 页码渲染成 Markdown 标题）。

    【调用链】被 _prefix_for_chunk() 调用

    【为什么每个 chunk 都带章节前缀】切片脱离原文后，模型不知道它在讲什么。
        带上"## 安装 > ### 依赖"这样的前缀，检索命中后能立刻知道上下文位置。

    【数据流向】
        入：section_path（("安装","依赖")）、page、limit（长度上限）
        出：形如 "# 安装\\n## 依赖\\nPage 3" 的字符串

    【为什么用 min(index+1, 6)】
        section_path 可能有 8 层，但 Markdown 最多 6 级标题。
        超过 6 级就都用 ######，避免生成非法 Markdown。
    """
    # 每一层标题渲染成对应级别的 #
    parts = [f"{'#' * min(index + 1, 6)} {title}" for index, title in enumerate(section_path)]
    if page is not None:
        parts.append(f"Page {page}")
    value = "\n".join(parts)

    # 不需要限制或没超限
    if limit is None or len(value) <= limit:
        return value
    # 上限太小，连省略号都放不下
    if limit <= 3:
        return value[:limit]
    # 截断 + 省略号
    return value[: limit - 3].rstrip() + "..."


def _prefix_for_chunk(
    section_path: tuple[str, ...],
    page: int | None,
    max_chars: int,
    *,
    has_body: bool,
) -> str:
    """计算前缀的可用长度（前缀最多占 max_chars 的 1/3，且不超过 512）。

    【调用链】_section_prefix() → 本函数
    【被谁调用】_emit_chunk() / _pack_blocks()

    【为什么要限制前缀长度】
        前缀是为了提供上下文，不是为了占满预算。
        如果章节路径特别深（8 层标题），前缀可能吃掉一半预算，
        正文就没空间了 —— 那就本末倒置。

    【has_body 的作用】
        True  → 有正文，前缀只能占 1/3（给正文留空间）
        False → 纯标题块（text 为空），前缀可以占满
    """
    # 没有任何章节信息 → 不生成前缀
    if not section_path and page is None:
        return ""

    if not has_body:
        limit = max_chars          # 纯标题块，前缀可以占满
    elif max_chars <= 3:
        limit = 0
    else:
        # 三重限制取最小：512 字符上限、1/3 预算、给省略号留 3 字符
        limit = min(512, max_chars // 3, max_chars - 3)

    return _section_prefix(section_path, page, limit=limit) if limit > 0 else ""


def _join_block_text(blocks: list[_Block]) -> str:
    """把多个块拼成正文（块之间空一行）。

    【调用链】被 _emit_chunk() / _pack_blocks() 调用

    为什么用 "\\n\\n" 而不是 "\\n"：
        块本身是段落，段落之间应该有视觉间隔，也符合 Markdown 习惯。
    """
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _emit_chunk(
    blocks: list[_Block],
    *,
    source_path: str,
    chunk_type: str,
    child_index: int,
    max_chars: int,
) -> dict[str, Any]:
    """把一组块输出成一个 chunk（含完整的检索元数据）。

    【调用链】_join_block_text() / _prefix_for_chunk() / _stable_id() → 本函数
    【被谁调用】_pack_blocks() 的 flush()

    【数据流向】
        入：blocks（一个 chunk 包含的块）、source_path、chunk_type、child_index
        出：dict，字段与 documents 表的 meta 字段对应

    【★ 父子块的关键就在这里】
        parent_id 由"来源路径 + 章节实例 + 页码"决定：
        同一章节下的所有切片共享 parent_id，检索后可以拼回完整章节。

        举例：一篇文档的"安装"章节被切成 3 块，
        这 3 块的 parent_id 相同，child_index 分别是 0/1/2。
        检索命中第 2 块后，expand_parent_context() 用 parent_id
        把 3 块拼回来，给模型看完整章节。
    """
    first = blocks[0]
    body = _join_block_text(blocks)
    # 前缀（章节标题 + 页码）
    prefix = _prefix_for_chunk(first.section_path, first.page, max_chars, has_body=bool(body))
    # 拼接：前缀 + 空行 + 正文
    content = f"{prefix}\n\n{body}" if prefix and body else (prefix or body)

    # 【父子块关键】parent_id 由"来源路径 + 章节实例 + 页码"决定：
    #   同一章节下的所有切片共享 parent_id，检索后可以拼回完整章节
    parent_id = _stable_id(
        "parent",
        source_path,
        # section_instance 为空时退化用 section_path（并给每层补序号 0）
        first.section_instance or tuple((title, 0) for title in first.section_path),
        first.page,
    )

    # 含代码块的 chunk 单独标类型（前端可以用不同样式渲染）
    effective_type = "markdown_code" if any(block.kind == "code_fence" for block in blocks) else chunk_type

    # 合并所有块的 metadata（后面的覆盖前面的）
    metadata: dict[str, Any] = {}
    for block in blocks:
        metadata.update(block.metadata)

    return {
        "content": content,
        # chunk_id 里包含 content 的哈希 → 内容变了 ID 就变，便于增量更新
        "chunk_id": _stable_id("chunk", parent_id, child_index, content),
        "parent_id": parent_id,
        "child_index": child_index,           # 在父块里的序号
        "chunk_type": effective_type,
        "section_path": list(first.section_path),
        # section_title 只取最后一级（用于引用展示"来自 3.2 节"）
        "section_title": first.section_path[-1] if first.section_path else None,
        "page": first.page,
        # 行号取所有块的最小/最大值（一个 chunk 可能跨多行）
        "start_line": min((block.start_line for block in blocks if block.start_line is not None), default=None),
        "end_line": max((block.end_line for block in blocks if block.end_line is not None), default=None),
        **metadata,
    }


# ════════════════════════════════════════════════════════════════
# ★ 核心算法：把块打包成 chunk
# ════════════════════════════════════════════════════════════════

def _pack_blocks(
    blocks: list[_Block],
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    chunk_type: str,
) -> list[dict[str, Any]]:
    """把块打包成 chunk —— 结构感知切片的核心。

    【调用链】追这个函数会进入：
        1. _prefix_for_chunk()   → 算前缀占多少字符
        2. _expand_block()       → 超长的块先拆小
        3. _join_block_text()    → 试拼看会不会超
        4. _overlap_blocks()     → 换 chunk 时取重叠区
        5. _emit_chunk()         → 真正产出 chunk
    【被谁调用】chunk_document()（markdown / json 两条路径）

    【数据流向】
        入：blocks（已按结构切好的块）、source_path、max_chars、overlap、chunk_type
        出：chunk dict 列表

    打包规则（按优先级）：
        1. 章节变了（section_instance 或 page 变化）→ 强制换新 chunk
           【为什么】不同章节的内容混在一个 chunk 里，检索时会互相干扰
        2. 超预算 → 换新 chunk，并把上一块的尾部作为重叠区带过来
        3. 否则继续往当前 chunk 里塞

    【最后有一道断言】
        任何 chunk 超过 max_chars 都是 bug（说明预算算错了）。
        宁可抛 AssertionError 也不能把超长内容写进库 ——
        因为 Milvus 的 VARCHAR 有字节上限，超了会写失败。
    """
    chunks: list[dict[str, Any]] = []
    current: list[_Block] = []
    # 当前 chunk 的"章节键"：(章节实例, 页码)
    current_key: tuple[tuple[tuple[str, int], ...], int | None] | None = None
    # 每个章节键下已经产出了几个 chunk（用于算 child_index）
    child_counts: dict[tuple[tuple[tuple[str, int], ...], int | None], int] = {}

    def flush() -> None:
        """把当前累积的块输出成一个 chunk。"""
        nonlocal current
        if not current or current_key is None:
            return
        child_index = child_counts.get(current_key, 0)
        chunks.append(
            _emit_chunk(
                current,
                source_path=source_path,
                chunk_type=chunk_type,
                child_index=child_index,
                max_chars=max_chars,
            )
        )
        child_counts[current_key] = child_index + 1

    for raw_block in blocks:
        key = (raw_block.section_instance, raw_block.page)
        # 先算前缀，得出正文可用预算
        prefix = _prefix_for_chunk(
            raw_block.section_path,
            raw_block.page,
            max_chars,
            has_body=bool(raw_block.text),
        )
        prefix_size = len(prefix)
        # 正文可用预算 = 总预算 - 前缀 - 分隔空行
        body_limit = max(1, max_chars - prefix_size - (2 if prefix_size else 0))

        # 超长的块先拆成若干小块，再逐个打包
        for block in _expand_block(raw_block, body_limit):
            key = (block.section_instance, block.page)

            # ── 规则 1：章节变了 → 先结算当前 chunk ──
            if current and key != current_key:
                flush()
                current = []
            current_key = key

            # 试拼：当前块 + 新块
            candidate_body = _join_block_text([*current, block])
            prefix = _prefix_for_chunk(
                block.section_path,
                block.page,
                max_chars,
                has_body=bool(candidate_body),
            )
            candidate_size = len(candidate_body) + len(prefix) + (2 if prefix else 0)

            # ── 规则 2：超预算 → 结算并把尾部作为重叠区 ──
            if current and candidate_size > max_chars:
                previous = list(current)
                flush()
                # 取上一块的尾部作为新 chunk 的开头
                current = _overlap_blocks(previous, overlap)
                # 重叠区本身也可能超预算 → 从头砍（丢掉最旧的块）
                while current:
                    candidate_body = _join_block_text([*current, block])
                    if len(candidate_body) + len(prefix) + (2 if prefix else 0) <= max_chars:
                        break
                    current.pop(0)

            # ── 规则 3：塞进当前 chunk ──
            current.append(block)

    # 收尾：把最后一批块输出
    flush()

    # 最后一道防线：任何 chunk 超长都是 bug
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("chunk content exceeded max_chars")
    return chunks


# ════════════════════════════════════════════════════════════════
# 结构解析器 ①：纯文本（按空行分段）
# ════════════════════════════════════════════════════════════════

def _plain_blocks(text: str) -> list[_Block]:
    """纯文本按空行分段。

    【调用链】被 _plain_chunks()（间接）/ _json_blocks()（JSON 解析失败时退化）/ _csv_chunks() 调用

    【规则】连续非空行 = 一个段落；空行 = 段落结束。
        这是最朴素的"语义完整"假设：作者用空行分段，说明一段就是一个意思。
    """
    lines = text.splitlines()
    blocks: list[_Block] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        """结算当前段落。"""
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(_Block(content, start_line, max(start_line, end_line)))
        current = []

    for line_number, line in enumerate(lines, start=1):
        # 空行 → 段落结束（end_line 是上一行）
        if not line.strip():
            flush(line_number - 1)
            continue
        # 新段落开始 → 记下起始行
        if not current:
            start_line = line_number
        current.append(line)

    # 收尾：处理没有以空行结尾的最后一段
    flush(len(lines))
    return blocks


def _plain_chunks(
    text: str,
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """纯文本切片（滑动窗口 + 边界对齐）。

    【调用链】_stable_id() / _line_for_offset() → 本函数
    【被谁调用】chunk_document()（非 csv/markdown/json 的所有格式）

    和 _pack_blocks 的区别：这里没有结构信息，只能按字符窗口滑动。

    【数据流向】
        入：text、source_path、max_chars、overlap
        出：chunk dict 列表（chunk_type="text"，无 section_path）

    【为什么要"边界对齐"】
        直接从 cursor+max_chars 处切，会得到"…这"这种断片。
        所以先在 [cursor + max_chars/2, end] 区间里找最近的
        空行（\\n\\n）或换行（\\n），在那里切。

    【重叠的实现】下一块的起点 = end - overlap
        而不是 end（否则上下文会断）。
    """
    chunks: list[dict[str, Any]] = []
    # 纯文本没有章节概念，整个文件就是一个父块
    parent_id = _stable_id("parent", source_path, "plain")
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        end = min(text_length, cursor + max_chars)

        # ── 边界对齐：只在后半段找边界，避免切得太短 ──
        if end < text_length:
            search_start = cursor + max(1, max_chars // 2)
            candidates = [
                text.rfind("\n\n", search_start, end),   # 优先空行（段落边界）
                text.rfind("\n", search_start, end),     # 其次换行
            ]
            boundary = max(candidates)
            if boundary > cursor:
                # +2 表示吃掉了 "\n\n"，+1 表示只吃掉一个 "\n"
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1)

        # ── 去掉首尾空白（但要记住去掉了多少，用于算行号）──
        raw = text[cursor:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        content = raw[left_trim:right_trim]

        if content:
            start_offset = cursor + left_trim
            end_offset = cursor + right_trim
            child_index = len(chunks)
            chunks.append(
                {
                    "content": content,
                    "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "chunk_type": "text",
                    "section_path": [],
                    "section_title": None,
                    "page": None,
                    "start_line": _line_for_offset(text, start_offset, 1),
                    "end_line": _line_for_offset(text, max(start_offset, end_offset - 1), 1),
                }
            )
            # 切片数上限保护
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")

        if end >= text_length:
            break
        # 下一块从 end - overlap 开始（保证重叠）
        # max(cursor + 1, ...) 保证一定前进，否则 overlap 配置不当会死循环
        cursor = max(cursor + 1, end - overlap)

    return chunks


# ════════════════════════════════════════════════════════════════
# 结构解析器 ②：Markdown（标题栈 + 代码围栏 + 页码）
# ════════════════════════════════════════════════════════════════

def _markdown_blocks(text: str) -> list[_Block]:
    """Markdown 按标题层级分块（同时识别 PDF 页码注释和代码围栏）。

    【调用链】被 chunk_document() 调用（.md/.pdf/.docx）
    【下游】产出的 blocks 交给 _pack_blocks() 打包

    【数据流向】
        入：Markdown 文本（PDF/DOCX 已在解析阶段转成 Markdown）
        出：_Block 列表，每个块都带 section_path / section_instance / page

    三种特殊行：
        # 标题           → 更新章节栈，产出一个 heading 块
        <!-- page: n --> → 更新当前页码（PDF 解析时插入的）
        ``` / ~~~        → 进入代码围栏模式，直到遇到闭合标记

    【标题栈（headings）是怎么工作的】
        遇到 "# 安装"   → 栈变 [(1,"安装",0)]
        遇到 "## 依赖"  → 栈变 [(1,"安装",0), (2,"依赖",0)]
        遇到 "## 配置"  → 弹到同级：[(1,"安装",0)] 再压入 → [(1,"安装",0),(2,"配置",0)]

        所以任何一段正文，它的 section_path 就是"当前栈里的所有标题"，
        天然表达了层级关系（"安装 > 依赖"）。

    【为什么标题也要产出一个块（text 为空的 heading 块）】
        用来触发 _pack_blocks 的"章节变了就换 chunk"规则。
        如果不产出这个块，第一个段落会归属到上一个章节。
    """
    lines = text.splitlines()
    blocks: list[_Block] = []
    # 标题栈：(层级, 标题文本, 出现序号)
    headings: list[tuple[int, str, int]] = []
    # 同名标题的出现次数统计，key = (父路径, 层级, 标题)
    heading_occurrences: dict[tuple[tuple[tuple[str, int], ...], int, str], int] = {}
    current_page: int | None = None
    current: list[str] = []
    current_start = 1
    current_kind = "paragraph"
    fence_marker: str | None = None     # 非 None 表示"当前在代码围栏内"
    # 标题正则：1~6 个 # + 空格 + 标题（允许结尾有 #）
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    # 页码注释正则：<!-- page: 3 -->
    page_re = re.compile(r"^<!--\s*page\s*:\s*(\d+)\s*-->$", re.IGNORECASE)

    def section_path() -> tuple[str, ...]:
        """当前章节路径（只有标题文本）。"""
        return tuple(title for _, title, _ in headings)

    def section_instance() -> tuple[tuple[str, int], ...]:
        """带出现序号的章节路径（区分"安装"出现两次的情况）。"""
        return tuple((title, occurrence) for _, title, occurrence in headings)

    def flush(end_line: int) -> None:
        """结算当前段落（带上当前的章节信息和页码）。"""
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(
                _Block(
                    content,
                    current_start,
                    max(current_start, end_line),
                    kind=current_kind,
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                )
            )
        current = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        # ── 情况 1：在代码围栏内 → 只等闭合标记 ──
        # 注意：围栏内的 # 不是标题，不能当标题处理
        if fence_marker is not None:
            current.append(line)
            if stripped.startswith(fence_marker):
                flush(line_number)
                fence_marker = None
                current_kind = "paragraph"
            continue

        # ── 情况 2：PDF 页码标记 ──
        page_match = page_re.match(stripped)
        if page_match:
            flush(line_number - 1)
            current_page = int(page_match.group(1))
            continue

        # ── 情况 3：标题 ──
        heading_match = heading_re.match(line)
        if heading_match:
            flush(line_number - 1)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            # 弹栈：同级或更高级的标题要先把栈里更深的清掉
            while headings and headings[-1][0] >= level:
                headings.pop()
            # 算这个标题的出现序号（同名标题第几次出现）
            parent_instance = section_instance()
            occurrence_key = (parent_instance, level, title)
            occurrence = heading_occurrences.get(occurrence_key, 0)
            heading_occurrences[occurrence_key] = occurrence + 1
            headings.append((level, title, occurrence))
            # 标题本身也产出块（纯标题块，text 为空，只带章节信息）
            blocks.append(
                _Block(
                    "",
                    line_number,
                    line_number,
                    kind="heading",
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                    metadata={"heading_line": line_number},
                )
            )
            continue

        # ── 情况 4：代码围栏开始 ──
        if stripped.startswith(("```", "~~~")):
            flush(line_number - 1)
            current_start = line_number
            current_kind = "code_fence"
            fence_marker = stripped[:3]      # 记录是 ``` 还是 ~~~
            current = [line]
            continue

        # ── 情况 5：空行 → 段落结束 ──
        if not stripped:
            flush(line_number - 1)
            continue

        # ── 情况 6：普通正文行 ──
        if not current:
            current_start = line_number
        current.append(line)

    # 收尾
    flush(len(lines))
    return blocks


# ════════════════════════════════════════════════════════════════
# 结构解析器 ③：JSON（按顶层键拆）
# ════════════════════════════════════════════════════════════════

def _json_blocks(text: str, max_chars: int) -> list[_Block]:
    """JSON 按顶层键/数组元素分块（每个键一个块）。

    【调用链】_plain_blocks()（解析失败时退化）→ 本函数
    【被谁调用】chunk_document()（.json）

    【为什么按键切】JSON 里每个键往往是一个独立概念，
        按键切出来的块语义最完整。

    【数据流向】
        入：JSON 文本、max_chars
        出：_Block 列表，section_path 是 JSON 路径（如 "$.database"）

    【为什么用 JSON 路径当章节名】
        检索命中后能告诉模型"这段来自 config.database 这个键"，
        比"这段来自第 42 行"有用得多。
    """
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        # 不是合法 JSON（可能是 JSONL 或注释掉的 JSON）→ 退化成纯文本
        return _plain_blocks(text)

    # 整体够小 → 整份文档就是一个块
    formatted = json.dumps(value, ensure_ascii=False, indent=2)
    if len(formatted) <= max_chars:
        return [
            _Block(
                formatted,
                1,
                max(1, len(text.splitlines())),
                kind="json_value",
                metadata={"json_path": "$"},
            )
        ]

    # ── 按顶层结构拆 ──
    blocks: list[_Block] = []
    if isinstance(value, dict):
        # 对象：每个键一个块
        items = [(f"$.{key}", {key: item}) for key, item in value.items()]
    elif isinstance(value, list):
        # 数组：每个元素一个块
        items = [(f"$[{index}]", item) for index, item in enumerate(value)]
    else:
        # 标量：没法拆
        items = [("$", value)]

    # 用 JSON 路径当章节名，检索命中后能知道"来自哪个键"
    for json_path, item in items:
        blocks.append(
            _Block(
                json.dumps(item, ensure_ascii=False, indent=2),
                None,       # 格式化后行号对不上原文，所以置 None
                None,
                kind="json_value",
                section_path=(json_path,),
                section_instance=((json_path, 0),),
                metadata={"json_path": json_path},
            )
        )
    return blocks


# ════════════════════════════════════════════════════════════════
# 结构解析器 ④：CSV（按行切，每块带表头）
# ════════════════════════════════════════════════════════════════

def _csv_chunks(text: str, source_path: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    """CSV 按行分块，每个块都带表头。

    【调用链】_pack_blocks()（解析失败时退化）/ _stable_id() → 本函数
    【被谁调用】chunk_document()（.csv）

    【为什么每块都重复表头】没有表头的数据行是"1,foo,2026-01-01"这种裸数据，
        模型完全看不懂每列是什么意思。

    【数据流向】
        入：CSV 文本、source_path、max_chars、overlap
        出：chunk dict 列表，每块 content = 表头 + 若干数据行

    【四个特殊处理】
        ① 表头本身就超长 → 单独切成几块（csv_header_part）
        ② 单行数据超长 → 该行也要切（row_part / row_parts 记录是第几段）
        ③ 只有表头没数据 → 返回一个只有表头的块
        ④ 重叠按"行"回退，而不是按字符（否则行会被切一半）
    """
    try:
        # csv 模块默认字段大小上限是 128KB，超过会报错。
        # 这里按文本长度动态放宽，否则长文本字段（如 Issue 正文）会解析失败。
        if len(text) > csv.field_size_limit():
            csv.field_size_limit(len(text))
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        # 不是合法 CSV → 退化成纯文本切片
        return _pack_blocks(
            _plain_blocks(text),
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            chunk_type="text",
        )

    if not rows:
        return []

    def render(row: list[str]) -> str:
        """把一行渲染回 CSV 格式（正确处理引号和逗号）。

        【为什么不用 ",".join(row)】
            如果某个单元格里含逗号或引号，直接 join 会破坏 CSV 结构。
            用 csv.writer 渲染才能正确加引号转义。
        """
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow(row)
        return buffer.getvalue()

    header = render(rows[0])
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "csv")

    def append_chunk(
        content: str,
        *,
        row_start: int,
        row_end: int,
        chunk_type: str = "csv_rows",
        row_part: int | None = None,
        row_parts: int | None = None,
        precise_lines: bool = True,
    ) -> None:
        """追加一个 CSV chunk（统一处理 chunk_id / 行号 / 上限检查）。"""
        child_index = len(chunks)
        chunks.append(
            {
                "content": content,
                "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                "parent_id": parent_id,
                "child_index": child_index,
                "chunk_type": chunk_type,
                "section_path": [],
                "section_title": None,
                "page": None,
                "start_line": row_start if precise_lines else None,
                "end_line": row_end if precise_lines else None,
                "row_start": row_start,     # CSV 特有：来自第几行（数据行，1-based，含表头）
                "row_end": row_end,
                # 只有被切开的行才带这两个字段（表示"这是第 i 段，共 n 段"）
                **({"row_part": row_part, "row_parts": row_parts} if row_part is not None else {}),
            }
        )
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")

    # ── 特殊情况 ①：表头本身就超长 → 单独切 ──
    if header and len(header) + 1 >= max_chars:
        header_parts = [header[index : index + max_chars] for index in range(0, len(header), max_chars)] or [""]
        for part_index, part in enumerate(header_parts):
            append_chunk(
                part,
                row_start=1,
                row_end=1,
                chunk_type="csv_header_part",
                row_part=part_index + 1,
                row_parts=len(header_parts),
                precise_lines="\n" not in part,     # 被切开的片段行号不准
            )
        header_prefix = ""      # 表头已单独成块，后面不再重复
    else:
        header_prefix = header

    # ── 特殊情况 ②：只有表头没有数据行 ──
    if not rows[1:]:
        if not chunks:
            append_chunk(header, row_start=1, row_end=1, precise_lines="\n" not in header)
        return chunks

    # ── 准备：算每块可用预算 ──
    prefix_size = len(header_prefix) + (1 if header_prefix else 0)
    row_budget = max_chars - prefix_size

    # 先把每行渲染成文本（超长的行也要切）
    # 元组含义：(文本, 行号, 第几段, 共几段, 行号是否精确)
    records: list[tuple[str, int, int | None, int | None, bool]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        rendered = render(row)
        parts = [rendered[index : index + row_budget] for index in range(0, len(rendered), row_budget)] or [""]
        for part_index, part in enumerate(parts):
            records.append((part, row_number, part_index + 1 if len(parts) > 1 else None, len(parts) if len(parts) > 1 else None, "\n" not in rendered))

    # ── 贪心装行 ──
    start = 0
    while start < len(records):
        selected: list[tuple[str, int, int | None, int | None, bool]] = []
        index = start

        # 一行一行往里加，直到加不下
        while index < len(records):
            candidate_rows = [item[0] for item in selected]
            candidate_rows.append(records[index][0])
            candidate = "\n".join([header_prefix, *candidate_rows]) if header_prefix else "\n".join(candidate_rows)
            # 至少装一行（selected 非空才判断超限），避免死循环
            if selected and len(candidate) > max_chars:
                break
            selected.append(records[index])
            index += 1

        # 组装这一块：表头 + 选中的行
        selected_text = [item[0] for item in selected]
        content = "\n".join([header_prefix, *selected_text]) if header_prefix else "\n".join(selected_text)
        row_start = min(item[1] for item in selected)
        row_end = max(item[1] for item in selected)
        # 只有"单行被切成多段"时才有意义
        row_part = selected[0][2] if len(selected) == 1 else None
        row_parts = selected[0][3] if len(selected) == 1 else None

        append_chunk(
            content,
            row_start=row_start,
            row_end=row_end,
            row_part=row_part,
            row_parts=row_parts,
            precise_lines=all(item[4] for item in selected),
        )

        if index >= len(records):
            break

        # ── 算下一块的起点（按行回退做重叠）──
        if overlap <= 0:
            start = index
            continue

        size = 0
        overlap_records = 0
        for record in reversed(selected):
            record_size = len(record[0])
            # 至少回退 1 行（overlap_records 为 0 时不判断）
            if overlap_records and size + record_size > overlap:
                break
            size += record_size
            overlap_records += 1
            if size >= overlap:
                break
        # max(start + 1, ...) 保证前进
        start = max(start + 1, index - overlap_records)

    # 同样的最后防线
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("CSV chunk content exceeded max_chars")
    return chunks


# ════════════════════════════════════════════════════════════════
# ★ 总入口：按后缀分发
# ════════════════════════════════════════════════════════════════

def chunk_document(
    text: str,
    source_path: str = "document.txt",
    max_chars: int = 1600,
    overlap: int = 160,
) -> list[dict[str, Any]]:
    """文档切片总入口 —— 按文件类型分发到五种策略之一。

    【调用链】追这个函数会进入：
        1. _validate_limits()      → 参数校验
        2. _normalize_text()       → 统一换行符
        3. 按后缀分发：
           .csv           → _csv_chunks()
           .md/.pdf/.docx → _markdown_blocks() → _pack_blocks()
           .json          → _json_blocks() → _pack_blocks()
           其他           → _plain_chunks()
    【被谁调用】
        services/rag/document_processing.py  ← 用户上传的文档（主路径）
        services/rag/indexing.py             ← Issue/PR/CI 日志
        chunk_text() / chunk_code_text()     ← 内部复用

    【数据流向】
        入：text（纯文本）、source_path（只用后缀）、max_chars、overlap
        出：chunk dict 列表，每个含 content / chunk_id / parent_id / child_index / section_path / page / 行号

    【默认参数怎么来的】
        max_chars=1600：约 800 汉字，是一般 embedding 模型的最佳区间
        overlap=160：约 10%，足够覆盖跨界的半句话，又不会太浪费
        实际调用时用的是 knowledge_base_configs 表里的 chunk_size / chunk_overlap

    【注意】这里没有 .py 分支
        代码不走这个函数，走 chunk_code_text()（按函数/类切，而不是按段落）。
    """
    _validate_limits(max_chars, overlap)

    # 空输入 → 空结果（不报错，因为空文件是合法输入）
    if not text:
        return []
    normalized = _normalize_text(text)
    if not normalized.strip():
        return []

    suffix = Path(source_path).suffix.lower()

    # ── CSV：自己一套逻辑（要处理表头）──
    if suffix == ".csv":
        return _csv_chunks(normalized, source_path, max_chars, overlap)

    # ── Markdown / JSON：先解析成 blocks，再统一打包 ──
    if suffix in MARKDOWN_SUFFIXES:
        blocks = _markdown_blocks(normalized)
        chunk_type = "markdown_section"
    elif suffix == ".json":
        blocks = _json_blocks(normalized, max_chars)
        chunk_type = "json_value"
    else:
        # ── 其他一切：滑动窗口 ──
        return _plain_chunks(
            normalized,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
        )

    chunks = _pack_blocks(
        blocks,
        source_path=source_path,
        max_chars=max_chars,
        overlap=overlap,
        chunk_type=chunk_type,
    )
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
    return chunks


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    """兼容旧接口：只返回文本，不要元数据。

    【调用链】chunk_document() → 本函数
    【被谁调用】chunk_pr_files()（PR diff 切片）、以及一些只关心文本的老代码

    【为什么保留】
        有些场景（如 diff 切片）只需要文本，不需要 parent_id 等元数据。
        但注意：用这个接口会丢掉父子块信息，能用 chunk_document 就别用它。
    """
    return [
        chunk["content"]
        for chunk in chunk_document(text, "document.txt", max_chars=max_chars, overlap=overlap)
    ]


# ════════════════════════════════════════════════════════════════
# 代码切片（另一条链路：不进 RAG，进代码图谱/工作区检索）
# ════════════════════════════════════════════════════════════════

def _line_windows(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
    overlap_lines: int = 6,
) -> list[tuple[str, int, int]]:
    """把一段行区间按字数切成多个窗口（代码切分的兜底方案）。

    【调用链】被 chunk_code_text() 调用
    【什么时候用】一个函数/类本身就超过 max_chars（比如 2000 行的巨型类）

    【为什么代码要单独处理】代码没有空行分段，必须按行累积到接近 max_chars。

    【数据流向】
        入：lines（全部代码行）、start_line / end_line（1-based，闭区间）
        出：(文本, 起始行, 结束行) 的列表

    【overlap_lines=6 的意义】
        按"行"重叠而不是按字符，因为代码的行是有意义的单位。
        6 行足够覆盖一个函数签名 + 几行上下文。
    """
    output: list[tuple[str, int, int]] = []
    cursor = max(0, start_line - 1)      # 转成 0-based
    stop = min(len(lines), end_line)

    while cursor < stop:
        # 单行超长（压缩过的 JS / 超长 SQL）→ 硬切
        if len(lines[cursor]) > max_chars:
            for offset in range(0, len(lines[cursor]), max_chars):
                output.append((lines[cursor][offset : offset + max_chars], cursor + 1, cursor + 1))
            cursor += 1
            continue

        # 按行累积
        size = 0
        end = cursor
        while end < stop:
            line_size = len(lines[end]) + (1 if end > cursor else 0)
            if end > cursor and size + line_size > max_chars:
                break
            size += line_size
            end += 1

        if end > cursor:
            content = "\n".join(lines[cursor:end]).strip("\n")
            if content:
                output.append((content, cursor + 1, end))

        if end >= stop:
            break
        # 回退 overlap_lines 行做重叠
        cursor = max(cursor + 1, end - overlap_lines)

    return output


def _python_regions(text: str, max_chars: int) -> list[tuple[int, int, str, str]]:
    """用 AST 把 Python 代码切成函数/类区域（这是最准的切法）。

    【调用链】被 chunk_code_text() 调用（suffix == ".py"）
    【下游】产出的 regions 交给 _line_windows() 兜底切分

    返回：(起始行, 结束行, 符号名, 符号类型)

    【类太大时按方法切】一个 2000 行的类整体成块检索效果很差，
        按方法切能让"找某个方法"的查询命中率大幅提升。

    【数据流向】
        入：Python 源码、max_chars
        出：region 列表，符号类型有 module / class / class_header / method / function

    【为什么用 AST 而不是正则】
        正则无法正确处理装饰器、多行签名、嵌套函数、注释里的假函数定义。
        AST 是 Python 官方解析器，100% 准确。
        代价：语法错误的文件解析不了 → 返回空列表，让上层退化成按行切。
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []       # 语法错误 → 让上层退化成按行切

    lines = text.splitlines()
    regions: list[tuple[int, int, str, str]] = []
    cursor = 1

    def node_start(node: ast.AST) -> int:
        """节点起始行（装饰器算在函数之前，要取最小值）。

        【为什么不能直接用 node.lineno】
            对带装饰器的函数，lineno 指向 `def` 那一行，
            但装饰器在它上面几行。切片时如果漏掉装饰器，
            `@app.get("/x")` 就和函数分家了，语义就断了。
        """
        decorators = getattr(node, "decorator_list", []) or []
        positions = [getattr(node, "lineno", 1), *(getattr(item, "lineno", 1) for item in decorators)]
        return min(positions)

    def append_node(node: ast.AST, parent: str | None = None) -> None:
        """把一个 AST 节点转成 region（类太大时递归拆成方法）。"""
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        name = str(getattr(node, "name", "module"))
        # 方法名用 "类名.方法名" 形式，便于检索
        symbol = f"{parent}.{name}" if parent else name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"

        # 类整体超长 → 拆成"类头 + 各方法"
        if isinstance(node, ast.ClassDef) and len("\n".join(lines[start - 1 : end])) > max_chars:
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                method_cursor = start
                first_method_start = node_start(methods[0])
                # 类头（class 声明 + 类属性）单独成一块
                if first_method_start > method_cursor:
                    regions.append((method_cursor, first_method_start - 1, symbol, "class_header"))
                # 每个方法一块（结束行 = 下一个方法的起始行 - 1）
                for index, method in enumerate(methods):
                    method_start = node_start(method)
                    next_start = node_start(methods[index + 1]) if index + 1 < len(methods) else end + 1
                    regions.append((method_start, next_start - 1, f"{symbol}.{method.name}", "method"))
                return

        regions.append((start, end, symbol, kind))

    # 顶层节点之间可能夹着模块级代码（import、常量）→ 用 cursor 补上
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in top_level:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        # 这个节点之前有"模块级代码" → 单独成一块
        if start > cursor:
            regions.append((cursor, start - 1, "module", "module"))
        append_node(node)
        cursor = max(cursor, end + 1)

    # 最后一个节点之后还有代码 → 补一块
    if cursor <= len(lines):
        regions.append((cursor, len(lines), "module", "module"))

    # 过滤非法区间（start > end 的不要）
    return [region for region in regions if region[0] <= region[1]]


def _generic_symbol(line: str, suffix: str) -> tuple[str, str] | None:
    """非 Python 语言用正则识别符号（没有 AST 可用时的近似方案）。

    【调用链】被 _generic_regions() 调用
    【支持语言】JS/TS、Go、Rust、Java/C#/C++/PHP/Ruby/Swift/Scala、SQL、Shell

    【局限】正则无法处理嵌套作用域，只能识别"顶层声明"。
        对检索来说够用：用户通常找的就是顶层函数/类。

    【返回值】(符号名, 符号类型)；不是声明行则返回 None

    注意：这里只识别"行首的声明"，缩进很深的嵌套函数会被漏掉 ——
    这是刻意取舍，因为嵌套函数通常不需要单独检索。
    """
    patterns: list[tuple[re.Pattern[str], str]] = []

    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        patterns = [
            # function / class / interface / type / enum 声明
            (
                re.compile(
                    r"^\s*(?:export\s+(?:default\s+)?)?(?:(?:async\s+)?function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
                ),
                "symbol",
            ),
            # const foo = () => {} / const foo = function() {}
            (
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
                ),
                "function",
            ),
        ]
    elif suffix == ".go":
        patterns = [(re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_]\w*)"), "symbol")]
    elif suffix == ".rs":
        patterns = [
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)"), "symbol")
        ]
    elif suffix in {".java", ".kt", ".kts", ".cs", ".cpp", ".c", ".h", ".hpp", ".php", ".rb", ".swift", ".scala"}:
        patterns = [
            (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed|data|open)\s+)*(?:class|interface|enum|struct|trait|record)\s+([A-Za-z_]\w*)"), "type")
        ]
    elif suffix == ".sql":
        patterns = [(re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TABLE|VIEW)\s+([\w.]+)", re.IGNORECASE), "sql_object")]
    elif suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        patterns = [(re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*(?:\(\))?\s*\{"), "function")]

    for pattern, kind in patterns:
        match = pattern.match(line)
        if match:
            return match.group(1), kind
    return None


def _generic_regions(text: str, suffix: str) -> list[tuple[int, int, str, str]]:
    """用正则找出所有符号起始行，相邻两个起点之间就是一个区域。

    【调用链】_generic_symbol() → 本函数
    【被谁调用】chunk_code_text()（非 Python 语言）

    【算法】只有"起点"信息，没有"终点"信息。
        所以用"下一个起点 - 1"当作当前符号的终点。
        这是近似方案：如果函数后面有空行或注释，会被算进上一个函数。

    【第一段之前的内容】如果第一个符号不在第 1 行，
        第 1 行到它之前就是模块级代码（import 等），单独成一块。
    """
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []

    for line_number, line in enumerate(lines, start=1):
        symbol = _generic_symbol(line, suffix)
        if symbol:
            starts.append((line_number, symbol[0], symbol[1]))

    # 一个符号都没识别到 → 整份文件当模块
    if not starts:
        return [(1, len(lines), "module", "module")] if lines else []

    regions: list[tuple[int, int, str, str]] = []

    # 第一个符号之前的模块级代码
    if starts[0][0] > 1:
        regions.append((1, starts[0][0] - 1, "module", "module"))

    # 每个符号：从自己的起点到下一个符号的前一行
    for index, (start, name, kind) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        regions.append((start, end, name, kind))
    return regions


def _language_for_path(path: str) -> str:
    """从路径推断语言名（就用后缀，如 "py"、"ts"）。

    【调用链】被 chunk_code_text() 调用

    为什么不用更智能的映射（如 .py → "python"）：
        这里只是给 header 展示用，后缀已经足够表意，不必维护一张映射表。
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or Path(path).name.lower()


def chunk_code_text(text: str, path: str, max_chars: int = 2200) -> list[dict[str, Any]]:
    """代码切片：按语言符号切，带 header 前缀。

    【调用链】追这个函数会进入：
        1. chunk_document()      → 文档类/数据类文件走这条（复用文档切片）
        2. _python_regions()     → .py 用 AST
        3. _generic_regions()    → 其他语言用正则
        4. _line_windows()       → 单个符号还超长时兜底
    【被谁调用】services/code_analysis.py（建代码图谱时）

    【数据流向】
        入：源码文本、文件路径、max_chars（默认 2200，比文档大因为代码密度低）
        出：chunk dict 列表，每个含 symbol / symbol_kind / language / header

    【为什么 header 很重要】一个函数切片单独看，模型不知道它来自哪个文件。
        header 里写明"File / Language / Symbol / Lines"，
        检索命中后模型能立刻建立位置感。

    【注意：这条链路不进 RAG】
        源码走"代码图谱 + 实时文件读取"，不进向量库。
        原因见 source_policy.py 末尾的论证。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []

    suffix = Path(path).suffix.lower()
    language = _language_for_path(path)

    # ── 文档类/数据类文件 → 复用文档切片，再补代码风格的 header ──
    # 为什么这样处理：README.md / config.json 本质是文档，
    # 按标题/键切比按"符号"切更合理（它们没有函数）
    if suffix in MARKDOWN_SUFFIXES | STRUCTURED_DATA_SUFFIXES | {".txt"}:
        document_chunks = chunk_document(normalized, path, max_chars=max_chars, overlap=max(0, min(160, max_chars // 10)))
        output: list[dict[str, Any]] = []
        for chunk in document_chunks:
            symbol = chunk.get("section_title") or "document"
            header = f"File: {path}\nSection: {symbol}\nLines: {chunk['start_line']}-{chunk['end_line']}"
            output.append({**chunk, "symbol": symbol, "symbol_kind": chunk["chunk_type"], "language": language, "header": header})
        return output

    lines = normalized.splitlines()
    # Python 用 AST（准），其他语言用正则（近似）
    regions = _python_regions(normalized, max_chars) if suffix == ".py" else _generic_regions(normalized, suffix)
    # 解析失败 → 整份文件当一个模块
    if not regions:
        regions = [(1, len(lines), "module", "module")]

    chunks: list[dict[str, Any]] = []
    symbol_occurrences: dict[str, int] = {}

    for start_line, end_line, symbol, symbol_kind in regions:
        # 同名符号可能出现多次（重载/重定义），用序号区分
        symbol_occurrence = symbol_occurrences.get(symbol, 0)
        symbol_occurrences[symbol] = symbol_occurrence + 1
        # 同一符号的所有分片共享 parent_id（便于拼回完整函数）
        parent_id = _stable_id("code-parent", path, symbol, symbol_occurrence)

        # 一个符号可能还是超过 max_chars → 再按行窗口切
        for part_index, (content, part_start, part_end) in enumerate(
            _line_windows(lines, start_line, end_line, max_chars=max_chars)
        ):
            if not content.strip():
                continue
            chunk_id = _stable_id("code", parent_id, part_index, content)
            # 只有被切开的分片才加 "(part 2)" 后缀
            part_suffix = f" (part {part_index + 1})" if part_index else ""
            chunks.append(
                {
                    "content": content,
                    "start_line": part_start,
                    "end_line": part_end,
                    "symbol": symbol,
                    "symbol_kind": symbol_kind,
                    "language": language,
                    # module 块单独标类型（它不是函数，是 import/常量区）
                    "chunk_type": "code_symbol" if symbol != "module" else "code_module",
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "child_index": part_index,
                    "symbol_occurrence": symbol_occurrence,
                    "header": (
                        f"File: {path}\nLanguage: {language}\nSymbol: {symbol}{part_suffix}\n"
                        f"Lines: {part_start}-{part_end}"
                    ),
                }
            )
    return chunks


def chunk_pr_files(files: list[dict[str, Any]], max_chars: int = 1800) -> list[dict[str, Any]]:
    """把 PR 的 diff 按文件切片（overlap=0，因为 diff 不需要上下文重叠）。

    【调用链】chunk_text() → 本函数
    【被谁调用】services/rag/indexing.py（把 PR 改动写进 RAG）

    【为什么 overlap=0】diff 的行号是有意义的，重叠会让"第几行改动"产生歧义。

    【数据流向】
        入：files（GitHub PR files 列表，每项含 filename / patch）
        出：chunk dict 列表，含 filename / chunk_index / chunk_id / content

    【为什么按文件切而不是整个 PR 一起切】
        一个 PR 可能改 50 个文件。按文件切能保证"一块只讲一个文件"，
        检索命中后可以直接定位到文件。
    """
    chunks: list[dict[str, Any]] = []

    for file in files:
        patch = file.get("patch") or ""
        filename = str(file.get("filename") or "patch.diff")

        for index, content in enumerate(chunk_text(patch, max_chars=max_chars, overlap=0)):
            chunks.append(
                {
                    "filename": filename,
                    "chunk_index": index,
                    "chunk_id": _stable_id("patch", filename, content),
                    "content": content,
                }
            )
    return chunks
