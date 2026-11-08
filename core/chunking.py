"""分块策略：固定长度 vs 语义分块。

设计要点：
- 分块是检索质量的第一决定因素，且「没有银弹」——最佳策略由评测集决定。
- 两个策略都实现，让你能跑出对照实验（docs/02 里那张表）。

当前实现不引入外部 NLP 依赖，用轻量启发式近似「语义分块」：
按空行和标题行（Markdown 的 # 开头）切，能保留段落/章节的语义边界。
"""

from __future__ import annotations

import re
from typing import Iterable

from core.config import settings

HEADING_RE = re.compile(r"^#{1,6}\s+")


def fixed_chunks(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """固定长度切块，近似按 token（1 字符 ≈ 0.5 token 的粗略换算），带重叠。"""
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    # 简单按字符切，用 size*2 近似 token 数（中文 1 字 ≈ 1 token，英文 1 词 ≈ 1 token，
    # 这里取个折中：按字符数 size 近似，文档里注明这是近似，生产应换成真实 tokenizer）
    step = max(1, size - overlap)
    chunks: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
        if i + size >= len(text):
            break
    return chunks


def semantic_chunks(text: str) -> list[str]:
    """按段落/标题边界切：比固定长度更尊重语义，但长度不可控。

    对短 FAQ 反而可能切太碎（把问题和答案分开），这是要在评测里验证的失败点。
    """
    blocks = re.split(r"\n\s*\n", text)  # 空行分段
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        is_heading = bool(HEADING_RE.match(b))
        # 遇到标题，或累积长度超限，就收口一段
        if (is_heading and buf) or buf_len > settings.chunk_size * 2:
            chunks.append("\n".join(buf))
            buf = []
            buf_len = 0
        buf.append(b)
        buf_len += len(b)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def chunk_text(text: str, strategy: str = "fixed") -> list[str]:
    if strategy == "semantic":
        return semantic_chunks(text)
    return fixed_chunks(text)


def chunk_documents(docs: Iterable[tuple[str, str]], strategy: str = "fixed") -> list[dict]:
    """把 (doc_id, text) 流变成 (doc_id, chunk_index, content) 列表。"""
    out: list[dict] = []
    for doc_id, text in docs:
        for i, c in enumerate(chunk_text(text, strategy)):
            out.append({"doc_id": doc_id, "chunk_index": i, "content": c})
    return out
