"""可插拔嵌入层。

设计要点：
1. mock provider 用确定性哈希生成向量，保证离线、无网络、无模型下载时也能跑通全链路。
   —— 这让你「先跑通再换真模型」成为可能，而不是一上来就被模型下载卡住。
2. 嵌入是 RAG 里「最贵但最不显眼」的一环：检索质量、缓存、成本都挂在它上面。
   所以 provider 切换必须是一行配置，而不是散落各处的 if。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from core.config import settings


def _hash_embed(text: str, dim: int) -> list[float]:
    """确定性伪嵌入：把文本散列成 dim 维单位向量。

    不是真语义嵌入，但满足两个关键性质：
    - 相同/相似输入 → 相同向量（可测缓存、可测幂等）
    - 单位向量 → 余弦相似度计算有意义
    """
    v = [0.0] * dim
    # 用多个不同 salt 散列填充不同维度，避免所有维同源导致退化
    for i in range(dim):
        h = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
        v[i] = ((h[0] * 256 + h[1]) / 65535.0) * 2 - 1
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class BaseEmbedder:
    def embed(self, text: str) -> list[float]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        return settings.embedding_dim


class MockEmbedder(BaseEmbedder):
    def embed(self, text: str) -> list[float]:
        return _hash_embed(text, settings.embedding_dim)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 兼容嵌入接口（也覆盖 vLLM / 本地 embedding server）。"""

    def __init__(self) -> None:
        import openai  # 延迟导入，避免 mock 模式下必须装

        self.client = openai.OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )
        self.model = settings.embedding_model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """本地 sentence-transformers（离线、隐私敏感场景）。"""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(settings.embedding_model)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


def get_embedder() -> BaseEmbedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder()
    if settings.embedding_provider == "local":
        return SentenceTransformerEmbedder()
    return MockEmbedder()
