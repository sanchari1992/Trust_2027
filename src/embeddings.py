"""Local embeddings (DeepSeek has no embedding API)."""
import asyncio

import numpy as np
from lightrag.utils import EmbeddingFunc


def get_embedding_func(model_name):
    """Async version, used by LightRAG."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    max_tokens = model.max_seq_length

    async def embed(texts, **kwargs):
        vectors = await asyncio.to_thread(
            model.encode, texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=dim, max_token_size=max_tokens, func=embed, model_name=model_name
    )


class SyncEmbedder:
    """Plain (non-async) version, used by RAPTOR. encode(list_of_texts) -> array."""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


def get_sync_embedder(model_name):
    return SyncEmbedder(model_name)
