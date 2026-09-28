"""LightRAG (Guo et al., EMNLP 2025) wrapped in our shared interface."""
import asyncio
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

from src.config import corpus_dir, index_dir
from src.embeddings import get_embedding_func
from src.llm import get_llm_func
from src.rags.base import BaseRAG


class LightRAGRag(BaseRAG):
    name = "lightrag"

    def __init__(self, cfg):
        self.cfg = cfg
        lc = cfg["lightrag"]
        self.working_dir = index_dir(cfg, "lightrag")
        self.corpus_dir = corpus_dir(cfg, "lightrag")
        self.query_mode = lc["query_mode"]
        self.chunk_size = lc["chunk_token_size"]
        self.chunk_overlap = lc["chunk_overlap_token_size"]
        # LightRAG is asynchronous. One loop is kept for the whole session so
        # that our normal (non-async) code can call it.
        self._loop = asyncio.new_event_loop()
        self._rag = None

    def _run(self, coroutine):
        return self._loop.run_until_complete(coroutine)

    async def _get_rag(self):
        if self._rag is None:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=get_llm_func(self.cfg),
                embedding_func=get_embedding_func(self.cfg["embedding"]["model"]),
                chunk_token_size=self.chunk_size,
                chunk_overlap_token_size=self.chunk_overlap,
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()
            self._rag = rag
        return self._rag

    # ---- the three functions every RAG provides -------------------------
    def load_corpus(self):
        docs = []
        for path in sorted(Path(self.corpus_dir).glob("*.txt")):
            docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
        return docs

    def build_index(self):
        docs = self.load_corpus()
        if not docs:
            raise RuntimeError(f"No .txt files found in {self.corpus_dir}")

        async def _build():
            rag = await self._get_rag()
            # LightRAG skips documents it has already indexed, so this is safe to re-run.
            await rag.ainsert(
                [d["text"] for d in docs],
                ids=[d["id"] for d in docs],
                file_paths=[d["id"] + ".txt" for d in docs],
            )

        self._run(_build())

    def answer(self, question):
        async def _answer():
            rag = await self._get_rag()
            param = QueryParam(
                mode=self.query_mode, stream=False, enable_rerank=False
            )
            result = await rag.aquery_llm(question, param=param)
            if result.get("status") != "success":
                raise RuntimeError(f"LightRAG query failed: {result.get('message')}")
            data = result.get("data") or {}
            return {
                "answer": result["llm_response"]["content"],
                # Original source text that was retrieved.
                "contexts": [c["content"] for c in data.get("chunks", [])],
                # Graph elements (LLM-written descriptions), kept separate on purpose.
                "graph_entities": data.get("entities", []),
                "graph_relationships": data.get("relationships", []),
            }

        return self._run(_answer())

    def close(self):
        if self._rag is not None:
            self._run(self._rag.finalize_storages())
            self._rag = None
