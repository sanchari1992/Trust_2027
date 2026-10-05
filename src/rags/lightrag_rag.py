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

    # ---- the functions every RAG provides --------------------------------
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

    async def _answer_one(self, rag, question):
        param = QueryParam(mode=self.query_mode, stream=False, enable_rerank=False)
        result = await rag.aquery_llm(question, param=param)
        if result.get("status") != "success":
            raise RuntimeError(f"LightRAG query failed: {result.get('message')}")
        data = result.get("data") or {}
        chunks = [c["content"] for c in data.get("chunks", [])]
        entities = data.get("entities", [])
        relations = data.get("relationships", [])
        # LightRAG also shows the model LLM-written descriptions of graph entities
        # and relations. They are part of `contexts` (what the model saw) but not
        # of `source_contexts` (original text only).
        graph_text = [
            f"{e.get('entity_name')} ({e.get('entity_type')}): {e.get('description')}"
            for e in entities
        ] + [
            f"{r.get('src_id')} -> {r.get('tgt_id')}: {r.get('description')}"
            for r in relations
        ]
        return {
            "answer": result["llm_response"]["content"],
            "contexts": chunks + graph_text,
            "source_contexts": chunks,
            "graph_entities": entities,
            "graph_relationships": relations,
        }

    def answer(self, question):
        async def _answer():
            rag = await self._get_rag()
            return await self._answer_one(rag, question)

        return self._run(_answer())

    def answer_batch(self, questions, concurrency=4):
        """Many questions at once, inside LightRAG's own event loop (its wrapper
        is not safe to call from several threads)."""
        if not questions:
            return []

        async def _all():
            rag = await self._get_rag()
            gate = asyncio.Semaphore(max(1, concurrency))

            async def one(q):
                async with gate:
                    try:
                        return await self._answer_one(rag, q)
                    except Exception as e:
                        return {"error": f"{type(e).__name__}: {e}"[:300]}

            return await asyncio.gather(*[one(q) for q in questions])

        return self._run(_all())

    def close(self):
        if self._rag is not None:
            self._run(self._rag.finalize_storages())
            self._rag = None
