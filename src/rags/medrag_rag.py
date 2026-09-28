"""MedRAG (Xiong et al., ACL Findings 2024) wrapped in our shared interface.

IMPORTANT (disclose in the paper): this is a re-implementation of MedRAG's
pipeline, not a call into its released code. Same corpus (MedRAG Textbooks),
same retriever (MedCPT, dense inner-product search), same default k=32, and a
prompt adapted from MedRAG's. We did this because MedRAG's released code pins
old library versions (and its custom pooling hook silently stops working with
current sentence-transformers), which would break the other RAGs.
"""
import json
from pathlib import Path

import numpy as np

from src.config import corpus_dir, index_dir
from src.llm import get_chat_func
from src.rags.base import BaseRAG

SYSTEM_PROMPT = (
    "You are a helpful medical expert, and your task is to answer a medical "
    "question using the relevant documents. Give a concise, definite answer "
    "based on the documents. Your responses will be used for research purposes only."
)


class MedCPTEncoder:
    """MedCPT loaded with plain `transformers`, exactly as its model card shows:
    the embedding is the [CLS] token; articles are (title, text) pairs."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.AutoModel = AutoModel
        self.AutoTokenizer = AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = {}

    def _load(self, name):
        if name not in self._loaded:
            tok = self.AutoTokenizer.from_pretrained(name)
            model = self.AutoModel.from_pretrained(name).to(self.device).eval()
            self._loaded[name] = (tok, model)
        return self._loaded[name]

    def _encode(self, name, inputs, max_length):
        tok, model = self._load(name)
        with self.torch.no_grad():
            enc = tok(
                inputs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(self.device)
            out = model(**enc).last_hidden_state[:, 0, :]
        return out.cpu().numpy().astype(np.float32)

    def encode_articles(self, pairs):
        return self._encode("ncbi/MedCPT-Article-Encoder", pairs, 512)

    def encode_queries(self, texts):
        return self._encode("ncbi/MedCPT-Query-Encoder", texts, 64)


class MedRagRag(BaseRAG):
    name = "medrag"

    def __init__(self, cfg):
        self.cfg = cfg
        self.mc = cfg["medrag"]
        self.corpus_dir = corpus_dir(cfg, "medrag")
        self.index_dir = index_dir(cfg, "medrag")
        self._snippets = None
        self._index = None
        self._encoder = None
        self._chat = None

    def _chunk_files(self):
        return sorted((Path(self.corpus_dir) / "chunk").glob("*.jsonl"))

    def _get_snippets(self):
        if self._snippets is None:
            rows = []
            for path in self._chunk_files():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        row["_source"] = path.stem  # which textbook this snippet is from
                        rows.append(row)
            self._snippets = rows
        return self._snippets

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = MedCPTEncoder()
        return self._encoder

    # ---- the three functions every RAG provides -------------------------
    def load_corpus(self):
        return [
            {"id": s["id"], "text": s["content"], "title": s["title"], "group": s["_source"]}
            for s in self._get_snippets()
        ]

    def build_index(self):
        import faiss

        index_path = self.index_dir / "faiss.index"
        if index_path.exists():
            print("MedCPT index already built; reusing it.")
            return
        files = self._chunk_files()
        if not files:
            raise RuntimeError(
                f"No chunk files in {self.corpus_dir}/chunk. "
                "Run: python scripts/prepare_corpora.py medrag"
            )
        emb_dir = self.index_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        encoder = self._get_encoder()
        batch = self.mc["batch_size"]

        # One saved file per textbook, so an interrupted run can resume.
        all_vectors = []
        for path in files:
            emb_path = emb_dir / (path.stem + ".npy")
            if not emb_path.exists():
                rows = [
                    json.loads(l)
                    for l in path.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
                print(f"Embedding {path.stem}: {len(rows)} snippets...")
                parts = []
                for i in range(0, len(rows), batch):
                    chunk = rows[i : i + batch]
                    parts.append(
                        encoder.encode_articles(
                            [[r["title"], r["content"]] for r in chunk]
                        )
                    )
                np.save(emb_path, np.concatenate(parts))
            all_vectors.append(np.load(emb_path))
        vectors = np.concatenate(all_vectors)
        assert len(vectors) == len(self._get_snippets()), "embedding/snippet mismatch"

        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product, as in MedRAG
        index.add(vectors)
        faiss.write_index(index, str(index_path))

    def _get_index(self):
        if self._index is None:
            import faiss

            self._index = faiss.read_index(str(self.index_dir / "faiss.index"))
        return self._index

    def answer(self, question):
        k = self.mc["top_k"]
        query_vec = self._get_encoder().encode_queries([question])
        scores, ids = self._get_index().search(query_vec, k)
        snippets = self._get_snippets()
        hits = [(snippets[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

        context = "\n".join(
            f"Document [{n}] (Title: {s['title']}) {s['content']}"
            for n, (s, _) in enumerate(hits)
        )
        if self._chat is None:
            self._chat = get_chat_func(self.cfg)
        answer = self._chat(
            SYSTEM_PROMPT,
            f"Here are the relevant documents:\n{context}\n\n"
            f"Here is the question:\n{question}\n\nAnswer:",
        )
        return {
            "answer": answer,
            "contexts": [s["content"] for s, _ in hits],
            "source_contexts": [s["content"] for s, _ in hits],
            "context_meta": [
                {"id": s["id"], "title": s["title"], "score": sc} for s, sc in hits
            ],
        }
