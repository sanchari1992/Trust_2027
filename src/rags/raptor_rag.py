"""RAPTOR (Sarthi et al., ICLR 2024) wrapped in our shared interface.

We use the authors' own code (cloned into third_party/raptor) and only plug in
our DeepSeek chat model and our local embedding model.

Design choice to disclose in the paper: RAPTOR's paper builds one tree per
document. Here ALL documents of the corpus go into ONE tree, so a question can
be answered from any document without being told which one.
"""
import pickle
import sys
from pathlib import Path

from src.config import ROOT, corpus_dir, index_dir
from src.embeddings import get_sync_embedder
from src.llm import get_chat_func
from src.rags.base import BaseRAG


class RaptorRag(BaseRAG):
    name = "raptor"

    def __init__(self, cfg):
        self.cfg = cfg
        self.rc = cfg["raptor"]
        self.corpus_dir = corpus_dir(cfg, "raptor")
        self.index_dir = index_dir(cfg, "raptor")
        self.tree_path = self.index_dir / "tree.pkl"
        self._raptor = None
        self._ra = None

    # ---- setup ----------------------------------------------------------
    def _import_raptor(self):
        if self._raptor is None:
            repo = ROOT / self.rc["repo_dir"]
            if not repo.exists():
                raise RuntimeError(
                    f"RAPTOR code not found at {repo}. "
                    "Run: python scripts/prepare_corpora.py raptor"
                )
            sys.path.insert(0, str(repo))
            import raptor

            self._raptor = raptor
        return self._raptor

    def _make_config(self):
        raptor = self._import_raptor()
        chat = get_chat_func(self.cfg)
        embedder = get_sync_embedder(self.cfg["embedding"]["model"])

        # The three plug-in parts RAPTOR asks for. Prompts follow the ones in
        # RAPTOR's own GPT-3.5 classes.
        class Summarizer(raptor.BaseSummarizationModel):
            def summarize(self, context, max_tokens=150):
                return chat(
                    "You are a helpful assistant.",
                    "Write a summary of the following, including as many key "
                    f"details as possible: {context}:",
                    max_tokens=max_tokens,
                )

        class QA(raptor.BaseQAModel):
            def answer_question(self, context, question):
                return chat(
                    "You are Question Answering Portal",
                    f"Given Context: {context} Give the best full answer to "
                    f"question {question}",
                )

        class Embedder(raptor.BaseEmbeddingModel):
            def create_embedding(self, text):
                return embedder.encode([text.replace("\n", " ")])[0]

        rc = self.rc
        return raptor.RetrievalAugmentationConfig(
            summarization_model=Summarizer(),
            qa_model=QA(),
            embedding_model=Embedder(),
            tb_max_tokens=rc["chunk_tokens"],
            tb_num_layers=rc["num_layers"],
            tb_summarization_length=rc["summary_tokens"],
            tr_top_k=rc["top_k"],
        )

    def _get_ra(self):
        if self._ra is None:
            raptor = self._import_raptor()
            config = self._make_config()
            if self.tree_path.exists():
                self._ra = raptor.RetrievalAugmentation(
                    config=config, tree=str(self.tree_path)
                )
            else:
                self._ra = raptor.RetrievalAugmentation(config=config)
        return self._ra

    # ---- the three functions every RAG provides -------------------------
    def load_corpus(self):
        docs = []
        for path in sorted(Path(self.corpus_dir).glob("*.txt")):
            docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
        return docs

    def build_index(self):
        if self.tree_path.exists():
            print("RAPTOR tree already built; reusing it.")
            return
        docs = self.load_corpus()
        if not docs:
            raise RuntimeError(f"No .txt files found in {self.corpus_dir}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        ra = self._get_ra()
        ra.add_documents("\n\n".join(d["text"] for d in docs))
        ra.save(str(self.tree_path))

    def answer(self, question):
        ra = self._get_ra()
        rc = self.rc
        answer, layer_info = ra.answer_question(
            question=question,
            top_k=rc["top_k"],
            max_tokens=rc["max_context_tokens"],
            return_layer_information=True,
        )
        nodes = ra.tree.all_nodes
        # Layer 0 = original text chunks. Higher layers = LLM-written summaries.
        # We keep them apart so faithfulness can be judged against real source text.
        contexts, summaries = [], []
        for info in layer_info:
            text = nodes[info["node_index"]].text
            (contexts if info["layer_number"] == 0 else summaries).append(text)
        return {
            "answer": answer,
            "contexts": contexts,
            "summaries": summaries,
            "layers": [i["layer_number"] for i in layer_info],
        }
