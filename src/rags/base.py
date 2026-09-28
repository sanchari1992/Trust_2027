"""The shared shape every RAG in this project follows."""


class BaseRAG:
    name = "base"

    def load_corpus(self):
        """Return the documents this RAG indexes: [{"id": "...", "text": "..."}].
        The benchmark generators (RAGAS, ARES, DataMorgana) read this."""
        raise NotImplementedError

    def build_index(self):
        """One-time step: index the corpus and save it under indexes/."""
        raise NotImplementedError

    def answer(self, question):
        """Answer one question. Return a dict with:
          "answer":          str
          "contexts":        [str, ...]  EVERYTHING the generating model was shown, in
                                         pieces. Faithfulness is judged against this
                                         by default (this is what RAGAS/ARES assume).
          "source_contexts": [str, ...]  only the ORIGINAL source text among those pieces
                                         (may be empty, e.g. when RAPTOR retrieves only
                                         summaries). Use it for a stricter check.
        Some RAGs add extra fields (summaries, graph_entities, ...)."""
        raise NotImplementedError

    def close(self):
        pass
