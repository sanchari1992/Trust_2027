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
        """Answer one question. Return:
        {"answer": str, "contexts": [str, ...]}
        'contexts' must be the retrieved SOURCE TEXT, because faithfulness
        scoring compares the answer against it."""
        raise NotImplementedError

    def close(self):
        pass
