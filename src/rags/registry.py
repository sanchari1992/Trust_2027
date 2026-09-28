"""Turns a RAG name into a ready-to-use RAG object."""

RAG_NAMES = ["lightrag", "raptor", "medrag"]


def get_rag(name, cfg):
    # Imported inside each branch so a missing library for one RAG
    # never breaks the others.
    if name == "lightrag":
        from src.rags.lightrag_rag import LightRAGRag

        return LightRAGRag(cfg)
    if name == "raptor":
        from src.rags.raptor_rag import RaptorRag

        return RaptorRag(cfg)
    if name == "medrag":
        from src.rags.medrag_rag import MedRagRag

        return MedRagRag(cfg)
    raise ValueError(f"Unknown RAG '{name}'. Choose from: {RAG_NAMES}")
