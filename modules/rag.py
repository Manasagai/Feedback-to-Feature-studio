"""
modules/rag.py
==============
Phase 5 will implement this module.

Responsibilities:
-----------------
- Load the product knowledge base from data/knowledge_base.txt.
- Split the knowledge base into manageable text chunks.
- Build a vector index over the chunks (using FAISS or ChromaDB).
- Given a cluster theme or query, retrieve the most relevant knowledge chunks.
- Return retrieved context as a string to be injected into the LLM prompt.

Planned dependencies (to be added in Phase 5):
    faiss-cpu  OR  chromadb
    sentence-transformers (shared with embeddings module)
"""

from typing import List


def load_knowledge_base(filepath: str) -> str:
    """
    Load the full text of the product knowledge base.

    Args:
        filepath (str): Path to knowledge_base.txt.

    Returns:
        str: Full knowledge base text.
    """
    # TODO (Phase 5): Read and return the contents of the knowledge base file.
    raise NotImplementedError("load_knowledge_base() will be implemented in Phase 5.")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split the knowledge base text into overlapping chunks for indexing.

    Args:
        text (str): Full knowledge base string.
        chunk_size (int): Approximate number of characters per chunk.
        overlap (int): Character overlap between adjacent chunks.

    Returns:
        List[str]: List of text chunks.
    """
    # TODO (Phase 5): Implement sliding-window chunking.
    raise NotImplementedError("chunk_text() will be implemented in Phase 5.")


def build_index(chunks: List[str], embedding_model):
    """
    Build a vector index over knowledge base chunks.

    Args:
        chunks (List[str]): Text chunks from the knowledge base.
        embedding_model: Loaded sentence transformer model.

    Returns:
        A searchable index object (FAISS index or ChromaDB collection).
    """
    # TODO (Phase 5): Encode chunks and build FAISS/ChromaDB index.
    raise NotImplementedError("build_index() will be implemented in Phase 5.")


def retrieve_context(query: str, index, chunks: List[str], top_k: int = 3) -> str:
    """
    Retrieve the top-k most relevant knowledge base chunks for a query.

    Args:
        query (str): A theme description or cluster summary.
        index: The vector index built from the knowledge base.
        chunks (List[str]): Original text chunks corresponding to index entries.
        top_k (int): Number of top chunks to retrieve.

    Returns:
        str: Concatenated relevant context string for LLM injection.
    """
    # TODO (Phase 5): Embed query, search index, return matching chunks.
    raise NotImplementedError("retrieve_context() will be implemented in Phase 5.")
