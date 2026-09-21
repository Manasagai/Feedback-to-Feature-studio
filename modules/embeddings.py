"""Local sentence-transformer embeddings for the small feedback dataset."""

from typing import List
import numpy as np
import streamlit as st


@st.cache_resource(show_spinner="Loading the local embedding model...")
def _load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache a local sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a pre-trained sentence transformer model.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """
    return _load_embedding_model(model_name)


def generate_embeddings(texts: List[str], model=None) -> np.ndarray:
    """
    Generate semantic embeddings for a list of feedback strings.

    Args:
        texts (List[str]): List of cleaned feedback texts.
        model: A loaded sentence transformer model.

    Returns:
        np.ndarray: 2D array of shape (num_texts, embedding_dim).
    """
    if model is None:
        model = load_embedding_model()
    return np.asarray(
        model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
