"""Embedding models for the RAG framework."""

from .base import Embeddings
from .openai import OpenAIEmbeddings

__all__ = [
    'Embeddings',
    'OpenAIEmbeddings',
]
