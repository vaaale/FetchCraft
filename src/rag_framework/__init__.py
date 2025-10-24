"""
RAG Framework - A flexible framework for building RAG applications.

This package provides a modular and extensible framework for building
Retrieval-Augmented Generation (RAG) applications with support for
multiple vector store backends.
"""

from .vector_index import VectorIndex
from .vector_store.base import VectorStore
from .vector_store.qdrant_store import QdrantVectorStore, QdrantConfig
from .node import Node, Chunk
from .parser import DocumentParser, TextFileDocumentParser
from .embeddings import Embeddings, OpenAIEmbeddings

__all__ = [
    'VectorIndex',
    'VectorStore',
    'QdrantVectorStore',
    'QdrantConfig',
    'Node',
    'Chunk',
    'DocumentParser',
    'TextFileDocumentParser',
    'Embeddings',
    'OpenAIEmbeddings',
]
