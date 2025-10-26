"""
RAG Framework - A flexible framework for building RAG applications.

This package provides a modular and extensible framework for building
Retrieval-Augmented Generation (RAG) applications with support for
multiple vector store backends.
"""

from fetchcraft.index.vector_index import VectorIndex
from .vector_store.base import VectorStore
from .vector_store.qdrant_store import QdrantVectorStore, QdrantConfig
from .node import Node, Chunk, SymNode, NodeWithScore
from .parser import DocumentParser, TextFileDocumentParser
from .chunking import ChunkingStrategy, CharacterChunkingStrategy, HierarchicalChunkingStrategy
from .embeddings import Embeddings, OpenAIEmbeddings
from .retriever import Retriever, VectorIndexRetriever
from .agents import BaseAgent, ReActAgent, RetrieverTool

__all__ = [
    'VectorIndex',
    'VectorStore',
    'QdrantVectorStore',
    'QdrantConfig',
    'Node',
    'Chunk',
    'SymNode',
    'NodeWithScore',
    'DocumentParser',
    'TextFileDocumentParser',
    'ChunkingStrategy',
    'CharacterChunkingStrategy',
    'HierarchicalChunkingStrategy',
    'Embeddings',
    'OpenAIEmbeddings',
    'Retriever',
    'VectorIndexRetriever',
    'BaseAgent',
    'ReActAgent',
    'RetrieverTool',
]
