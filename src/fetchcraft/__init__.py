"""
RAG Framework - A flexible framework for building RAG applications.

This package provides a modular and extensible framework for building
Retrieval-Augmented Generation (RAG) applications with support for
multiple vector store backends.
"""

from fetchcraft.index.vector_index import VectorIndex
from .vector_store.base import VectorStore
from .vector_store.qdrant_store import QdrantVectorStore, QdrantConfig
from .vector_store.chroma_store import ChromaVectorStore, ChromaConfig
from .document_store.base import DocumentStore
from .document_store.mongodb_store import MongoDBDocumentStore, MongoDBConfig
from .node import Node, DocumentNode, Chunk, SymNode, NodeWithScore
from .parser import DocumentParser, TextFileDocumentParser
from .chunking import ChunkingStrategy, CharacterChunkingStrategy, HierarchicalChunkingStrategy
from .embeddings import Embeddings, OpenAIEmbeddings
from .retriever import Retriever, VectorIndexRetriever
from .agents import BaseAgent, ReActAgent, RetrieverTool, FileSearchTool, FileSearchResult

__all__ = [
    'VectorIndex',
    'VectorStore',
    'QdrantVectorStore',
    'QdrantConfig',
    'ChromaVectorStore',
    'ChromaConfig',
    'DocumentStore',
    'MongoDBDocumentStore',
    'MongoDBConfig',
    'Node',
    'DocumentNode',
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
    'FileSearchTool',
    'FileSearchResult',
]
