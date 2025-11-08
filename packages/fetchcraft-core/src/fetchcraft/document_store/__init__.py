"""
Document store implementations for storing full documents.
"""

from .base import DocumentStore
from .mongodb_store import MongoDBDocumentStore, MongoDBConfig

__all__ = [
    'DocumentStore',
    'MongoDBDocumentStore',
    'MongoDBConfig',
]
