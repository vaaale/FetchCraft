from abc import ABC, abstractmethod
from typing import List, Union

from pydantic import BaseModel, ConfigDict


class Embeddings(BaseModel, ABC):
    """
    Abstract base class for embedding models.
    
    This class defines the interface that all embedding implementations must follow.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous version of embed_documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            raise RuntimeError(
                "Cannot use sync method when async loop is already running. "
                "Use embed_documents() instead."
            )
        
        return loop.run_until_complete(self.embed_documents(texts))
    
    def embed_query_sync(self, text: str) -> List[float]:
        """
        Synchronous version of embed_query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            raise RuntimeError(
                "Cannot use sync method when async loop is already running. "
                "Use embed_query() instead."
            )
        
        return loop.run_until_complete(self.embed_query(text))
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        pass
