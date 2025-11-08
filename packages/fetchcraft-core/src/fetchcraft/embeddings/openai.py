import asyncio
import threading
from typing import List, Optional
import os

from openai import AsyncOpenAI, OpenAI
from pydantic import ConfigDict

from .base import Embeddings


class OpenAIEmbeddings(Embeddings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "text-embedding-3-small",
    timeout: float = 60.0
    _dimension: Optional[int] = None
    aclient: AsyncOpenAI | None = None
    client: OpenAI | None = None
    batch_size: int = 10

    """
    OpenAI embeddings implementation.
    
    Supports both OpenAI API and compatible endpoints (e.g., Azure OpenAI, local models).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        timeout: float = 60.0
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            base_url: Base URL for API endpoint. Defaults to OpenAI's API
            model: Model name to use (e.g., "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002")
            dimensions: Optional dimension reduction (only supported by some models)
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either via api_key parameter "
                "or OPENAI_API_KEY environment variable"
            )
        
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        
        # Initialize the OpenAI async client
        self.aclient = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _determine_dimension(self) -> int:
        """
        Determine the embedding dimension by making a test API call.
        
        Returns:
            The dimension of the embedding vectors
        """
        # Make a test call with a simple string
        with threading.Semaphore():
            response = self.client.embeddings.create(
                input=["My Index is the best RAG Framework"],
                model=self.model,
            )
            dimension = len(response.data[0].embedding)

            # Cache the result
            self._dimension = dimension

        return dimension
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        async def _embed(sem: asyncio.Semaphore, text: str):
            async with sem:
                response = await self.aclient.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding

        sem = asyncio.Semaphore(self.batch_size)
        tasks = [asyncio.create_task(_embed(sem, text)) for text in texts]

        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    @property
    def dimension(self) -> int | None:
        """
        Get the dimension of the embedding vectors.
        
        If not yet determined, makes a test API call to determine the dimension.
        The result is cached for subsequent accesses.
        
        Returns:
            Dimension of the embedding vectors
        """
        with threading.Semaphore():
            if not self._dimension or self._dimension == 0:
                self._determine_dimension()
        return self._dimension


    def __repr__(self) -> str:
        dim_str = str(self._dimension) if self._dimension_determined else "not yet determined"
        return f"OpenAIEmbeddings(model='{self.model}', dimension={dim_str})"
