from typing import List, Optional
import os

from openai import AsyncOpenAI

from .base import Embeddings


class OpenAIEmbeddings(Embeddings):
    """
    OpenAI embeddings implementation.
    
    Supports both OpenAI API and compatible endpoints (e.g., Azure OpenAI, local models).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either via api_key parameter "
                "or OPENAI_API_KEY environment variable"
            )
        
        self.base_url = base_url
        self.model = model
        self.dimensions_param = dimensions
        self.timeout = timeout
        
        # Initialize the OpenAI async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # Cache for the determined dimension
        self._dimension: Optional[int] = dimensions  # Only set if explicitly provided
        self._dimension_determined = dimensions is not None
    
    async def _determine_dimension(self) -> int:
        """
        Determine the embedding dimension by making a test API call.
        
        Returns:
            The dimension of the embedding vectors
        """
        # Make a test call with a simple string
        kwargs = {"model": self.model, "input": ["test"]}
        if self.dimensions_param:
            kwargs["dimensions"] = self.dimensions_param
        
        response = await self.client.embeddings.create(**kwargs)
        dimension = len(response.data[0].embedding)
        
        # Cache the result
        self._dimension = dimension
        self._dimension_determined = True
        
        return dimension
    
    async def aget_dimension(self) -> int:
        """
        Asynchronously get the dimension of the embedding vectors.
        
        If not yet determined, makes a test API call to determine the dimension.
        The result is cached for subsequent accesses.
        
        Returns:
            Dimension of the embedding vectors
        """
        if not self._dimension_determined:
            await self._determine_dimension()
        return self._dimension  # type: ignore
    
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
        
        # Prepare kwargs for the API call
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions_param:
            kwargs["dimensions"] = self.dimensions_param
        
        # Call OpenAI API
        response = await self.client.embeddings.create(**kwargs)
        
        # Extract embeddings and determine dimension on first call
        embeddings = [item.embedding for item in response.data]
        
        if not self._dimension_determined and embeddings:
            self._dimension = len(embeddings[0])
            self._dimension_determined = True
        
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
    def dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        If not yet determined, makes a test API call to determine the dimension.
        The result is cached for subsequent accesses.
        
        Returns:
            Dimension of the embedding vectors
        """
        if not self._dimension_determined:
            # Need to determine dimension - must use async
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Cannot determine dimension synchronously while event loop is running. "
                        "Call 'await embeddings.embed_query(\"test\")' first to determine dimension, "
                        "or provide 'dimensions' parameter during initialization."
                    )
                self._dimension = loop.run_until_complete(self._determine_dimension())
            except RuntimeError as e:
                if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                    # Create new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        self._dimension = loop.run_until_complete(self._determine_dimension())
                    finally:
                        loop.close()
                else:
                    raise
        
        return self._dimension  # type: ignore
    
    def __repr__(self) -> str:
        dim_str = str(self._dimension) if self._dimension_determined else "not yet determined"
        return f"OpenAIEmbeddings(model='{self.model}', dimension={dim_str})"
