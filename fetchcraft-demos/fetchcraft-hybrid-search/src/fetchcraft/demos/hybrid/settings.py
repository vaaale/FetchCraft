from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the hybrid search demo."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "fetchcraft_chatbot"

    # Documents configuration
    documents_path: str = "Documents"

    # Embeddings configuration
    embedding_model: str = "bge-m3"
    openai_api_key: str = "sk-321"
    embedding_base_url: str | None = None

    # Index configuration
    index_id: str = "docs-index"

    # LLM configuration
    llm_model: str = "gpt-4-turbo"

    # Chunking configuration
    chunk_size: int = 8192
    child_sizes: list[int] = [4096, 1024]
    chunk_overlap: int = 200
    use_hierarchical_chunking: bool = True

    # Hybrid search configuration
    enable_hybrid: bool = True
    fusion_method: str = "rrf"
