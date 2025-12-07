from pydantic_settings import SettingsConfigDict, BaseSettings


class MCPServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    frontend_base_url: str = "http://localhost:8003"
    frontend_port: int = 8003
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    database_name: str = "fetchcraft"
    collection_name: str = "fetchcraft_chatbot"
    documents_path: str = "Documents"
    embedding_model: str = "bge-m3"
    embedding_api_key: str = "sk-321"
    embedding_base_url: str = "http://wingman.akhbar.home:8000/v1"
    opanai_api_key: str = "sk-321"
    openai_base_url: str = "http://wingman.akhbar.home:8000/v1"
    index_id: str = "docs-index"
    llm_model: str = "gpt-4-turbo"
    enable_hybrid: bool = True
    fusion_method: str = "rrf"
    host: str = "0.0.0.0"
    port: int = 8001
