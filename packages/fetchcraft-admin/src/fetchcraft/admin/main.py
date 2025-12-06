"""
Fetchcraft Admin Server - Entry point using the new framework.

This module provides a convenient entry point for running the admin server
with the default ingestion handler and configuration.

For custom configurations, use the framework directly:
    from fetchcraft.admin import (
        FetchcraftAdminServer,
        FetchcraftIngestionAdminHandler,
        FetchcraftIngestionPipelineFactory,
        IngestionConfig,
    )
"""
from pathlib import Path

from qdrant_client import QdrantClient

from fetchcraft.admin import (
    FetchcraftAdminServer,
    FetchcraftIngestionAdminHandler,
    FetchcraftIngestionPipelineFactory,
    IngestionConfig, DefaultIndexFactory,
)
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.ingestion.pipeline import TrackedIngestionPipeline
from fetchcraft.ingestion.transformations import (
    AsyncParsingTransformation,
    ExtractKeywords,
    ChunkingTransformation,
)
from fetchcraft.ingestion.sinks import VectorIndexSink
from dotenv import load_dotenv

from fetchcraft.node_parser import HierarchicalNodeParser
from fetchcraft.parsing.base import DocumentParser
from fetchcraft.parsing.docling.client.docling_parser import RemoteDoclingParser
from fetchcraft.parsing.text_file_parser import TextFileParser
from fetchcraft.vector_store import QdrantVectorStore

load_dotenv()

# Frontend dist path
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"



class DefaultPipelineFactory(FetchcraftIngestionPipelineFactory):
    index_factory: DefaultIndexFactory
    chunker: HierarchicalNodeParser
    parser_map: dict[str, DocumentParser]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    """
    Default pipeline factory with standard transformations.
    
    This factory creates pipelines with:
    - Async parsing transformation
    - Keyword extraction
    - Chunking transformation
    - Vector index sink
    """
    
    def configure_pipeline(self, pipeline: TrackedIngestionPipeline) -> None:
        """Configure the pipeline with default transformations and sinks."""
        pipeline.add_transformation(AsyncParsingTransformation(parser_map=self.parser_map))
        pipeline.add_transformation(ExtractKeywords())
        pipeline.add_transformation(ChunkingTransformation(chunker=self.chunker))
        pipeline.add_sink(VectorIndexSink(index_factory=self.index_factory))



# =============================================================================
# Helper Functions
# =============================================================================

def get_ingestion_dependencies(settings: IngestionConfig):


    """Get common dependencies needed for ingestion jobs."""
    doc_store = MongoDBDocumentStore(
        connection_string=settings.mongo_uri,
        database_name="fetchcraft",
        collection_name=settings.collection_name,
    )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url
    )

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=settings.enable_hybrid,
        fusion_method=settings.fusion_method
    )

    chunker = HierarchicalNodeParser(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        child_sizes=settings.child_chunks,
        child_overlap=50
    )

    index_factory = DefaultIndexFactory(
        vector_store=vector_store,
        doc_store=doc_store,
        index_id=settings.index_id
    )

    # Build callback URL for docling async parsing
    callback_url = f"{settings.callback_base_url}/api/tasks/callback"

    parser_map = {
        "default": TextFileParser(),
        "application/pdf": RemoteDoclingParser(
            docling_url=settings.docling_server,
            callback_url=callback_url
        )
    }

    return {
        "index_factory": index_factory,
        "chunker": chunker,
        "parser_map": parser_map,
    }


def create_server(config: IngestionConfig = None, deps: dict = None) -> FetchcraftAdminServer:
    """
    Create a configured admin server instance.
    
    Args:
        config: Optional configuration. If not provided, uses default IngestionConfig.
    
    Returns:
        Configured FetchcraftAdminServer instance
    """
    if config is None:
        config = IngestionConfig()
    
    handler = FetchcraftIngestionAdminHandler(
        pipeline_factory=DefaultPipelineFactory(**deps),
        frontend_dist=FRONTEND_DIST if FRONTEND_DIST.exists() else None,
    )
    
    server = FetchcraftAdminServer(
        handlers=[handler],
        config=config,
        title="Fetchcraft Admin",
        description="Administration interface with job and document tracking",
        version="2.0.0",
        frontend_dist=FRONTEND_DIST if FRONTEND_DIST.exists() else None,
    )
    
    return server


def main():
    """Run the admin server with default configuration."""
    config = IngestionConfig()
    dependencies = get_ingestion_dependencies(config)
    server = create_server(config, dependencies)
    server.run()


if __name__ == "__main__":
    main()
