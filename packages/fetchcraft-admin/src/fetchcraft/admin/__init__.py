"""
Fetchcraft Admin - Modular administration framework.

This package provides a framework for building administration interfaces
with pluggable handler modules.

Example usage:
    from fetchcraft.admin import (
        FetchcraftAdminServer,
        FetchcraftIngestionAdminHandler,
        FetchcraftIngestionPipelineFactory,
        IngestionConfig,
    )
    from fetchcraft.ingestion.pipeline import TrackedIngestionPipeline
    from fetchcraft.ingestion.transformations import (
        ParsingTransformation,
        ExtractKeywords,
        ChunkingTransformation,
    )
    from fetchcraft.ingestion.sinks import VectorIndexSink
    
    class MyPipelineFactory(FetchcraftIngestionPipelineFactory):
        async def configure_pipeline(self, pipeline: TrackedIngestionPipeline) -> None:
            pipeline.add_transformation(ParsingTransformation(parser_map=self.parser_map))
            pipeline.add_transformation(ExtractKeywords())
            pipeline.add_transformation(ChunkingTransformation(chunker=self.chunker))
            pipeline.add_sink(VectorIndexSink(index_factory=self.index_factory, index_id=self.index_id))
    
    config = IngestionConfig()
    handler = FetchcraftIngestionAdminHandler(pipeline_factory=MyPipelineFactory())
    server = FetchcraftAdminServer(handlers=[handler], config=config)
    server.run()
"""

__version__ = "2.0.0"

# Framework exports
from fetchcraft.admin.handler import FetchcraftAdminHandler
from fetchcraft.admin.context import HandlerContext
from fetchcraft.admin.config import FetchcraftAdminConfig
from fetchcraft.admin.server import FetchcraftAdminServer

# Ingestion module exports
from fetchcraft.admin.ingestion import (
    FetchcraftIngestionAdminHandler,
    FetchcraftIngestionPipelineFactory,
    IngestionConfig,
    DefaultIndexFactory,
)



__all__ = [
    # Framework
    "FetchcraftAdminServer",
    "FetchcraftAdminHandler",
    "FetchcraftAdminConfig",
    "HandlerContext",
    # Ingestion
    "FetchcraftIngestionAdminHandler",
    "FetchcraftIngestionPipelineFactory",
    "IngestionConfig",
    "DefaultIndexFactory",
]
