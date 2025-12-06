"""
Fetchcraft Admin Ingestion Module.

This module provides the ingestion handler and pipeline factory for
document ingestion functionality.
"""
from fetchcraft.admin.ingestion.pipeline_factory import (
    FetchcraftIngestionPipelineFactory,
    DefaultIndexFactory,
)
from fetchcraft.admin.ingestion.handler import FetchcraftIngestionAdminHandler
from fetchcraft.admin.ingestion.config import IngestionConfig

__all__ = [
    "FetchcraftIngestionPipelineFactory",
    "FetchcraftIngestionAdminHandler",
    "IngestionConfig",
    "DefaultIndexFactory",
]
