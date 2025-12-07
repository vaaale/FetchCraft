"""
Legacy transformation implementations.

DEPRECATED: Use fetchcraft.ingestion.transformations instead.
These are kept for backwards compatibility with the legacy IngestionPipeline.
"""
from fetchcraft.ingestion.base import Record as LegacyRecord, Transformation as LegacyTransformation
from fetchcraft.node import DocumentNode
from fetchcraft.node_parser import NodeParser


class ExtractKeywords(LegacyTransformation):
    """Legacy keyword extraction - use fetchcraft.ingestion.transformations.ExtractKeywords instead."""
    
    async def process(self, record: LegacyRecord) -> LegacyRecord:
        doc = DocumentNode.model_validate(record.payload["document"])
        text = doc.text
        words = [w.strip(".,!?;:\"'()[]{}-").lower() for w in text.split()]
        counts: dict[str, int] = {}
        for w in words:
            if len(w) >= 4:
                counts[w] = counts.get(w, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        record.meta["keywords"] = [k for k, _ in top]
        return record


class DocumentSummarization(LegacyTransformation):
    """Legacy summarization - use fetchcraft.ingestion.transformations.DocumentSummarization instead."""
    
    def __init__(self, max_sentences: int = 30):
        self.max_sentences = max_sentences

    async def process(self, record: LegacyRecord) -> LegacyRecord:
        doc = DocumentNode.model_validate(record.payload["document"])
        text = doc.text

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        selected = sentences[: self.max_sentences]
        record.meta["summary"] = ". ".join(selected)
        doc.metadata['summary'] = ". ".join(selected)
        record.payload["document"] = doc.model_dump()
        return record


class ChunkingTransformation(LegacyTransformation):
    """Legacy chunking - use fetchcraft.ingestion.transformations.ChunkingTransformation instead."""
    
    def __init__(self, chunker: NodeParser):
        self.chunker = chunker

    async def process(self, record: LegacyRecord) -> LegacyRecord:
        doc = DocumentNode.model_validate(record.payload["document"])
        nodes = self.chunker.get_nodes([doc])
        record.payload["document"] = doc.model_dump()
        record.payload["chunks"] = nodes
        return record
