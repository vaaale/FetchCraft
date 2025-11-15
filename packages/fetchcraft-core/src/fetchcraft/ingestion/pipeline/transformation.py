import asyncio

from fetchcraft.ingestion.base import Record, Transformation
from fetchcraft.node import DocumentNode
from fetchcraft.node_parser import NodeParser


class ExtractKeywords(Transformation):
    async def process(self, record: Record) -> Record:
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


class DocumentSummarization(Transformation):
    def __init__(self, max_sentences: int = 30):
        self.max_sentences = max_sentences

    async def process(self, record: Record) -> Record:
        doc = DocumentNode.model_validate(record.payload["document"])
        text = doc.text

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        selected = sentences[: self.max_sentences]
        record.meta["summary"] = ". ".join(selected)
        doc.metadata['summary'] = ". ".join(selected)
        record.payload["document"] = doc.model_dump()
        return record

class ChunkingTransformation(Transformation):
    def __init__(self, chunker: NodeParser):
        self.chunker = chunker

    async def process(self, record: Record) -> Record:
        doc = DocumentNode.model_validate(record.payload["document"])
        nodes = self.chunker.get_nodes([doc])
        record.payload["document"] = doc.model_dump()
        record.payload["chunks"] = nodes
        return record
