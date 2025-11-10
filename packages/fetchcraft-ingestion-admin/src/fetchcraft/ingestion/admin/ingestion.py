import asyncio
import os
import pathlib
import traceback

from qdrant_client import QdrantClient

from fetchcraft.connector.filesystem import FilesystemConnector, LocalFile
from fetchcraft.document_store import MongoDBDocumentStore, DocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion.base import ConnectorSource, IngestionPipeline, Record, Sink
from fetchcraft.ingestion.pipeline.transformation import ChunkingTransformation
from fetchcraft.ingestion.sqlite_backend import AsyncSQLiteQueue
from fetchcraft.node import DocumentNode, Node
from fetchcraft.node_parser import HierarchicalNodeParser
from fetchcraft.parsing.docling import DoclingDocumentParser
from fetchcraft.parsing.docling.client.docling_parser import RemoteDoclingParser
from fetchcraft.parsing.text_file_parser import TextFileParser
from fetchcraft.vector_store import QdrantVectorStore

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DOCUMENT_DB = os.getenv("DOCUMENT_DB", "fetchcraft")  # Different collection for hybrid search
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_chatbot")
DOCUMENTS_PATH = pathlib.Path(os.getenv("DOCUMENTS_PATH", "Documents"))
DOCLING_SERVER = os.getenv("DOCLING_SERVER", "http://localhost:8001")
INGESTION_DB = os.getenv("INGESTION_DB", "ingestion_queue.db")

# Embeddings configuration (adjust based on your setup)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "sk-321")
EMBEDDING_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # None = use OpenAI default
INDEX_ID = os.getenv("INDEX_ID", "docs-index")

# LLM configuration for the agent
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8192"))
CHILD_CHUNKS = [int(chunk_size) for chunk_size in os.getenv("CHILD_CHUNKS", "4096,1024").split(",")]
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# HYBRID SEARCH CONFIGURATION
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")  # "rrf" or "dbsf"


class VectorIndexSink(Sink):

    def __init__(self, vector_index: VectorIndex[Node], index_id: str = "vector_index"):
        self.index_id = index_id
        self.vector_index = vector_index
        self._lock = asyncio.Lock()
        self.counter = 0

    async def write(self, record: Record) -> None:
        async with self._lock:
            nodes = record.payload.get("chunks", [])
            doc = DocumentNode.model_validate(record.payload["document"])
            await self.vector_index.delete_document_nodes(doc)
            await self.vector_index.add_nodes(nodes)
            self.counter += 1
            print(f"{self.index_id} Indexed[{self.counter}]: {record.id}")

    def num_records(self) -> int:
        return self.counter

class DocumentStoreSink(Sink):

    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        self._lock = asyncio.Lock()

    async def write(self, record: Record) -> None:
        async with self._lock:
            try:
                doc = DocumentNode.model_validate(record.payload["document"])
                node_id = await self.doc_store.add_document(doc)
                doc.id = node_id
                record.payload["document"] = doc.model_dump()
                print(f"DocumentStoreSink Indexed: {record.id}")
            except Exception as e:
                print(f"DocumentStoreSink Error: {e}")
                traceback.print_exc()



# -----------------------------
# Demo usage
# -----------------------------

async def main():
    document_path = DOCUMENTS_PATH
    doc_store = MongoDBDocumentStore(
        connection_string="mongodb://mongodb:27017",
        database_name="fetchcraft",
        collection_name=COLLECTION_NAME,
    )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=ENABLE_HYBRID,
        fusion_method=FUSION_METHOD
    )
    initial_doc_count = await doc_store.count_documents()
    initial_node_count = client.get_collection(COLLECTION_NAME).points_count

    chunker = HierarchicalNodeParser(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        child_sizes=CHILD_CHUNKS,
        child_overlap=50
    )
    vector_index = VectorIndex(
        vector_store=vector_store,
        doc_store=doc_store,
        index_id=INDEX_ID
    )
    def filter_fn(file: LocalFile) -> bool:
        filepath = str(file.path)
        count = asyncio.get_event_loop().run_until_complete(doc_store.count_documents(filters={"metadata.source": filepath}))
        return count == 0


    backend = AsyncSQLiteQueue(INGESTION_DB)

    index_sink = VectorIndexSink(vector_index=vector_index, index_id=INDEX_ID)
    pipeline = (
        IngestionPipeline(backend=backend)
        .source(ConnectorSource(
            connector=FilesystemConnector(
                path=document_path,
                filter=None
            ),
            parser_map={
                "text/plain": TextFileParser(),
                "text": TextFileParser(),
                "default": RemoteDoclingParser(docling_url=DOCLING_SERVER)
            }
        ))
        # .add_transformation(DocumentSummarization(max_sentences=2, simulate_latency=0.2), deferred=True)
        # .add_transformation(ExtractKeywords())
        .add_transformation(ChunkingTransformation(chunker=chunker))
        .add_sink(DocumentStoreSink(doc_store=doc_store))
        .add_sink(index_sink)
    )

    # This returns only when:
    # - all source docs have been enqueued,
    # - main + deferred queues are drained,
    # - all sinks have finished writes.
    await pipeline.run_job()

    print("Ingestion job finished, pipeline shut down.")
    print(f"Indexed {index_sink.num_records()} documents.")

    final_node_count = client.get_collection(COLLECTION_NAME).points_count
    final_doc_count = await doc_store.count_documents()
    print(f"\n\nInitial document count: {initial_doc_count}")
    print(f"Final document count: {final_doc_count}")
    print(f"Documents added: {final_doc_count - initial_doc_count}")

    print(f"Initial node count: {initial_node_count}")
    print(f"Final node count: {final_node_count}")
    print(f"Nodes added: {final_node_count - initial_node_count}")

    # await pipeline.run()
    #
    # # Keep event loop alive to let workers process; in real apps use a proper lifetime controller
    # try:
    #     while True:
    #         await asyncio.sleep(1)
    #         print("Pipeline running...")
    # except KeyboardInterrupt:
    #     pass


if __name__ == "__main__":
    import contextlib
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
