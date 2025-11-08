import asyncio
import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient

from fetchcraft.document_store import MongoDBDocumentStore, DocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.evaluation import DatasetGenerator
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node_parser import HierarchicalNodeParser, SimpleNodeParser
from fetchcraft.parsing import FilesystemDocumentParser
from fetchcraft.vector_store import QdrantVectorStore

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# Configuration
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_chatbot")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

# Hybrid search configuration
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")


# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8192"))
CHILD_SIZES = [4096, 1024]
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
USE_HIERARCHICAL_CHUNKING = os.getenv("USE_HIERARCHICAL_CHUNKING", "true").lower() == "true"

# Index configuration
INDEX_ID = "docs-index"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "gpt-4-turbo"


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists in Qdrant."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if collection_name in collection_names:
        count_result = client.count(
            collection_name=collection_name,
            exact=True  # Set to True for an exact count, False for approximate
        )
        return count_result.count > 0
    return collection_name in collection_names


async def load_and_index_documents(
    vector_index: VectorIndex,
    document_store: DocumentStore,
    documents_path: Path,
    chunk_size: int = 8192,
    child_sizes=[4096, 1024],
    overlap: int = 200,
    use_hierarchical: bool = True
) -> int:
    """Load documents from a directory and index them."""
    logger.info(f"Loading documents from: {documents_path}")

    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_path}")

    # Step 1: Load documents from filesystem
    logger.info("Loading documents...")
    source = FilesystemDocumentParser.from_directory(
        directory=documents_path,
        pattern="*",
        recursive=True
    )

    documents = [doc async for doc in source.get_documents()]

    await document_store.add_documents(documents)

    if not documents:
        logger.warning("No text files found in the specified directory!")
        return 0

    logger.info(f"Loaded {len(documents)} documents")

    # Step 2: Parse documents into nodes
    if use_hierarchical:
        logger.info(f"Using HierarchicalNodeParser (parent={chunk_size}, children={child_sizes})")
        parser = HierarchicalNodeParser(
            chunk_size=chunk_size,
            overlap=overlap,
            child_sizes=child_sizes,
            child_overlap=50
        )
    else:
        logger.info(f"Using SimpleNodeParser (chunk_size={chunk_size})")
        parser = SimpleNodeParser(
            chunk_size=chunk_size,
            overlap=overlap
        )

    all_nodes = parser.get_nodes(documents)
    # doc_map = {doc.id: doc for doc in documents}
    # for node in all_nodes:
    #     doc_map[node.doc_id].children_ids.append(node.id)



    logger.info(f"Indexing {len(all_nodes)} chunks with hybrid search...")
    await vector_index.add_nodes(all_nodes, show_progress=True)

    logger.info(f"Successfully indexed {len(all_nodes)} chunks!")
    return len(all_nodes)


async def main():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        enable_hybrid=True,
        fusion_method="rrf"
    )
    needs_indexing = not collection_exists(qdrant_client, COLLECTION_NAME)

    document_store = MongoDBDocumentStore(
        connection_string=MONGODB_URI,
        database_name=COLLECTION_NAME,
        collection_name=COLLECTION_NAME
    )

    generator = DatasetGenerator(
        model=LLM_MODEL,
    )

    vector_index = VectorIndex(
        vector_store=vector_store,
        doc_store=document_store,
        index_id=INDEX_ID
    )

    if needs_indexing:
        num_chunks = await load_and_index_documents(
            vector_index=vector_index,
            document_store=document_store,
            documents_path=DOCUMENTS_PATH,
            chunk_size=CHUNK_SIZE,
            child_sizes=CHILD_SIZES,
            overlap=CHUNK_OVERLAP,
            use_hierarchical=USE_HIERARCHICAL_CHUNKING
        )
        if num_chunks == 0:
            logger.warning("No documents were indexed!")

    dataset = await generator.generate_dataset(
        document_store=document_store,
        vector_store=vector_store,
        index_id=INDEX_ID,
        num_documents=10,
        questions_per_node=3,
        max_nodes_per_document=5,
        show_progress=True
    )

    print(dataset)




if __name__ == '__main__':
    asyncio.run(main())
