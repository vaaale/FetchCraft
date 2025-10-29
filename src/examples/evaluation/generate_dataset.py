from typing import *
import asyncio
import os

from qdrant_client import QdrantClient

from fetchcraft import OpenAIEmbeddings, QdrantVectorStore, MongoDBDocumentStore, VectorIndex, DatasetGenerator

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "fetchcraft_docs"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

# Index configuration
INDEX_ID = "docs-index"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "gpt-4-turbo"


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

    document_store = MongoDBDocumentStore(
        connection_string=MONGODB_URI,
        database_name="fetchcraft",
        collection_name="documents"
    )

    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )

    generator = DatasetGenerator(
        model=LLM_MODEL,
    )

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
