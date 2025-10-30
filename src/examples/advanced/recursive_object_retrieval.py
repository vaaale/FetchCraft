import asyncio
import os
from typing import List, Any, Dict

from mongomock.mongo_client import MongoClient
from qdrant_client import QdrantClient

from examples.test_data import AI_CHUNKS, PHYSICS_CHUNKS, DAD_JOKES_CHUNKS
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Node, ObjectNode, DefaultObjectMapper, ObjectType
from fetchcraft.retriever import VectorIndexRetriever
from fetchcraft.vector_store import QdrantVectorStore

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://wingman:8000/v1")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_objects")

mongo_client = MongoClient()
client = QdrantClient(":memory:")


async def get_vector_index(index_id: str, nodes: List[Node]):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    #
    # document_store = MongoDBDocumentStore(
    #     client=mongo_client,
    #     database_name=COLLECTION_NAME,
    #     collection_name=COLLECTION_NAME
    # )
    #
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    index = VectorIndex(
        vector_store=vector_store,
        # doc_store=document_store,
        index_id=index_id
    )
    await index.add_nodes(nodes)

    return index


async def main():
    # Get Vector Indices
    ai_index_name = "AI-INDEX"
    ai_vector_index = await get_vector_index(ai_index_name, AI_CHUNKS)

    physics_index_name = "PHYSICS-INDEX"
    physics_vector_index = await get_vector_index(physics_index_name, PHYSICS_CHUNKS)

    jokes_index_name = "JOKES-INDEX"
    jokes_vector_index = await get_vector_index(jokes_index_name, DAD_JOKES_CHUNKS)

    object_nodes = [
        ObjectNode.from_retriever(
            text="topics about AI, algorithms, deep learning, machine learning",
            retriever=ai_vector_index.as_retriever(top_k=2)
        ),
        ObjectNode.from_retriever(
            text="information about physics like quantum physics, relativity, mechanics, etc.",
            retriever=physics_vector_index.as_retriever(top_k=2)
        ),
        ObjectNode.from_retriever(
            text="relax with som funny jokes and humor",
            retriever=jokes_vector_index.as_retriever(top_k=2)
        ),
    ]

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    def _create_vector_index_retriever(node_data: Dict[str, Any]):
        return VectorIndexRetriever(
            vector_index=VectorIndex(
                vector_store=vector_store,
                index_id=(node_data["index_id"])
            ),
            top_k=(node_data["top_k"]),
            resolve_parents=(node_data["resolve_parents"]),
            **node_data["search_kwargs"]
        )

    object_mapper = DefaultObjectMapper(
        factories={
            ObjectType.VECTOR_INDEX_RETRIEVER: _create_vector_index_retriever
        }
    )
    retriever_index = await get_vector_index("RETRIEVER-INDEX", object_nodes)

    retriever = retriever_index.as_retriever(top_k=2, object_mapper=object_mapper)

    nodes = retriever.retrieve("what is quantum physics?")

    print(nodes)



if __name__ == '__main__':
    asyncio.run(main())
