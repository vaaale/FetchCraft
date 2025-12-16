

import asyncio
import os
from typing import List, Any, Dict

import openai
from pydantic_ai import Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from qdrant_client import QdrantClient

from examples.test_data import AI_CHUNKS, PHYSICS_CHUNKS, DAD_JOKES_CHUNKS
from fetchcraft.agents import PydanticAgent, RetrieverTool
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Node, ObjectNode, DefaultObjectMapper, ObjectType, DocumentNode
from fetchcraft.retriever import VectorIndexRetriever
from fetchcraft.vector_store import QdrantVectorStore

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://wingman:8000/v1")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_objects")

client = QdrantClient(":memory:")


async def get_vector_index(index_id: str, nodes: List[Node]):
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

    index = VectorIndex(
        vector_store=vector_store,
        index_id=index_id
    )
    _ids = await index.add_nodes(doc=None, nodes=nodes)

    return index

async def create_agent(name: str, description: str, ai_vector_index: VectorIndex[Node | Any]) -> PydanticAgent:
    return PydanticAgent.create(
        model=await get_model(),
        tools=[
            Tool(
                RetrieverTool.from_retriever(
                    ai_vector_index.as_retriever(top_k=2),
                    name=name,
                    description=description
                ).get_tool_function(),
                takes_ctx=True,
                max_retries=3
            )
        ],
        name=name
    )


async def main():
    # Get Vector Indices
    ai_index_name = "AI-INDEX"
    ai_vector_index = await get_vector_index(ai_index_name, AI_CHUNKS)

    physics_index_name = "PHYSICS-INDEX"
    physics_vector_index = await get_vector_index(physics_index_name, PHYSICS_CHUNKS)

    jokes_index_name = "JOKES-INDEX"
    jokes_vector_index = await get_vector_index(jokes_index_name, DAD_JOKES_CHUNKS)

    agents = {
        "ai_agent" : {
            "description": "topics about AI, algorithms, deep learning, machine learning",
            "vector_index": ai_vector_index
        },
        "physics_agent" : {
            "description": "information about physics like quantum physics, relativity, mechanics, etc.",
            "vector_index": physics_vector_index
        },
        "jokes_agent" : {
            "description": "relax with som funny jokes and humor",
            "vector_index": jokes_vector_index
        }
    }

    object_nodes = [
        ObjectNode.from_agent(
            text=agent_data["description"],
            agent=await create_agent(
                agent_name,
                agent_data["description"],
                agent_data["vector_index"]
            )
        )
        for agent_name, agent_data in agents.items()
    ]
    def _create_agent(node_data: Dict[str, Any]):
        agent_name = node_data["agent_kwargs"]["name"]
        agent_def = agents[agent_name]
        agent = asyncio.get_event_loop().run_until_complete(create_agent(agent_name, agent_def["description"], agent_def["vector_index"]))
        return agent

    object_mapper = DefaultObjectMapper(
        factories={
            ObjectType.AGENT: _create_agent
        }
    )
    retriever_index = await get_vector_index("RETRIEVER-INDEX", object_nodes)

    retriever = retriever_index.as_retriever(top_k=2, object_mapper=object_mapper)

    query = "what is quantum physics?"
    nodes = retriever.retrieve(query)

    print(f"Question: {query}")
    for i, node in enumerate(nodes):
        print(f"Answer {i+1}: {node.text}")

    print(nodes)


# noinspection PyTypeChecker
async def get_model():
    api_key = os.environ.get("OPENAI_API_KEY", "sk-123")
    api_base = os.environ.get("OPENAI_API_URL", "http://wingman:8000/v1")

    model = OpenAIChatModel(
        os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
        provider=OpenAIProvider(
            openai_client=openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
        )
    )

    return model


if __name__ == '__main__':
    asyncio.run(main())
