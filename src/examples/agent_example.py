"""
Example demonstrating the use of ReAct agents with RAG.

This example shows how to:
1. Create a vector index with documents
2. Create a retriever from the index
3. Create a ReAct agent that uses the retriever
4. Query the agent with questions
"""

import asyncio
import os

import openai
from pydantic_ai import Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from qdrant_client import QdrantClient

from rag_framework import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Node,
    ReActAgent,
    RetrieverTool
)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

api_key = os.environ.get("OPENAI_API_KEY", "sk-123")
api_base = os.environ.get("OPENAI_API_URL", "http://wingman:8000/v1")

model = OpenAIChatModel(
    os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
    provider=OpenAIProvider(
        openai_client=openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
    )
)


async def basic_agent_example():
    """Basic example of using a ReAct agent with a retriever."""
    
    print("="*60)
    print("Basic ReAct Agent Example")
    print("="*60 + "\n")

    # Step 1: Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )
    dimension = embeddings.dimension
    print(f"✓ Initialized embeddings (dimension: {dimension})\n")

    # Step 2: Create knowledge base
    documents_text = [
        "Bill Gates was born on October 28, 1955, in Seattle, Washington.",
        "Bill Gates co-founded Microsoft with Paul Allen in 1975.",
        "Bill Gates stepped down as CEO of Microsoft in 2000 but remained chairman.",
        "Bill Gates is known for his philanthropic work through the Bill & Melinda Gates Foundation.",
        "The Gates Foundation focuses on global health, education, and poverty alleviation.",
        "Python is a high-level programming language created by Guido van Rossum.",
        "Python was first released in 1991 and emphasizes code readability.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neural networks in the brain.",
        "The Turing test was proposed by Alan Turing in 1950.",
    ]
    
    print(f"Creating knowledge base with {len(documents_text)} documents...")
    nodes = [Node(text=text) for text in documents_text]
    
    # Step 3: Setup vector store and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="knowledge_base",
        vector_size=dimension
    )
    
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings
    )
    
    await index.add_documents(nodes)
    print(f"✓ Indexed {len(nodes)} documents (embeddings auto-generated!)\n")
    
    # Step 4: Create retriever from index
    retriever = index.as_retriever(top_k=3)
    print(f"✓ Created retriever\n")
    
    # Step 5: Create retriever tool
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

    print(f"✓ Created retriever tool\n")
    
    # Step 6: Create ReAct agent with the retriever tool
    agent = ReActAgent.create(
        model=model,
        tools=tools,
        retries=3
    )
    print(f"✓ Created ReAct agent: {agent}\n")
    
    # Step 7: Query the agent
    questions = [
        "When was Bill Gates born?",
        "What company did Bill Gates co-found?",
        "What does the Gates Foundation focus on?",
        "Who created Python?",
        "How old was Bill Gates when he died?",  # This should say he's still alive or no info
    ]
    
    print("="*60)
    print("Asking Questions")
    print("="*60 + "\n")
    
    for question in questions:
        print(f"Q: {question}")
        try:
            answer = await agent.query(question)
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


async def custom_system_prompt_example():
    """Example with custom system prompt for the agent."""
    
    print("\n" + "="*60)
    print("Custom System Prompt Example")
    print("="*60 + "\n")
    
    # Setup (abbreviated)
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )

    documents_text = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower was completed in 1889.",
        "The Eiffel Tower is 330 meters tall.",
        "Gustave Eiffel designed the Eiffel Tower.",
        "The Eiffel Tower was built for the 1889 World's Fair.",
    ]
    
    nodes = [Node(text=text) for text in documents_text]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="landmarks",
        vector_size=embeddings.dimension
    )
    
    index = VectorIndex(vector_store=vector_store, embeddings=embeddings)
    await index.add_documents(nodes)
    
    retriever = index.as_retriever(top_k=2)
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

    # Create agent with custom system prompt
    custom_prompt = """You are an expert tour guide assistant.

When answering questions about landmarks:
1. Search for relevant information using the search_documents tool
2. Provide detailed, engaging answers
3. Include historical context when available
4. If information is not in the documents, politely say so"""
    
    agent = ReActAgent.create(
        tools=tools,
        model=model,
        system_prompt=custom_prompt
    )
    
    print("✓ Created agent with custom system prompt\n")
    
    question = "Tell me about the Eiffel Tower"
    print(f"Q: {question}")
    
    try:
        answer = await agent.query(question)
        print(f"A: {answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")


async def multi_step_reasoning_example():
    """Example showing multi-step reasoning capabilities."""
    
    print("\n" + "="*60)
    print("Multi-Step Reasoning Example")
    print("="*60 + "\n")
    
    # Setup
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )

    documents_text = [
        "Microsoft was founded in 1975.",
        "Bill Gates co-founded Microsoft with Paul Allen.",
        "Microsoft's first product was a BASIC interpreter for the Altair 8800.",
        "Windows 1.0 was released in 1985.",
        "Microsoft went public in 1986.",
        "Bill Gates was CEO of Microsoft from 1975 to 2000.",
        "Steve Ballmer succeeded Bill Gates as CEO in 2000.",
        "Satya Nadella became CEO of Microsoft in 2014.",
    ]
    
    nodes = [Node(text=text) for text in documents_text]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="tech_history",
        vector_size=embeddings.dimension
    )
    
    index = VectorIndex(vector_store=vector_store, embeddings=embeddings)
    await index.add_documents(nodes)
    
    retriever = index.as_retriever(top_k=4)
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]
    agent = ReActAgent.create(tools=tools, model=model)
    
    print("✓ Setup complete\n")
    
    # Complex question requiring multi-step reasoning
    question = "How many years was Bill Gates the CEO of Microsoft?"
    print(f"Q: {question}")
    print("(This requires finding when he became CEO and when he stepped down)\n")
    
    try:
        answer = await agent.query(question)
        print(f"A: {answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")


async def main():
    """Run all examples."""
    try:
        await basic_agent_example()
        await custom_system_prompt_example()
        await multi_step_reasoning_example()
        
        print("\n" + "="*60)
        print("All examples completed successfully! ✓")
        print("="*60)
        print("\nNote: Actual results depend on the LLM model used.")
        print("Make sure you have OPENAI_API_KEY set in your environment.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. OPENAI_API_KEY environment variable set")
        print("  2. pydantic-ai installed: pip install pydantic-ai")
        print("  3. An OpenAI-compatible model running if using custom base_url")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
