"""
Example client demonstrating how to integrate the RAG Tool API.

This shows different ways to use the RAG Tool API:
1. Simple HTTP requests
2. As an OpenAI-compatible tool
3. Integration with other AI agents
"""

import asyncio
import json
import requests
from typing import Dict, Any, Optional


class RAGToolClient:
    """Client for the RAG Tool API."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        """
        Initialize the RAG Tool client.
        
        Args:
            base_url: Base URL of the RAG Tool API
        """
        self.base_url = base_url
        self.tool_definition = None
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            top_k: Number of documents to retrieve
            include_citations: Whether to include citations
            
        Returns:
            Response dictionary with answer and optional citations
        """
        url = f"{self.base_url}/query"
        payload = {
            "question": question,
            "top_k": top_k,
            "include_citations": include_citations
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status of the service."""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for integration."""
        if self.tool_definition is None:
            url = f"{self.base_url}/tool-definition"
            response = requests.get(url)
            response.raise_for_status()
            self.tool_definition = response.json()
        return self.tool_definition
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Get the full OpenAPI specification."""
        url = f"{self.base_url}/openapi.json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_compact_openapi_spec(self) -> Dict[str, Any]:
        """Get the compact OpenAPI specification for tool integration."""
        url = f"{self.base_url}/openapi-tool-spec"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


# ============================================================================
# Example 1: Simple Usage
# ============================================================================

def example_simple_usage():
    """Example of simple usage of the RAG Tool API."""
    print("="*70)
    print("Example 1: Simple Usage")
    print("="*70)
    
    client = RAGToolClient()
    
    # Check health
    print("\n1. Checking service health...")
    health = client.get_health()
    print(f"   Status: {health['status']}")
    print(f"   Model: {health['model']}")
    print(f"   Collection: {health['collection']}")
    
    # Query the RAG system
    print("\n2. Querying RAG system...")
    question = "What are the main topics in the documents?"
    result = client.query(question)
    
    print(f"\n   Question: {question}")
    print(f"   Answer: {result['answer']}")
    print(f"   Processing time: {result['processing_time_ms']}ms")
    
    if result.get('citations'):
        print(f"\n   Citations ({len(result['citations'])}):")
        for i, citation in enumerate(result['citations'], 1):
            print(f"     {i}. {citation['filename']} (score: {citation['score']:.3f})")
    
    print("\n" + "="*70)


# ============================================================================
# Example 2: OpenAI Function Calling Integration
# ============================================================================

def example_openai_function_integration():
    """Example of how to integrate as an OpenAI function."""
    print("\n" + "="*70)
    print("Example 2: OpenAI Function Calling Integration")
    print("="*70)
    
    client = RAGToolClient()
    
    # Get tool definition
    tool_def = client.get_tool_definition()
    
    print("\n1. Tool Definition:")
    print(json.dumps(tool_def, indent=2))
    
    print("\n2. How to use with OpenAI:")
    print("""
    from openai import OpenAI
    
    client = OpenAI()
    
    # Define the function that calls our RAG API
    def query_rag_system(question: str, top_k: int = 3, include_citations: bool = True):
        rag_client = RAGToolClient()
        return rag_client.query(question, top_k, include_citations)
    
    # Use with OpenAI function calling
    messages = [{"role": "user", "content": "What information do we have about X?"}]
    tools = [tool_definition]  # From get_tool_definition()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # If the model wants to call the tool
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "query_rag_system":
            args = json.loads(tool_call.function.arguments)
            result = query_rag_system(**args)
            # Continue conversation with the result...
    """)
    
    print("\n" + "="*70)


# ============================================================================
# Example 3: Pydantic AI Tool Integration
# ============================================================================

def example_pydantic_ai_integration():
    """Example of how to integrate with Pydantic AI."""
    print("\n" + "="*70)
    print("Example 3: Pydantic AI Tool Integration")
    print("="*70)
    
    print("""
    from pydantic_ai import Agent, Tool
    import requests
    
    # Define the RAG query function
    def query_rag_system(question: str) -> str:
        '''Query the RAG system for information.'''
        response = requests.post(
            "http://localhost:8002/query",
            json={"question": question}
        )
        result = response.json()
        return result['answer']
    
    # Create a Pydantic AI agent with the RAG tool
    agent = Agent(
        model='openai:gpt-4',
        tools=[Tool(query_rag_system)]
    )
    
    # Use the agent
    result = await agent.run("What information do we have about X?")
    print(result)
    
    # The agent will automatically call the RAG tool when needed!
    """)
    
    print("\n" + "="*70)


# ============================================================================
# Example 4: LangChain Integration
# ============================================================================

def example_langchain_integration():
    """Example of how to integrate with LangChain."""
    print("\n" + "="*70)
    print("Example 4: LangChain Integration")
    print("="*70)
    
    print("""
    from langchain.tools import StructuredTool
    from langchain.agents import initialize_agent, AgentType
    from langchain.chat_models import ChatOpenAI
    import requests
    
    # Define the RAG query function
    def query_rag_system(question: str, top_k: int = 3) -> dict:
        '''Query the RAG system for information from documents.
        
        Args:
            question: The question to ask
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and citations
        '''
        response = requests.post(
            "http://localhost:8002/query",
            json={"question": question, "top_k": top_k}
        )
        return response.json()
    
    # Create a LangChain tool
    rag_tool = StructuredTool.from_function(
        func=query_rag_system,
        name="query_rag_system",
        description="Query a RAG system to get answers from a document collection"
    )
    
    # Create an agent with the tool
    llm = ChatOpenAI(temperature=0)
    agent = initialize_agent(
        tools=[rag_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    
    # Use the agent
    result = agent.run("What information do we have about X?")
    print(result)
    """)
    
    print("\n" + "="*70)


# ============================================================================
# Example 5: Direct HTTP Integration
# ============================================================================

def example_direct_http():
    """Example of direct HTTP integration (curl equivalent)."""
    print("\n" + "="*70)
    print("Example 5: Direct HTTP Integration")
    print("="*70)
    
    print("""
    # Using curl:
    curl -X POST http://localhost:8002/query \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "What are the main topics?",
        "top_k": 3,
        "include_citations": true
      }'
    
    # Using Python requests:
    import requests
    
    response = requests.post(
        "http://localhost:8002/query",
        json={
            "question": "What are the main topics?",
            "top_k": 3,
            "include_citations": True
        }
    )
    result = response.json()
    print(result['answer'])
    
    # Using httpx (async):
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8002/query",
            json={"question": "What are the main topics?"}
        )
        result = response.json()
        print(result['answer'])
    """)
    
    print("\n" + "="*70)


# ============================================================================
# Example 6: OpenAPI Spec for GPT Actions
# ============================================================================

def example_gpt_actions():
    """Example of getting OpenAPI spec for GPT Actions."""
    print("\n" + "="*70)
    print("Example 6: GPT Actions Integration (ChatGPT)")
    print("="*70)
    
    client = RAGToolClient()
    
    print("\n1. Get the compact OpenAPI spec:")
    spec = client.get_compact_openapi_spec()
    print(json.dumps(spec, indent=2))
    
    print("\n2. How to use with ChatGPT:")
    print("""
    1. Go to https://chat.openai.com/gpts/editor
    2. Create a new GPT or edit an existing one
    3. Go to "Configure" > "Actions"
    4. Click "Create new action"
    5. Paste the OpenAPI spec from /openapi-tool-spec
    6. Or use the full spec from /openapi.json
    7. Test the action and save
    
    Your GPT can now query your RAG system!
    """)
    
    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("RAG Tool API - Integration Examples")
    print("="*70)
    print("\nMake sure the RAG Tool API server is running:")
    print("  python -m demo.rag_tool_api")
    print("="*70)
    
    try:
        # Example 1: Simple usage
        example_simple_usage()
        
        # Example 2: OpenAI function calling
        example_openai_function_integration()
        
        # Example 3: Pydantic AI
        example_pydantic_ai_integration()
        
        # Example 4: LangChain
        example_langchain_integration()
        
        # Example 5: Direct HTTP
        example_direct_http()
        
        # Example 6: GPT Actions
        example_gpt_actions()
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to RAG Tool API")
        print("   Make sure the server is running: python -m demo.rag_tool_api")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
