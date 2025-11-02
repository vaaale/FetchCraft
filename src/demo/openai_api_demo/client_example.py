"""
Example client for testing the FastAPI RAG server.

This demonstrates how to interact with the OpenAI-compatible API.

Usage:
    python -m demo.openai_api_demo.client_example
"""

import os
from openai import OpenAI


def main():
    """Run example queries against the RAG API."""
    # Configure client to point to local server
    client = OpenAI(
        api_key="not-needed",  # API key not required for local server
        base_url="http://localhost:8000/v1"
    )
    
    print("="*70)
    print("🤖 FastAPI RAG Client Example")
    print("="*70)
    print("\nMake sure the server is running:")
    print("  python -m demo.openai_api_demo.server")
    print("="*70 + "\n")
    
    # Example 1: Simple query
    print("📝 Example 1: Simple Non-Streaming Query")
    print("-" * 70)
    
    try:
        response = client.chat.completions.create(
            model="rag-hybrid",
            messages=[
                {"role": "user", "content": "What is hybrid search?"}
            ],
            temperature=0.7
        )
        
        print(f"Question: What is hybrid search?")
        print(f"\nAnswer: {response.choices[0].message.content}\n")
        print("✅ Success!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the server is running at http://localhost:8000")
        return
    
    # Example 2: Streaming query
    print("\n" + "="*70)
    print("📝 Example 2: Streaming Response")
    print("-" * 70)
    
    try:
        print("Question: Explain RAG systems\n")
        print("Answer (streaming): ", end="")
        
        stream = client.chat.completions.create(
            model="rag-hybrid",
            messages=[
                {"role": "user", "content": "Explain RAG systems"}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print("\n\n✅ Success!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Example 3: Multi-turn conversation
    print("\n" + "="*70)
    print("📝 Example 3: Multi-turn Conversation")
    print("-" * 70)
    
    try:
        messages = [
            {"role": "user", "content": "What is vector search?"}
        ]
        
        print(f"User: {messages[0]['content']}")
        
        # First turn
        response = client.chat.completions.create(
            model="rag-hybrid",
            messages=messages
        )
        
        assistant_msg = response.choices[0].message.content
        print(f"Assistant: {assistant_msg}\n")
        
        # Add to conversation history
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": "How does it differ from traditional search?"})
        
        print(f"User: {messages[-1]['content']}")
        
        # Second turn
        response = client.chat.completions.create(
            model="rag-hybrid",
            messages=messages
        )
        
        print(f"Assistant: {response.choices[0].message.content}\n")
        print("✅ Success!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: With citations (using requests for full response)
    print("\n" + "="*70)
    print("📝 Example 4: Getting Citations")
    print("-" * 70)
    
    try:
        import requests
        
        url = "http://localhost:8000/v1/chat/completions"
        data = {
            "model": "rag-hybrid",
            "messages": [
                {"role": "user", "content": "What topics are covered?"}
            ],
            "stream": False
        }
        
        response = requests.post(url, json=data)
        result = response.json()
        
        print(f"Question: What topics are covered?")
        print(f"\nAnswer: {result['choices'][0]['message']['content']}")
        
        if result.get("citations"):
            print(f"\n📚 Citations ({len(result['citations'])} sources):")
            for i, citation in enumerate(result["citations"], 1):
                print(f"\n  [{i}] {citation['filename']}")
                print(f"      Score: {citation['score']:.3f}")
                print(f"      Preview: {citation['text_preview'][:100]}...")
        else:
            print("\n📚 No citations available")
        
        print("\n✅ Success!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*70)
    print("🎉 All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
