# FastAPI RAG Server with Hybrid Search

An **OpenAI-compatible API** server for RAG (Retrieval-Augmented Generation) with hybrid search capabilities. This demo provides a production-ready REST API that can be used with any OpenAI-compatible client.

## 🌟 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions endpoint
- **Hybrid Search**: Combines dense (semantic) + sparse (keyword) vectors for superior results
- **Streaming Support**: Real-time streaming responses using Server-Sent Events (SSE)
- **Citations**: Returns source documents and similarity scores with each response
- **Async/Await**: Fully asynchronous for high performance
- **Docker Ready**: Easy deployment with environment variables

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install fastapi uvicorn qdrant-client pydantic-ai openai fastembed

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Start the Server

```bash
# Basic usage
python -m demo.openai_api_demo.server

# With custom configuration
ENABLE_HYBRID=true \
FUSION_METHOD=rrf \
DOCUMENTS_PATH=./docs \
python -m demo.openai_api_demo.server
```

The server will start at `http://localhost:8000` with:
- OpenAI-compatible endpoint: `http://localhost:8000/v1/chat/completions`
- Health check: `http://localhost:8000/health`
- Models list: `http://localhost:8000/v1/models`

## 📡 API Usage

### 1. Using cURL

#### Non-streaming request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-hybrid",
    "messages": [
      {"role": "user", "content": "What is hybrid search?"}
    ],
    "stream": false
  }'
```

#### Streaming request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-hybrid",
    "messages": [
      {"role": "user", "content": "Explain RAG systems"}
    ],
    "stream": true
  }'
```

### 2. Using Python Requests

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Non-streaming
data = {
    "model": "rag-hybrid",
    "messages": [
        {"role": "user", "content": "What is hybrid search?"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print("Answer:", result["choices"][0]["message"]["content"])
if result.get("citations"):
    print("\nCitations:")
    for i, citation in enumerate(result["citations"], 1):
        print(f"  [{i}] {citation['filename']} (score: {citation['score']:.3f})")
```

### 3. Using OpenAI Python Client

The API is fully compatible with the OpenAI Python client:

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    api_key="not-needed",  # API key not required for local server
    base_url="http://localhost:8000/v1"
)

# Non-streaming
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=[
        {"role": "user", "content": "What is hybrid search?"}
    ]
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="rag-hybrid",
    messages=[
        {"role": "user", "content": "Explain RAG systems"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 4. Using LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="rag-hybrid",
    openai_api_key="not-needed",
    openai_api_base="http://localhost:8000/v1",
    streaming=True
)

response = llm.invoke("What is hybrid search?")
print(response.content)
```

### 5. Multi-turn Conversations

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

messages = [
    {"role": "user", "content": "What is hybrid search?"}
]

# First turn
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=messages
)

print("Assistant:", response.choices[0].message.content)

# Add to conversation
messages.append(response.choices[0].message.model_dump())
messages.append({"role": "user", "content": "How does it compare to dense search?"})

# Second turn
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=messages
)

print("Assistant:", response.choices[0].message.content)
```

## 🎯 API Endpoints

### POST `/v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request Body:**

```json
{
  "model": "rag-hybrid",
  "messages": [
    {"role": "user", "content": "Your question here"}
  ],
  "temperature": 0.7,
  "stream": false,
  "max_tokens": null
}
```

**Response (Non-streaming):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "rag-hybrid",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Answer based on retrieved documents..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  },
  "citations": [
    {
      "source": "/path/to/doc.txt",
      "filename": "doc.txt",
      "score": 0.892,
      "text_preview": "Preview of the parsing text..."
    }
  ]
}
```

### GET `/health`

Health check endpoint.

```json
{
  "status": "healthy",
  "hybrid_search_enabled": true,
  "fusion_method": "rrf"
}
```

### GET `/v1/models`

List available models (OpenAI-compatible).

```json
{
  "object": "list",
  "data": [
    {
      "id": "rag-hybrid",
      "object": "model",
      "created": 1677652288,
      "owned_by": "fetchcraft"
    }
  ]
}
```

## ⚙️ Configuration

Configure the server using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `fetchcraft_hybrid_api` | Qdrant collection name |
| `DOCUMENTS_PATH` | `Documents` | Path to documents directory |
| `ENABLE_HYBRID` | `true` | Enable hybrid search |
| `FUSION_METHOD` | `rrf` | Fusion method (`rrf` or `dbsf`) |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model name |
| `LLM_MODEL` | `gpt-4-turbo` | LLM model for responses |
| `CHUNK_SIZE` | `8192` | Chunk size for documents |
| `USE_HIERARCHICAL_CHUNKING` | `true` | Use hierarchical chunking |

### Example with Custom Configuration

```bash
HOST=0.0.0.0 \
PORT=9000 \
ENABLE_HYBRID=true \
FUSION_METHOD=rrf \
DOCUMENTS_PATH=./my_docs \
EMBEDDING_MODEL=text-embedding-3-small \
LLM_MODEL=gpt-4 \
python -m demo.openai_api_demo.server
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY Documents/ ./Documents/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "demo.fastapi_demo.server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - ENABLE_HYBRID=true
      - FUSION_METHOD=rrf
      - DOCUMENTS_PATH=/app/Documents
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
    volumes:
      - ./Documents:/app/Documents

volumes:
  qdrant_storage:
```

Run with:

```bash
docker-compose up
```

## 🧪 Testing

### Test the Server

```python
# test_api.py
import requests

def test_chat_completion():
    """Test non-streaming chat completion."""
    url = "http://localhost:8000/v1/chat/completions"
    data = {
        "model": "rag-hybrid",
        "messages": [
            {"role": "user", "content": "What is this about?"}
        ],
        "stream": False
    }
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    
    result = response.json()
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert result["choices"][0]["message"]["content"]
    
    print("✅ Test passed!")
    print("Answer:", result["choices"][0]["message"]["content"])
    
    if result.get("citations"):
        print(f"\nFound {len(result['citations'])} citations")

if __name__ == "__main__":
    test_chat_completion()
```

Run the test:

```bash
python test_api.py
```

## 🔧 Advanced Features

### Custom System Prompts

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers based on provided documents. Always cite your sources."
    },
    {
        "role": "user",
        "content": "What is hybrid search?"
    }
]

response = client.chat.completions.create(
    model="rag-hybrid",
    messages=messages
)
```

### Temperature Control

```python
# More creative (higher temperature)
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=[{"role": "user", "content": "Explain hybrid search"}],
    temperature=0.9
)

# More focused (lower temperature)
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=[{"role": "user", "content": "What is the exact definition?"}],
    temperature=0.1
)
```

### Accessing Citations

```python
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=[{"role": "user", "content": "Your question"}]
)

# Note: Citations are in the raw response, not in the OpenAI client object
# Use requests library to get full response with citations
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "rag-hybrid",
        "messages": [{"role": "user", "content": "Your question"}]
    }
)

result = response.json()
if result.get("citations"):
    for citation in result["citations"]:
        print(f"Source: {citation['filename']}")
        print(f"Score: {citation['score']:.3f}")
        print(f"Preview: {citation['text_preview']}\n")
```

## 🎨 Integration Examples

### Integrate with Existing Applications

```python
# Your existing application code
from openai import OpenAI

# Simply change the base_url to point to your RAG server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"  # Point to RAG server
)

# All existing OpenAI code works as-is!
response = client.chat.completions.create(
    model="rag-hybrid",  # Use your RAG model
    messages=[{"role": "user", "content": "Your question"}]
)
```

### Build a Chat Interface

```python
import streamlit as st
from openai import OpenAI

st.title("RAG Chat with Hybrid Search")

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="rag-hybrid",
            messages=st.session_state.messages,
            stream=True
        )
        
        full_response = ""
        message_placeholder = st.empty()
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

## 📊 Performance

- **Hybrid Search**: Combines semantic + keyword matching for best results
- **Async Processing**: Non-blocking I/O for high concurrency
- **Caching**: Qdrant handles vector caching automatically
- **Streaming**: Reduces time-to-first-token

### Benchmarking

```bash
# Install apache bench
sudo apt-get install apache2-utils

# Test throughput
ab -n 100 -c 10 -p request.json -T application/json \
  http://localhost:8000/v1/chat/completions
```

## 🐛 Troubleshooting

### Server won't start

- Check that Qdrant is running: `docker ps`
- Verify port 8000 is not in use: `lsof -i :8000`
- Check environment variables are set correctly

### No documents indexed

- Verify `DOCUMENTS_PATH` exists and contains files
- Check file permissions
- Look for indexing logs in server output

### Hybrid search not working

- Install fastembed: `pip install fastembed`
- Verify `ENABLE_HYBRID=true` environment variable
- Check Qdrant collection supports sparse vectors

## 📝 API Comparison

This server implements the following OpenAI endpoints:

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/v1/chat/completions` | ✅ Implemented | Full support with RAG |
| `/v1/models` | ✅ Implemented | Lists available models |
| `/v1/embeddings` | ❌ Not implemented | Use embeddings service directly |
| `/v1/completions` | ❌ Not implemented | Use chat completions instead |

## 🚀 Production Deployment

### Recommendations

1. **Use HTTPS**: Deploy behind a reverse proxy (nginx, Caddy)
2. **Add Authentication**: Implement API key validation
3. **Rate Limiting**: Use middleware for rate limiting
4. **Monitoring**: Add logging and metrics (Prometheus, Grafana)
5. **Load Balancing**: Deploy multiple instances behind a load balancer
6. **Persistent Storage**: Mount Qdrant volumes for data persistence

### Example nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;  # Important for streaming
    }
}
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Fetchcraft Framework](../../../README.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

Same as the main Fetchcraft project.
