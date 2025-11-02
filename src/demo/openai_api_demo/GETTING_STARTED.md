# Getting Started with FastAPI RAG Demo

## 🎯 What You'll Learn

In this guide, you'll:
1. Start a production-ready RAG API server
2. Query it using OpenAI-compatible clients
3. Integrate it into your applications
4. Deploy it with Docker

## ⚡ Quick Start (60 seconds)

### Option 1: One-Command Start (Easiest)

```bash
# From the repository root
./src/demo/openai_api_demo/quickstart.sh
```

This automatically:
- ✅ Starts Qdrant if not running
- ✅ Creates sample documents
- ✅ Installs dependencies
- ✅ Starts the server

### Option 2: Manual Start

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Install dependencies
pip install fastapi uvicorn pydantic-ai qdrant-client openai fastembed

# 3. Create sample document
mkdir -p Documents
echo "Sample text about hybrid search..." > Documents/sample.txt

# 4. Start server
python -m demo.openai_api_demo.server
```

The server starts at: **http://localhost:8000**

## 🧪 Test It

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "hybrid_search_enabled": true,
  "fusion_method": "rrf"
}
```

### Test 2: Simple Query

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-hybrid",
    "messages": [{"role": "user", "content": "What is hybrid search?"}],
    "stream": false
  }'
```

### Test 3: Run Example Client

```bash
python -m demo.openai_api_demo.client_example
```

This will run 4 examples:
1. Simple non-streaming query
2. Streaming response
3. Multi-turn conversation
4. Getting citations

## 📱 Use It in Your App

### Python with OpenAI SDK

```python
from openai import OpenAI

# Point to your local RAG server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Use exactly like OpenAI API
response = client.chat.completions.create(
    model="rag-hybrid",
    messages=[
        {"role": "user", "content": "Your question here"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/TypeScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'not-needed',
  baseURL: 'http://localhost:8000/v1'
});

const response = await client.chat.completions.create({
  model: 'rag-hybrid',
  messages: [
    { role: 'user', content: 'Your question here' }
  ]
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-hybrid",
    "messages": [
      {"role": "user", "content": "Your question"}
    ]
  }'
```

## 🎨 Common Use Cases

### 1. Chat Interface (Streamlit)

```python
import streamlit as st
from openai import OpenAI

st.title("RAG Chat")

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
        placeholder = st.empty()
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })
```

### 2. API Gateway Pattern

```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
rag_client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

@app.post("/ask")
async def ask_question(question: str):
    """Your custom API that uses RAG internally."""
    response = rag_client.chat.completions.create(
        model="rag-hybrid",
        messages=[{"role": "user", "content": question}]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "model": "rag-hybrid"
    }
```

### 3. Batch Processing

```python
from openai import OpenAI
import asyncio

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

questions = [
    "What is hybrid search?",
    "How does RAG work?",
    "What are the benefits?"
]

# Process all questions
for i, question in enumerate(questions, 1):
    print(f"\nQuestion {i}: {question}")
    
    response = client.chat.completions.create(
        model="rag-hybrid",
        messages=[{"role": "user", "content": question}]
    )
    
    print(f"Answer: {response.choices[0].message.content}\n")
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Documents
DOCUMENTS_PATH=./Documents

# Hybrid Search
ENABLE_HYBRID=true
FUSION_METHOD=rrf

# Models
EMBEDDING_MODEL=bge-m3
LLM_MODEL=gpt-4-turbo

# Server
HOST=0.0.0.0
PORT=8000
```

Then load it:

```bash
export $(cat .env | xargs)
python -m demo.openai_api_demo.server
```

### Custom Configuration

```bash
# Use custom documents directory
DOCUMENTS_PATH=/path/to/docs python -m demo.openai_api_demo.server

# Use different embedding model
EMBEDDING_MODEL=text-embedding-3-small python -m demo.openai_api_demo.server

# Disable hybrid search (dense only)
ENABLE_HYBRID=false python -m demo.openai_api_demo.server
```

## 🐳 Docker Deployment

### Quick Docker Start

```bash
cd src/demo/openai_api_demo
docker-compose up
```

This starts:
- Qdrant (vector database)
- RAG API server

Both services with health checks and proper networking.

### Access Services

- **RAG API**: http://localhost:8000
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Just the API
docker-compose logs -f rag-api

# Just Qdrant
docker-compose logs -f qdrant
```

## 🔍 Troubleshooting

### Issue: Server won't start

**Check Qdrant is running:**
```bash
curl http://localhost:6333/health
```

If not:
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### Issue: No documents found

**Check Documents directory:**
```bash
ls -la Documents/
```

Create sample documents:
```bash
mkdir -p Documents
echo "Your content here" > Documents/sample.txt
```

### Issue: Import errors

**Install all dependencies:**
```bash
pip install -r src/demo/openai_api_demo/requirements.txt
```

Or manually:
```bash
pip install fastapi uvicorn pydantic-ai qdrant-client openai fastembed
```

### Issue: Port 8000 already in use

**Use a different port:**
```bash
PORT=9000 python -m demo.openai_api_demo.server
```

Then access at http://localhost:9000

### Issue: Slow first query

**Expected behavior:**
- First query builds the index (slower)
- Subsequent queries use cached index (faster)
- Wait time depends on document count

## 📊 Performance Tips

### 1. Pre-build Index

Start server once to build index, then restart for fast queries:

```bash
# First run - builds index
python -m demo.openai_api_demo.server
# Wait for "System Ready" message, then Ctrl+C

# Subsequent runs - uses cached index
python -m demo.openai_api_demo.server
```

### 2. Optimize Chunk Size

Smaller chunks = more precise but slower:
```bash
CHUNK_SIZE=2048 python -m demo.openai_api_demo.server
```

Larger chunks = faster but less precise:
```bash
CHUNK_SIZE=16384 python -m demo.openai_api_demo.server
```

### 3. Use Appropriate Models

**For speed:**
```bash
EMBEDDING_MODEL=bge-small
LLM_MODEL=gpt-3.5-turbo
```

**For quality:**
```bash
EMBEDDING_MODEL=bge-m3
LLM_MODEL=gpt-4-turbo
```

## 🎓 Next Steps

### Learn More

1. **Read the full documentation**: [README.md](README.md)
2. **Study the code**: [server.py](server.py)
3. **Try examples**: [client_example.py](client_example.py)

### Customize

1. **Add authentication** (see README.md)
2. **Add rate limiting** (see README.md)
3. **Add custom prompts**
4. **Integrate with your UI**

### Deploy

1. **Docker**: Use provided Dockerfile
2. **Docker Compose**: Use docker-compose.yml
3. **Cloud**: Deploy to AWS, GCP, Azure
4. **Reverse Proxy**: Put behind nginx

## 💡 Tips

### Development

- Use `reload=True` for auto-reload during development:
  ```python
  uvicorn.run("demo.openai_api_demo.server:app", reload=True)
  ```

### Production

- Use multiple workers for better performance:
  ```bash
  uvicorn demo.openai_api_demo.server:app --workers 4
  ```

- Enable access logs:
  ```bash
  uvicorn demo.openai_api_demo.server:app --access-log
  ```

### Monitoring

- Check health endpoint periodically:
  ```bash
  while true; do curl http://localhost:8000/health; sleep 10; done
  ```

## 🤝 Get Help

- **Documentation**: See [README.md](README.md)
- **Examples**: Run [client_example.py](client_example.py)
- **Issues**: Check error messages and logs
- **Community**: Open an issue on GitHub

## ✨ Summary

You now have a production-ready RAG API that:

✅ Works with any OpenAI client  
✅ Supports streaming responses  
✅ Uses hybrid search for better results  
✅ Returns citations with answers  
✅ Can be deployed with Docker  
✅ Is fully customizable

**Start building!** 🚀
