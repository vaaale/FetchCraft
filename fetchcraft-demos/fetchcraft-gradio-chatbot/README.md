# 🤖 RAG Chatbot with Hybrid Search - Gradio Web App

A modern, web-based chatbot interface powered by hybrid search (dense + sparse vectors) for superior document retrieval.

## 🌟 Features

- **🎨 Modern Web UI**: Clean, responsive Gradio interface
- **🔍 Hybrid Search**: Combines dense (semantic) + sparse (keyword) vectors
- **💬 Multi-turn Conversations**: Context-aware chat with memory
- **📚 Inline Citations**: Expandable source details within each bot response
- **📊 Citations Panel**: Separate panel showing all sources at a glance
- **⚡ Real-time Responses**: Async processing for fast interactions
- **🎯 Relevance Scores**: Color-coded confidence indicators
- **🔧 Configurable**: Environment variables for easy customization

## 📸 Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 RAG Chatbot with Hybrid Search                         │
├─────────────────────────────────┬───────────────────────────┤
│                                 │                           │
│  Chat History                   │  📚 Sources               │
│  ┌───────────────────────────┐  │  ┌─────────────────────┐ │
│  │ User: What is...?         │  │  │ [1] doc.pdf         │ │
│  │ Bot: Based on...          │  │  │     Score: 0.892    │ │
│  │                           │  │  │     Preview: ...    │ │
│  │ User: Tell me more        │  │  │                     │ │
│  │ Bot: According to...      │  │  │ [2] guide.txt       │ │
│  └───────────────────────────┘  │  │     Score: 0.745    │ │
│                                 │  └─────────────────────┘ │
│  [Ask a question...]            │                           │
│  [Send 📤]  [Clear 🗑️]         │                           │
└─────────────────────────────────┴───────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Qdrant running (vector database)
- OpenAI API key (or compatible endpoint)

### 2. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Install Dependencies

```bash
pip install gradio
pip install fastembed  # For hybrid search
```

### 4. Set Up Documents

Place your documents in a directory (default: `Documents/`):

```bash
mkdir Documents
cp your-documents/*.txt Documents/
```

### 5. Configure Environment

```bash
export OPENAI_API_KEY="your-api-key"
export DOCUMENTS_PATH="Documents"
export LLM_MODEL="gpt-4-turbo"
export EMBEDDING_MODEL="bge-m3"
```

### 6. Run the App

```bash
cd /path/to/fetchcraft
python -m demo.gradio_chatbot.app
```

The app will be available at: **http://localhost:7860**

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `COLLECTION_NAME` | Qdrant collection name | `fetchcraft_chatbot` |
| `DOCUMENTS_PATH` | Path to documents directory | `Documents` |
| `LLM_MODEL` | LLM model for chat | `gpt-4-turbo` |
| `EMBEDDING_MODEL` | Embedding model | `bge-m3` |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `OPENAI_BASE_URL` | Custom OpenAI endpoint | (optional) |
| `ENABLE_HYBRID` | Enable hybrid search | `true` |
| `FUSION_METHOD` | Fusion method (rrf/dbsf) | `rrf` |
| `CHUNK_SIZE` | Chunk size in characters | `8192` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `USE_HIERARCHICAL_CHUNKING` | Use hierarchical chunking | `true` |

### Example Configuration

```bash
# Use a local LLM endpoint
export OPENAI_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="mistral-7b"
export EMBEDDING_MODEL="bge-small-en-v1.5"

# Configure chunking
export CHUNK_SIZE="4096"
export CHUNK_OVERLAP="100"
export USE_HIERARCHICAL_CHUNKING="false"

# Hybrid search settings
export ENABLE_HYBRID="true"
export FUSION_METHOD="rrf"

# Custom documents path
export DOCUMENTS_PATH="/data/my-documents"
export COLLECTION_NAME="my-custom-collection"
```

## 📁 Project Structure

```
demo/gradio_chatbot/
├── app.py              # Main application
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## 🎯 Usage Guide

### Starting a Conversation

1. **Open the app** in your browser (http://localhost:7860)
2. **Type your question** in the message box
3. **Click "Send 📤"** or press Enter
4. **View the response** in the chat history
5. **Check citations** in the right panel

### Multi-turn Conversations

The chatbot maintains conversation context:

```
You: What is machine learning?
Bot: Machine learning is a subset of AI that...

You: What are its main types?
Bot: Based on our previous discussion, the main types are...
```

### Clearing History

Click the **"Clear 🗑️"** button to start a fresh conversation.

### Citations Panel

The citations panel shows:
- **Document name** and source path
- **Relevance score** (0.0 to 1.0)
- **Text preview** from the source
- **Color coding**:
  - 🟢 Green: High relevance (> 0.8)
  - 🟡 Yellow: Medium relevance (0.6 - 0.8)
  - ⚪ Gray: Lower relevance (< 0.6)

## 🔍 Hybrid Search Explained

### What is Hybrid Search?

Hybrid search combines two types of vectors:

1. **Dense Vectors** (Semantic)
   - Captures meaning and context
   - Good for: conceptual queries, synonyms
   - Example: "ML algorithm" matches "machine learning model"

2. **Sparse Vectors** (Keyword)
   - Exact keyword matching (BM25-style)
   - Good for: specific terms, model numbers, codes
   - Example: "GPT-4" matches exactly "GPT-4"

### Fusion Methods

**RRF (Reciprocal Rank Fusion)** - Default
- Balanced approach
- Combines rankings from both search types
- Good general-purpose choice

**DBSF (Distribution-Based Score Fusion)**
- Normalizes scores statistically
- Better for diverse result sets

### When to Use Hybrid Search

✅ **Use hybrid search for:**
- Technical documentation
- Code repositories
- Product manuals with model numbers
- Medical/legal documents with specific terms
- Mixed semantic + keyword queries

❌ **Dense-only might be sufficient for:**
- General knowledge questions
- Purely semantic queries
- Creative content

## 🛠️ Troubleshooting

### App Won't Start

**Error: Connection refused**
```bash
# Make sure Qdrant is running
docker run -p 6333:6333 qdrant/qdrant
```

**Error: No module named 'gradio'**
```bash
pip install gradio
```

**Error: fastembed required**
```bash
pip install fastembed
```

### No Documents Indexed

**Check documents path:**
```bash
ls $DOCUMENTS_PATH
# Should show .txt, .md, or other text files
```

**Re-index documents:**
```bash
# Delete the collection and restart
qdrant-cli collection delete fetchcraft_chatbot
python -m demo.gradio_chatbot.app
```

### Poor Results

**Try these solutions:**

1. **Adjust chunk size:**
   ```bash
   export CHUNK_SIZE="4096"  # Smaller chunks
   ```

2. **Enable hybrid search:**
   ```bash
   export ENABLE_HYBRID="true"
   ```

3. **Try different fusion method:**
   ```bash
   export FUSION_METHOD="dbsf"
   ```

4. **Use hierarchical chunking:**
   ```bash
   export USE_HIERARCHICAL_CHUNKING="true"
   ```

### API Key Issues

**Invalid API key:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

**Using local LLM:**
```bash
# Point to your local endpoint
export OPENAI_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="your-model-name"
```

## 🔒 Security Considerations

### API Keys

- Never commit API keys to version control
- Use environment variables or `.env` files
- Rotate keys regularly

### Network Access

By default, the app binds to `0.0.0.0` (all interfaces):

```python
# In app.py, change to localhost only:
interface.launch(
    server_name="127.0.0.1",  # Localhost only
    server_port=7860
)
```

### Document Access

- The chatbot can access ALL documents in `DOCUMENTS_PATH`
- Only index documents you want to be searchable
- Consider access controls for sensitive data

## 📊 Performance Tips

### Faster Indexing

```bash
# Use smaller chunks for faster indexing
export CHUNK_SIZE="2048"

# Disable hierarchical chunking
export USE_HIERARCHICAL_CHUNKING="false"
```

### Better Results

```bash
# Use larger chunks for more context
export CHUNK_SIZE="8192"

# Enable hierarchical chunking
export USE_HIERARCHICAL_CHUNKING="true"

# Increase overlap
export CHUNK_OVERLAP="400"
```

### Production Deployment

For production use:

1. **Use a reverse proxy** (nginx, Caddy)
2. **Enable HTTPS**
3. **Add authentication** (Gradio auth or external)
4. **Set up monitoring** (logs, metrics)
5. **Use persistent storage** for Qdrant
6. **Scale Qdrant** for high traffic

Example with Gradio auth:
```python
interface.launch(
    auth=("username", "password"),
    server_name="0.0.0.0",
    ssl_certfile="cert.pem",
    ssl_keyfile="key.pem"
)
```

## 🎨 Customization

### Change Theme

```python
# In app.py
interface = gr.Blocks(
    theme=gr.themes.Glass(),  # Or Soft(), Monochrome(), etc.
    ...
)
```

### Add More Tools

```python
# In setup_rag_system()
from fetchcraft import FileSearchTool

file_tool = FileSearchTool(root_path="./data")
tools = [
    Tool(retriever_tool.get_tool_function(), takes_ctx=True),
    Tool(file_tool.get_tool_function(), takes_ctx=True)
]
```

### Custom System Prompt

```python
# In setup_rag_system()
agent = ReActAgent.create(
    model=LLM_MODEL,
    tools=tools,
    system_prompt="""You are a helpful assistant specializing in...
    Your custom instructions here..."""
)
```

## 🧪 Development

### Running in Development Mode

```bash
# Enable debug logging
export LOG_LEVEL="DEBUG"

# Use in-memory Qdrant
export QDRANT_HOST=":memory:"

# Smaller dataset for testing
export DOCUMENTS_PATH="test_docs"
```

### Testing Locally

```bash
# Create test documents
mkdir test_docs
echo "Test document content" > test_docs/test.txt

# Run with test configuration
DOCUMENTS_PATH=test_docs python -m demo.gradio_chatbot.app
```

## 📚 Related Documentation

- [Hybrid Search Demo](../hybrid_demo/README.md) - CLI version
- [Agent Implementation](../../AGENT_IMPLEMENTATION.md)
- [Retriever Tool Usage](../../RETRIEVER_TOOL_UPDATE.md)
- [Main README](../../../README.md)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add user authentication
- [ ] Support for more document formats (PDF, DOCX, etc.)
- [ ] Conversation export/import
- [ ] Multi-index support
- [ ] Streaming responses
- [ ] Voice input/output
- [ ] Dark mode toggle
- [ ] Custom CSS themes

## 📝 License

Same as the parent project.

## 🙏 Acknowledgments

Built with:
- [Gradio](https://gradio.app/) - Web UI framework
- [Qdrant](https://qdrant.tech/) - Vector database
- [Pydantic AI](https://ai.pydantic.dev/) - Agent framework
- [FastEmbed](https://qdrant.github.io/fastembed/) - Sparse embeddings

## 💡 Tips & Tricks

### Best Questions to Ask

✅ **Good questions:**
- "What are the main features of X?"
- "How do I configure Y?"
- "Compare X and Y"
- "Give me examples of Z"

❌ **Less effective:**
- "Hi" / "Hello" (too vague)
- "Everything about X" (too broad)
- Questions about non-indexed content

### Using Citations

- Citations show **which documents** were used
- **Relevance scores** indicate confidence
- Click [number] references in responses (if enabled)
- Check sources for more context

### Conversation Tips

- Be specific in your questions
- Reference previous answers with "it", "this", "that"
- Ask follow-up questions naturally
- Clear conversation when switching topics

---

**Enjoy chatting with your documents! 🚀**
