# 🚀 Quick Start Guide - Gradio Chatbot

Get the chatbot running in 5 minutes!

## Prerequisites

- Python 3.10+
- Docker (for Qdrant)

## Step-by-Step Setup

### 1. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Leave this running in a terminal.

### 2. Install Gradio

```bash
pip install gradio fastembed
```

### 3. Set Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key-here"

# Optional - customize if needed
export DOCUMENTS_PATH="Documents"
export LLM_MODEL="gpt-4-turbo"
export ENABLE_HYBRID="true"
```

Or copy and edit the `.env` file:

```bash
cd src/demo/gradio_chatbot
cp .env.example .env
# Edit .env with your settings
```

### 4. Prepare Documents

```bash
# Create documents directory
mkdir -p Documents

# Add your documents
cp /path/to/your/docs/*.txt Documents/
```

### 5. Launch the App

**Option A: Using Python**
```bash
cd /path/to/MyIndex
python -m demo.gradio_chatbot.app
```

**Option B: Using the launch script**
```bash
cd src/demo/gradio_chatbot
chmod +x launch.sh
./launch.sh
```

### 6. Open in Browser

Navigate to: **http://localhost:7860**

## What You'll See

1. **First Run**: The system will index your documents (may take a few minutes)
2. **Subsequent Runs**: Will use the existing index (instant startup)
3. **Web Interface**: Chat interface with citations panel

## Example Usage

### Try These Questions

```
Q: What topics are covered in the documents?
Q: Tell me about [specific topic]
Q: Compare X and Y
Q: Give me examples of Z
```

### Follow-up Questions

```
Q: What is machine learning?
A: [Answer with citations]

Q: What are its main applications?
A: [Contextual answer based on previous question]
```

## Troubleshooting

### "Connection refused"

Qdrant isn't running. Start it:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### "No documents indexed"

Check your documents path:
```bash
ls $DOCUMENTS_PATH  # Should show files
```

### "Module not found"

Install missing packages:
```bash
pip install gradio fastembed
```

### "Invalid API key"

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Using Local LLM

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="your-model-name"
```

## Configuration Tips

### Faster Indexing

```bash
export CHUNK_SIZE="2048"
export USE_HIERARCHICAL_CHUNKING="false"
```

### Better Results

```bash
export CHUNK_SIZE="8192"
export USE_HIERARCHICAL_CHUNKING="true"
export ENABLE_HYBRID="true"
```

### Different Collection

To use a different collection (new index):
```bash
export COLLECTION_NAME="my-custom-collection"
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [environment variables](README.md#environment-variables)
- Learn about [hybrid search](README.md#hybrid-search-explained)
- Customize the [interface](README.md#customization)

## Common Commands

```bash
# View running containers
docker ps

# Stop Qdrant
docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)

# Clear the collection (re-index)
# Delete collection in Qdrant UI: http://localhost:6333/dashboard

# Update dependencies
pip install --upgrade gradio fastembed
```

## Support

- Check logs for detailed error messages
- See [Troubleshooting](README.md#troubleshooting) section
- Review configuration in the web UI's info accordion

---

**Ready to chat! 🎉**
