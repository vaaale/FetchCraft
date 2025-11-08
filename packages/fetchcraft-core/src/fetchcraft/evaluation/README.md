# Evaluation Module

Comprehensive evaluation tools for testing retriever performance in RAG systems.

## Overview

This module provides:
- **DatasetGenerator**: Generate evaluation datasets using LLM-generated questions
- **RetrieverEvaluator**: Evaluate retriever performance with industry-standard metrics
- **EvaluationDataset**: Persistent dataset storage and management
- **EvaluationMetrics**: Comprehensive metrics including MRR, NDCG, Precision, Recall

## Quick Example

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from qdrant_client import QdrantClient

from fetchcraft.evaluation import (
    DatasetGenerator,
    RetrieverEvaluator,
    EvaluationDataset
)
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex

# 1. Setup stores and index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="my_collection",
    embeddings=embeddings
)

doc_store = MongoDBDocumentStore(
    connection_string="mongodb://localhost:27017",
    database_name="fetchcraft",
    collection_name="documents"
)

# 2. Generate dataset
model = OpenAIChatModel(
    model_name="gpt-4-turbo",
    provider=OpenAIProvider(api_key="...")
)

generator = DatasetGenerator(
    model=model,
    document_store=doc_store,
    vector_store=vector_store
)

dataset = await generator.generate_dataset(
    num_documents=10,
    questions_per_node=3
)
dataset.save("eval_dataset.json")

# 3. Evaluate retriever
vector_index = VectorIndex(vector_store=vector_store)
retriever = vector_index.as_retriever(top_k=5)

evaluator = RetrieverEvaluator(retriever=retriever)
metrics = await evaluator.evaluate(dataset)
print(metrics)
```

## Components

### DatasetGenerator

Generates question-answer pairs from indexed documents.

**Key Methods:**
- `generate_dataset()`: Sample documents and generate questions
- `generate_from_specific_nodes()`: Generate from specific node IDs
- `_generate_questions_for_node()`: LLM-based question generation

**Parameters:**
- `model`: Pydantic AI model (OpenAIChatModel) for question generation
- `document_store`: MongoDBDocumentStore for accessing full documents
- `vector_store`: QdrantVectorStore or ChromaVectorStore for accessing nodes/chunks
- `index_id`: Optional index identifier for filtering nodes

### RetrieverEvaluator

Evaluates retriever performance using standard IR metrics.

**Key Methods:**
- `evaluate()`: Run full evaluation on dataset
- `evaluate_with_different_k()`: Test multiple k values
- `get_failed_queries()`: Analyze failures
- `save_results()`: Save detailed results

**Metrics Calculated:**
- Hit Rate / Recall@k
- Mean Reciprocal Rank (MRR)
- Precision@k (at 1, 3, 5, k)
- Recall@k (at 1, 3, 5, k)
- NDCG@k
- Average Rank
- Rank Distribution

### EvaluationDataset

Container for question-answer pairs with persistence.

**Methods:**
- `save()`: Save to JSON file
- `load()`: Load from JSON file
- `__len__()`: Get number of pairs

**Structure:**
```python
from fetchcraft.evaluation import EvaluationDataset, QuestionContextPair

dataset = EvaluationDataset(
    qa_pairs=[
        QuestionContextPair(
            question="What is X?",
            node_id="node-123",
            context="X is...",
            metadata={"parsing": "doc.txt"}
        ),
        # ... more pairs
    ],
    metadata={
        "num_documents": 10,
        "total_pairs": 30,
        "model": "gpt-4-turbo"
    }
)
```

## Metrics Explained

### Hit Rate / Recall@k
Percentage of queries where the correct node appears in top-k results.
- **Good**: > 0.7
- **Excellent**: > 0.85

### Mean Reciprocal Rank (MRR)
Average of reciprocal ranks (1/rank) across all queries.
- **Formula**: `MRR = (1/n) * Σ(1/rank_i)`
- **Good**: > 0.5
- **Excellent**: > 0.7

### NDCG@k
Normalized Discounted Cumulative Gain - measures ranking quality.
- **Good**: > 0.6
- **Excellent**: > 0.75

### Precision@k
Proportion of retrieved documents that are relevant.
- For single-document relevance: either 0 or 1/k per query

## Workflow

### 1. Dataset Generation

Generate once, use multiple times for different retriever configurations:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from fetchcraft.evaluation import DatasetGenerator

# Initialize model for question generation
model = OpenAIChatModel(
    model_name="gpt-4-turbo",
    provider=OpenAIProvider(api_key="your-api-key")
)

# Create generator
generator = DatasetGenerator(
    model=model,
    document_store=doc_store,
    vector_store=vector_store
)

# Generate comprehensive dataset
dataset = await generator.generate_dataset(
    num_documents=50,
    questions_per_node=3,
    max_nodes_per_document=5,
    show_progress=True
)

# Save for reuse
dataset.save("my_eval_dataset.json")
```

### 2. Baseline Evaluation

Establish baseline performance:

```python
from fetchcraft.evaluation import RetrieverEvaluator
from fetchcraft.index.vector_index import VectorIndex

# Create baseline retriever
baseline_index = VectorIndex(vector_store=vector_store)
baseline_retriever = baseline_index.as_retriever(top_k=5)

# Evaluate
evaluator = RetrieverEvaluator(retriever=baseline_retriever)
baseline_metrics = await evaluator.evaluate(dataset, show_progress=True)
print(f"Baseline Hit Rate: {baseline_metrics.hit_rate:.2%}")
print(f"Baseline MRR: {baseline_metrics.mrr:.4f}")
```

### 3. Iterative Improvement

Test improvements against baseline:

```python
from fetchcraft.vector_store import QdrantVectorStore

# Test with hybrid search enabled
improved_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="improved_collection",
    embeddings=embeddings,
    enable_hybrid=True,  # Enable hybrid search
    fusion_method="rrf"
)

improved_index = VectorIndex(vector_store=improved_vector_store)
improved_retriever = improved_index.as_retriever(top_k=5)

# Evaluate improved version
evaluator = RetrieverEvaluator(retriever=improved_retriever)
improved_metrics = await evaluator.evaluate(dataset, show_progress=True)

# Compare
delta = improved_metrics.hit_rate - baseline_metrics.hit_rate
print(f"Improvement: {delta:+.2%}")
print(f"Hit Rate: {baseline_metrics.hit_rate:.2%} → {improved_metrics.hit_rate:.2%}")
```

### 4. Analysis

Investigate failures and optimize:

```python
# Find problematic queries
failed = evaluator.get_failed_queries()

for result in failed[:10]:
    print(f"Question: {result.question}")
    print(f"Expected: {result.expected_node_id}")
    print(f"Got: {result.retrieved_node_ids[:3]}")
    print()
```

## Best Practices

1. **Dataset Size**: 30-100+ documents for reliable metrics
2. **Question Quality**: Use capable models (GPT-4) for generation
3. **Reusability**: Generate dataset once, reuse for all tests
4. **Reproducibility**: Save datasets and results with timestamps
5. **Error Analysis**: Always examine failed queries for insights

## Examples

See:
- [evaluate_retriever.py](../../examples/evaluation/evaluate_retriever.py) - Complete workflow
- [test_evaluation.py](../../../tests/test_evaluation.py) - Unit tests
- [EVALUATION.md](../../../docs/EVALUATION.md) - Detailed guide

## Dependencies

Required:
- `pydantic-ai` - For question generation using LLM agents
- `openai` - OpenAI API client (or compatible)
- `tqdm` - Progress bars (optional but recommended)

Recommended:
- `motor` - For MongoDBDocumentStore
- `qdrant-client` - For QdrantVectorStore
- `chromadb` - For ChromaVectorStore

The module works with any DocumentStore and VectorStore implementation that follows the base interfaces.
