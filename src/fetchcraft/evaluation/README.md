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
from openai import AsyncOpenAI
from fetchcraft.evaluation import (
    DatasetGenerator,
    RetrieverEvaluator,
    EvaluationDataset
)
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.vector_store import QdrantVectorStore

# 1. Generate dataset
client = AsyncOpenAI(api_key="...")
doc_store = MongoDBDocumentStore(
    client=client,
    database="fetchcraft"
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="fetchcraft"
)
generator = DatasetGenerator(
    client=client,
    document_store=doc_store,
    vector_store=vector_store
)

dataset = await generator.generate_dataset(
    num_documents=10,
    questions_per_node=3
)
dataset.save("eval_dataset.json")

# 2. Evaluate retriever
evaluator = RetrieverEvaluator(retriever=my_retriever)
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
- `client`: OpenAI AsyncClient for question generation
- `document_store`: Source of documents
- `vector_store`: Source of nodes/chunks
- `model`: LLM model (default: "gpt-4")
- `index_id`: Optional index identifier

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
dataset = EvaluationDataset(
    qa_pairs=[
        QuestionAnswerPair(
            question="What is X?",
            node_id="node-123",
            context="X is...",
            metadata={"source": "doc.txt"}
        ),
        # ... more pairs
    ],
    metadata={
        "num_documents": 10,
        "total_pairs": 30,
        "model": "gpt-4"
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
evaluator = RetrieverEvaluator(retriever=baseline_retriever)
baseline_metrics = await evaluator.evaluate(dataset)
print(f"Baseline Hit Rate: {baseline_metrics.hit_rate:.2%}")
```

### 3. Iterative Improvement

Test improvements against baseline:

```python
# Test with hybrid search
improved_retriever.enable_hybrid = True
evaluator = RetrieverEvaluator(retriever=improved_retriever)
improved_metrics = await evaluator.evaluate(dataset)

# Compare
delta = improved_metrics.hit_rate - baseline_metrics.hit_rate
print(f"Improvement: {delta:+.2%}")
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
- `openai` - For question generation
- `tqdm` - Progress bars (optional)

The module works with any DocumentStore and VectorStore implementation.
