# Retriever Evaluation Guide

This guide explains how to evaluate retriever performance using the evaluation module.

## Overview

The evaluation module provides tools to:
1. **Generate evaluation datasets** from your indexed documents
2. **Evaluate retriever performance** using comprehensive metrics
3. **Analyze results** to improve your RAG system

## Key Components

### DatasetGenerator

Generates question-answer pairs from indexed documents using an LLM.

**Features:**
- Samples documents from DocumentStore and VectorStore
- Fetches top-level nodes for each document
- Generates multiple questions per node using OpenAI
- Creates structured datasets with node IDs as ground truth

### RetrieverEvaluator

Evaluates retriever performance using standard IR metrics.

**Metrics Calculated:**
- **Hit Rate / Recall@k**: Proportion of queries where correct node was retrieved
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of correct results
- **Precision@k**: Precision at various cutoffs (1, 3, 5, k)
- **Recall@k**: Recall at various cutoffs
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Average Rank**: Mean position of correct results
- **Rank Distribution**: Histogram of where correct results appear

## Quick Start

### 1. Generate Dataset

```python
from openai import AsyncOpenAI
from fetchcraft import DatasetGenerator

# Initialize OpenAI client
client = AsyncOpenAI(api_key="your-key")

# Create generator
generator = DatasetGenerator(
    client=client,
    document_store=doc_store,
    vector_store=vector_store,
    model="gpt-4",
    index_id="my-index"
)

# Generate dataset
dataset = await generator.generate_dataset(
    num_documents=10,           # Sample 10 documents
    questions_per_node=3,        # 3 questions per node
    max_nodes_per_document=5,    # Use up to 5 nodes per doc
    show_progress=True
)

# Save for later use
dataset.save("eval_dataset.json")
```

### 2. Evaluate Retriever

```python
from fetchcraft import RetrieverEvaluator, EvaluationDataset

# Load dataset
dataset = EvaluationDataset.load("eval_dataset.json")

# Create evaluator
evaluator = RetrieverEvaluator(retriever=my_retriever)

# Run evaluation
metrics = await evaluator.evaluate(dataset, show_progress=True)

# Print results
print(metrics)

# Save detailed results
evaluator.save_results("results.json")
```

## Understanding Metrics

### Hit Rate / Recall@k
- **Definition**: Percentage of queries where the correct node appears in top-k results
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - 0.80 = Correct answer found 80% of the time
  - Essential baseline metric for retrieval quality

### Mean Reciprocal Rank (MRR)
- **Definition**: Average of 1/rank for each query
- **Range**: 0.0 to 1.0 (higher is better)
- **Formula**: `MRR = (1/n) * Σ(1/rank_i)`
- **Interpretation**:
  - 1.0 = Always returns correct result at rank 1
  - 0.5 = Average rank of 2
  - 0.33 = Average rank of 3
  - Penalizes lower-ranked correct results

### Precision@k
- **Definition**: Proportion of retrieved docs that are relevant
- **Range**: 0.0 to 1.0 (higher is better)
- **For single relevant doc**: Either 0 or 1/k

### NDCG@k
- **Definition**: Normalized Discounted Cumulative Gain
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Measures ranking quality with position-based discounting
- **Formula**: `NDCG = DCG / IDCG`
  - DCG penalizes relevant docs appearing lower in ranking
  - Higher ranks contribute more to the score

### Average Rank
- **Definition**: Mean position of correct results (when found)
- **Range**: 1 to k (lower is better)
- **Interpretation**: 
  - 1.0 = Always at top position
  - 2.5 = Average between positions 2-3

## Advanced Usage

### Compare Different k Values

```python
evaluator = RetrieverEvaluator(retriever=my_retriever)

k_results = await evaluator.evaluate_with_different_k(
    dataset=dataset,
    k_values=[1, 3, 5, 10, 20],
    show_progress=True
)

# Compare performance
for k, metrics in k_results.items():
    print(f"k={k}: Hit Rate={metrics.hit_rate:.3f}, MRR={metrics.mrr:.3f}")
```

### Analyze Failed Queries

```python
# Get all failed queries
failed = evaluator.get_failed_queries()

for result in failed[:5]:
    print(f"Question: {result.question}")
    print(f"Expected: {result.expected_node_id}")
    print(f"Retrieved: {result.retrieved_node_ids}")
    print()
```

### Generate from Specific Nodes

```python
# Generate questions for specific nodes only
node_ids = ["node-1", "node-2", "node-3"]

dataset = await generator.generate_from_specific_nodes(
    node_ids=node_ids,
    questions_per_node=5,
    show_progress=True
)
```

## Best Practices

### Dataset Generation

1. **Sample Size**
   - Start with 10-20 documents for quick iteration
   - Use 50-100+ documents for comprehensive evaluation
   - More questions per node = more reliable metrics

2. **Node Selection**
   - Focus on top-level nodes (parent chunks)
   - Ensure nodes have sufficient context (>100 tokens)
   - Mix different document types/topics

3. **Question Quality**
   - Use a capable model (GPT-4) for question generation
   - Review sample questions to ensure quality
   - Filter out overly simple or ambiguous questions

### Evaluation

1. **Baseline Metrics**
   - Hit Rate@5 > 0.7: Good retrieval
   - MRR > 0.5: Good ranking quality
   - NDCG@5 > 0.6: Acceptable ranking

2. **Iteration**
   - Generate dataset once, evaluate multiple configurations
   - Test different: embedding models, chunk sizes, retrieval strategies
   - Track metrics over time as you improve the system

3. **Error Analysis**
   - Always examine failed queries
   - Look for patterns in failures (topic, length, complexity)
   - Use insights to improve chunking or retrieval

## Example Workflow

```python
import asyncio
from openai import AsyncOpenAI
from fetchcraft import (
    DatasetGenerator,
    RetrieverEvaluator,
    EvaluationDataset
)

async def evaluate_retriever_pipeline():
    # 1. Generate dataset (do this once)
    openai_client = AsyncOpenAI(api_key="...")
    generator = DatasetGenerator(
        client=openai_client,
        document_store=doc_store,
        vector_store=vector_store,
        model="gpt-4"
    )
    
    dataset = await generator.generate_dataset(
        num_documents=20,
        questions_per_node=3,
        show_progress=True
    )
    dataset.save("eval_dataset.json")
    
    # 2. Evaluate baseline retriever
    evaluator = RetrieverEvaluator(retriever=baseline_retriever)
    baseline_metrics = await evaluator.evaluate(dataset)
    print("Baseline:", baseline_metrics.hit_rate, baseline_metrics.mrr)
    
    # 3. Evaluate improved retriever
    evaluator = RetrieverEvaluator(retriever=improved_retriever)
    improved_metrics = await evaluator.evaluate(dataset)
    print("Improved:", improved_metrics.hit_rate, improved_metrics.mrr)
    
    # 4. Compare
    improvement = improved_metrics.hit_rate - baseline_metrics.hit_rate
    print(f"Hit Rate Improvement: {improvement:+.2%}")

asyncio.run(evaluate_retriever_pipeline())
```

## Interpreting Results

### Good Performance
```
Hit Rate@5: 0.85 (85%)
MRR: 0.72
NDCG@5: 0.78
Average Rank: 1.4
```
- Finds correct answer 85% of the time
- Usually in top 1-2 positions
- Strong ranking quality

### Poor Performance
```
Hit Rate@5: 0.45 (45%)
MRR: 0.28
NDCG@5: 0.32
Average Rank: 3.8
```
- Misses correct answer 55% of the time
- Low ranking positions
- Needs improvement in retrieval strategy

## Troubleshooting

### Low Hit Rate
- **Check**: Are chunks too small/large?
- **Try**: Adjust chunk size and overlap
- **Try**: Enable hybrid search (dense + sparse)
- **Try**: Different embedding model

### Good Hit Rate, Low MRR
- **Issue**: Correct results retrieved but poorly ranked
- **Try**: Adjust fusion method (RRF vs DBSF)
- **Try**: Tune score combination weights
- **Try**: Re-ranking with cross-encoder

### High Variance
- **Issue**: Inconsistent performance across queries
- **Check**: Dataset quality and diversity
- **Try**: Larger evaluation dataset
- **Try**: Stratified sampling by topic/type

## See Also

- [Retriever Usage Guide](RETRIEVER_USAGE.md)
- [Chunking Strategies](CHUNKING_STRATEGIES.md)
- [Example: evaluate_retriever.py](../src/examples/evaluate_retriever.py)
