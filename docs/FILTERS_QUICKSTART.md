# Metadata Filters - Quick Start Guide

## Installation

No additional installation needed - filters are built into Fetchcraft!

```python
from fetchcraft import EQ, AND, OR, GTE, IN
```

## 5-Minute Quick Start

### 1. Basic Filter

```python
from fetchcraft import EQ
from fetchcraft.index.vector_index import VectorIndex

# Create your index (assuming it's already set up)
index = VectorIndex(vector_store=vector_store)

# Search with a simple filter
results = await index.search_by_text(
    "machine learning tutorials",
    k=10,
    filters=EQ("category", "tutorial")  # Only return tutorials
)
```

### 2. Numeric Range Filter

```python
from fetchcraft import GTE, LTE, AND

# Find articles from 2023-2024
results = await index.search_by_text(
    "AI developments",
    k=10,
    filters=AND(
        GTE("year", 2023),
        LTE("year", 2024)
    )
)
```

### 3. Multiple Values Filter

```python
from fetchcraft import IN

# Find content in any of these languages
results = await index.search_by_text(
    "programming tutorials",
    k=10,
    filters=IN("language", ["python", "javascript", "rust"])
)
```

### 4. Complex Nested Filter

```python
from fetchcraft import AND, OR, EQ, GTE

# (category == "tutorial" OR category == "guide") AND year >= 2023
filter = AND(
    OR(
        EQ("category", "tutorial"),
        EQ("category", "guide")
    ),
    GTE("year", 2023)
)

results = await index.search_by_text("learning resources", k=10, filters=filter)
```

### 5. Using with Retriever (with Default Filters)

```python
from fetchcraft import EQ

# Create a retriever with default filters in constructor
retriever = index.as_retriever(
    top_k=5,
    filters=EQ("status", "published")  # Default filter for ALL queries
)

# Use it - default filters are applied automatically
results = retriever.retrieve("machine learning")
# All results will have status == "published"

# Or override filters per query
results = retriever.retrieve(
    "machine learning",
    filters=EQ("difficulty", "beginner")  # Override default filter
)
# Now results will have difficulty == "beginner" (status filter is replaced)

# Without default filters (pass filters each time)
retriever = index.as_retriever(top_k=5)  # No default filter
results = retriever.retrieve("ML", filters=EQ("category", "tutorial"))
```

## All Available Operators

```python
from fetchcraft import EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS

# Equality
EQ("key", "value")  # key == value
NE("key", "value")  # key != value

# Comparison
GT("score", 0.8)  # score > 0.8
GTE("score", 0.8)  # score >= 0.8
LT("age", 18)  # age < 18
LTE("age", 18)  # age <= 18

# Collections
IN("tag", ["ai", "ml", "dl"])  # tag in list

# Text
CONTAINS("title", "machine")  # title contains "machine"
```

## All Logical Conditions

```python
from fetchcraft import AND, OR, NOT

# AND - all must match
AND(filter1, filter2, filter3)

# OR - any must match
OR(filter1, filter2, filter3)

# NOT - negate condition
NOT(filter1)

# Nested
AND(
    OR(filter1, filter2),
    filter3
)
```

## Common Patterns

### Filter by Date Range

```python
from fetchcraft import AND, GTE, LTE

filters = AND(
    GTE("published_date", "2024-01-01"),
    LTE("published_date", "2024-12-31")
)
```

### Exclude Content

```python
from fetchcraft import NE

filters = NE("status", "draft")  # Exclude drafts
```

### Multiple Categories

```python
from fetchcraft import OR, EQ

filters = OR(
    EQ("category", "ai"),
    EQ("category", "ml"),
    EQ("category", "deep-learning")
)
```

### Advanced Business Logic

```python
from fetchcraft import AND, OR, EQ, GTE, NE

filters = AND(
    EQ("tier", "premium"),  # Premium content
    GTE("year", 2023),  # Recent
    NE("status", "archived"),  # Not archived
    OR(  # Tutorial or guide
        EQ("type", "tutorial"),
        EQ("type", "guide")
    )
)
```

## Where Can I Use Filters?

### 1. Direct Search on Index
```python
results = await index.search_by_text(query, k=10, filters=my_filter)
```

### 2. With Retriever (per-query filters)
```python
retriever = index.as_retriever(top_k=5)
results = retriever.retrieve(query, filters=my_filter)
```

### 3. Default Filters in Retriever Constructor ⭐
```python
# Set default filters when creating the retriever
retriever = index.as_retriever(
    top_k=5, 
    filters=eq("status", "published")  # Applied to ALL queries
)

# Use without specifying filters each time
results = retriever.retrieve(query)  # Default filters applied automatically

# Or override for specific queries
results = retriever.retrieve(query, filters=eq("category", "tutorial"))
```

### 4. With Vector Store Directly
```python
results = await vector_store.search_by_text(
    query,
    k=10,
    filters=my_filter
)
```

## Tips & Best Practices

1. **Start Simple**: Begin with single field filters, then add complexity
2. **Index Metadata**: Ensure the fields you filter on are indexed in your vector store
3. **Test Filters**: Use specific values to verify filters work as expected
4. **Combine with Semantic Search**: Filters refine results, don't replace semantic search
5. **Document Schema**: Keep a consistent metadata schema across your documents

## Troubleshooting

### No Results Returned
- Check field names match exactly (case-sensitive)
- Verify values exist in your documents
- Test without filters first to ensure documents exist

### Slow Queries
- Ensure metadata fields are indexed
- Simplify complex nested filters
- Consider pre-filtering large datasets

### Type Errors
- Use appropriate operators for data types (e.g., GT/LT for numbers, not strings)
- Ensure IN receives a list
- Use boolean values (True/False), not strings ("true"/"false")

## Next Steps

- See [METADATA_FILTERS.md](METADATA_FILTERS.md) for complete documentation
- Check [filter_example.py](../examples/filter_example.py) for working examples
- Run tests: `pytest tests/test_filters.py -v`

## Examples

Full working examples are available in:
- `src/examples/filter_example.py` - Comprehensive examples
- `tests/test_filters.py` - Test suite with examples
