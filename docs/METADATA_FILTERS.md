# Metadata Filtering

Fetchcraft provides a powerful metadata filtering system that allows you to refine search results based on document metadata. Filters are defined using a simple, expressive API and are automatically translated to native vector store filter formats.

## Features

- **Rich Operators**: EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS
- **Logical Conditions**: AND, OR, NOT
- **Nested Filters**: Combine filters for complex queries
- **Type-Safe**: Pydantic-based filter classes with validation
- **Vector Store Agnostic**: Filters are translated to native formats (Qdrant, Pinecone, etc.)

## Basic Usage

```python
from fetchcraft import EQ, GT, IN, CONTAINS
from fetchcraft.index.vector_index import VectorIndex

# Create a retriever with filters
retriever = index.as_retriever(top_k=5)

# Search with a simple equality filter
results = retriever.retrieve(
    "machine learning",
    filters=EQ("category", "ai")
)

# Search with a range filter
results = retriever.retrieve(
    "recent articles",
    filters=GT("year", 2022)
)

# Search with an IN filter
results = retriever.retrieve(
    "programming tutorials",
    filters=IN("language", ["python", "javascript", "rust"])
)

# Search with a text contains filter
results = retriever.retrieve(
    "type systems",
    filters=CONTAINS("title", "TypeScript")
)
```

## Filter Operators

### Equality Operators

#### EQ (Equal)

```python
from fetchcraft import EQ

# Match exact value
filter = EQ("status", "active")
filter = EQ("count", 42)
filter = EQ("is_published", True)
```

#### NE (Not Equal)

```python
from fetchcraft import NE

# Match anything except this value
filter = NE("status", "archived")
filter = NE("priority", "low")
```

### Comparison Operators

#### GT (Greater Than)

```python
from fetchcraft import GT

# Match values greater than threshold
filter = GT("year", 2020)
filter = GT("score", 0.85)
filter = GT("price", 99.99)
```

#### GTE (Greater Than or Equal)

```python
from fetchcraft import GTE

# Match values greater than or equal to threshold
filter = GTE("year", 2023)
filter = GTE("rating", 4.0)
```

#### LT (Less Than)

```python
from fetchcraft import LT

# Match values less than threshold
filter = LT("age", 18)
filter = LT("file_size", 1000000)
```

#### LTE (Less Than or Equal)

```python
from fetchcraft import LTE

# Match values less than or equal to threshold
filter = LTE("year", 2023)
filter = LTE("price", 50.0)
```

### Collection Operators

#### IN (In List)

```python
from fetchcraft import IN

# Match any value in the list
filter = IN("category", ["tech", "science", "engineering"])
filter = IN("status", ["active", "pending"])
filter = IN("priority", [1, 2, 3])
```

### String Operators

#### CONTAINS (Text Contains)

```python
from fetchcraft import CONTAINS

# Match documents where field contains text (case-insensitive)
filter = CONTAINS("title", "machine learning")
filter = CONTAINS("description", "tutorial")
```

## Composite Filters

### AND Condition

Combine multiple filters where all must match:

```python
from fetchcraft import AND, EQ, GTE

# Match documents that satisfy ALL conditions
filter = AND(
    EQ("category", "ai"),
    GTE("year", 2023),
    EQ("language", "python")
)

results = retriever.retrieve("deep learning", filters=filter)
```

### OR Condition

Combine multiple filters where any can match:

```python
from fetchcraft import OR, EQ

# Match documents that satisfy ANY condition
filter = OR(
    EQ("category", "ai"),
    EQ("category", "ml"),
    EQ("category", "data-science")
)

results = retriever.retrieve("neural networks", filters=filter)
```

### NOT Condition

Negate a filter:

```python
from fetchcraft import NOT, EQ

# Match documents that do NOT satisfy the condition
filter = NOT(EQ("status", "archived"))

results = retriever.retrieve("articles", filters=filter)
```

## Nested Filters

Combine conditions for complex queries:

```python
from fetchcraft import AND, OR, EQ, GTE, LTE

# Complex nested filter:
# (category == "ai" AND year >= 2023) OR (category == "ml" AND difficulty == "beginner")
filter = OR(
    AND(
        EQ("category", "ai"),
        GTE("year", 2023)
    ),
    AND(
        EQ("category", "ml"),
        EQ("difficulty", "beginner")
    )
)

results = retriever.retrieve("machine learning", filters=filter)
```

Another example with date ranges:

```python
from fetchcraft import AND, OR, EQ, GTE, LTE

# Articles from 2023-2024 about Python or JavaScript
filter = AND(
    GTE("year", 2023),
    LTE("year", 2024),
    OR(
        EQ("language", "python"),
        EQ("language", "javascript")
    )
)
```

## Using with Different Components

### With VectorIndex

```python
from fetchcraft import EQ

# Direct search with filters
results = await index.search_by_text(
    "machine learning",
    k=10,
    filters=EQ("category", "tutorial")
)
```

### With Retriever

```python
from fetchcraft import AND, EQ, GTE

# Option 1: Pass filters per query
retriever = index.as_retriever(top_k=5)
results = retriever.retrieve(
    "deep learning",
    filters=AND(
        EQ("category", "ai"),
        GTE("year", 2023)
    )
)

# Option 2: Set default filters in constructor (recommended for common filters)
retriever = index.as_retriever(
    top_k=5,
    filters=EQ("status", "published")  # Applied to all queries by default
)
results = retriever.retrieve("deep learning")  # Default filter applied

# Option 3: Override default filters for specific queries
results = retriever.retrieve(
    "deep learning",
    filters=GTE("year", 2024)  # Overrides default filter
)
```

### With Async Retrieval

```python
from fetchcraft import IN

results = await retriever.aretrieve(
    "programming",
    filters=IN("language", ["python", "rust", "go"])
)
```

## Advanced Usage

### Using Class-Based API

If you prefer explicit class instantiation:

```python
from fetchcraft.filters import (
    FieldFilter, 
    CompositeFilter, 
    FilterOperator, 
    FilterCondition
)

# Create filters using classes
field_filter = FieldFilter(
    key="category",
    operator=FilterOperator.EQ,
    value="ai"
)

composite_filter = CompositeFilter(
    condition=FilterCondition.AND,
    filters=[
        FieldFilter(key="year", operator=FilterOperator.GTE, value=2023),
        FieldFilter(key="status", operator=FilterOperator.EQ, value="published")
    ]
)
```

### Filter Representation

Filters can be converted to dictionaries for inspection or serialization:

```python
from fetchcraft import AND, EQ, GTE

filter = AND(
    EQ("category", "ai"),
    GTE("year", 2023)
)

# Convert to dict
filter_dict = filter.to_dict()
print(filter_dict)
# {
#     "type": "composite",
#     "condition": "AND",
#     "filters": [
#         {"type": "field", "key": "category", "operator": "==", "value": "ai"},
#         {"type": "field", "key": "year", "operator": ">=", "value": 2023}
#     ]
# }
```

## Vector Store Translation

Filters are automatically translated to native vector store formats:

### Qdrant

Fetchcraft filters are translated to Qdrant's filter format:

```python
# Fetchcraft filter
from fetchcraft import AND, EQ, GTE

filter = AND(
    EQ("category", "ai"),
    GTE("year", 2023)
)

# Automatically translated to Qdrant format:
# models.Filter(
#     must=[
#         models.FieldCondition(key="category", match=models.MatchValue(value="ai")),
#         models.FieldCondition(key="year", range=models.Range(gte=2023))
#     ]
# )
```

### ChromaDB

Fetchcraft filters are translated to ChromaDB's where clause format:

```python
# Fetchcraft filter
from fetchcraft import AND, EQ, GTE

filter = AND(
    EQ("category", "ai"),
    GTE("year", 2023)
)

# Automatically translated to ChromaDB format:
# {
#     "$and": [
#         {"category": {"$eq": "ai"}},
#         {"year": {"$gte": 2023}}
#     ]
# }
```

### Adding Support for Other Vector Stores

To add filter support for a new vector store, implement the `_translate_filter` method:

```python
from fetchcraft.vector_store.base import VectorStore
from fetchcraft.filters import MetadataFilter, FieldFilter, CompositeFilter


class MyVectorStore(VectorStore):
    def _translate_filter(self, filter_obj: MetadataFilter):
        """Translate Fetchcraft filter to native format."""
        if isinstance(filter_obj, FieldFilter):
            # Translate field filter
            return self._translate_field_filter(filter_obj)
        elif isinstance(filter_obj, CompositeFilter):
            # Translate composite filter
            return self._translate_composite_filter(filter_obj)
```

## Best Practices

1. **Index Metadata Fields**: Ensure metadata fields you want to filter on are indexed in your vector store
2. **Use Appropriate Operators**: Choose the right operator for your data type (e.g., ranges for numbers, contains for text)
3. **Combine with Semantic Search**: Filters refine semantic search results, they don't replace it
4. **Test Filter Performance**: Complex nested filters may impact query performance
5. **Document Metadata Schema**: Maintain consistent metadata schema across documents

## Common Patterns

### Filter by Date Range

```python
from fetchcraft import AND, GTE, LTE

# Documents from 2023
filter = AND(
    GTE("year", 2023),
    LTE("year", 2023)
)
```

### Filter by Multiple Categories

```python
from fetchcraft import OR, EQ

# Documents in any of these categories
filter = OR(
    EQ("category", "ai"),
    EQ("category", "ml"),
    EQ("category", "deep-learning")
)
```

### Exclude Drafts

```python
from fetchcraft import NE

# Only published documents
filter = NE("status", "draft")
```

### Filter by Tags

```python
from fetchcraft import IN

# Documents with specific tags
filter = IN("tags", ["python", "tutorial", "beginner"])
```

### Complex Business Logic

```python
from fetchcraft import AND, OR, EQ, GTE, NE

# Premium content published in the last year, not archived
filter = AND(
    EQ("tier", "premium"),
    GTE("published_date", "2023-01-01"),
    NE("status", "archived"),
    OR(
        EQ("category", "tutorial"),
        EQ("category", "guide")
    )
)
```

## Examples

See the complete working examples in:
- `src/examples/filter_example.py` - Comprehensive filter examples
- `src/examples/advanced/filter_advanced.py` - Advanced filtering patterns

## API Reference

### Filter Functions

- `eq(key, value)` - Equality filter
- `ne(key, value)` - Not equal filter
- `gt(key, value)` - Greater than filter
- `lt(key, value)` - Less than filter
- `gte(key, value)` - Greater than or equal filter
- `lte(key, value)` - Less than or equal filter
- `in_(key, values)` - In list filter
- `contains(key, text)` - Text contains filter
- `and_(*filters)` - AND composite filter
- `or_(*filters)` - OR composite filter
- `not_(filter)` - NOT composite filter

### Classes

- `FilterOperator` - Enum of filter operators
- `FilterCondition` - Enum of logical conditions
- `MetadataFilter` - Base filter class
- `FieldFilter` - Single field filter
- `CompositeFilter` - Composite filter with multiple conditions

## Troubleshooting

### Filter Not Applied

**Problem**: Results don't respect the filter

**Solution**: Check that:
1. Metadata fields exist in your documents
2. Field names match exactly (case-sensitive)
3. Values match the expected type (string vs number vs boolean)

### Performance Issues

**Problem**: Queries with filters are slow

**Solution**:
1. Ensure metadata fields are indexed in your vector store
2. Simplify complex nested filters
3. Use more specific filters to reduce candidate set
4. Consider pre-filtering large datasets

### Type Errors

**Problem**: Filter fails with type error

**Solution**:
1. Ensure operator matches data type (e.g., don't use GT on strings)
2. Check that IN filter receives a list
3. Verify boolean fields use boolean values (not "true"/"false" strings)
