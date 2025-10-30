# Metadata Filters - Cheat Sheet

## Import

```python
from fetchcraft import EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS, AND, OR, NOT
```

## Basic Operators

| Operator | Syntax | Example |
|----------|--------|---------|
| Equal | `eq(key, value)` | `eq("status", "active")` |
| Not Equal | `ne(key, value)` | `ne("type", "draft")` |
| Greater Than | `gt(key, value)` | `gt("score", 0.8)` |
| Greater or Equal | `gte(key, value)` | `gte("year", 2023)` |
| Less Than | `lt(key, value)` | `lt("price", 100)` |
| Less or Equal | `lte(key, value)` | `lte("age", 18)` |
| In List | `in_(key, list)` | `in_("tag", ["ai", "ml"])` |
| Contains | `contains(key, text)` | `contains("title", "python")` |

## Logical Conditions

| Condition | Syntax | Example |
|-----------|--------|---------|
| AND | `and_(*filters)` | `and_(f1, f2, f3)` |
| OR | `or_(*filters)` | `or_(f1, f2)` |
| NOT | `not_(filter)` | `not_(f1)` |

## Usage Patterns

### With Index
```python
results = await index.search_by_text(
    "query",
    k=10,
    filters=eq("category", "tutorial")
)
```

### With Retriever (per-query)
```python
retriever = index.as_retriever(top_k=5)
results = retriever.retrieve("query", filters=eq("status", "published"))
```

### With Retriever (default filters in constructor)
```python
# Set default filters once
retriever = index.as_retriever(
    top_k=5,
    filters=eq("category", "tutorial")  # Applied to ALL queries
)

# Use without specifying filters
results = retriever.retrieve("query")

# Or override for specific queries
results = retriever.retrieve("query", filters=eq("status", "draft"))
```

## Common Patterns

### Date Range
```python
and_(gte("year", 2023), lte("year", 2024))
```

### Multiple Values
```python
in_("category", ["ai", "ml", "data-science"])
```

### Exclude
```python
ne("status", "archived")
# or
not_(eq("status", "archived"))
```

### Complex AND
```python
and_(
    eq("category", "tutorial"),
    gte("year", 2023),
    eq("level", "beginner")
)
```

### Complex OR
```python
or_(
    eq("category", "tutorial"),
    eq("category", "guide"),
    eq("category", "reference")
)
```

### Nested
```python
and_(
    gte("year", 2023),
    or_(
        eq("category", "tutorial"),
        eq("category", "guide")
    ),
    in_("language", ["python", "rust"])
)
```

## Real-World Examples

### Recent Published Content
```python
and_(
    eq("status", "published"),
    gte("publish_date", "2024-01-01"),
    ne("archived", True)
)
```

### Beginner Python Tutorials from 2023+
```python
and_(
    eq("language", "python"),
    eq("level", "beginner"),
    eq("type", "tutorial"),
    gte("year", 2023)
)
```

### Premium Content in Multiple Categories
```python
and_(
    eq("tier", "premium"),
    or_(
        eq("category", "advanced"),
        eq("category", "expert")
    ),
    ne("status", "draft")
)
```

### Popular Recent Articles
```python
and_(
    gte("views", 1000),
    gte("published_date", "2024-01-01"),
    in_("category", ["tech", "science", "ai"])
)
```

## Tips

✅ **DO**
- Use appropriate operators for data types
- Start simple, add complexity as needed
- Index metadata fields for performance
- Test filters with known data

❌ **DON'T**
- Use GT/LT on string fields (use EQ/NE)
- Forget to index metadata fields
- Over-complicate nested filters
- Use string "true"/"false" for booleans

## Quick Debug

```python
# Print filter structure
filter = and_(eq("key1", "val1"), gte("key2", 2023))
print(filter)  # Shows filter representation

# Convert to dict
filter_dict = filter.to_dict()
print(filter_dict)  # Shows JSON structure
```

## Performance

- Simple filters: ⚡ Fast
- Nested AND: ⚡ Fast (reduces candidates)
- Nested OR: ⚡⚡ Moderate (more candidates)
- Deep nesting: ⚡⚡⚡ Slower

## Supported Vector Stores

- ✅ **Qdrant** - Full support (see `src/examples/filter_example.py`)
- ✅ **ChromaDB** - Full support (see `src/examples/filter_example_chroma.py`)

Same filter API works with both!

## Full Docs

- Quick Start: `docs/FILTERS_QUICKSTART.md`
- Complete Guide: `docs/METADATA_FILTERS.md`
- Examples: `src/examples/filter_example.py` (Qdrant)
- Examples: `src/examples/filter_example_chroma.py` (ChromaDB)
- Tests: `tests/test_filters.py`
- ChromaDB Docs: `CHROMADB_FILTERS_IMPLEMENTATION.md`
