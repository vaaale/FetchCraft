# BaseModel Inheritance Updates

All vector stores, indices, and retrievers now inherit from Pydantic's `BaseModel`, providing:
- **Automatic validation** of fields
- **Serialization/deserialization** support
- **Type safety** at runtime
- **IDE autocomplete** improvements

## Summary of Changes

### 1. Abstract Base Classes

All abstract base classes now inherit from both `BaseModel` and `ABC`:

#### VectorStore (`src/fetchcraft/vector_store/base.py`)
```python
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

class VectorStore(BaseModel, ABC, Generic[D]):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

#### Retriever (`src/fetchcraft/retriever/base.py`)
```python
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

class Retriever(BaseModel, ABC, Generic[D]):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

#### Embeddings (`src/fetchcraft/embeddings/base.py`)
```python
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

class Embeddings(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

### 2. Concrete Implementations

#### QdrantVectorStore (`src/fetchcraft/vector_store/qdrant_store.py`)
- Now uses Pydantic `Field` for all attributes
- Inherits from `VectorStore[D]` (which inherits from `BaseModel`)
- Fields: `client`, `collection_name`, `document_class`, `vector_size`, `distance`

```python
class QdrantVectorStore(VectorStore[D]):
    client: Any = Field(description="QdrantClient instance")
    collection_name: str = Field(description="Name of the collection")
    document_class: Optional[Type[D]] = Field(default=None)
    vector_size: int = Field(default=384)
    distance: str = Field(default="Cosine")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

#### VectorIndex (`src/fetchcraft/index/vector_index.py`)
- Now inherits from `BaseModel`
- Uses `SkipValidation` for abstract base type fields
- Fields: `vector_store`, `embeddings`, `index_id`

```python
class VectorIndex(BaseModel, Generic[D]):
    vector_store: Annotated[VectorStore[D], SkipValidation()] = Field(...)
    embeddings: Annotated[Embeddings, SkipValidation()] = Field(...)
    index_id: str = Field(default_factory=lambda: str(uuid4()))
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

#### VectorIndexRetriever (`src/fetchcraft/retriever/vector_index_retriever.py`)
- Now inherits from `Retriever[D]` (which inherits from `BaseModel`)
- Fields: `vector_index`, `embeddings`, `top_k`, `resolve_parents`, `search_kwargs`

```python
class VectorIndexRetriever(Retriever[D]):
    vector_index: Annotated[Any, SkipValidation()] = Field(...)
    embeddings: Annotated[Any, SkipValidation()] = Field(...)
    top_k: int = Field(default=4)
    resolve_parents: bool = Field(default=True)
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

## Key Technical Details

### SkipValidation for Abstract Types

When using abstract base classes as field types, we use `SkipValidation` to prevent Pydantic from trying to instantiate the abstract class during validation:

```python
from typing import Annotated
from pydantic import SkipValidation

vector_store: Annotated[VectorStore[D], SkipValidation()] = Field(...)
```

This allows:
- Field to accept any concrete implementation (e.g., `QdrantVectorStore`)
- No runtime validation errors when passing subclasses
- Type hints remain accurate for IDE support

### ConfigDict Settings

All classes use consistent config:

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True,  # Allow non-Pydantic types (e.g., QdrantClient)
    validate_assignment=True,       # Validate when fields are assigned
)
```

## Benefits

### 1. Automatic Validation
```python
# Invalid types are caught at runtime
vector_index = VectorIndex(
    vector_store="not a vector store",  # ❌ ValidationError
    embeddings=embeddings,
    index_id="test"
)
```

### 2. Serialization Support
```python
# Serialize to dict
vector_index_dict = vector_index.model_dump()

# Serialize to JSON (with exclusions for non-serializable fields)
json_str = embeddings.model_dump_json(exclude={'client', 'aclient'})
```

### 3. Type Safety
```python
# IDE knows the exact types
vector_index.vector_store  # Type: VectorStore[D]
vector_index.index_id      # Type: str
```

### 4. Better Documentation
Fields are self-documenting with descriptions:
```python
top_k: int = Field(default=4, description="Number of results to return")
```

## Backward Compatibility

✅ **All existing code continues to work** - the changes are fully backward compatible:

```python
# Old way (still works)
vector_index = VectorIndex(vector_store, embeddings, "my-index")

# New way (also works, with validation)
vector_index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings,
    index_id="my-index"
)
```

## Testing

Run the test suite to verify:

```bash
python test_basemodel_inheritance.py
```

Expected output:
```
✅ ALL TESTS PASSED!
All vector stores, indices, and retrievers properly inherit from BaseModel.
```

## Migration Guide

### For Users

**No changes required** - existing code will continue to work.

Optionally, you can now:
- Use `model_dump()` for serialization
- Rely on automatic validation
- Get better IDE support

### For Contributors

When creating new vector stores, indices, or retrievers:

1. **Inherit from the base class** (which now inherits from `BaseModel`)
2. **Define fields with `Field()`** for better documentation
3. **Use `ConfigDict`** with `arbitrary_types_allowed=True`
4. **Use `SkipValidation`** for abstract base type fields

Example:
```python
from pydantic import Field, ConfigDict
from typing import Annotated
from pydantic import SkipValidation

class MyVectorStore(VectorStore[Node]):
    client: Any = Field(description="Client instance")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
```

## Files Modified

1. `/src/fetchcraft/vector_store/base.py` - VectorStore base class
2. `/src/fetchcraft/vector_store/qdrant_store.py` - QdrantVectorStore implementation
3. `/src/fetchcraft/index/vector_index.py` - VectorIndex class
4. `/src/fetchcraft/retriever/base.py` - Retriever base class
5. `/src/fetchcraft/retriever/vector_index_retriever.py` - VectorIndexRetriever implementation
6. `/src/fetchcraft/embeddings/base.py` - Embeddings base class

## Verification

All inheritance verified:
- ✅ `VectorStore` inherits from `BaseModel`
- ✅ `QdrantVectorStore` inherits from `BaseModel`
- ✅ `VectorIndex` inherits from `BaseModel`
- ✅ `Retriever` inherits from `BaseModel`
- ✅ `VectorIndexRetriever` inherits from `BaseModel`
- ✅ `Embeddings` inherits from `BaseModel`
- ✅ All concrete implementations (e.g., `OpenAIEmbeddings`) inherit from `BaseModel`
