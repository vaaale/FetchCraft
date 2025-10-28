"""
Example demonstrating multiple indices in the same vector store.
"""

import asyncio
from typing import List, Any

from pydantic import PrivateAttr
from qdrant_client import QdrantClient

from fetchcraft import VectorIndex, QdrantVectorStore, Node, Chunk, Embeddings


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing without API calls."""

    _dimension: int = PrivateAttr()

    def __init__(self, /, dimension=384, **data: Any):
        super().__init__(**data)
        self._dimension = dimension

    async def embed_query(self, text: str):
        """Return a mock embedding based on text hash."""
        return [hash(text) % 100 / 100.0] * self.dimension
    
    async def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return [await self.embed_query(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension




async def demo_multiple_indices():
    """
    Demonstrate how to use multiple indices in the same Qdrant collection.
    
    This allows you to:
    - Separate different types of documents in the same collection
    - Maintain multiple independent indices for different projects
    - Share infrastructure while keeping data logically separated
    """
    
    # Initialize Qdrant client
    client = QdrantClient(":memory:")  # Using in-memory for demo
    # client = QdrantClient("http://localhost:6333")  # Or use actual Qdrant server

    # Initialize embeddings (mock for demo)
    embeddings = MockEmbeddings(dimension=384)
    
    # Create a shared vector store with embeddings configured
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="shared_collection",
        embeddings=embeddings,
        distance="Cosine"
    )
    
    print("="*60)
    print("Multiple Indices Demo")
    print("="*60 + "\n")
    
    # Create Index 1: Technical documentation
    tech_index = VectorIndex(
        vector_store=vector_store,
        index_id="tech_docs"
    )
    
    # Create Index 2: Marketing content
    marketing_index = VectorIndex(
        vector_store=vector_store,
        index_id="marketing_content"
    )
    
    # Create Index 3: Customer support
    support_index = VectorIndex(
        vector_store=vector_store,
        index_id="customer_support"
    )
    
    print(f"Created 3 indices:")
    print(f"  - Technical docs: {tech_index.index_id}")
    print(f"  - Marketing: {marketing_index.index_id}")
    print(f"  - Support: {support_index.index_id}")
    print()
    
    # Add documents to technical index (no embeddings needed!)
    tech_docs = [
        Node(
            text="API authentication using OAuth2",
            metadata={"category": "api", "version": "v2"}
        ),
        Node(
            text="Database schema design patterns",
            metadata={"category": "database", "version": "v2"}
        ),
    ]
    
    tech_ids = await tech_index.add_documents(tech_docs)
    print(f"✓ Added {len(tech_ids)} documents to technical index")
    
    # Add documents to marketing index (no embeddings needed!)
    marketing_docs = [
        Node(
            text="New product launch announcement",
            metadata={"campaign": "product_launch"}
        ),
        Node(
            text="Customer success stories",
            metadata={"campaign": "testimonials"}
        ),
    ]
    
    marketing_ids = await marketing_index.add_documents(marketing_docs)
    print(f"✓ Added {len(marketing_ids)} documents to marketing index")
    
    # Add documents to support index (no embeddings needed!)
    support_docs = [
        Node(
            text="How to reset your password",
            metadata={"topic": "account"}
        ),
        Node(
            text="Troubleshooting connection issues",
            metadata={"topic": "technical"}
        ),
    ]
    
    support_ids = await support_index.add_documents(support_docs)
    print(f"✓ Added {len(support_ids)} documents to support index")
    print()
    
    # Demonstrate isolated searches
    print("="*60)
    print("Searching within specific indices")
    print("="*60 + "\n")
    
    query = "authentication and security"
    
    # Search only in technical index
    print(f"Searching technical docs index for '{query}':")
    tech_results = await tech_index.search_by_text(query, k=2)
    for doc, score in tech_results:
        print(f"  - Score: {score:.3f} | {doc.text}")
    print()
    
    # Search only in marketing index
    print(f"Searching marketing index for '{query}':")
    marketing_results = await marketing_index.search_by_text(query, k=2)
    for doc, score in marketing_results:
        print(f"  - Score: {score:.3f} | {doc.text}")
    print()
    
    # Search only in support index
    print(f"Searching support index for '{query}':")
    support_results = await support_index.search_by_text(query, k=2)
    for doc, score in support_results:
        print(f"  - Score: {score:.3f} | {doc.text}")
    print()
    
    # Demonstrate isolated retrieval
    print("="*60)
    print("Retrieving documents by ID with index isolation")
    print("="*60 + "\n")
    
    tech_doc_id = tech_ids[0]
    
    # Can retrieve from correct index
    doc = await tech_index.get_document(tech_doc_id)
    print(f"✓ Retrieved from tech index: {doc.text if doc else 'Not found'}")
    
    # Cannot retrieve from wrong index (returns None)
    doc = await marketing_index.get_document(tech_doc_id)
    print(f"✗ Retrieved from marketing index: {doc.text if doc else 'Not found (isolated)'}")
    print()
    
    # Demonstrate isolated deletion
    print("="*60)
    print("Deleting documents with index isolation")
    print("="*60 + "\n")
    
    # Delete from marketing index
    deleted = await marketing_index.delete_documents([marketing_ids[0]])
    print(f"✓ Deleted 1 document from marketing index")
    
    # Verify it's gone from marketing index
    doc = await marketing_index.get_document(marketing_ids[0])
    print(f"  Marketing index: {doc.text if doc else 'Document deleted ✓'}")
    
    # Verify tech and support indices are unaffected
    tech_doc = await tech_index.get_document(tech_ids[0])
    support_doc = await support_index.get_document(support_ids[0])
    print(f"  Tech index still has: {len([tech_doc]) if tech_doc else 0} document(s)")
    print(f"  Support index still has: {len([support_doc]) if support_doc else 0} document(s)")
    print()
    
    # Advanced: Create indices with default index_id (auto-generated UUID)
    print("="*60)
    print("Auto-generated index IDs")
    print("="*60 + "\n")
    
    auto_index1 = VectorIndex(vector_store=vector_store)
    auto_index2 = VectorIndex(vector_store=vector_store)
    
    print(f"Auto-generated index 1 ID: {auto_index1.index_id}")
    print(f"Auto-generated index 2 ID: {auto_index2.index_id}")
    print(f"IDs are unique: {auto_index1.index_id != auto_index2.index_id}")
    print()
    
    print("="*60)
    print("Demo completed successfully!")
    print("="*60)


async def demo_use_cases():
    """Show practical use cases for multiple indices."""
    
    print("\n" + "="*60)
    print("Practical Use Cases for Multiple Indices")
    print("="*60 + "\n")
    
    client = QdrantClient(":memory:")
    
    # Mock embeddings for use cases
    embeddings = MockEmbeddings(dimension=384)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="multi_tenant",
        embeddings=embeddings
    )
    
    # Use Case 1: Multi-tenant application
    print("1. Multi-tenant SaaS application:")
    tenant_a_index = VectorIndex(vector_store, index_id="tenant_company_a")
    tenant_b_index = VectorIndex(vector_store, index_id="tenant_company_b")
    print(f"   - Company A has isolated index: {tenant_a_index.index_id}")
    print(f"   - Company B has isolated index: {tenant_b_index.index_id}")
    print(f"   - Both share the same Qdrant collection\n")
    
    # Use Case 2: Environment separation
    print("2. Environment separation (dev/staging/prod):")
    dev_index = VectorIndex(vector_store, index_id="env_development")
    staging_index = VectorIndex(vector_store, index_id="env_staging")
    prod_index = VectorIndex(vector_store, index_id="env_production")
    print(f"   - Development: {dev_index.index_id}")
    print(f"   - Staging: {staging_index.index_id}")
    print(f"   - Production: {prod_index.index_id}\n")
    
    # Use Case 3: Language-specific indices
    print("3. Multi-language content:")
    en_index = VectorIndex(vector_store, index_id="lang_english")
    es_index = VectorIndex(vector_store, index_id="lang_spanish")
    fr_index = VectorIndex(vector_store, index_id="lang_french")
    print(f"   - English: {en_index.index_id}")
    print(f"   - Spanish: {es_index.index_id}")
    print(f"   - French: {fr_index.index_id}\n")
    
    # Use Case 4: Version control
    print("4. Document version control:")
    v1_index = VectorIndex(vector_store, index_id="version_1.0")
    v2_index = VectorIndex(vector_store, index_id="version_2.0")
    print(f"   - Version 1.0: {v1_index.index_id}")
    print(f"   - Version 2.0: {v2_index.index_id}\n")


if __name__ == "__main__":
    asyncio.run(demo_multiple_indices())
    asyncio.run(demo_use_cases())
