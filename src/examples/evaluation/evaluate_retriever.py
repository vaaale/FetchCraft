"""
Example: Evaluating Retriever Performance

This script demonstrates a complete workflow:
1. Ingest documents from a directory (if not already indexed)
2. Generate an evaluation dataset from indexed documents
3. Evaluate a retriever's performance with comprehensive metrics
4. Analyze and visualize the results

The script supports:
- Automatic document ingestion with hierarchical chunking
- LLM-based question generation for evaluation
- Comprehensive metrics (Hit Rate, MRR, NDCG, Precision, Recall)
- Testing multiple k values
- Error analysis and result visualization

Environment Variables:
- DOCUMENTS_PATH: Path to documents directory (default: "Documents")
- QDRANT_HOST: Qdrant host (default: "localhost")
- QDRANT_PORT: Qdrant port (default: "6333")
- MONGODB_URI: MongoDB connection string
- OPENAI_API_KEY: OpenAI API key for embeddings and question generation
- CHUNK_SIZE: Parent chunk size (default: 8192)
- CHUNK_OVERLAP: Chunk overlap (default: 200)
"""

import asyncio
import os
from pathlib import Path
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from qdrant_client import QdrantClient

from fetchcraft import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    MongoDBDocumentStore,
    DatasetGenerator,
    RetrieverEvaluator,
    EvaluationDataset,
    TextFileDocumentParser,
    HierarchicalChunkingStrategy,
    CharacterChunkingStrategy,
    Chunk,
)


# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "fetchcraft_docs"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

# Index configuration
INDEX_ID = "docs-index"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "gpt-4-turbo"

# Document ingestion configuration
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8192"))
CHILD_SIZES = [4096, 1024]
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
USE_HIERARCHICAL_CHUNKING = os.getenv("USE_HIERARCHICAL_CHUNKING", "true").lower() == "true"


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists in Qdrant."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    return collection_name in collection_names


async def load_and_index_documents(
    vector_index: VectorIndex,
    document_store: MongoDBDocumentStore,
    documents_path: Path,
    chunk_size: int = 8192,
    child_sizes = [4096, 1024],
    overlap: int = 200,
    use_hierarchical: bool = True
) -> int:
    """
    Load documents from a directory and index them.
    
    Args:
        vector_index: Vector index to add documents to
        document_store: Document store to save full documents
        documents_path: Path to directory containing documents
        chunk_size: Size of parent chunks
        child_sizes: Sizes of child chunks for hierarchical chunking
        overlap: Overlap between chunks
        use_hierarchical: Whether to use hierarchical chunking
        
    Returns:
        Number of chunks indexed
    """
    print(f"   Loading documents from: {documents_path}")
    
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_path}")
    
    # Create chunking strategy
    if use_hierarchical:
        chunker = HierarchicalChunkingStrategy(
            chunk_size=chunk_size,
            overlap=overlap,
            child_chunks=child_sizes,
            child_overlap=50
        )
        print(f"   Using hierarchical chunking: {chunk_size} -> {child_sizes}")
    else:
        chunker = CharacterChunkingStrategy(
            chunk_size=chunk_size,
            overlap=overlap
        )
        print(f"   Using character chunking: {chunk_size}")
    
    # Create parser
    parser = TextFileDocumentParser(chunker=chunker)
    
    # Parse documents
    results = await parser.parse_directory(
        directory_path=documents_path,
        pattern="*",
        recursive=True
    )
    
    if not results:
        print("   ⚠️  No text files found in the specified directory!")
        return 0
    
    # Separate document nodes from chunks
    # The parser returns [DocumentNode, chunk1, chunk2, ...]
    all_chunks: list[Chunk] = []
    document_nodes = []
    
    for file_path, nodes in results.items():
        if not nodes:
            continue
        
        # First element is the DocumentNode
        doc_node = nodes[0]
        chunks = nodes[1:]  # Rest are chunks
        
        print(f"   Loaded {len(chunks)} chunks from {Path(file_path).name}")
        
        # Update document node with children IDs
        # For hierarchical chunking, find top-level chunks (those not referenced as parents by others)
        # Collect all chunk IDs and all parent IDs
        chunk_ids = {c.id for c in chunks}
        parent_ids_in_chunks = {c.parent_id for c in chunks if c.parent_id and c.parent_id != doc_node.id}
        
        # Top-level chunks are those whose IDs are NOT in parent_ids_in_chunks
        # These are the largest chunks that have children but are not children themselves
        top_level_chunks = [c for c in chunks if c.id not in parent_ids_in_chunks]
        
        # Set their parent_id to doc_node and update doc_node's children_ids
        for chunk in top_level_chunks:
            chunk.parent_id = doc_node.id
        
        doc_node.children_ids = [c.id for c in top_level_chunks]
        print(f"   ✓ Document has {len(doc_node.children_ids)} top-level children")
        
        # Add to collections
        document_nodes.append(doc_node)
        all_chunks.extend(chunks)
    
    # Store document nodes first (they are parent nodes with children_ids populated)
    print(f"   Storing {len(document_nodes)} documents to document store...")
    if document_nodes:
        await document_store.add_documents(document_nodes)
        
        # Verify: Check that documents were stored with children
        print(f"   Verifying stored documents...")
        sample_doc = await document_store.get_document(document_nodes[0].id)
        if sample_doc:
            print(f"   ✓ Sample document retrieved with {len(sample_doc.children_ids)} children_ids")
            if not sample_doc.children_ids:
                print(f"   ⚠️  WARNING: Document has no children_ids after storage!")
        else:
            print(f"   ⚠️  WARNING: Could not retrieve sample document after storage!")
    
    # Then index chunks to vector store
    print(f"   Indexing {len(all_chunks)} chunks to vector store...")
    await vector_index.add_documents(all_chunks, show_progress=True)
    
    print(f"   ✓ Successfully indexed {len(all_chunks)} chunks from {len(document_nodes)} documents!")
    return len(all_chunks)


async def main():
    """Main evaluation workflow."""
    
    print("=" * 70)
    print("Retriever Performance Evaluation")
    print("=" * 70)
    
    # Initialize embeddings
    print("\n1. Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    # Connect to Qdrant
    print("2. Connecting to Qdrant...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Check if we need to index documents
    needs_indexing = not collection_exists(qdrant_client, COLLECTION_NAME)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=True,  # Test with hybrid search
        fusion_method="rrf"
    )

    # Connect to MongoDB for document store
    print("3. Connecting to MongoDB...")
    document_store = MongoDBDocumentStore(
        connection_string=MONGODB_URI,
        database_name="fetchcraft",
        collection_name="documents"
    )
    
    # Create vector index and retriever
    print("4. Creating vector index...")
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )
    
    retriever = vector_index.as_retriever(
        top_k=5,
        resolve_parents=True
    )
    
    # ========================================================================
    # STEP 0: Index Documents (if needed)
    # ========================================================================
    
    if needs_indexing:
        print(f"\n4a. Collection '{COLLECTION_NAME}' not found, indexing documents...")
        num_chunks = await load_and_index_documents(
            vector_index=vector_index,
            document_store=document_store,
            documents_path=DOCUMENTS_PATH,
            chunk_size=CHUNK_SIZE,
            child_sizes=CHILD_SIZES,
            overlap=CHUNK_OVERLAP,
            use_hierarchical=USE_HIERARCHICAL_CHUNKING
        )
        if num_chunks == 0:
            print("   ⚠️  No documents were indexed! Check DOCUMENTS_PATH.")
            return
    else:
        print(f"\n4a. Collection '{COLLECTION_NAME}' already exists, skipping indexing")
        # Count existing documents
        doc_count = await document_store.count_documents()
        print(f"   Found {doc_count} documents in document store")
    
    # ========================================================================
    # STEP 1: Generate Evaluation Dataset
    # ========================================================================
    
    dataset_path = Path("evaluation_dataset.json")
    
    if dataset_path.exists():
        print(f"\n5. Loading existing dataset from {dataset_path}...")
        dataset = EvaluationDataset.load(str(dataset_path))
        print(f"   Loaded {len(dataset)} question-answer pairs")
    else:
        print("\n5. Generating evaluation dataset...")
        
        # Initialize OpenAI client for question generation
        openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        model = OpenAIChatModel(
            os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
            provider=OpenAIProvider(
                openai_client=openai_client
            )
        )
        # Create dataset generator
        generator = DatasetGenerator(
            model=model,
        )
        
        # Generate dataset
        dataset = await generator.generate_dataset(
            num_documents=10,  # Sample 10 documents
            document_store=document_store,
            vector_store=vector_store,
            index_id=INDEX_ID,
            questions_per_node=3,  # Generate 3 questions per node
            max_nodes_per_document=5,  # Use up to 5 nodes per document
            show_progress=True
        )
        
        # Save dataset
        dataset.save(str(dataset_path))
        print(f"   Generated {len(dataset)} question-answer pairs")
        print(f"   Dataset saved to {dataset_path}")
    
    # ========================================================================
    # STEP 2: Evaluate Retriever
    # ========================================================================
    
    print("\n6. Evaluating retriever...")
    evaluator = RetrieverEvaluator(retriever=retriever)
    
    metrics = await evaluator.evaluate(
        dataset=dataset,
        show_progress=True
    )
    
    # Print metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print(metrics)
    
    # ========================================================================
    # STEP 3: Analyze Results
    # ========================================================================
    
    print("\n7. Analyzing results...")
    
    # Save detailed results
    results_path = Path("evaluation_results.json")
    evaluator.save_results(str(results_path))
    print(f"   Detailed results saved to {results_path}")
    
    # Analyze failed queries
    failed_queries = evaluator.get_failed_queries()
    print(f"\n   Failed Queries: {len(failed_queries)}")
    if failed_queries:
        print("\n   Sample Failed Queries:")
        for i, result in enumerate(failed_queries[:3], 1):
            print(f"   {i}. {result.question}")
            print(f"      Expected: {result.expected_node_id[:16]}...")
            print(f"      Retrieved: {[nid[:16] + '...' for nid in result.retrieved_node_ids[:3]]}")
    
    # Analyze top-ranked hits
    top_rank_hits = evaluator.get_queries_by_rank(1)
    print(f"\n   Queries with Perfect Rank (1): {len(top_rank_hits)}")
    
    # ========================================================================
    # STEP 4: Test Different k Values
    # ========================================================================
    
    print("\n8. Testing different k values...")
    k_values = [1, 3, 5, 10]
    k_results = await evaluator.evaluate_with_different_k(
        dataset=dataset,
        k_values=k_values,
        show_progress=False
    )
    
    print("\n   Performance across different k values:")
    print("   " + "-" * 66)
    print(f"   {'k':<5} {'Hit Rate':<12} {'MRR':<12} {'Recall@k':<12} {'NDCG@k':<12}")
    print("   " + "-" * 66)
    for k in k_values:
        m = k_results[k]
        print(f"   {k:<5} {m.hit_rate:<12.4f} {m.mrr:<12.4f} {m.recall_at_k:<12.4f} {m.ndcg_at_k:<12.4f}")
    print("   " + "-" * 66)
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Results: {results_path}")
    print(f"\nKey Metrics:")
    print(f"  - Hit Rate@{metrics.k}: {metrics.hit_rate:.2%}")
    print(f"  - MRR: {metrics.mrr:.4f}")
    print(f"  - NDCG@{metrics.k}: {metrics.ndcg_at_k:.4f}")
    print(f"  - Average Rank (when found): {metrics.average_rank:.2f}")
    print("=" * 70)


async def generate_dataset_only():
    """
    Separate function to only generate a dataset.
    Useful when you want to generate once and evaluate multiple times.
    """
    print("Generating evaluation dataset only...")
    
    # Initialize clients
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    
    document_store = MongoDBDocumentStore(
        connection_string=MONGODB_URI,
        database_name="fetchcraft",
        collection_name="documents"
    )
    
    openai_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    generator = DatasetGenerator(
        client=openai_client,
        document_store=document_store,
        vector_store=vector_store,
        model=LLM_MODEL,
        index_id=INDEX_ID
    )
    
    dataset = await generator.generate_dataset(
        num_documents=20,
        questions_per_node=3,
        show_progress=True
    )
    
    dataset.save("evaluation_dataset.json")
    print(f"Dataset with {len(dataset)} pairs saved!")


async def ingest_documents_only():
    """
    Separate function to only ingest documents.
    Useful for initial setup or re-indexing.
    """
    print("=" * 70)
    print("Document Ingestion")
    print("=" * 70)
    
    # Initialize components
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        enable_hybrid=True,
        fusion_method="rrf"
    )
    
    document_store = MongoDBDocumentStore(
        connection_string=MONGODB_URI,
        database_name="fetchcraft",
        collection_name="documents"
    )
    
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )
    
    # Ingest documents
    print("\nIndexing documents...")
    num_chunks = await load_and_index_documents(
        vector_index=vector_index,
        document_store=document_store,
        documents_path=DOCUMENTS_PATH,
        chunk_size=CHUNK_SIZE,
        child_sizes=CHILD_SIZES,
        overlap=CHUNK_OVERLAP,
        use_hierarchical=USE_HIERARCHICAL_CHUNKING
    )
    
    print("\n" + "=" * 70)
    print(f"✓ Ingestion complete! Indexed {num_chunks} chunks")
    print("=" * 70)


async def evaluate_only():
    """
    Separate function to only evaluate using an existing dataset.
    """
    print("Evaluating retriever with existing dataset...")
    
    # Load dataset
    dataset = EvaluationDataset.load("evaluation_dataset.json")
    
    # Setup retriever
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        enable_hybrid=True
    )
    
    vector_index = VectorIndex(vector_store=vector_store, index_id=INDEX_ID)
    retriever = vector_index.as_retriever(top_k=5, resolve_parents=True)
    
    # Evaluate
    evaluator = RetrieverEvaluator(retriever=retriever)
    metrics = await evaluator.evaluate(dataset, show_progress=True)
    
    print(metrics)
    evaluator.save_results("evaluation_results.json")


if __name__ == "__main__":
    """
    Usage:
    
    1. Full workflow (recommended for first run):
       python -m examples.evaluate_retriever
       
       This will:
       - Check if documents are indexed, index them if not
       - Generate evaluation dataset (or load if exists)
       - Evaluate retriever performance
       - Show comprehensive metrics and analysis
    
    2. Individual steps:
       
       a) Just index documents:
          Uncomment: asyncio.run(ingest_documents_only())
          
       b) Just generate dataset (requires indexed docs):
          Uncomment: asyncio.run(generate_dataset_only())
          
       c) Just evaluate (requires dataset):
          Uncomment: asyncio.run(evaluate_only())
    
    Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - MongoDB running: docker run -p 27017:27017 mongo
    - Documents in ./Documents/ directory (or set DOCUMENTS_PATH)
    - OPENAI_API_KEY environment variable set
    """
    
    # Run full evaluation workflow (includes document ingestion if needed)
    asyncio.run(main())
    
    # Or run individual steps:
    # asyncio.run(ingest_documents_only())     # Just index documents
    # asyncio.run(generate_dataset_only())     # Just generate evaluation dataset
    # asyncio.run(evaluate_only())             # Just evaluate with existing dataset
