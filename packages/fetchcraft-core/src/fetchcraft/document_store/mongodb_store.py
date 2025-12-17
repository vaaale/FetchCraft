"""
MongoDB document store implementation.
"""

from typing import List, Optional, Dict, Any, Type, Union
from pydantic import BaseModel, Field, ConfigDict

from .base import DocumentStore
from ..node import Node, DocumentNode, Chunk, SymNode


try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    AsyncIOMotorClient = None
    AsyncIOMotorDatabase = None
    AsyncIOMotorCollection = None


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB document store."""
    connection_string: str = "mongodb://localhost:27017"
    database_name: str = "fetchcraft"
    collection_name: str = "documents"
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MongoDBDocumentStore(DocumentStore[Node]):
    """
    MongoDB implementation of the DocumentStore interface.
    
    Stores full documents in MongoDB for efficient retrieval and management.
    Uses Motor (async MongoDB driver) for async operations.
    
    Example:
        ```python
        from fetchcraft.document_store import MongoDBDocumentStore
        from fetchcraft.node import DocumentNode
        
        # Create store
        store = MongoDBDocumentStore(
            connection_string="mongodb://localhost:27017",
            database_name="my_rag_app",
            collection_name="documents"
        )
        
        # Store a document
        doc = DocumentNode.from_text("Hello world", metadata={"parsing": "test"})
        await store.add_document(doc)
        
        # Retrieve it
        retrieved = await store.get_document(doc.id)
        ```
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database_name: str = "fetchcraft",
        collection_name: str = "documents",
        document_class: Type[Node] = DocumentNode,
        client: Optional[Any] = None
    ):
        """
        Initialize MongoDB document store.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database
            collection_name: Name of the collection
            document_class: Default document class for reconstruction
            client: Optional pre-configured Motor client
        """
        if not MOTOR_AVAILABLE:
            raise ImportError(
                "motor package is required for MongoDBDocumentStore. "
                "Install it with: pip install motor"
            )
        
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.document_class = document_class
        
        # Initialize Motor client
        if client:
            self.client = client
        else:
            self.client: AsyncIOMotorClient = AsyncIOMotorClient(connection_string)
        
        self.database: AsyncIOMotorDatabase = self.client[database_name]
        self.collection: AsyncIOMotorCollection = self.database[collection_name]
        
        # Track if indexes are created
        self._indexes_created = False
    
    async def _ensure_indexes(self):
        """Create indexes for efficient querying."""
        if self._indexes_created:
            return
        
        # Create index on document ID for fast lookups
        await self.collection.create_index("id", unique=True)
        
        # Create index on doc_id for document-level queries
        await self.collection.create_index("doc_id")
        
        # Create index on metadata fields (if needed)
        await self.collection.create_index([("metadata.parsing", 1)])
        
        self._indexes_created = True
    
    def _get_doc_class(self, class_name: Optional[str]) -> Type[Node]:
        """
        Get the document class based on the stored class name.
        
        Args:
            class_name: Name of the class stored in the document
            
        Returns:
            The appropriate document class
        """
        if class_name == 'SymNode':
            return SymNode  # type: ignore
        elif class_name == 'Chunk':
            return Chunk  # type: ignore
        elif class_name == 'DocumentNode':
            return DocumentNode  # type: ignore
        elif class_name == 'Node':
            return Node  # type: ignore
        else:
            return self.document_class
    
    async def add_document(self, document: Node) -> str:
        """
        Add a single document to MongoDB.
        
        Args:
            document: The document to store
            
        Returns:
            The document ID
        """
        existing = await self.list_documents(filters={"metadata.source": document.metadata["source"]})
        if existing:
            if len(existing) > 1:
                raise ValueError(f"Multiple documents with hash {document.hash} found")

            existing_doc = existing[0]

            if document.hash != existing_doc.hash:
                # Document has changed. Replace the old with the new
                await self.delete_document(existing_doc.id)
            else:
                document.doc_id = existing_doc.doc_id
                document.id = existing_doc.id


        # Convert document to dict
        doc_dict = document.model_dump()
        doc_dict['_doc_class'] = document.__class__.__name__
        
        # Insert into MongoDB (replace if exists)
        await self.collection.replace_one(
            {"id": document.id},
            doc_dict,
            upsert=True
        )
        
        return document.id
    
    async def add_documents(self, documents: List[Node]) -> List[str]:
        """
        Add multiple documents to MongoDB.
        
        Args:
            documents: List of documents to store
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # await self._ensure_indexes()
        
        # Convert all documents to dicts
        doc_dicts = []
        for doc in documents:
            doc_dict = doc.model_dump()
            doc_dict['_doc_class'] = doc.__class__.__name__
            doc_dicts.append(doc_dict)
        
        # Bulk insert (replace if exists)
        # Use individual update_one operations for mongomock compatibility
        # bulk_write with UpdateOne/ReplaceOne has issues with mongomock and newer pymongo
        for doc_dict in doc_dicts:
            await self.collection.update_one(
                {'id': doc_dict['id']},
                {'$set': doc_dict},
                upsert=True
            )
        
        return [doc.id for doc in documents]
    
    async def get_document(self, document_id: str) -> Optional[Node]:
        """
        Retrieve a document by its ID from MongoDB.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        # Find document by ID
        doc_dict = await self.collection.find_one({"id": document_id})
        
        if not doc_dict:
            return None
        
        # Remove MongoDB's _id field
        doc_dict.pop('_id', None)
        
        # Get document class
        doc_class_name = doc_dict.pop('_doc_class', None)
        doc_class = self._get_doc_class(doc_class_name)
        
        # Reconstruct document
        return doc_class(**doc_dict)
    
    async def get_documents(self, document_ids: List[str]) -> List[Node]:
        """
        Retrieve multiple documents by their IDs from MongoDB.
        
        Args:
            document_ids: List of document IDs to retrieve
            
        Returns:
            List of documents (may be fewer than requested if some not found)
        """
        if not document_ids:
            return []
        
        # Find all documents with matching IDs
        cursor = self.collection.find({"id": {"$in": document_ids}})
        
        documents = []
        async for doc_dict in cursor:
            # Remove MongoDB's _id field
            doc_dict.pop('_id', None)
            
            # Get document class
            doc_class_name = doc_dict.pop('_doc_class', None)
            doc_class = self._get_doc_class(doc_class_name)
            
            # Reconstruct document
            documents.append(doc_class(**doc_dict))
        
        return documents
    
    async def update_document(self, document: Node) -> bool:
        """
        Update an existing document in MongoDB.
        
        Args:
            document: The document with updated content
            
        Returns:
            True if update was successful, False otherwise
        """
        # Convert document to dict
        doc_dict = document.model_dump()
        doc_dict['_doc_class'] = document.__class__.__name__
        
        # Update in MongoDB
        result = await self.collection.replace_one(
            {"id": document.id},
            doc_dict
        )
        
        return result.modified_count > 0
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by its ID from MongoDB.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        result = await self.collection.delete_one({"id": document_id})
        return result.deleted_count > 0
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete multiple documents by their IDs from MongoDB.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not document_ids:
            return True
        
        result = await self.collection.delete_many({"id": {"$in": document_ids}})
        return result.deleted_count > 0
    
    async def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in MongoDB.
        
        Args:
            document_id: The ID of the document to check
            
        Returns:
            True if document exists, False otherwise
        """
        count = await self.collection.count_documents({"id": document_id}, limit=1)
        return count > 0

    async def all_ids(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Return all document IDs in the store.

        Returns:
            List of document IDs
        """
        query = filters or {}
        node_ids = []
        cursor = self.collection.find(query)
        async for doc in cursor:
            node_ids.append(doc["id"])
        return node_ids

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        List documents with pagination and optional filtering.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filters: Optional MongoDB query filters
            
        Returns:
            List of documents
        """
        query = filters or {}
        
        # Find documents with pagination
        cursor = self.collection.find(query).skip(offset).limit(limit)
        
        documents = []
        async for doc_dict in cursor:
            # Remove MongoDB's _id field
            doc_dict.pop('_id', None)
            
            # Get document class
            doc_class_name = doc_dict.pop('_doc_class', None)
            doc_class = self._get_doc_class(doc_class_name)
            
            # Reconstruct document
            documents.append(doc_class(**doc_dict))
        
        return documents
    
    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in MongoDB.
        
        Args:
            filters: Optional MongoDB query filters
            
        Returns:
            Number of documents
        """
        query = filters or {}
        return await self.collection.count_documents(query)
    
    async def get_documents_by_doc_id(self, doc_id: str) -> List[Node]:
        """
        Get all nodes belonging to a document (by doc_id).
        
        This retrieves the DocumentNode and all its chunks/children.
        
        Args:
            doc_id: The document ID
            
        Returns:
            List of all nodes with this doc_id
        """
        cursor = self.collection.find({"doc_id": doc_id})
        
        documents = []
        async for doc_dict in cursor:
            # Remove MongoDB's _id field
            doc_dict.pop('_id', None)
            
            # Get document class
            doc_class_name = doc_dict.pop('_doc_class', None)
            doc_class = self._get_doc_class(doc_class_name)
            
            # Reconstruct document
            documents.append(doc_class(**doc_dict))
        
        return documents

    async def find(
        self,
        query: Dict[str, Any],
        values: List[str],
        unique: bool = False,
    ) -> Union[List[Any], List[Dict[str, Any]]]:
        """
        Find arbitrary values from a MongoDB collection.

        Args:
            query: MongoDB filter query
            values: list of dot-path fields to return (e.g. ["metadata.source"])
            unique: whether to return unique values only

        Returns:
            - If one value field is provided:
                List[Any]
            - If multiple value fields are provided:
                List[Dict[str, Any]]
        """

        # Build projection
        projection = {field: 1 for field in values}
        projection["_id"] = 0

        cursor = await self.collection.find(query, projection)

        def extract(doc: Dict[str, Any], path: str):
            """Safely extract dotted-path values."""
            current = doc
            for key in path.split("."):
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current

        results = []

        for doc in cursor:
            if len(values) == 1:
                results.append(extract(doc, values[0]))
            else:
                results.append({v: extract(doc, v) for v in values})

        if unique:
            if len(values) == 1:
                return list({v for v in results if v is not None})
            else:
                # Deduplicate dicts
                seen = set()
                unique_results = []
                for item in results:
                    key = tuple(sorted(item.items()))
                    if key not in seen:
                        seen.add(key)
                        unique_results.append(item)
                return unique_results

        return results

    async def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
