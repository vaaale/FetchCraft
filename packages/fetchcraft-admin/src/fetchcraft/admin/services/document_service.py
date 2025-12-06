"""Document Service - Business logic for document operations."""
import logging
from typing import List, Optional

from fetchcraft.ingestion.models import DocumentRecord, DocumentStatus
from fetchcraft.ingestion.repository import DocumentRepository
from fetchcraft.admin.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service layer for document-related business logic.
    
    Encapsulates all document management operations including retrieval,
    listing, and retry operations.
    """
    
    def __init__(
        self,
        doc_repo: DocumentRepository,
        ingestion_service: IngestionService
    ):
        """
        Initialize document service.
        
        Args:
            doc_repo: Repository for document persistence
            ingestion_service: Service for managing ingestion pipeline
        """
        self.doc_repo = doc_repo
        self.ingestion_service = ingestion_service
    
    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        return await self.doc_repo.get_document(doc_id)
    
    async def list_job_documents(
        self,
        job_id: str,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """
        List documents for a job with optional filtering.
        
        Args:
            job_id: Job identifier
            status: Optional status filter
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of documents matching criteria
        """
        return await self.ingestion_service.get_job_documents(
            job_id=job_id,
            status=status,
            limit=limit,
            offset=offset
        )
    
    async def retry_failed_documents(self, job_id: str) -> int:
        """
        Retry all failed documents for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Number of documents retried
        """
        count = await self.ingestion_service.retry_failed_documents(job_id)
        logger.info(f"Retried {count} failed documents for job {job_id}")
        return count
    
    async def get_document_counts(self, job_id: str) -> dict:
        """
        Get document counts by status for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary mapping status to count
        """
        counts = {}
        for status in DocumentStatus:
            docs = await self.doc_repo.get_documents_by_status(job_id, status)
            counts[status.value] = len(docs)
        
        return counts
    
    async def get_parent_document_by_job_id(
        self,
        docling_job_id: str
    ) -> Optional[DocumentRecord]:
        """
        Find parent document by docling job ID.
        
        Searches for a document with matching docling_job_id in metadata
        and is_parent_document flag set.
        
        Args:
            docling_job_id: Docling job identifier from callback
            
        Returns:
            Parent document if found, None otherwise
        """
        # This requires a custom query - delegate to repository
        # For now, we'd need to add this method to the repository interface
        logger.warning(
            "get_parent_document_by_job_id not yet implemented in repository"
        )
        return None
