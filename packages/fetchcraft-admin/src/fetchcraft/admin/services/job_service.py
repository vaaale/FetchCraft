"""Job Service - Business logic for job operations."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fetchcraft.admin.models.models import Job
from fetchcraft.ingestion.models import JobStatus
from fetchcraft.ingestion.repository import JobRepository, DocumentRepository
from fetchcraft.admin.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


class JobService:
    """
    Service layer for job-related business logic.
    
    Encapsulates all job management operations including creation,
    retrieval, deletion, and restart operations.
    """
    
    def __init__(
        self,
        job_repo: JobRepository,
        doc_repo: DocumentRepository,
        ingestion_service: IngestionService,
        document_root: Path
    ):
        """
        Initialize job service.
        
        Args:
            job_repo: Repository for job persistence
            doc_repo: Repository for document persistence
            ingestion_service: Service for managing ingestion pipeline
            document_root: Root directory for document files
        """
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.ingestion_service = ingestion_service
        self.document_root = document_root
    
    async def create_job(
        self,
        name: str,
        source_path: str,
        parser_map: Dict,
        chunker,
        vector_index,
        index_id: str
    ) -> Job:
        """
        Create and start a new ingestion job.
        
        Args:
            name: Job name
            source_path: Path to source documents (relative to document_root)
            parser_map: Map of file extensions to parsers
            chunker: Document chunker
            vector_index: Vector index for storage
            index_id: Index identifier
            
        Returns:
            Created job
            
        Raises:
            ValueError: If source path is invalid or job creation fails
        """
        # Validate source path
        full_path = self.document_root / source_path
        if not full_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Create job via ingestion service
        try:
            job_id = await self.ingestion_service.create_job(
                name=name,
                source_path=source_path,
                parser_map=parser_map,
                chunker=chunker,
                vector_index=vector_index,
                index_id=index_id
            )
        except Exception as e:
            logger.error(f"Failed to create job '{name}': {e}", exc_info=True)
            raise ValueError(f"Job creation failed: {e}")
        
        # Retrieve created job
        job = await self.job_repo.get_job(job_id)
        if not job:
            raise ValueError(f"Job created but not found: {job_id}")
        
        logger.info(f"Created job '{name}' with ID {job_id}")
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job if found, None otherwise
        """
        return await self.job_repo.get_job(job_id)
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Job]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            List of jobs matching criteria
        """
        return await self.job_repo.list_jobs(status, limit, offset)
    
    async def delete_job(
        self,
        job_id: str,
        delete_documents: bool = True
    ) -> bool:
        """
        Delete a job and optionally its documents.
        
        Args:
            job_id: Job identifier
            delete_documents: Whether to delete associated documents
            
        Returns:
            True if job was deleted, False if not found
            
        Raises:
            ValueError: If job is in invalid state for deletion
        """
        # Check if job exists and can be deleted
        job = await self.job_repo.get_job(job_id)
        if not job:
            return False
        
        # Only allow deletion of completed, failed, or cancelled jobs
        if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise ValueError(
                f"Cannot delete job in state '{job.status}'. "
                "Only completed, failed, or cancelled jobs can be deleted."
            )
        
        # Delete via ingestion service
        deleted = await self.ingestion_service.delete_job(job_id, delete_documents)
        if deleted:
            logger.info(f"Deleted job {job_id} (delete_documents={delete_documents})")
        
        return deleted
    
    async def stop_job(self, job_id: str) -> bool:
        """
        Stop a running job.
        
        Stops the pipeline workers and marks the job as cancelled.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was stopped, False if not found or not running
            
        Raises:
            ValueError: If job cannot be stopped (wrong state)
        """
        # Delegate to ingestion service
        stopped = await self.ingestion_service.stop_job(job_id)
        if stopped:
            logger.info(f"Stopped job {job_id}")
        
        return stopped
    
    async def restart_job(
        self,
        job_id: str,
        parser_map: Dict,
        chunker,
        vector_index,
        index_id: str
    ) -> Job:
        """
        Restart a completed or failed job.
        
        Creates a new job with the same configuration.
        
        Args:
            job_id: ID of job to restart
            parser_map: Map of file extensions to parsers
            chunker: Document chunker
            vector_index: Vector index for storage
            index_id: Index identifier
            
        Returns:
            New job
            
        Raises:
            ValueError: If original job not found or restart fails
        """
        # Verify original job exists
        original_job = await self.job_repo.get_job(job_id)
        if not original_job:
            raise ValueError(f"Job not found: {job_id}")
        
        # Restart via ingestion service
        try:
            new_job_id = await self.ingestion_service.restart_job(
                job_id=job_id,
                parser_map=parser_map,
                chunker=chunker,
                vector_index=vector_index,
                index_id=index_id
            )
        except Exception as e:
            logger.error(f"Failed to restart job {job_id}: {e}", exc_info=True)
            raise ValueError(f"Job restart failed: {e}")
        
        # Retrieve new job
        new_job = await self.job_repo.get_job(new_job_id)
        if not new_job:
            raise ValueError(f"Restarted job not found: {new_job_id}")
        
        logger.info(f"Restarted job {job_id} as {new_job_id}")
        return new_job
    
    async def get_job_statistics(self, job_id: str) -> Optional[Dict]:
        """
        Get statistics for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job statistics or None if job not found
        """
        job = await self.job_repo.get_job(job_id)
        if not job:
            return None
        
        # Get document counts by status
        from fetchcraft.ingestion.models import DocumentStatus
        
        total_docs = 0
        pending_docs = 0
        processing_docs = 0
        completed_docs = 0
        failed_docs = 0
        
        # Count documents in each status
        for status in DocumentStatus:
            docs = await self.doc_repo.get_documents_by_status(job_id, status)
            count = len(docs)
            total_docs += count
            
            if status == DocumentStatus.PENDING:
                pending_docs = count
            elif status == DocumentStatus.PROCESSING:
                processing_docs = count
            elif status == DocumentStatus.COMPLETED:
                completed_docs = count
            elif status == DocumentStatus.FAILED:
                failed_docs = count
        
        return {
            "job_id": job_id,
            "job_name": job.name,
            "job_status": job.status.value,
            "total_documents": total_docs,
            "pending": pending_docs,
            "processing": processing_docs,
            "completed": completed_docs,
            "failed": failed_docs,
            "success_rate": (completed_docs / total_docs * 100) if total_docs > 0 else 0
        }
