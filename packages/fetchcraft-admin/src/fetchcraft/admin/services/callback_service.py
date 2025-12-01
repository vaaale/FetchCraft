"""Callback Service - Business logic for handling async parsing callbacks."""
import json
import logging
from typing import Any, Dict, Optional, Tuple

from fetchcraft.node import DocumentNode
from fetchcraft.ingestion.models import DocumentRecord, DocumentStatus
from fetchcraft.ingestion.repository import IDocumentRepository, IJobRepository
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue

logger = logging.getLogger(__name__)


class CallbackService:
    """
    Service layer for handling async parsing callbacks.
    
    Handles node and completion callbacks from the docling parsing server,
    creating child documents and managing parent document completion.
    """
    
    def __init__(
        self,
        doc_repo: IDocumentRepository,
        job_repo: IJobRepository,
        queue_backend: AsyncPostgresQueue
    ):
        """
        Initialize callback service.
        
        Args:
            doc_repo: Repository for document persistence
            job_repo: Repository for job retrieval
            queue_backend: Queue backend for enqueueing documents
        """
        self.doc_repo = doc_repo
        self.job_repo = job_repo
        self.queue_backend = queue_backend
    
    async def find_parent_document(
        self,
        docling_job_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[dict]]:
        """
        Find parent document and related info by docling job ID.
        
        Args:
            docling_job_id: Job ID from docling server
            
        Returns:
            Tuple of (parent_doc_id, parent_job_id, parent_source, parent_metadata)
            or (None, None, None, None) if not found
        """
        # This requires direct database access
        from fetchcraft.ingestion.repository import PostgresDocumentRepository
        
        if not isinstance(self.doc_repo, PostgresDocumentRepository):
            logger.error("Cannot find parent document without PostgreSQL repository")
            return None, None, None, None
        
        async with self.doc_repo.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, job_id, source, metadata 
                FROM ingestion_documents
                WHERE metadata->>'docling_job_id' = $1
                AND metadata->>'is_parent_document' = 'true'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                docling_job_id
            )
            
            if not row:
                return None, None, None, None
            
            # Parse metadata
            metadata = row['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            return row['id'], row['job_id'], row['source'], metadata
    
    async def handle_node_callback(
        self,
        job_id: str,
        filename: str,
        node_index: int,
        total_nodes: int,
        node: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle a node callback from docling server.
        
        Creates a child document record and enqueues it for processing.
        
        Args:
            job_id: Docling job identifier
            filename: Original filename (temp file from docling)
            node_index: Index of this node
            total_nodes: Total number of nodes expected
            node: Parsed node data
            
        Returns:
            Created document ID or None if parent not found
        """
        # Find parent document
        parent_id, parent_job_id, parent_source, parent_metadata = \
            await self.find_parent_document(job_id)
        
        if not parent_id:
            logger.error(f"Parent document not found for docling_job_id={job_id}")
            return None
        
        # Merge parent document metadata into node metadata if available
        if parent_metadata and parent_metadata.get("document"):
            try:
                parent_doc = DocumentNode.model_validate(parent_metadata["document"])
                if "metadata" in node:
                    node["metadata"].update(parent_doc.metadata)
            except Exception as e:
                logger.warning(f"Failed to merge parent metadata: {e}")
        
        # Update parent document node count
        nodes_received = parent_metadata.get('nodes_received_count', 0) + 1
        parent_metadata['nodes_received_count'] = nodes_received
        await self.doc_repo.update_document_metadata(parent_id, parent_metadata)
        
        # Get job pipeline steps
        from fetchcraft.ingestion.repository import PostgresJobRepository
        if isinstance(self.job_repo, PostgresJobRepository):
            async with self.job_repo.pool.acquire() as conn:
                job_row = await conn.fetchrow(
                    "SELECT pipeline_steps FROM ingestion_jobs WHERE id = $1",
                    parent_job_id
                )
                if not job_row:
                    logger.error(f"Job not found: {parent_job_id}")
                    return None
                
                pipeline_steps = job_row['pipeline_steps']
                if isinstance(pipeline_steps, str):
                    pipeline_steps = json.loads(pipeline_steps)
        else:
            logger.error("Cannot get pipeline steps without PostgreSQL repository")
            return None
        
        # Create child document record
        from fetchcraft.ingestion.models import utcnow
        
        child_doc = DocumentRecord(
            job_id=parent_job_id,
            source=f"{parent_source} [node {node_index + 1}/{total_nodes}]",
            status=DocumentStatus.PENDING,
            step_statuses={step: "pending" for step in pipeline_steps},
            metadata={
                'parent_document_id': parent_id,
                'docling_job_id': job_id,
                'node_index': node_index,
                'total_nodes': total_nodes,
                'filename': parent_source,
                'source': parent_source,
                'document': node,
            }
        )
        
        # Persist the document
        await self.doc_repo.create_document(child_doc)
        
        # Find parsing step index
        parsing_step_idx = None
        for idx, step in enumerate(pipeline_steps):
            if "ParsingTransformation" in step or "parsing" in step.lower():
                parsing_step_idx = idx
                break
        
        # Enqueue document to start after parsing step
        next_step_idx = parsing_step_idx + 1 if parsing_step_idx is not None else 0
        
        await self.queue_backend.enqueue(
            "ingest.main",
            body={
                "type": "document",
                "doc_id": child_doc.id,
                "job_id": parent_job_id,
                "current_step_idx": next_step_idx,
            }
        )
        
        logger.info(
            f"Enqueued node document {child_doc.id} for processing "
            f"(parent: {parent_id}, nodes_received: {nodes_received})"
        )
        
        return child_doc.id
    
    async def handle_completion_callback(
        self,
        job_id: str,
        filename: str,
        status: str,
        total_nodes: Optional[int] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Handle a completion callback from docling server.
        
        Updates parent document metadata with parsing completion status.
        Parent will be marked as completed when all child nodes finish.
        
        Args:
            job_id: Docling job identifier
            filename: Original filename
            status: Completion status ("completed" or "failed")
            total_nodes: Total number of nodes parsed
            error: Error message if failed
            
        Returns:
            True if handled successfully, False if parent not found
        """
        # Find parent document
        parent_id, _, _, parent_metadata = await self.find_parent_document(job_id)
        
        if not parent_id:
            logger.error(f"Parent document not found for docling_job_id={job_id}")
            return False
        
        # Update parent metadata with completion info
        parent_metadata['parsing_completed'] = True
        parent_metadata['parsing_status'] = status
        if total_nodes is not None:
            parent_metadata['total_nodes'] = total_nodes
        
        if status == "completed":
            # Parent stays in PROCESSING - will be marked COMPLETED when all children finish
            logger.info(
                f"Parsing completed for parent {parent_id}, "
                f"waiting for {total_nodes} child nodes to finish processing"
            )
        else:
            # Parsing failed - mark parent as failed immediately
            await self.doc_repo.update_document_status(
                parent_id,
                DocumentStatus.FAILED,
                error_message=error or "Parsing failed"
            )
            logger.error(f"Marked parent document {parent_id} as failed: {error}")
        
        # Update metadata
        await self.doc_repo.update_document_metadata(parent_id, parent_metadata)
        
        return True
