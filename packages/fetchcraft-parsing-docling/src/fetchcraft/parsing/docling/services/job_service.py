"""
Job service for job management business logic.

This module handles the core business logic for job lifecycle management.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Tuple

from ..models import BatchParseResponse, JobStatusEnum
from ..repositories.job_repository import Job, JobRepository
from .parsing_service import ParsingService


class JobService:
    """Service for managing parsing jobs."""
    
    def __init__(
        self,
        job_repository: JobRepository,
        parsing_service: ParsingService,
        max_file_size_bytes: int,
        callback_service = None
    ):
        """
        Initialize the job service.
        
        Args:
            job_repository: Repository for job persistence
            parsing_service: Service for parsing documents
            max_file_size_bytes: Maximum allowed file size in bytes
            callback_service: Optional service for HTTP callbacks
        """
        self.repository = job_repository
        self.parsing_service = parsing_service
        self.max_file_size_bytes = max_file_size_bytes
        self.callback_service = callback_service
        
        # In-memory job cache
        self.jobs: Dict[str, Job] = {}
    
    def load_jobs(self) -> None:
        """Load all jobs from repository into memory."""
        self.jobs = self.repository.load_all_jobs()
    
    def save_jobs(self) -> None:
        """Save all jobs from memory to repository."""
        self.repository.save_all_jobs(self.jobs)
    
    def create_job(
        self,
        files_data: List[Tuple[str, bytes]],
        callback_url: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Create a new job and save uploaded files.
        
        Args:
            files_data: List of (filename, content) tuples
            callback_url: Optional URL to receive parsing callbacks
            
        Returns:
            Tuple of (job_id, list of filenames)
            
        Raises:
            ValueError: If file size exceeds maximum
        """
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save files and collect filenames
        filenames = []
        for filename, content in files_data:
            # Check file size
            if len(content) > self.max_file_size_bytes:
                raise ValueError(
                    f"File {filename} size ({len(content) / 1024 / 1024:.2f} MB) "
                    f"exceeds maximum ({self.max_file_size_bytes / 1024 / 1024:.2f} MB)"
                )
            
            # Save file to job directory
            self.repository.save_file(job_id, filename, content)
            filenames.append(filename)
        
        # Create job
        job = Job(
            job_id=job_id,
            files=filenames,
            status=JobStatusEnum.PENDING,
            callback_url=callback_url
        )
        
        # Store job in memory and persist
        self.jobs[job_id] = job
        self.save_jobs()
        
        return job_id, filenames
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatusEnum,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        results: Optional[BatchParseResponse] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update job status and related fields.
        
        Args:
            job_id: The job ID
            status: New status
            started_at: Start timestamp (optional)
            completed_at: Completion timestamp (optional)
            results: Parsing results (optional)
            error: Error message (optional)
        """
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job.status = status
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if results is not None:
            job.results = results
        if error is not None:
            job.error = error
        
        self.save_jobs()
    
    async def process_job(
        self,
        job_id: str,
        executor,
        file_semaphore: asyncio.Semaphore
    ) -> None:
        """
        Process a single job.
        
        Args:
            job_id: The job ID to process
            executor: ThreadPoolExecutor for parsing
            file_semaphore: Semaphore for controlling concurrent file processing
        """
        job = self.jobs.get(job_id)
        if not job:
            return
        
        # Update job status to processing
        job.status = JobStatusEnum.PROCESSING
        job.started_at = time.time()
        self.save_jobs()
        
        print(f"⚙️  Processing job {job_id} with {len(job.files)} files")
        
        try:
            batch_start_time = time.time()
            
            # Get job directory where files are stored
            job_dir = self.repository.get_job_directory(job_id)
            
            # Prepare callback handlers if callback_url is provided
            callback_handler = None
            completion_callback_handler = None
            
            if job.callback_url and self.callback_service:
                async def node_callback(node_dict: dict, node_index: int, metadata: dict):
                    """Handler to send node callbacks."""
                    await self.callback_service.send_node_callback(
                        url=job.callback_url,
                        job_id=metadata['job_id'],
                        filename=metadata['filename'],
                        node_index=node_index,
                        total_nodes=metadata.get('total_nodes', -1),  # Unknown until complete
                        node_data=node_dict
                    )
                
                async def completion_callback(
                    success: bool,
                    total_nodes: int,
                    processing_time_ms: float,
                    error: str,
                    metadata: dict
                ):
                    """Handler to send completion/failure callbacks."""
                    if success:
                        await self.callback_service.send_completion_callback(
                            url=job.callback_url,
                            job_id=metadata['job_id'],
                            filename=metadata['filename'],
                            total_nodes=total_nodes,
                            processing_time_ms=processing_time_ms
                        )
                    else:
                        await self.callback_service.send_failure_callback(
                            url=job.callback_url,
                            job_id=metadata['job_id'],
                            filename=metadata['filename'],
                            error=error,
                            processing_time_ms=processing_time_ms
                        )
                
                callback_handler = node_callback
                completion_callback_handler = completion_callback
            
            # Parse files individually to support callbacks
            results = []
            loop = asyncio.get_event_loop()
            
            for filename in job.files:
                file_path = job_dir / filename
                
                # Prepare callback metadata for this file
                callback_metadata = {
                    'job_id': job_id,
                    'filename': filename
                } if callback_handler else None
                
                # Parse file with callback support
                async with file_semaphore:
                    result = await loop.run_in_executor(
                        executor,
                        self.parsing_service.parse_file_sync,
                        file_path,
                        callback_handler,
                        callback_metadata,
                        completion_callback_handler
                    )
                    results.append(result)
                    await asyncio.sleep(0)  # Yield control
            
            batch_processing_time = (time.time() - batch_start_time) * 1000
            
            # Aggregate statistics
            total_files = len(results)
            successful = sum(1 for r in results if r.success)
            failed = total_files - successful
            total_nodes = sum(r.num_nodes for r in results)
            
            job.results = BatchParseResponse(
                results=results,
                total_files=total_files,
                successful=successful,
                failed=failed,
                total_nodes=total_nodes,
                total_processing_time_ms=round(batch_processing_time, 2)
            )
            job.status = JobStatusEnum.COMPLETED
            print(f"✅ Job {job_id} completed: {successful}/{total_files} files successful")
            
        except Exception as e:
            job.status = JobStatusEnum.FAILED
            job.error = str(e)
            print(f"❌ Job {job_id} failed: {e}")
        
        finally:
            job.completed_at = time.time()
            self.save_jobs()
    
    async def resume_unfinished_jobs(self) -> List[str]:
        """
        Resume processing of unfinished jobs on startup.
        
        This function:
        1. Finds all jobs in PENDING or PROCESSING state
        2. Validates that job files still exist
        3. Resets PROCESSING jobs to PENDING (they were interrupted)
        4. Returns list of job IDs to queue
        
        Returns:
            List of job IDs to queue for processing
        """
        unfinished_jobs = []
        failed_jobs = []
        
        for job_id, job in self.jobs.items():
            if job.status in [JobStatusEnum.PENDING, JobStatusEnum.PROCESSING]:
                # Verify that all job files exist
                missing_files = []
                
                for filename in job.files:
                    if not self.repository.file_exists(job_id, filename):
                        missing_files.append(filename)
                
                if missing_files:
                    # Mark job as failed if files are missing
                    print(f"❌ Job {job_id} has missing files: {missing_files}")
                    job.status = JobStatusEnum.FAILED
                    job.error = f"Missing files: {', '.join(missing_files)}"
                    job.completed_at = time.time()
                    failed_jobs.append(job_id)
                else:
                    # Reset PROCESSING jobs to PENDING since they were interrupted
                    if job.status == JobStatusEnum.PROCESSING:
                        print(f"🔄 Resetting interrupted job {job_id} to PENDING")
                        job.status = JobStatusEnum.PENDING
                        job.started_at = None
                    
                    unfinished_jobs.append(job_id)
        
        # Report summary
        if unfinished_jobs:
            print(f"📥 Found {len(unfinished_jobs)} unfinished job(s) to resume")
        
        if failed_jobs:
            print(f"⚠️  Marked {len(failed_jobs)} job(s) as failed due to missing files")
        
        if not unfinished_jobs and not failed_jobs:
            print("✅ No unfinished jobs to resume")
        
        # Save updated job states
        if unfinished_jobs or failed_jobs:
            self.save_jobs()
        
        return unfinished_jobs
