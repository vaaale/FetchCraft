"""
Callback service for HTTP notifications.

This module handles sending HTTP callbacks to external endpoints.
"""

import asyncio
from typing import Optional, Dict, Any

try:
    import httpx
except ImportError:
    httpx = None


class CallbackService:
    """Service for sending HTTP callbacks."""
    
    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        """
        Initialize the callback service.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for callback functionality. "
                "Install it with: pip install httpx"
            )
        
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def send_callback(
        self,
        url: str,
        payload: Dict[str, Any],
        retry_count: int = 0
    ) -> bool:
        """
        Send HTTP POST callback to the specified URL.
        
        Args:
            url: The callback URL
            payload: JSON payload to send
            retry_count: Current retry attempt (internal)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                print(f"⚠️  Callback failed with status {response.status_code}: {url}")
                
                # Retry on server errors
                if response.status_code >= 500 and retry_count < self.max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    return await self.send_callback(url, payload, retry_count + 1)
                
                return False
                
        except httpx.TimeoutException:
            print(f"⚠️  Callback timeout: {url}")
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self.send_callback(url, payload, retry_count + 1)
            return False
            
        except httpx.RequestError as e:
            print(f"⚠️  Callback request error: {url} - {e}")
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self.send_callback(url, payload, retry_count + 1)
            return False
            
        except Exception as e:
            print(f"⚠️  Unexpected callback error: {url} - {e}")
            return False
    
    async def send_node_callback(
        self,
        url: str,
        job_id: str,
        filename: str,
        node_index: int,
        total_nodes: int,
        node_data: Dict[str, Any]
    ) -> bool:
        """
        Send a node callback with metadata.
        
        Args:
            url: The callback URL
            job_id: Job identifier
            filename: Source filename
            node_index: Index of this node (0-based)
            total_nodes: Total number of nodes in the document
            node_data: The parsed node data
            
        Returns:
            True if successful, False otherwise
        """
        payload = {
            "job_id": job_id,
            "filename": filename,
            "node_index": node_index,
            "total_nodes": total_nodes,
            "node": node_data
        }
        
        return await self.send_callback(url, payload)
    
    async def send_completion_callback(
        self,
        url: str,
        job_id: str,
        filename: str,
        total_nodes: int,
        processing_time_ms: float
    ) -> bool:
        """
        Send a completion callback indicating file processing is complete.
        
        Args:
            url: The callback URL
            job_id: Job identifier
            filename: Source filename
            total_nodes: Total number of nodes parsed
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        payload = {
            "job_id": job_id,
            "filename": filename,
            "status": "completed",
            "total_nodes": total_nodes,
            "processing_time_ms": processing_time_ms
        }
        
        return await self.send_callback(url, payload)
    
    async def send_failure_callback(
        self,
        url: str,
        job_id: str,
        filename: str,
        error: str,
        processing_time_ms: float
    ) -> bool:
        """
        Send a failure callback indicating file processing failed.
        
        Args:
            url: The callback URL
            job_id: Job identifier
            filename: Source filename
            error: Error message
            processing_time_ms: Processing time before failure in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        payload = {
            "job_id": job_id,
            "filename": filename,
            "status": "failed",
            "error": error,
            "processing_time_ms": processing_time_ms
        }
        
        return await self.send_callback(url, payload)
    
    # =========================================================================
    # New CallbackMessage format methods (for async transformation support)
    # =========================================================================
    
    async def send_task_callback(
        self,
        url: str,
        task_id: str,
        status: str,
        message: Dict[str, Any],
        error: Optional[str] = None
    ) -> bool:
        """
        Send a callback in the new CallbackMessage format.
        
        This is the standard format for async transformation callbacks.
        
        Args:
            url: The callback URL
            task_id: Task identifier for correlation
            status: Status ('PROCESSING', 'COMPLETED', 'FAILED')
            message: Callback payload (e.g., parsed node data)
            error: Error message if status is FAILED
            
        Returns:
            True if successful, False otherwise
        """
        payload = {
            "task_id": task_id,
            "status": status,
            "message": message,
        }
        if error:
            payload["error"] = error
        
        return await self.send_callback(url, payload)
    
    async def send_task_node_callback(
        self,
        url: str,
        task_id: str,
        node_data: Dict[str, Any],
        node_index: int,
        total_nodes: int,
        filename: str
    ) -> bool:
        """
        Send a node callback in CallbackMessage format.
        
        Args:
            url: The callback URL
            task_id: Task identifier
            node_data: The parsed node data
            node_index: Index of this node (0-based)
            total_nodes: Total number of nodes
            filename: Source filename
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "node",
            "node": node_data,
            "node_index": node_index,
            "total_nodes": total_nodes,
            "filename": filename,
        }
        
        return await self.send_task_callback(url, task_id, "PROCESSING", message)
    
    async def send_task_completion_callback(
        self,
        url: str,
        task_id: str,
        total_nodes: int,
        filename: str,
        processing_time_ms: float
    ) -> bool:
        """
        Send a completion callback in CallbackMessage format.
        
        Args:
            url: The callback URL
            task_id: Task identifier
            total_nodes: Total number of nodes parsed
            filename: Source filename
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "completion",
            "total_nodes": total_nodes,
            "filename": filename,
            "processing_time_ms": processing_time_ms,
        }
        
        return await self.send_task_callback(url, task_id, "COMPLETED", message)
    
    async def send_task_failure_callback(
        self,
        url: str,
        task_id: str,
        error: str,
        filename: str,
        processing_time_ms: float
    ) -> bool:
        """
        Send a failure callback in CallbackMessage format.
        
        Args:
            url: The callback URL
            task_id: Task identifier
            error: Error message
            filename: Source filename
            processing_time_ms: Processing time before failure
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "failure",
            "filename": filename,
            "processing_time_ms": processing_time_ms,
        }
        
        return await self.send_task_callback(url, task_id, "FAILED", message, error=error)
