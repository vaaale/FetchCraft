"""
Simple client example for the Docling parsing server.

This example demonstrates basic usage of the parsing API.
"""

import time
import requests
from pathlib import Path
from typing import Optional


class DoclingParserClient:
    """Simple client for the Docling parsing server."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the parsing server
        """
        self.base_url = base_url
    
    def health(self) -> dict:
        """
        Check server health.
        
        Returns:
            Health status and configuration
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def parse(self, *file_paths) -> dict:
        """
        Parse one or more documents.
        
        Args:
            *file_paths: Variable number of file paths (str or Path)
            
        Returns:
            Parsing results with nodes for each file
        """
        files = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            files.append(("files", (path.name, open(path, "rb"))))
        
        try:
            response = requests.post(f"{self.base_url}/parse", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            # Close file handles
            for _, (_, f) in files:
                f.close()
    
    def parse_single(self, file_path) -> dict:
        """
        Parse a single document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parsing result for the file
        """
        result = self.parse(file_path)
        return result['results'][0]

    def submit_job(self, *file_paths) -> dict:
        """
        Submit files for parsing and return immediately with a job ID.
        
        This is non-blocking - it returns a job ID immediately without
        waiting for parsing to complete. Use get_job_status() to poll
        for completion and get_job_results() to retrieve results.
        
        Args:
            *file_paths: Variable number of file paths (str or Path)
            
        Returns:
            Job submission response with job_id
        """
        files = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            files.append(("files", (path.name, open(path, "rb"))))
        
        try:
            response = requests.post(f"{self.base_url}/submit", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            # Close file handles
            for _, (_, f) in files:
                f.close()

    def get_job_status(self, job_id: str) -> dict:
        """
        Get the current status of a submitted job.
        
        Args:
            job_id: The job identifier returned from submit_job()
            
        Returns:
            Job status information including current state and timestamps
        """
        response = requests.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def get_job_results(self, job_id: str) -> dict:
        """
        Get the results of a completed job.
        
        Args:
            job_id: The job identifier returned from submit_job()
            
        Returns:
            Job results including parsing output if completed
        """
        response = requests.get(f"{self.base_url}/jobs/{job_id}/results")
        response.raise_for_status()
        return response.json()

    def submit_and_wait(
        self, 
        *file_paths,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None
    ) -> dict:
        """
        Submit files for parsing and wait for completion.
        
        This is a convenience method that submits a job and polls until
        it completes, then returns the results. Use this for a simpler
        API when you want to wait for results.
        
        Args:
            *file_paths: Variable number of file paths (str or Path)
            poll_interval: Seconds to wait between status checks (default: 1.0)
            timeout: Maximum seconds to wait for completion (default: None = no timeout)
            
        Returns:
            Parsing results when job completes
            
        Raises:
            TimeoutError: If timeout is reached before job completes
            RuntimeError: If job fails
        """
        # Submit job
        submit_response = self.submit_job(*file_paths)
        job_id = submit_response['job_id']
        
        print(f"Job {job_id} submitted, waiting for completion...")
        
        start_time = time.time()
        
        while True:
            # Check status
            status = self.get_job_status(job_id)
            
            if status['status'] == 'completed':
                # Get results
                results = self.get_job_results(job_id)
                print(f"Job {job_id} completed successfully")
                return results['results']
            
            elif status['status'] == 'failed':
                error = status.get('error', 'Unknown error')
                raise RuntimeError(f"Job {job_id} failed: {error}")
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
            
            # Wait before next poll
            time.sleep(poll_interval)


def main():
    """Example usage."""
    # Create client
    client = DoclingParserClient()
    
    # Check health
    print("Checking server health...")
    health = client.health()
    print(f"Server status: {health['status']}")
    print(f"Version: {health['version']}")
    
    # Example 1: Parse a single file
    print("\n--- Example 1: Single file ---")
    result = client.parse_single("/mnt/storage/data/Finance/Crayon/Crayon_annual-report_2023.pdf")
    print(f"Filename: {result['filename']}")
    print(f"Success: {result['success']}")
    print(f"Nodes: {result['num_nodes']}")
    
    if result['nodes']:
        first_node = result['nodes'][0]
        print(f"First node text: {first_node['text'][:100]}...")
    
    # Example 2: Parse multiple files
    print("\n--- Example 2: Multiple files ---")
    results = client.parse("document1.pdf", "document2.docx", "report.pptx")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Total nodes: {results['total_nodes']}")
    
    for result in results['results']:
        print(f"\n  {result['filename']}: {result['num_nodes']} nodes")
    
    # Example 3: Access metadata
    print("\n--- Example 3: Metadata ---")
    result = client.parse_single("document.pdf")
    if result['nodes']:
        node = result['nodes'][0]
        metadata = node['metadata']
        print(f"Page: {metadata.get('page_number')}")
        print(f"Total pages: {metadata.get('total_pages')}")
        print(f"File size: {metadata.get('file_size')} bytes")

    # Example 4: Async job submission (non-blocking)
    print("\n--- Example 4: Async job submission ---")
    # submit_response = client.submit_job("document.pdf")
    # job_id = submit_response['job_id']
    # print(f"Job submitted: {job_id}")
    # 
    # # Poll for status
    # status = client.get_job_status(job_id)
    # print(f"Job status: {status['status']}")
    # 
    # # Wait for completion and get results
    # while status['status'] not in ['completed', 'failed']:
    #     time.sleep(1)
    #     status = client.get_job_status(job_id)
    # 
    # if status['status'] == 'completed':
    #     results = client.get_job_results(job_id)
    #     print(f"Job completed with {results['results']['total_nodes']} nodes")

    # Example 5: Submit and wait (convenience method)
    print("\n--- Example 5: Submit and wait ---")
    # results = client.submit_and_wait("document.pdf", poll_interval=1.0)
    # print(f"Parsing completed with {results['total_nodes']} nodes")


if __name__ == "__main__":
    main()
