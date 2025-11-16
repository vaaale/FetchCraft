"""
Example demonstrating async job-based document parsing.

This example shows how to use the new async job interface where documents
are submitted for parsing and the client can poll for completion rather than
keeping a connection open.
"""

import asyncio
import time
from pathlib import Path
from fetchcraft.parsing.docling import AsyncDoclingParserClient
import os

TEST_PDF = os.getenv("TEST_PDF", "/mnt/storage/data/knowledge/textfiles_tiny/finance/Crayon/Crayon_annual-report_2021.pdf")
TEST_PDF2 = os.getenv("TEST_PDF", "/mnt/storage/data/knowledge/textfiles_tiny/finance/Crayon/Crayon_annual-report_2023.pdf")

async def example_submit_and_poll(document_path: str):
    """Example: Submit job and manually poll for completion."""
    client = AsyncDoclingParserClient(base_url="http://localhost:8003")
    
    print("=" * 70)
    print("Example 1: Submit and Poll Manually")
    print("=" * 70)
    
    # Submit job - returns immediately with job_id
    print("\n1. Submitting job...")
    submit_response = await client.submit_job(document_path)
    job_id = submit_response['job_id']
    print(f"   ✓ Job submitted: {job_id}")
    print(f"   Status: {submit_response['status']}")
    print(f"   Message: {submit_response['message']}")
    
    # Poll for status
    print("\n2. Polling for completion...")
    while True:
        status = await client.get_job_status(job_id)
        print(f"   Status: {status['status']}")
        
        if status['status'] == 'completed':
            break
        elif status['status'] == 'failed':
            print(f"   Error: {status['error']}")
            return
        
        # Wait before next poll
        await asyncio.sleep(1)
    
    # Get results
    print("\n3. Fetching results...")
    results = await client.get_job_results(job_id)
    batch_results = results['results']
    
    print(f"   ✓ Total files: {batch_results['total_files']}")
    print(f"   ✓ Successful: {batch_results['successful']}")
    print(f"   ✓ Total nodes: {batch_results['total_nodes']}")
    print(f"   ✓ Processing time: {batch_results['total_processing_time_ms']:.2f}ms")


async def example_submit_and_wait(document_path: str, document_path2: str):
    """Example: Submit job and wait using convenience method."""
    client = AsyncDoclingParserClient()
    
    print("\n" + "=" * 70)
    print("Example 2: Submit and Wait (Convenience Method)")
    print("=" * 70)
    
    # This is simpler - just submit and wait
    print("\n1. Submitting and waiting for completion...")
    results = await client.submit_and_wait(
        document_path,
        document_path2,
        poll_interval=1.0,  # Poll every 1 second
        timeout=300  # Timeout after 5 minutes
    )
    
    print(f"   ✓ Total files: {results['total_files']}")
    print(f"   ✓ Successful: {results['successful']}")
    print(f"   ✓ Total nodes: {results['total_nodes']}")
    print(f"   ✓ Processing time: {results['total_processing_time_ms']:.2f}ms")
    
    # Access individual file results
    print("\n2. Individual file results:")
    for result in results['results']:
        print(f"   • {result['filename']}: {result['num_nodes']} nodes")


async def example_multiple_jobs(document_path: str, document_path2: str):
    """Example: Submit multiple jobs concurrently."""
    client = AsyncDoclingParserClient()
    
    print("\n" + "=" * 70)
    print("Example 3: Multiple Jobs Concurrently")
    print("=" * 70)
    
    # Submit multiple jobs
    print("\n1. Submitting multiple jobs...")
    job_ids = []
    
    for i, doc in enumerate([document_path, document_path2]):
        response = await client.submit_job(doc)
        job_id = response['job_id']
        job_ids.append(job_id)
        print(f"   ✓ Job {i+1} submitted: {job_id}")
    
    # Wait for all to complete
    print("\n2. Waiting for all jobs to complete...")
    completed = []
    
    while len(completed) < len(job_ids):
        for job_id in job_ids:
            if job_id in completed:
                continue
            
            status = await client.get_job_status(job_id)
            if status['status'] == 'completed':
                completed.append(job_id)
                print(f"   ✓ Job {job_id} completed")
            elif status['status'] == 'failed':
                completed.append(job_id)
                print(f"   ✗ Job {job_id} failed: {status['error']}")
        
        if len(completed) < len(job_ids):
            await asyncio.sleep(1)
    
    # Fetch all results
    print("\n3. Fetching results...")
    for job_id in job_ids:
        results = await client.get_job_results(job_id)
        if results['status'] == 'completed':
            batch = results['results']
            print(f"   Job {job_id}: {batch['total_nodes']} nodes")


async def main():
    """Run all examples."""
    client = AsyncDoclingParserClient()
    
    # Check server health
    print("Checking server health...")
    health = await client.health()
    print(f"Server status: {health['status']}")
    print(f"Version: {health['version']}\n")
    
    # Run examples
    # Note: Uncomment the examples you want to run and provide valid file paths
    
    await example_submit_and_poll(TEST_PDF)
    # await example_submit_and_wait()
    # await example_multiple_jobs()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
