#!/usr/bin/env python3
"""
Test script to verify that job status can be checked while processing is happening.

This script:
1. Submits a job to parse a document
2. Immediately starts polling for status
3. Verifies that status requests are handled promptly during processing
"""

import asyncio
import time
from pathlib import Path
from fetchcraft.parsing.docling import AsyncDoclingParserClient


async def test_status_during_processing():
    """Test that status checks work while parsing is in progress."""
    
    # Use the test PDF from environment or default
    import os
    test_pdf = os.getenv("TEST_PDF", "/mnt/storage/data/knowledge/textfiles_tiny/finance/Crayon/Crayon_annual-report_2021.pdf")
    
    if not Path(test_pdf).exists():
        print(f"❌ Test file not found: {test_pdf}")
        print("   Set TEST_PDF environment variable to a valid PDF file")
        return
    
    client = AsyncDoclingParserClient(base_url="http://localhost:8003")
    
    print("=" * 70)
    print("Testing Job Status During Processing")
    print("=" * 70)
    
    try:
        # Check server health
        print("\n1. Checking server health...")
        health = await client.health()
        print(f"   ✓ Server status: {health['status']}")
        
        # Submit job
        print("\n2. Submitting parsing job...")
        submit_start = time.time()
        submit_response = await client.submit_job(test_pdf)
        submit_time = time.time() - submit_start
        job_id = submit_response['job_id']
        
        print(f"   ✓ Job submitted: {job_id}")
        print(f"   ✓ Submit time: {submit_time*1000:.2f}ms")
        print(f"   ✓ Initial status: {submit_response['status']}")
        
        # Poll for status with timing
        print("\n3. Polling for status (measuring response times)...")
        poll_count = 0
        response_times = []
        
        while True:
            poll_start = time.time()
            status = await client.get_job_status(job_id)
            poll_time = time.time() - poll_start
            response_times.append(poll_time)
            poll_count += 1
            
            print(f"   Poll #{poll_count}: status={status['status']}, response_time={poll_time*1000:.2f}ms")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            # Small delay between polls
            await asyncio.sleep(0.5)
        
        # Get results
        print("\n4. Fetching results...")
        results = await client.get_job_results(job_id)
        
        if results['status'] == 'completed':
            batch = results['results']
            print(f"   ✓ Job completed successfully")
            print(f"   ✓ Total nodes: {batch['total_nodes']}")
            print(f"   ✓ Processing time: {batch['total_processing_time_ms']:.2f}ms")
        else:
            print(f"   ✗ Job failed: {results['error']}")
        
        # Analysis
        print("\n5. Response Time Analysis:")
        print(f"   • Total status checks: {poll_count}")
        print(f"   • Average response time: {sum(response_times)/len(response_times)*1000:.2f}ms")
        print(f"   • Min response time: {min(response_times)*1000:.2f}ms")
        print(f"   • Max response time: {max(response_times)*1000:.2f}ms")
        
        # Success criteria
        max_acceptable_response_time = 1.0  # 1 second
        if max(response_times) < max_acceptable_response_time:
            print(f"\n✅ SUCCESS: All status checks responded within {max_acceptable_response_time}s")
            print("   The server is properly handling status requests during processing!")
        else:
            print(f"\n⚠️  WARNING: Some status checks took longer than {max_acceptable_response_time}s")
            print("   This suggests the event loop may still be blocked")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n🧪 Starting test...\n")
    asyncio.run(test_status_during_processing())
