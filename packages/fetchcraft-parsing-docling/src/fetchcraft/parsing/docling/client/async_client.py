"""
Async client example for the Docling parsing server.

This example demonstrates async usage with aiohttp for better performance
when parsing many documents.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

import aiohttp


class AsyncDoclingParserClient:
    """Async client for the Docling parsing server."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 60 * 40):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the parsing server
        """
        self.base_url = base_url
        self.timeout = timeout

    async def health(self) -> Dict[str, Any]:
        """
        Check server health.
        
        Returns:
            Health status and configuration
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                return await response.json()

    async def parse(self, *file_paths) -> Dict[str, Any]:
        """
        Parse one or more documents.
        
        Args:
            *file_paths: Variable number of file paths (str or Path)
            
        Returns:
            Parsing results with nodes for each file
        """
        data = aiohttp.FormData()

        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            data.add_field('files', open(path, 'rb'), filename=path.name)

        async with aiohttp.ClientSession(read_timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/parse", data=data) as response:
                response.raise_for_status()
                return await response.json()

    async def parse_single(self, file_path) -> Dict[str, Any]:
        """
        Parse a single document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parsing result for the file
        """
        result = await self.parse(file_path)
        return result['results'][0]

    async def parse_batch(self, file_paths: List[Path], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Parse multiple documents in batches.
        
        This is useful when you have many files and want to control
        how many requests are sent to the server concurrently.
        
        Args:
            file_paths: List of file paths to parse
            batch_size: Number of files to send per request
            
        Returns:
            List of all parsing results
        """
        all_results = []

        # Split files into batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            results = await self.parse(*batch)
            all_results.extend(results['results'])

        return all_results

    async def parse_parallel(self, file_paths: List[Path], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Parse multiple documents in parallel with concurrency limit.
        
        Each file is sent as a separate request, but with a limit on
        concurrent requests.
        
        Args:
            file_paths: List of file paths to parse
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of all parsing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_with_semaphore(path):
            async with semaphore:
                return await self.parse_single(path)

        tasks = [parse_with_semaphore(path) for path in file_paths]
        return await asyncio.gather(*tasks)


async def main():
    """Example usage."""
    client = AsyncDoclingParserClient()

    # Check health
    print("Checking server health...")
    health = await client.health()
    print(f"Server status: {health['status']}")

    # Example 1: Parse single file
    print("\n--- Example 1: Single file ---")
    result = await client.parse_single("document.pdf")
    print(f"Filename: {result['filename']}")
    print(f"Nodes: {result['num_nodes']}")

    # Example 2: Parse multiple files in one request
    print("\n--- Example 2: Multiple files (single request) ---")
    results = await client.parse("doc1.pdf", "doc2.docx", "doc3.pptx")
    print(f"Total files: {results['total_files']}")
    print(f"Total nodes: {results['total_nodes']}")

    # Example 3: Parse many files in batches
    print("\n--- Example 3: Batch processing ---")
    file_paths = [Path(f"document_{i}.pdf") for i in range(20)]
    # Only create this if files exist
    # results = await client.parse_batch(file_paths, batch_size=5)
    # print(f"Processed {len(results)} files")

    # Example 4: Parse files in parallel with concurrency limit
    print("\n--- Example 4: Parallel processing ---")
    file_paths = [Path(f"document_{i}.pdf") for i in range(10)]
    # Only create this if files exist
    # results = await client.parse_parallel(file_paths, max_concurrent=3)
    # print(f"Processed {len(results)} files in parallel")

    print("\n✓ Examples completed")


if __name__ == "__main__":
    asyncio.run(main())
