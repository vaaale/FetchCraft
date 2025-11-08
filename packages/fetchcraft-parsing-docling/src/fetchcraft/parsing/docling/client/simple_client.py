"""
Simple client example for the Docling parsing server.

This example demonstrates basic usage of the parsing API.
"""

import requests
from pathlib import Path


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


if __name__ == "__main__":
    main()
