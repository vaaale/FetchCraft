"""
Test client for the Docling parsing server.

This script demonstrates how to use the parsing API with both single
and multiple file uploads.

Usage:
    python test_client.py <file1> [file2] [file3] ...
"""

import sys
import json
import requests
from pathlib import Path
from typing import List


def parse_documents(server_url: str, file_paths: List[Path]) -> dict:
    """
    Parse documents using the Docling parsing server.
    
    Args:
        server_url: Base URL of the parsing server
        file_paths: List of file paths to parse
        
    Returns:
        Response dictionary with parsing results
    """
    endpoint = f"{server_url}/parse"
    
    # Prepare files for upload
    files = []
    for file_path in file_paths:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        files.append(("files", (file_path.name, open(file_path, "rb"))))
    
    if not files:
        print("Error: No valid files to upload")
        sys.exit(1)
    
    print(f"Uploading {len(files)} file(s) to {endpoint}...")
    
    try:
        response = requests.post(endpoint, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to server: {e}")
        sys.exit(1)
    finally:
        # Close file handles
        for _, (_, file_handle) in files:
            file_handle.close()


def print_results(results: dict):
    """
    Print parsing results in a readable format.
    
    Args:
        results: Response dictionary from the parsing server
    """
    print("\n" + "=" * 70)
    print("PARSING RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Statistics:")
    print(f"  • Total Files: {results['total_files']}")
    print(f"  • Successful: {results['successful']}")
    print(f"  • Failed: {results['failed']}")
    print(f"  • Total Nodes: {results['total_nodes']}")
    print(f"  • Total Processing Time: {results['total_processing_time_ms']:.2f} ms")
    
    print(f"\nIndividual Results:")
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}. {result['filename']}")
        print(f"   Status: {'✓ Success' if result['success'] else '✗ Failed'}")
        print(f"   Nodes: {result['num_nodes']}")
        print(f"   Processing Time: {result['processing_time_ms']:.2f} ms")
        
        if result['error']:
            print(f"   Error: {result['error']}")
        
        if result['nodes'] and result['success']:
            print(f"\n   Sample Node (first of {len(result['nodes'])}):")
            node = result['nodes'][0]
            print(f"   - ID: {node['id']}")
            print(f"   - Type: {node['node_type']}")
            print(f"   - Text Length: {len(node['text'])} characters")
            
            # Print metadata
            if node.get('metadata'):
                print(f"   - Metadata:")
                for key, value in node['metadata'].items():
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:50] + "..."
                    print(f"     • {key}: {value}")
            
            # Print text preview
            text_preview = node['text'][:200]
            if len(node['text']) > 200:
                text_preview += "..."
            print(f"\n   Text Preview:")
            print(f"   {text_preview}")


def check_health(server_url: str):
    """
    Check server health and display configuration.
    
    Args:
        server_url: Base URL of the parsing server
    """
    endpoint = f"{server_url}/health"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        health = response.json()
        
        print("\n" + "=" * 70)
        print("SERVER STATUS")
        print("=" * 70)
        print(f"\nStatus: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"\nConfiguration:")
        for key, value in health['config'].items():
            print(f"  • {key}: {value}")
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not check server health: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <file1> [file2] [file3] ...")
        print("\nExample:")
        print("  python test_client.py document.pdf report.docx")
        sys.exit(1)
    
    # Configuration
    server_url = "http://localhost:8080"
    
    # Parse file paths from command line
    file_paths = [Path(arg) for arg in sys.argv[1:]]
    
    # Check server health
    check_health(server_url)
    
    # Parse documents
    results = parse_documents(server_url, file_paths)
    
    # Print results
    print_results(results)
    
    # Optionally save results to JSON file
    output_file = Path("parsing_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to: {output_file}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
