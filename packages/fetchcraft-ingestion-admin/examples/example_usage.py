"""
Example usage of the Fetchcraft Ingestion Admin MCP Server tools.

This example demonstrates how to use the MCP server tools programmatically
if needed for testing or integration.
"""

import asyncio
import os
from pathlib import Path


async def example_usage():
    """Example of using the ingestion admin tools."""
    
    # These would be called via MCP protocol in actual usage
    # This is just to show the expected workflow
    
    print("=" * 70)
    print("Fetchcraft Ingestion Admin - Example Usage")
    print("=" * 70)
    
    # Example 1: Get queue statistics
    print("\n1. Getting queue statistics...")
    print("   Tool: get_queue_stats()")
    print("   Returns: total_messages, by_state, by_queue, failed_messages, oldest_pending")
    
    # Example 2: List messages
    print("\n2. Listing recent messages...")
    print("   Tool: list_messages(limit=10, offset=0)")
    print("   Returns: messages[], total, limit, offset, has_more")
    
    # Example 3: Filter by queue and state
    print("\n3. Listing messages in specific queue/state...")
    print("   Tool: list_messages(queue='ingest.main', state='ready', limit=20)")
    
    # Example 4: Get message details
    print("\n4. Getting message details...")
    print("   Tool: get_message(message_id='some-id')")
    print("   Returns: id, queue, state, attempts, available_at, lease_until, body")
    
    # Example 5: Retry a failed message
    print("\n5. Retrying a failed message...")
    print("   Tool: retry_message(message_id='failed-msg-id')")
    print("   Returns: success, message_id, old_state, new_state, message")
    
    # Example 6: Delete a message
    print("\n6. Deleting a specific message...")
    print("   Tool: delete_message(message_id='unwanted-msg-id')")
    print("   Returns: success, message")
    
    # Example 7: Clear completed messages
    print("\n7. Clearing completed messages...")
    print("   Tool: clear_queue(state='done', confirm=True)")
    print("   Returns: success, deleted_count, message")
    
    # Example 8: List all queues
    print("\n8. Listing all queues...")
    print("   Tool: list_queues()")
    print("   Returns: queues[], total_queues")
    
    print("\n" + "=" * 70)
    print("Usage Notes:")
    print("=" * 70)
    print("• Run the server: fetchcraft-ingestion-admin")
    print("• Or: python -m fetchcraft.ingestion.admin.server")
    print("• Configure DB_PATH in .env or environment")
    print("• Access via MCP protocol (Claude Desktop, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(example_usage())
