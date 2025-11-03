#!/usr/bin/env python3
"""
Test script for CLI functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.app import RAGCLI
from src.rag_pipeline_db import format_results_db

def test_retrieval_mode():
    """Test the retrieval mode functionality"""
    print("🧪 Testing CLI Retrieval Mode")

    # Create CLI instance
    cli = RAGCLI()

    # Initialize components
    if not cli.initialize_components():
        print("❌ Failed to initialize components")
        return False

    # Simulate retrieval mode input
    print("Testing retrieval with query: 'machine learning'")

    # Call retrieval_mode directly (bypassing input)
    # We'll simulate what happens inside retrieval_mode

    query = "machine learning"
    print("⏳ Searching...")

    start_time = 0
    if cli.retriever:
        import time
        start_time = time.time()
        results = cli.retriever.retrieve(query, top_k=3)
        search_time = time.time() - start_time

        print(f"   Search completed in {search_time:.2f}s")
        print(format_results_db(results))

        if results:
            print(f"\n📊 Found {len(results)} relevant document chunks")

        return True
    else:
        print("❌ Retriever not initialized")
        return False

if __name__ == "__main__":
    success = test_retrieval_mode()
    print(f"\n✅ Test {'PASSED' if success else 'FAILED'}")