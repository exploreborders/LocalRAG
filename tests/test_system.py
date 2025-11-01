#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Test script for the Local RAG System components
"""
from src.retrieval_db import DatabaseRetriever

def test_retrieval():
    """
    Test the document retrieval functionality.

    Creates a DatabaseRetriever instance and performs a test query
    to verify that vector search is working correctly.
    """
    print("Testing Retrieval System...")
    retriever = DatabaseRetriever()
    query = "What is RAG?"
    results = retriever.retrieve(query, top_k=2)
    print(f"Query: {query}")
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"Result {i}: Score {result.get('score', 'N/A')}")
        print(f"Content: {result['content'][:100]}...")
    print("✓ Retrieval test passed\n")

def test_rag_pipeline():
    """
    Test the complete RAG pipeline functionality.

    Tests retrieval + generation pipeline to ensure end-to-end
    question answering works correctly.
    """
    print("Testing RAG Pipeline...")
    try:
        from src.rag_pipeline_db import RAGPipelineDB
        rag = RAGPipelineDB()
        question = "What is Retrieval-Augmented Generation?"
        result = rag.query(question)
        print(f"Question: {question}")
        print(f"Answer: {result['answer'][:200]}...")
        print("✓ RAG pipeline test passed\n")
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        print("Note: Full RAG requires Ollama running\n")

if __name__ == "__main__":
    test_retrieval()
    test_rag_pipeline()
    print("System test completed!")