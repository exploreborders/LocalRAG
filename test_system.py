#!/usr/bin/env python3
"""
Test script for the Local RAG System components
"""
from src.retrieval import Retriever

def test_retrieval():
    print("Testing Retrieval System...")
    retriever = Retriever()
    query = "What is RAG?"
    results = retriever.retrieve(query, k=2)
    print(f"Query: {query}")
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"Result {i}: Distance {result['distance']:.4f}")
        print(f"Content: {result['document'].page_content[:100]}...")
    print("✓ Retrieval test passed\n")

def test_rag_pipeline():
    print("Testing RAG Pipeline...")
    try:
        from src.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        question = "What is Retrieval-Augmented Generation?"
        result = rag.query(question)
        print(f"Question: {question}")
        print(f"Answer: {result['result'][:200]}...")
        print("✓ RAG pipeline test passed\n")
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        print("Note: Full RAG requires Ollama running\n")

if __name__ == "__main__":
    test_retrieval()
    test_rag_pipeline()
    print("System test completed!")