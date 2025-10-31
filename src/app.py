#!/usr/bin/env python3
"""
Simple CLI for the Local RAG System
"""
try:
    from .retrieval_db import DatabaseRetriever
    from .rag_pipeline_db import RAGPipelineDB, format_results_db, format_answer_db
    from .document_processor import DocumentProcessor
except ImportError:
    from retrieval_db import DatabaseRetriever
    from rag_pipeline_db import RAGPipelineDB, format_results_db, format_answer_db
    from document_processor import DocumentProcessor

def main():
    """
    Main CLI entry point for the Local RAG System.

    Provides interactive menu for:
    1. Document retrieval only
    2. Full RAG pipeline with AI generation
    3. Batch document processing
    """
    print("Local RAG System")
    print("=================")
    print("Choose mode:")
    print("1. Retrieval only")
    print("2. Full RAG (requires Ollama)")
    print("3. Process existing documents")
    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        retriever = DatabaseRetriever()
        print("\nRetrieval Mode - Enter queries to find relevant documents")
        print("Type 'quit' to exit")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            results = retriever.retrieve(query, top_k=2)
            print(format_results_db(results))

    elif choice == "2":
        try:
            rag = RAGPipelineDB()
            print("\nRAG Mode - Enter questions for AI-powered answers")
            print("Type 'quit' to exit")
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() == 'quit':
                    break
                result = rag.query(question)
                print(format_answer_db(result))
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            print("Make sure Ollama is running: ollama serve")
            print("And pull the model: ollama pull llama2")

    elif choice == "3":
        try:
            processor = DocumentProcessor()
            processor.process_existing_documents()
        except Exception as e:
            print(f"Error processing documents: {e}")

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()