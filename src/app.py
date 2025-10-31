#!/usr/bin/env python3
"""
Simple CLI for the Local RAG System
"""
try:
    from .retrieval import Retriever, format_results
    from .rag_pipeline import RAGPipeline, format_answer
except ImportError:
    from retrieval import Retriever, format_results
    from rag_pipeline import RAGPipeline, format_answer

def main():
    print("Local RAG System")
    print("=================")
    print("Choose mode:")
    print("1. Retrieval only")
    print("2. Full RAG (requires Ollama)")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        retriever = Retriever()
        print("\nRetrieval Mode - Enter queries to find relevant documents")
        print("Type 'quit' to exit")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            results = retriever.retrieve(query, k=2)
            print(format_results(results))

    elif choice == "2":
        try:
            rag = RAGPipeline()
            print("\nRAG Mode - Enter questions for AI-powered answers")
            print("Type 'quit' to exit")
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() == 'quit':
                    break
                result = rag.query(question)
                print(format_answer(result))
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            print("Make sure Ollama is running: ollama serve")
            print("And pull the model: ollama pull llama2")

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()