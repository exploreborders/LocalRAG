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
            query_lang = result.get('query_language', 'unknown')
            if query_lang != 'unknown':
                lang_names = {
                    'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish',
                    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'sv': 'Swedish',
                    'pl': 'Polish', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'
                }
                lang_display = lang_names.get(query_lang, query_lang.upper())
                print(f"🌍 Detected query language: {lang_display}")

            print(format_answer_db(result['answer']))

            # Show source documents
            if 'retrieved_documents' in result and result['retrieved_documents']:
                print("\n📚 Source Documents Used:")
                doc_sources = {}
                for doc in result['retrieved_documents']:
                    doc_info = doc.get('document', {})
                    filename = doc_info.get('filename', 'Unknown')
                    if filename not in doc_sources:
                        doc_sources[filename] = {'count': 0, 'score': 0}
                    doc_sources[filename]['count'] += 1
                    doc_sources[filename]['score'] = max(doc_sources[filename]['score'], doc.get('score', 0))

                for i, (filename, info) in enumerate(doc_sources.items(), 1):
                    print(f"  {i}. {filename} (chunks: {info['count']}, relevance: {info['score']:.3f})")
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