#!/usr/bin/env python3
"""
Updated RAG Pipeline using database-backed retrieval.
"""

from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from .retrieval_db import DatabaseRetriever

class RAGPipelineDB:
    """
    Retrieval-Augmented Generation pipeline using database-backed retrieval
    and Ollama LLM for answer generation.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", llm_model: str = "llama2"):
        """
        Initialize the RAG pipeline.

        Args:
            model_name (str): Embedding model for retrieval
            llm_model (str): Ollama model for generation
        """
        self.retriever = DatabaseRetriever(model_name)
        self.llm = OllamaLLM(model=llm_model)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Please provide a comprehensive and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, say so.

Answer:"""
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query (str): Search query
            top_k (int): Number of results to retrieve

        Returns:
            list: Retrieved document chunks with content and metadata
        """
        return self.retriever.retrieve(query, top_k)

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using retrieved context and LLM.

        Args:
            query (str): Original question
            context_docs (list): Retrieved document chunks

        Returns:
            str: Generated answer text
        """
        # Combine context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in context_docs])

        # Create prompt
        prompt = self.prompt_template.format(context=context, question=query)

        # Generate answer
        try:
            answer = self.llm.invoke(prompt)
            return answer
        except Exception as e:
            return f"Error generating answer: {e}"

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieve relevant documents and generate answer.

        Args:
            question (str): Question to answer
            top_k (int): Number of documents to retrieve

        Returns:
            dict: Response containing question, answer, retrieved docs, and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k)

        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)

        return {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_docs': len(retrieved_docs)
        }

def format_results_db(results: List[Dict[str, Any]]) -> str:
    """Format retrieval results for display."""
    if not results:
        return "No relevant documents found."

    formatted = []
    for i, result in enumerate(results, 1):
        doc_info = result.get('document', {})
        formatted.append(f"""
**Document {i}:** {doc_info.get('filename', 'Unknown')}
**Relevance Score:** {result.get('score', 0):.3f}
**Content:** {result['content'][:200]}...
""")

    return "\n---\n".join(formatted)

def format_answer_db(answer: str) -> str:
    """Format generated answer for display."""
    return answer.strip()