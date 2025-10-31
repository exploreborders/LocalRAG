from langchain_ollama import OllamaLLM
try:
    from .retrieval import Retriever
except ImportError:
    from retrieval import Retriever

class RAGPipeline:
    def __init__(self, model_name="llama2", retriever=None):
        self.llm = OllamaLLM(model=model_name)
        if retriever is None:
            self.retriever = Retriever()
        else:
            self.retriever = retriever

    def query(self, question):
        """
        Answer a question using RAG.
        """
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, k=2)
        context = "\n\n".join([result['document'].page_content for result in results])

        # Create prompt
        prompt = f"""Use the following context to answer the question. If you cannot answer from the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Generate answer
        answer = self.llm.invoke(prompt)

        return {
            'result': answer,
            'source_documents': [result['document'] for result in results]
        }

def format_answer(result):
    """
    Format the answer for display.
    """
    answer = result['result']
    sources = result['source_documents']
    formatted_sources = "\n".join([f"- {doc.page_content[:200]}..." for doc in sources])
    return f"Answer: {answer}\n\nSources:\n{formatted_sources}"

if __name__ == "__main__":
    # Note: This requires Ollama to be running with llama2 model
    # Run: ollama serve
    # And: ollama pull llama2

    try:
        rag = RAGPipeline()
        question = "What is Retrieval-Augmented Generation?"
        result = rag.query(question)
        print(format_answer(result))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and the llama2 model is pulled.")