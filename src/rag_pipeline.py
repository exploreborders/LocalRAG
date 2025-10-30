from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from retrieval import Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

class RAGPipeline:
    def __init__(self, model_name="llama2", retriever=None):
        self.llm = Ollama(model=model_name)
        if retriever is None:
            self.retriever = Retriever()
        else:
            self.retriever = retriever

        # Create LangChain vector store from FAISS
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # Load FAISS index
        from vector_store import load_faiss_index
        from embeddings import load_embeddings
        index = load_faiss_index()
        _, documents = load_embeddings()

        # Create FAISS vector store
        self.vectorstore = FAISS(embeddings.embed_query, index, documents, relevance_score_fn=lambda x: 1/(1+x))

        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )

    def query(self, question):
        """
        Answer a question using RAG.
        """
        result = self.qa_chain({"query": question})
        return result

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