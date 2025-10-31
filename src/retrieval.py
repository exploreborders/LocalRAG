from sentence_transformers import SentenceTransformer
try:
    from .vector_store import load_faiss_index, search_similar
    from .embeddings import load_embeddings
except ImportError:
    from vector_store import load_faiss_index, search_similar
    from embeddings import load_embeddings

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path="models/faiss_index.pkl"):
        self.model = SentenceTransformer(model_name)
        self.index = load_faiss_index(index_path)
        _, self.documents = load_embeddings()

    def retrieve(self, query, k=3):
        """
        Retrieve relevant documents for a given query.
        """
        query_embedding = self.model.encode([query])[0]
        results = search_similar(query_embedding, self.index, self.documents, k)
        return results

def format_results(results):
    """
    Format retrieval results for display.
    """
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"Result {i} (Distance: {result['distance']:.4f}):\n{result['document'].page_content[:300]}...")
    return "\n\n".join(formatted)

if __name__ == "__main__":
    retriever = Retriever()

    # Test queries
    queries = [
        "What is RAG?",
        "How does RAG work?",
        "Benefits of RAG systems"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=2)
        print(format_results(results))