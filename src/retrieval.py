from sentence_transformers import SentenceTransformer
try:
    from .vector_store import load_faiss_index, search_similar
    from .embeddings import load_embeddings
except ImportError:
    from vector_store import load_faiss_index, search_similar
    from embeddings import load_embeddings

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name

        try:
            # Load model with device specification (same as embeddings.py)
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise Exception(f"Failed to load embedding model '{model_name}': {e}")

        try:
            self.index = load_faiss_index(model_name)
        except Exception as e:
            raise Exception(f"Failed to load FAISS index for model '{model_name}': {e}")

        try:
            _, self.documents = load_embeddings(model_name)[:2]
        except Exception as e:
            raise Exception(f"Failed to load embeddings for model '{model_name}': {e}")

        # Validate that we have compatible data
        if hasattr(self.index, 'd'):
            expected_dim = self.index.d
        else:
            expected_dim = self.model.get_sentence_embedding_dimension()

        if len(self.documents) != self.index.ntotal:
            raise Exception(f"Document count ({len(self.documents)}) doesn't match index size ({self.index.ntotal})")

    def retrieve(self, query, k=3):
        """
        Retrieve relevant documents for a given query with performance monitoring.
        """
        import time
        start_time = time.time()

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        # Search
        results = search_similar(query_embedding, self.index, self.documents, k)

        # Log performance (optional)
        retrieval_time = time.time() - start_time
        if retrieval_time > 1.0:  # Log slow queries
            print(".3f")

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