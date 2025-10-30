# Local RAG System

A local Retrieval-Augmented Generation system built with Python, Ollama, and FAISS.

## Features

- Document loading and text chunking
- Embedding creation using sentence-transformers
- Vector storage with FAISS
- Retrieval system for similarity search
- Integration with Ollama LLMs for generation
- Command-line interface

## Setup

1. **Clone or navigate to the project directory**

2. **Create virtual environment:**
   ```bash
   python3 -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama:**
   - Download from https://ollama.ai
   - Pull a model: `ollama pull llama2`
   - Start server: `ollama serve`

5. **Prepare data:**
   - Add your documents to the `data/` directory (currently supports .txt files)
   - Run data processing: `python src/embeddings.py`

## Usage

### Run the application:
```bash
python src/app.py
```

Choose between:
- **Retrieval only**: Search for relevant documents
- **Full RAG**: Get AI-generated answers with context

### Manual testing:

**Test retrieval:**
```bash
python src/retrieval.py
```

**Test vector store:**
```bash
python src/vector_store.py
```

## Project Structure

```
LocalRAG/
├── src/
│   ├── data_loader.py      # Document loading and chunking
│   ├── embeddings.py       # Embedding creation
│   ├── vector_store.py     # FAISS vector operations
│   ├── retrieval.py        # Retrieval system
│   ├── rag_pipeline.py     # RAG with Ollama integration
│   └── app.py              # CLI interface
├── data/                   # Document storage
├── models/                 # Saved models and indices
├── plan.md                 # Implementation plan and progress
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Implementation Steps Completed

See `plan.md` for detailed implementation progress.

## Notes

- The system currently uses sample data. Add your own documents to `data/` for real use cases.
- Full RAG functionality requires Ollama running locally.
- Embeddings are cached in `models/` for faster subsequent runs.