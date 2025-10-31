# Local RAG System

A local Retrieval-Augmented Generation system built with Python, Ollama, and FAISS.

## Features

- **Multi-Model Support**: Choose from multiple embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- **Smart Caching**: Avoid reprocessing documents when content hasn't changed
- **Batch Processing**: Process documents with multiple models simultaneously
- **Model Comparison**: Side-by-side performance comparison of different embedding models
- Document loading and text chunking (supports .txt, .pdf, .docx, .pptx, .xlsx)
- Embedding creation using sentence-transformers with GPU acceleration
- Optimized vector storage with FAISS (IVF-PQ for large datasets)
- Retrieval system for similarity search
- Integration with Ollama LLMs for generation
- Comprehensive web interface with analytics and settings
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
    - Run data processing: `python -m src.embeddings` (single model) or `python -m src.embeddings --all` (all models)

## Usage

### Web Interface (Recommended):
```bash
python run_web.py
```
Or directly:
```bash
streamlit run web_interface/app.py
```
Then open http://localhost:8501 in your browser for a comprehensive multipage experience with:
- **🏠 Home**: Query interface and system control
- **📁 Documents**: File upload, management, and multi-model processing
- **⚙️ Settings**: Configuration options with dynamic model detection
- **📊 Analytics**: Performance monitoring and model comparison

### Command Line Interface:
```bash
python -m src.app
```

Choose between:
- **Retrieval only**: Search for relevant documents
- **Full RAG**: Get AI-generated answers with context

### Manual testing:

**Test retrieval:**
```bash
python -m src.retrieval
```

**Test vector store:**
```bash
python -m src.vector_store
```

**Run web interface:**
```bash
streamlit run web_interface/app.py
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

- **Multi-Model Support**: The system supports multiple embedding models. Each model stores its own embeddings and vector index for optimal performance.
- **Smart Caching**: Documents are only reprocessed when their content changes, saving time and resources.
- **Model Selection**: Choose the best embedding model for your use case. Larger models generally provide better accuracy but require more resources.
- The system currently uses sample data. Add your own documents to `data/` for real use cases.
- Full RAG functionality requires Ollama running locally.
- Embeddings are cached in `models/` for faster subsequent runs.