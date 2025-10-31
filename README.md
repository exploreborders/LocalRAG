# Local RAG System

A local Retrieval-Augmented Generation system built with Python, Ollama, and FAISS.

## Features

- Document loading and text chunking (supports .txt, .pdf, .docx, .pptx, .xlsx)
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
    - Run data processing: `python -m src.embeddings`

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
- **📁 Documents**: File upload and management
- **⚙️ Settings**: Configuration options
- **📊 Analytics**: Performance monitoring

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

- The system currently uses sample data. Add your own documents to `data/` for real use cases.
- Full RAG functionality requires Ollama running locally.
- Embeddings are cached in `models/` for faster subsequent runs.