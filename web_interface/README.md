# Local RAG System - Web Interface

A modern multipage web interface for the Local RAG system, featuring database-backed document storage and Elasticsearch vector search, built with Streamlit.

## Features

- **🏠 Home Page**: Interactive query interface with dual mode support
- **📁 Documents Page**: File upload and document library management
- **⚙️ Settings Page**: Comprehensive configuration options
- **📊 Analytics Page**: Performance monitoring and usage statistics
- **Query History**: Track and revisit previous queries across sessions
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Feedback**: Loading indicators and status updates

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source rag_env/bin/activate
   ```

2. **Launch the web interface:**
   ```bash
   streamlit run web_interface/app.py
   ```

3. **Open your browser** to `http://localhost:8501`

4. **Navigate using the sidebar** to explore different features

## Page Overview

### 🏠 Home - Main Query Interface
- **Natural Language Queries**: Ask questions in plain English
- **Dual Mode Support**: Retrieval-only or AI-powered RAG responses
- **System Initialization**: One-click setup for the RAG system
- **Results Display**: Formatted answers with source attribution

### 📁 Documents - File Management
- **Multi-Format Upload**: Support for PDF, DOCX, XLSX, PPTX, TXT files
- **Document Library**: Browse and manage your knowledge base
- **Automatic Processing**: Documents are automatically chunked and embedded on upload
- **Database Storage**: All documents and chunks stored in PostgreSQL

### ⚙️ Settings - Configuration
- **AI Generation Settings**: Configure temperature, max tokens, and Ollama models
- **Interface Preferences**: Theme selection and display options

### 📊 Analytics - Performance Dashboard
- **Usage Metrics**: Query counts, response times, and trends
- **System Health**: Monitor component status and resource usage
- **Performance Charts**: Visualize query patterns and response times
- **Data Export**: Download query history and metrics as CSV/JSON

## Usage Guide

### Getting Started
1. **Visit the Home page** and initialize the system
2. **Go to Documents page** and upload your files
3. **Reprocess documents** to create embeddings and vector index
4. **Return to Home** and start asking questions!

### Query Modes

#### Retrieval Mode
- Searches your document library using semantic similarity
- Returns relevant text chunks ranked by relevance
- Fast and works without external AI services

#### RAG Mode
- Combines retrieval with AI generation for comprehensive answers
- Requires Ollama to be running locally
- Provides context-aware responses with citations

### Theme Switching

The interface supports light and dark themes:

1. **Go to Settings page** (⚙️ Settings)
2. **Select theme** from the dropdown (Light or Dark)
3. **Save settings** - you'll see a notification about app restart
4. **Restart the application** to apply the theme change

Theme changes require a full application restart because they modify the Streamlit configuration file.

### Document Management

#### Supported Formats
- **📄 Text files**: .txt (plain text)
- **📕 PDFs**: .pdf (Portable Document Format)
- **📝 Word documents**: .docx (Microsoft Word)
- **📊 Spreadsheets**: .xlsx (Microsoft Excel)
- **📈 Presentations**: .pptx (Microsoft PowerPoint)

#### Processing Pipeline
1. **Upload**: Files are processed temporarily (not permanently stored)
2. **Parsing**: Advanced text extraction using Docling with layout awareness
3. **Chunking**: Documents split into manageable pieces with overlap
4. **Embedding**: Text converted to vector representations using nomic-embed-text-v1.5
5. **Storage**: Chunks stored in PostgreSQL, vectors indexed in Elasticsearch

## Configuration

### Settings Categories

#### Retrieval Settings
```yaml
retrieval:
  chunk_size: 1000        # Size of text chunks
  chunk_overlap: 200      # Overlap between chunks
  k_retrieval: 3          # Number of results to retrieve
  embedding_model: "all-MiniLM-L6-v2"  # Available sentence transformer model
```

**Note:** Both LLM Model and Embedding Model dropdowns automatically detect and show only available models.

#### Generation Settings
```yaml
generation:
  model: "llama2"         # Ollama model name (from installed models only)
  temperature: 0.7        # Response randomness (0.0-1.0)
  max_tokens: 500         # Maximum response length
  ollama_host: "http://localhost:11434"  # Ollama server URL
```

**Note:** The LLM Model dropdown automatically detects and shows only your installed Ollama models.

#### Interface Settings
```yaml
interface:
  theme: "light"          # UI theme ("light" or "dark")
  max_results_display: 5  # Results per page
```

## System Requirements

### Core Requirements
- ✅ Python 3.8+ with virtual environment
- ✅ All dependencies from `requirements.txt`
- ✅ Document files in supported formats

### RAG Mode Requirements
- ✅ Ollama installed and running
- ✅ AI model downloaded (e.g., `ollama pull llama2`)
- ✅ Sufficient RAM for model loading

## Troubleshooting

### Common Issues

**"Could not import RAG system components"**
- Ensure you're running from the project root directory
- Check that all dependencies are installed

**"RAG mode unavailable"**
- Install Ollama: https://ollama.ai
- Start server: `ollama serve`
- Pull model: `ollama pull llama2`

**"No documents found"**
- Upload files using the Documents page
- Ensure files are in supported formats
- Reprocess documents after uploading

**Slow performance**
- Reduce `k_retrieval` value
- Use smaller chunk sizes
- Try a lighter embedding model

**Theme not changing**
- Theme changes require app restart
- After changing theme in Settings, restart the application
- Check that `.streamlit/config.toml` exists in the project root

**Model dropdown shows wrong models**
- LLM dropdown automatically detects installed Ollama models
- Embedding dropdown automatically detects available sentence-transformers models
- If Ollama is not running, only "llama2" will be shown as default
- If sentence-transformers models are not cached, only "all-MiniLM-L6-v2" will be shown
- Install additional models with `ollama pull <model_name>` or they will auto-download when first used

### Performance Optimization

- **For Speed**: Use smaller chunks and lower k-values
- **For Accuracy**: Increase k-value and use larger chunks
- **For Memory**: Choose efficient embedding models
- **For Quality**: Use higher-quality models like `all-mpnet-base-v2`

## Development

### Project Structure
```
web_interface/
├── app.py                    # Landing page
├── pages/
│   ├── 1_🏠_Home.py         # Query interface
│   ├── 2_📁_Documents.py    # File management
│   ├── 3_⚙️_Settings.py     # Configuration
│   └── 4_📊_Analytics.py    # Performance dashboard
├── components/
│   ├── query_interface.py   # Query components
│   └── results_display.py   # Results rendering
├── utils/
│   └── session_manager.py   # Session management
└── config/
    └── default_settings.yaml # Default configuration
```

### Adding New Features

1. **Create page files** in `pages/` directory (numbered for order)
2. **Add components** in `components/` for reusable UI elements
3. **Update session state** in `utils/session_manager.py`
4. **Test across all pages** for consistency

### Session State Management

The app uses Streamlit's session state to maintain:
- System initialization status
- Query history and results
- User preferences and settings
- Component availability flags

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive error handling and user feedback
3. Test features across all pages and modes
4. Update documentation for new functionality
5. Ensure responsive design works on mobile devices

## License

This project is part of the Local RAG System. See main project README for licensing information.