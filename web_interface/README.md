# Local RAG System - Web Interface

A modern Streamlit-based web interface for the Local RAG document intelligence system.

## Features

- **🏠 Home**: Query interface for AI-powered document Q&A with source citations
- **📁 Documents**: Upload, manage, and reprocess document collections with AI summaries
- **⚙️ Settings**: Configure AI models, caching, and system parameters
- **📊 Analytics**: Performance monitoring and usage statistics
- **AI-Powered Features**: Automatic tagging, categorization, and summarization
- **Full Summary Display**: Complete document summaries without truncation

## Quick Start

1. **Setup the system:**
    ```bash
    python setup_all.py
    ```

2. **Start the web interface:**
    ```bash
    python run_web.py
    ```

3. **Open your browser** to `http://localhost:8501`

## Pages

### 🏠 Home
- **Query Interface**: Ask questions about your documents in natural language
- **AI-Powered Responses**: Get contextual answers with source citations
- **Query History**: Track and revisit previous searches

### 📁 Documents
- **File Upload**: Support for PDF, DOCX, XLSX, PPTX, TXT files with automatic processing
- **Document Library**: Browse and manage uploaded documents with AI-generated summaries
- **Reprocessing**: Update existing documents with improved AI analysis
- **AI Organization**: Automatic tagging, categorization, and summarization
- **Full Summary Display**: Complete document descriptions without truncation

### ⚙️ Settings
- **AI Configuration**: Set Ollama model parameters and embedding options
- **Cache Settings**: Configure Redis caching behavior
- **System Preferences**: Customize interface and processing options

### 📊 Analytics
- **Performance Metrics**: Query response times and system statistics
- **Cache Monitoring**: Redis cache hit rates and memory usage
- **Usage Analytics**: Document processing and query patterns

## Usage Guide

### Getting Started
1. **Upload documents** using the Documents page
2. **Ask questions** on the Home page about your documents
3. **Monitor performance** via the Analytics page
4. **Configure settings** in the Settings page

### Document Processing
- **Supported Formats**: PDF, DOCX, XLSX, PPTX, TXT
- **Automatic Processing**: Documents are parsed, chunked, and embedded on upload
- **Batch Operations**: Process multiple documents efficiently
- **Progress Tracking**: Real-time status updates during processing

### Query Interface
- **Natural Language**: Ask questions in plain English
- **Source Citations**: All answers include document references
- **Context Awareness**: Responses consider document content and relationships

## Configuration

### AI Settings
- **LLM Model**: Select from installed Ollama models
- **Temperature**: Control response randomness (0.0-1.0)
- **Max Tokens**: Set maximum response length
- **Embedding Model**: Choose embedding model for document processing

### System Settings
- **Cache Configuration**: Enable/disable Redis caching
- **Batch Processing**: Configure parallel processing options
- **Database Settings**: Connection parameters for PostgreSQL/Elasticsearch

### Interface Settings
- **Theme**: Light or dark mode
- **Display Options**: Customize result presentation

## System Requirements

- **Python 3.8+** with virtual environment
- **Ollama** installed and running locally
- **AI Models**: llama3.2:latest and embeddinggemma:latest (via Ollama)
- **Web Browser**: Modern browser for Streamlit interface
- **RAM**: 16GB+ recommended for AI processing

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure you're running from the project root directory
- Check that virtual environment is activated
- Verify all dependencies are installed

**AI Model Issues**
- Install Ollama from https://ollama.ai
- Pull required models: `ollama pull llama3.2:latest`
- Ensure Ollama service is running

**Document Processing**
- Check file formats are supported (PDF, DOCX, XLSX, PPTX, TXT)
- Verify files are not corrupted or password-protected
- Monitor processing status in the interface

**Performance Issues**
- Reduce chunk overlap for faster processing
- Use smaller batch sizes for memory constraints
- Enable Redis caching for better performance

## Development

### Project Structure
```
web_interface/
├── app.py                    # Main application entry point
├── pages/                    # Streamlit pages (numbered for sidebar order)
│   ├── 1_🏠_Home.py         # Query interface
│   ├── 2_📁_Documents.py    # Document management
│   ├── 3_⚙️_Settings.py     # Configuration
│   └── 4_📊_Analytics.py    # Performance monitoring
├── components/               # Reusable UI components
│   ├── query_interface.py    # Query input components
│   └── results_display.py    # Results rendering
└── utils/
    └── session_manager.py    # Session state management
```

### Adding Features
1. Create new page files in `pages/` (use numbering for sidebar order)
2. Add reusable components in `components/`
3. Update session state management as needed
4. Test across all pages for consistency

## License

Part of the Local RAG System. See main project README for licensing information.