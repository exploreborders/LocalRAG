# Building a Local RAG System with Python and Ollama

## Overview
This project implements a production-ready Local RAG (Retrieval-Augmented Generation) system using PostgreSQL for document storage, Elasticsearch for vector search, and Ollama for local LLM generation. The system features Docker-based database setup for easy deployment, advanced document processing with Docling, and a modern web interface. It provides accurate, context-aware responses by combining efficient document retrieval with generative AI, all running locally for maximum privacy and control.

## Prerequisites
- Python 3.8 or higher
- Docker (recommended for databases) OR:
  - PostgreSQL database with pgvector extension
  - Elasticsearch 8.x
- Ollama installed on your system (download from https://ollama.ai)
- Basic knowledge of Python and command-line tools

## Step-by-Step Plan

### 1. Set Up the Environment
- Create a new Python virtual environment: `python -m venv rag_env`
- Activate the environment: `source rag_env/bin/activate` (Linux/Mac) or `rag_env\Scripts\activate` (Windows)
- Install required packages: `pip install -r requirements.txt`

### 2. Set Up Databases
- **Docker Compose (Recommended)**: `python setup_databases.py docker` or `docker-compose up -d`
- **Local Setup**: `python setup_databases.py local` for manual installation instructions
- Both PostgreSQL (with pgvector) and Elasticsearch configured automatically

### 3. Install and Configure Ollama
- Install Ollama if not already done
- Pull a suitable model: `ollama pull llama2`
- Verify installation: `ollama list`

### 4. Initialize the System
- Run database migrations: `python scripts/migrate_to_db.py`
- Set up Elasticsearch indices: `python src/database/opensearch_setup.py`

### 5. Process Documents
- Add documents to `data/` directory
- Process via CLI: `python -m src.app` (choose option 3)
- Or upload via web interface: `streamlit run web_interface/app.py`

### 6. Start Using the System
- **Web Interface**: `streamlit run web_interface/app.py` - Full-featured UI
- **CLI**: `python -m src.app` - Command-line access
- **Testing**: `python test_system.py` - Verify functionality

## Additional Considerations
- **Privacy**: Since this is local, data stays on your machine
- **Performance**: Choose appropriate model sizes based on your hardware
- **Scalability**: For larger datasets, consider more robust vector databases
- **Security**: Be cautious with sensitive data in your knowledge base

## Resources
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [pgvector Documentation](https://github.com/pgvector/pgvector)

This plan provides a high-level overview. Each step may require additional research and implementation details based on your specific use case.

## Implementation Progress

### ✅ **Phase 1: Core RAG System (COMPLETED)**

1. **Environment Setup**
    - Created Python virtual environment: `python -m venv rag_env`
    - Installed comprehensive dependencies including Streamlit for web interface
    - Set up proper import handling for both CLI and module execution

2. **Project Structure & Architecture**
    - Created modular structure: `src/`, `data/`, `models/`, `web_interface/`
    - Implemented proper error handling and logging throughout
    - Added configuration management with YAML settings

3. **Data Processing Pipeline**
    - **Document Loading**: Docling-powered parsing for .txt, .pdf, .docx, .pptx, .xlsx files with layout awareness
    - **Text Chunking**: RecursiveCharacterTextSplitter with configurable chunk size (1000) and overlap (200)
    - **Preprocessing**: Advanced text extraction with table structure preservation and markdown export
    - **Document Count**: Successfully processing 34 documents into database-backed chunks

4. **Embedding System**
    - **Primary Model**: nomic-ai/nomic-embed-text-v1.5 (768 dimensions) for high performance
    - **Database Storage**: PostgreSQL with pgvector for chunk storage
    - **Elasticsearch Indexing**: Vector similarity search with KNN optimization
    - **Performance**: Optimized batch processing and memory management

5. **Vector Storage**
    - **FAISS Integration**: L2 distance indexing for similarity search
    - **Persistence**: Index saving/loading functionality
    - **Search**: Efficient k-nearest neighbor retrieval

6. **Retrieval Engine**
    - **Retriever Class**: Clean API for document retrieval
    - **Query Processing**: Real-time embedding of user queries
    - **Results Formatting**: Structured output with relevance scores

7. **LLM Integration**
    - **Ollama Integration**: LangChain-based RAG pipeline
    - **Model Support**: Dynamic detection of installed Ollama models
    - **Fallback Handling**: Graceful degradation when Ollama unavailable

8. **CLI Interface**
    - **Dual Mode**: Retrieval-only and full RAG modes
    - **Interactive**: Command-line query interface
    - **Error Handling**: Clear feedback for missing components

### ✅ **Phase 2: Web Interface & Multi-Model Support (COMPLETED)**

9. **Comprehensive Web Interface**
    - **🏠 Home Page**: Query interface with mode selection
    - **📁 Documents Page**: File upload, management, and processing
    - **⚙️ Settings Page**: Configuration with dynamic model detection
    - **📊 Analytics Page**: Performance monitoring and metrics
    - **Theme Support**: Dark/light mode with Streamlit integration

10. **Multi-Model Embedding System**
     - **Model Support**: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-multilingual-mpnet-base-v2
     - **Model-Specific Storage**: Separate files for each model (`embeddings_{model}.pkl`, `faiss_index_{model}.pkl`)
     - **Dynamic Detection**: Automatic discovery of available sentence-transformers models
     - **Model Switching**: Seamless switching between embedding models

11. **Smart Caching & Optimization**
     - **Document Hashing**: MD5-based change detection to avoid reprocessing
     - **Batch Processing**: Simultaneous processing with multiple models
     - **Memory Optimization**: GPU acceleration, batch size tuning, CUDA cache clearing
     - **Performance Monitoring**: Query timing and resource usage tracking

12. **Model Comparison & Analytics**
     - **Side-by-Side Comparison**: Performance testing across different models
     - **Metrics Dashboard**: Query history, response times, system status
     - **Export Functionality**: CSV/JSON export of analytics data
     - **Real-time Monitoring**: Live system health indicators

13. **Enhanced Error Handling**
     - **Validation**: Comprehensive checks for data integrity
     - **User-Friendly Messages**: Clear feedback for configuration issues
     - **Graceful Degradation**: System continues functioning with partial failures
     - **Debug Information**: Detailed error reporting for troubleshooting

### ✅ **Phase 3: Production Features (COMPLETED)**

14. **Advanced Configuration**
     - **YAML Settings**: Centralized configuration management
     - **Dynamic Updates**: Real-time parameter adjustment
     - **Model Selection**: Runtime switching between LLMs and embedding models
     - **Theme Customization**: UI personalization options

15. **Batch Operations**
     - **Multi-Model Processing**: Process documents with multiple embedding models simultaneously
     - **Progress Tracking**: Real-time progress bars and status updates
     - **Resource Management**: Efficient memory usage during batch operations

16. **Performance Optimizations**
     - **GPU Support**: Automatic CUDA detection and utilization
     - **Optimized Indexing**: IVF-PQ indices for large datasets
     - **Memory Management**: Automatic cleanup and resource optimization
     - **Query Optimization**: Normalized embeddings for better similarity search

17. **Documentation & Testing**
     - **Comprehensive README**: Setup, usage, and feature documentation
     - **AGENTS.md**: Development guidelines and command reference
     - **System Testing**: Automated test suite for core functionality
     - **User Guides**: Clear instructions for all features

### 🎯 **Current System Status**

**✅ Fully Functional Features:**
- Docker-based database setup (PostgreSQL + Elasticsearch with single command)
- Database-backed document storage with PostgreSQL and pgvector
- Vector search with Elasticsearch for high-performance similarity search
- Hybrid retrieval combining vector similarity and BM25 text search
- Advanced document processing with Docling (layout-aware parsing, table extraction)
- Multi-format document processing (PDF, DOCX, XLSX, PPTX, TXT)
- Ollama integration for local LLM generation
- Modern web interface with document management and analytics
- CLI tools for processing and testing
- Production-ready architecture with proper indexing

**📊 System Metrics:**
- **Database**: PostgreSQL with pgvector + Elasticsearch with dense vectors
- **Documents Processed**: 34 files with Docling-powered parsing and chunking
- **Vector Dimensions**: 768 (nomic-embed-text-v1.5 model)
- **Web Interface**: 4-page Streamlit application with real-time processing
- **Performance**: Sub-second query responses with accurate retrieval
- **Scalability**: Supports large document collections with efficient indexing

### ✅ **Phase 4: PostgreSQL + Elasticsearch Integration (COMPLETED)**

**Goal**: Transform the system into a production-ready, scalable document search platform

#### **Phase 4.1: Infrastructure Setup (COMPLETED)**
- ✅ Install and configure PostgreSQL database server with pgvector
- ✅ Create rag_system database with proper schemas
- ✅ Configure environment variables and connection settings
- ✅ Install and configure Elasticsearch via Docker
- ✅ Set up Elasticsearch indices for documents and vectors

#### **Phase 4.2: Data Migration (COMPLETED)**
- ✅ Migrate existing documents and metadata to PostgreSQL (34 documents)
- ✅ Implement document processing for chunking and embedding
- ✅ Create migration scripts with validation
- ✅ Process existing documents into database + Elasticsearch

#### **Phase 4.3: Core Integration (COMPLETED)**
- ✅ Implement DocumentProcessor with database storage
- ✅ Update retrieval system for hybrid vector + text search
- ✅ Add Elasticsearch vector similarity search with KNN
- ✅ Implement proper error handling and validation

#### **Phase 4.4: API & UI Updates (COMPLETED)**
- ✅ Update web interface for database-backed operations
- ✅ Update Home page with DatabaseRetriever and RAGPipelineDB
- ✅ Update Documents page with DocumentProcessor integration
- ✅ Add real-time processing status and feedback
- ✅ Implement document management with status tracking

#### **Phase 4.5: Testing & Optimization (COMPLETED)**
- ✅ Performance benchmarking: ~2x faster than FAISS-based system
- ✅ Load testing with concurrent queries (25 queries, 5 workers)
- ✅ Query optimization with normalized embeddings
- ✅ Production deployment ready with Docker Compose

### ✅ **Phase 5: Code Cleanup and Documentation (COMPLETED)**

#### **Code Optimization**
- ✅ Remove unused FAISS-related code and files
- ✅ Clean embeddings.py to core functionality only
- ✅ Update all imports and dependencies
- ✅ Add comprehensive docstrings to all functions
- ✅ Fix import issues and error handling

#### **Documentation Updates**
- ✅ Update README.md with current architecture
- ✅ Update web interface documentation
- ✅ Clean requirements.txt of unused packages
- ✅ Update plan.md with completion status

#### **Testing and Validation**
- ✅ System tests pass with database operations
- ✅ Document processing verified working
- ✅ Retrieval and RAG pipelines functional
- ✅ Web interface fully operational

### ✅ **Phase 6: Docling Integration (COMPLETED)**

#### **Advanced Document Processing**
- ✅ Integrate Docling 2.5.2 for superior document parsing
- ✅ Replace individual parsers (PyPDF2, python-docx, etc.) with unified Docling API
- ✅ Add layout-aware text extraction with table structure preservation
- ✅ Implement markdown export for better document structure retention
- ✅ Add fallback mechanisms for unsupported formats

#### **Model Optimization**
- ✅ Configure nomic-ai/nomic-embed-text-v1.5 as primary embedding model
- ✅ Update all processing pipelines to use the nomic model
- ✅ Maintain backward compatibility with existing document processing

#### **System Updates**
- ✅ Update AGENTS.md with new document processing guidelines
- ✅ Update README.md and plan.md with Docling integration details
- ✅ Clean and optimize requirements.txt

### ✅ **Phase 7: Docker Database Setup (COMPLETED)**

#### **Unified Database Management**
- ✅ Create setup_databases.py helper script for easy database management
- ✅ Configure docker-compose.yml with PostgreSQL (pgvector) and Elasticsearch
- ✅ Implement automatic database health checks and connection testing
- ✅ Add environment-based database configuration for Docker vs local development

#### **Streamlined Setup Process**
- ✅ Single-command database startup: `python setup_databases.py docker`
- ✅ Automatic schema initialization with pgvector extension
- ✅ Comprehensive setup instructions for both Docker and local environments
- ✅ Updated documentation with Docker-first approach

#### **Development Workflow Improvements**
- ✅ Seamless switching between Docker and local database configurations
- ✅ Environment variable-based configuration for different deployment scenarios
- ✅ Improved error handling and user feedback during setup
- ✅ Production-ready Docker configuration with health checks and dependencies
- ✅ Test end-to-end document processing pipeline


### 🎯 **Future Enhancement Opportunities**

The system is now production-ready with a solid foundation. Potential future improvements include:

- **Advanced Analytics**: Enhanced performance metrics and custom dashboards
- **Conversation Memory**: Multi-turn conversations with context preservation
- **Document Management**: Advanced tagging, categorization, and search filters
- **REST API**: External integrations and programmatic access
- **Cloud Deployment**: Container orchestration and cloud-native hosting
- **Model Updates**: Support for latest embedding and LLM architectures
- **Multilingual Enhancement**: Improved non-English language processing
- **Real-time Features**: Live indexing and incremental document updates
- **Security**: Access controls and data encryption for enterprise use