# Building a Local RAG System with Python and Ollama

## Overview
Retrieval-Augmented Generation (RAG) combines retrieval of relevant information from a knowledge base with generative AI to provide accurate, context-aware responses. This plan outlines the steps to build a local RAG system using Python and Ollama for running large language models locally.

## Prerequisites
- Python 3.8 or higher
- Ollama installed on your system (download from https://ollama.ai)
- Basic knowledge of Python and command-line tools

## Step-by-Step Plan

### 1. Set Up the Environment
- Create a new Python virtual environment: `python -m venv rag_env`
- Activate the environment: `source rag_env/bin/activate` (Linux/Mac) or `rag_env\Scripts\activate` (Windows)
- Install required packages:
  ```
  pip install langchain langchain-community faiss-cpu sentence-transformers ollama python-dotenv
  ```

### 2. Install and Configure Ollama
- Install Ollama if not already done
- Pull a suitable model (e.g., Llama 2 or Mistral): `ollama pull llama2`
- Verify installation: `ollama list`

### 3. Prepare Your Data
- Collect documents (PDFs, text files, etc.) for your knowledge base
- Create a directory for data, e.g., `data/`
- If needed, implement text extraction (for PDFs: `pip install pypdf2` or similar)
- Clean and preprocess the text data

### 4. Create Embeddings
- Choose an embedding model (e.g., sentence-transformers' all-MiniLM-L6-v2)
- Implement embedding creation for your documents
- Split documents into chunks for better retrieval

### 5. Build the Vector Store
- Use FAISS or ChromaDB for vector storage
- Store document embeddings with metadata
- Implement persistence to save/load the vector store

### 6. Implement Retrieval
- Create a retrieval function to find relevant documents based on user queries
- Use similarity search on the vector store
- Return top-k most relevant chunks

### 7. Integrate with LLM Generation
- Use LangChain to connect retrieval with Ollama
- Create a chain that retrieves context and generates responses
- Implement prompt engineering for better results

### 8. Build the Application Interface
- Create a simple command-line interface or web app (using Streamlit or Flask)
- Allow users to input queries and receive generated responses
- Add options for configuration (model selection, chunk size, etc.)

### 9. Testing and Optimization
- Test the system with various queries
- Evaluate retrieval quality and generation accuracy
- Optimize parameters (chunk size, embedding model, number of retrieved docs)
- Add error handling and logging

### 10. Deployment and Maintenance
- Package the application for easy deployment
- Set up monitoring and logging
- Update the knowledge base as needed
- Keep Ollama and dependencies updated

## Additional Considerations
- **Privacy**: Since this is local, data stays on your machine
- **Performance**: Choose appropriate model sizes based on your hardware
- **Scalability**: For larger datasets, consider more robust vector databases
- **Security**: Be cautious with sensitive data in your knowledge base

## Resources
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

This plan provides a high-level overview. Each step may require additional research and implementation details based on your specific use case.

## Implementation Progress

### Completed Steps

1. **Set Up the Environment**
   - Created Python virtual environment: `python -m venv rag_env`
   - Activated the environment
   - Installed required packages: langchain, langchain-community, faiss-cpu, sentence-transformers, ollama, python-dotenv

2. **Project Structure**
   - Created directories: `src/`, `data/`, `models/`

3. **Data Preparation**
   - Created sample document: `data/sample.txt` with RAG overview
   - Implemented document loader: `src/data_loader.py`
     - Uses LangChain's TextLoader for .txt files
     - Implements text splitting with RecursiveCharacterTextSplitter
     - Chunk size: 1000 characters, overlap: 200 characters

4. **Embeddings Creation**
   - Implemented embeddings script: `src/embeddings.py`
     - Uses sentence-transformers with all-MiniLM-L6-v2 model
     - Created embeddings for 2 document chunks
     - Saved embeddings and documents to `models/embeddings.pkl`
     - Embedding dimensions: 384

5. **Vector Store**
   - Implemented FAISS vector store: `src/vector_store.py`
     - Created FAISS index with L2 distance
     - Saved index to `models/faiss_index.pkl`
     - Tested similarity search functionality

6. **Retrieval System**
   - Implemented retrieval class: `src/retrieval.py`
     - Created Retriever class for querying the vector store
     - Integrated with sentence-transformers for query encoding
     - Tested with sample queries

7. **LLM Integration**
   - Implemented RAG pipeline: `src/rag_pipeline.py`
     - Integrated Ollama LLM with retrieval using LangChain
     - Created RetrievalQA chain for question answering
     - Note: Requires Ollama server running with llama2 model

8. **Application Interface**
   - Created CLI application: `src/app.py`
     - Supports retrieval-only mode and full RAG mode
     - Interactive query interface
     - Error handling for missing Ollama

9. **Testing and Documentation**
   - Created README.md with setup and usage instructions
   - Created requirements.txt for dependency management
   - Tested retrieval system with sample queries
   - Verified embedding and vector store functionality

### Next Steps
- Install and configure Ollama (manual step for user)
- Prepare sample data
- Implement embeddings and vector store
- Build retrieval system
- Integrate with Ollama LLM
- Create application interface