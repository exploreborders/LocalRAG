"""
Shared pytest fixtures and configuration for LocalRAG unit tests.

This file provides common mocking fixtures for external dependencies
to enable isolated unit testing.
"""

import pytest
from unittest.mock import Mock, MagicMock
from sqlalchemy.orm import Session


@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy session for database operations."""
    session = MagicMock(spec=Session)
    session.commit.return_value = None
    session.rollback.return_value = None
    session.close.return_value = None
    return session


@pytest.fixture
def mock_elasticsearch():
    """Mock Elasticsearch client."""
    es = MagicMock()
    es.search.return_value = {"hits": {"hits": []}}
    es.index.return_value = {"_id": "test_id", "result": "created"}
    es.delete_by_query.return_value = {"deleted": 0}
    return es


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache client."""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = 1
    cache.exists.return_value = False
    return cache


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for LLM operations."""
    client = MagicMock()
    client.list.return_value = {"models": [{"name": "llama3.2:latest"}]}
    client.generate.return_value = {
        "response": "Mock AI response",
        "done": True,
        "context": [1, 2, 3],
    }
    client.show.return_value = {"modelfile": "# Mock model file"}
    return client


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for embeddings."""
    model = MagicMock()
    model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    model.get_sentence_embedding_dimension.return_value = 384
    return model


@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    fs = MagicMock()
    fs.exists.return_value = True
    fs.isfile.return_value = True
    fs.isdir.return_value = False
    fs.read_text.return_value = "Mock file content"
    fs.write_text.return_value = None
    return fs


@pytest.fixture
def mock_requests():
    """Mock requests library for HTTP calls."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"status": "ok"}
    response.text = "Mock response"

    requests_mock = MagicMock()
    requests_mock.get.return_value = response
    requests_mock.post.return_value = response
    return requests_mock


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "id": 1,
        "filename": "test_document.pdf",
        "filepath": "/tmp/test_document.pdf",
        "content_type": "application/pdf",
        "status": "processed",
        "full_content": "This is a test document content.",
        "document_summary": "A test document summary.",
        "key_topics": ["test", "document"],
        "chapter_content": {"chapter_1": "Chapter content"},
        "toc_content": [{"title": "Chapter 1", "page": 1}],
        "content_structure": {"chapters": 1, "pages": 5},
    }


@pytest.fixture
def sample_chunk_data():
    """Sample document chunk data for testing."""
    return {
        "id": 1,
        "document_id": 1,
        "content": "This is a test chunk of content.",
        "chunk_index": 0,
        "chapter_title": "Chapter 1",
        "page_number": 1,
        "token_count": 10,
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
    }


@pytest.fixture
def sample_query():
    """Sample query data for testing."""
    return {
        "question": "What is machine learning?",
        "top_k": 5,
        "language": "en",
        "filters": {"category": "technical"},
    }
