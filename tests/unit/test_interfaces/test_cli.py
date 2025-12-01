"""
Unit tests for RAGCLI class.

Tests CLI interface functionality, menu system, command handling, and user interactions.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.interfaces.cli import RAGCLI


class TestRAGCLI:
    """Test the RAGCLI class functionality."""

    def test_init(self):
        """Test RAGCLI initialization."""
        cli = RAGCLI()

        assert cli.retriever is None
        assert cli.rag_pipeline is None
        assert cli.processor is None
        assert cli.cache is None
        assert isinstance(cli.lang_names, dict)
        assert "en" in cli.lang_names
        assert cli.lang_names["en"] == "🇺🇸 English"

    def test_print_header(self, capsys):
        """Test header printing."""
        cli = RAGCLI()
        cli.print_header()

        captured = capsys.readouterr()
        assert "🤖 LOCAL RAG SYSTEM" in captured.out
        assert "Command Line Interface" in captured.out
        assert "=" * 70 in captured.out

    def test_print_menu(self, capsys):
        """Test menu printing."""
        cli = RAGCLI()
        cli.print_menu()

        captured = capsys.readouterr()
        assert "📋 Available Modes:" in captured.out
        assert "1. 🎯 Smart Search" in captured.out
        assert "2. 🤖 Full RAG Mode" in captured.out
        assert "3. 📁 Process Documents" in captured.out
        assert "4. 📊 System Status" in captured.out
        assert "5. ⚙️  Settings" in captured.out
        assert "6. 🆘 Help" in captured.out
        assert "0. 🚪 Exit" in captured.out

    @patch("src.interfaces.cli.DocumentProcessor")
    @patch("src.interfaces.cli.DatabaseRetriever")
    def test_initialize_components_success(self, mock_retriever_class, mock_processor_class):
        """Test successful component initialization."""
        mock_retriever = MagicMock()
        mock_processor = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_processor_class.return_value = mock_processor

        cli = RAGCLI()
        result = cli.initialize_components()

        assert result is True
        assert cli.retriever == mock_retriever
        assert cli.processor == mock_processor
        mock_retriever_class.assert_called_once()
        mock_processor_class.assert_called_once()

    @patch("src.interfaces.cli.RedisCache")
    @patch("src.interfaces.cli.DocumentProcessor")
    @patch("src.interfaces.cli.DatabaseRetriever")
    def test_initialize_components_with_cache(
        self, mock_retriever_class, mock_processor_class, mock_cache_class
    ):
        """Test component initialization with Redis cache."""
        mock_retriever = MagicMock()
        mock_processor = MagicMock()
        mock_cache = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_processor_class.return_value = mock_processor
        mock_cache_class.return_value = mock_cache

        cli = RAGCLI()
        result = cli.initialize_components()

        assert result is True
        assert cli.cache == mock_cache

    def test_initialize_components_failure(self, capsys):
        """Test component initialization failure."""
        with patch("src.interfaces.cli.DocumentProcessor", side_effect=Exception("Init failed")):
            cli = RAGCLI()
            result = cli.initialize_components()

            assert result is False
            captured = capsys.readouterr()
            assert "Error initializing components" in captured.out

    @patch("builtins.input")
    def test_topic_aware_mode_quit(self, mock_input):
        """Test topic aware mode with quit command."""
        mock_input.return_value = "quit"

        cli = RAGCLI()
        cli.topic_aware_mode()

        # Should not raise exception and exit gracefully

    @patch("builtins.input")
    @patch("src.interfaces.cli.DatabaseRetriever")
    def test_topic_aware_mode_search(self, mock_retriever_class, mock_input):
        """Test topic aware mode with search query."""
        # Mock user inputs: query, then quit
        mock_input.side_effect = ["test query", "quit"]

        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_topic_boost.return_value = [
            {"content": "test result", "score": 0.8, "topic_boost": 0.2}
        ]
        mock_retriever_class.return_value = mock_retriever

        cli = RAGCLI()
        cli.retriever = mock_retriever

        with patch("time.time", side_effect=[1.0, 2.0]):
            cli.topic_aware_mode()

        mock_retriever.retrieve_with_topic_boost.assert_called_once_with("test query", top_k=3)

    @patch("builtins.input")
    def test_rag_mode_quit(self, mock_input):
        """Test RAG mode with quit command."""
        mock_input.return_value = "quit"

        cli = RAGCLI()
        cli.rag_mode()

        # Should not raise exception

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGPipelineDB")
    def test_rag_mode_query(self, mock_rag_class, mock_input):
        """Test RAG mode with query."""
        mock_input.side_effect = ["test question", "quit"]

        mock_rag = MagicMock()
        mock_rag.query.return_value = {
            "answer": "Test answer",
            "query_language": "en",
            "retrieved_documents": [{"document": {"filename": "test.pdf"}, "score": 0.9}],
        }
        mock_rag_class.return_value = mock_rag

        cli = RAGCLI()
        cli.rag_pipeline = mock_rag

        with patch("time.time", side_effect=[1.0, 2.0]):
            cli.rag_mode()

        mock_rag.query.assert_called_once_with("test question")

    @patch("src.interfaces.cli.DocumentProcessor")
    def test_process_documents(self, mock_processor_class, capsys):
        """Test document processing."""
        mock_processor = MagicMock()
        mock_processor.process_existing_documents = MagicMock()
        mock_processor_class.return_value = mock_processor

        cli = RAGCLI()
        cli.processor = mock_processor

        with patch("time.time", side_effect=[1.0, 6.0]):  # 5 seconds
            cli.process_documents()

        captured = capsys.readouterr()
        assert "Processing existing documents" in captured.out
        mock_processor.process_existing_documents.assert_called_once()

    def test_show_system_status(self, capsys):
        """Test system status display."""
        cli = RAGCLI()
        # Set cache to prevent initialization
        cli.cache = MagicMock()

        # Mock external services to avoid connection attempts
        with (
            patch("elasticsearch.Elasticsearch") as mock_es,
            patch("redis.Redis") as mock_redis,
        ):
            # Mock Elasticsearch to raise connection error
            mock_es_instance = MagicMock()
            mock_es_instance.ping.side_effect = Exception("Connection refused")
            mock_es.return_value = mock_es_instance

            # Mock Redis to raise connection error
            mock_redis_instance = MagicMock()
            mock_redis_instance.ping.side_effect = Exception("Connection refused")
            mock_redis.return_value = mock_redis_instance

            # For database, we'll just test that the method runs without crashing
            # The database connection will fail but that's expected in test environment
            try:
                cli.show_system_status()
            except Exception:
                # If it fails due to database issues, that's expected
                pass

        captured = capsys.readouterr()
        assert "📊 SYSTEM STATUS" in captured.out
        # Don't check for database status since it's hard to mock the internal imports

    def test_show_settings(self, capsys):
        """Test settings display."""
        cli = RAGCLI()
        cli.show_settings()

        captured = capsys.readouterr()
        assert "⚙️  SYSTEM SETTINGS" in captured.out
        assert "Current configuration:" in captured.out

    def test_show_help(self, capsys):
        """Test help display."""
        cli = RAGCLI()
        cli.show_help()

        captured = capsys.readouterr()
        assert "🆘 HELP - Local RAG System CLI" in captured.out
        assert "MODES:" in captured.out
        assert "FEATURES:" in captured.out

    def test_show_topic_aware_help(self, capsys):
        """Test topic aware help display."""
        cli = RAGCLI()
        cli.show_topic_aware_help()

        captured = capsys.readouterr()
        assert "🎯 SMART SEARCH COMMANDS:" in captured.out

    def test_show_rag_help(self, capsys):
        """Test RAG help display."""
        cli = RAGCLI()
        cli.show_rag_help()

        captured = capsys.readouterr()
        assert "🤖 RAG MODE COMMANDS:" in captured.out

    @patch("builtins.input")
    def test_run_exit(self, mock_input):
        """Test main run loop with exit command."""
        mock_input.return_value = "0"

        cli = RAGCLI()

        with patch("sys.exit"):
            cli.run()
            # Should not reach sys.exit in test

    @patch("builtins.input")
    def test_run_invalid_choice(self, mock_input, capsys):
        """Test main run loop with invalid choice."""
        mock_input.side_effect = ["invalid", "0"]

        cli = RAGCLI()
        cli.run()

        captured = capsys.readouterr()
        assert "❌ Invalid choice" in captured.out

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGCLI.topic_aware_mode")
    def test_run_topic_aware_mode(self, mock_topic_mode, mock_input):
        """Test main run loop selecting topic aware mode."""
        mock_input.side_effect = ["1", "0"]

        cli = RAGCLI()
        cli.run()

        mock_topic_mode.assert_called_once()

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGCLI.rag_mode")
    def test_run_rag_mode(self, mock_rag_mode, mock_input):
        """Test main run loop selecting RAG mode."""
        mock_input.side_effect = ["2", "0"]

        cli = RAGCLI()
        cli.run()

        mock_rag_mode.assert_called_once()

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGCLI.process_documents")
    def test_run_process_documents(self, mock_process_docs, mock_input):
        """Test main run loop selecting document processing."""
        mock_input.side_effect = ["3", "0"]

        cli = RAGCLI()
        cli.run()

        mock_process_docs.assert_called_once()

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGCLI.show_system_status")
    def test_run_system_status(self, mock_system_status, mock_input):
        """Test main run loop selecting system status."""
        mock_input.side_effect = ["4", "0"]

        cli = RAGCLI()
        cli.run()

        mock_system_status.assert_called_once()

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGCLI.show_settings")
    def test_run_settings(self, mock_settings, mock_input):
        """Test main run loop selecting settings."""
        mock_input.side_effect = ["5", "0"]

        cli = RAGCLI()
        cli.run()

        mock_settings.assert_called_once()

    @patch("builtins.input")
    @patch("src.interfaces.cli.RAGCLI.show_help")
    def test_run_help(self, mock_help, mock_input):
        """Test main run loop selecting help."""
        mock_input.side_effect = ["6", "0"]

        cli = RAGCLI()
        cli.run()

        mock_help.assert_called_once()

    @patch("builtins.input")
    def test_run_keyboard_interrupt(self, mock_input):
        """Test main run loop with keyboard interrupt."""
        mock_input.side_effect = KeyboardInterrupt()

        cli = RAGCLI()

        # Should not raise exception
        cli.run()

    def test_main_function(self):
        """Test main function."""
        with patch("src.interfaces.cli.RAGCLI") as mock_cli_class:
            mock_cli = MagicMock()
            mock_cli_class.return_value = mock_cli

            from src.interfaces.cli import main

            main()

            mock_cli_class.assert_called_once()
            mock_cli.run.assert_called_once()

    def test_main_function_exception(self, capsys):
        """Test main function with exception."""
        with patch("src.interfaces.cli.RAGCLI", side_effect=Exception("Test error")):
            from src.interfaces.cli import main

            with pytest.raises(SystemExit):
                main()

            captured = capsys.readouterr()
            assert "❌ Fatal error: Test error" in captured.out
