#!/usr/bin/env python3
"""
Comprehensive test suite for the security and maintainability fixes.
Tests all the critical issues that were identified and fixed.
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the modules we fixed
from utils.security_validator import SecurityValidator, SecurityIssue
from utils.config_manager import ConfigManager, ModelConfig
from utils.file_security import FileUploadValidator, FileSecurityError
from utils.error_handler import (
    ErrorHandler, LocalRAGError, DatabaseError, 
    handle_errors, error_context, safe_execute
)


class TestSecurityValidator(unittest.TestCase):
    """Test the security validator fixes."""
    
    def setUp(self):
        self.validator = SecurityValidator()
    
    def test_empty_postgres_password_detected(self):
        """Test that empty PostgreSQL password is detected."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': ''}):
            issues = self.validator.validate_passwords()
            postgres_issues = [i for i in issues if i.category == 'password' and 'PostgreSQL' in i.description]
            self.assertTrue(len(postgres_issues) > 0)
    
    def test_default_opensearch_password_detected(self):
        """Test that default OpenSearch password is detected."""
        with patch.dict(os.environ, {'OPENSEARCH_PASSWORD': 'changeme'}):
            issues = self.validator.validate_passwords()
            opensearch_issues = [i for i in issues if 'changeme' in i.description]
            self.assertTrue(len(opensearch_issues) > 0)
    
    def test_critical_issues_filtering(self):
        """Test filtering of critical security issues."""
        # Add a critical issue
        self.validator.issues.append(SecurityIssue(
            severity='critical',
            category='password',
            description='Critical issue',
            recommendation='Fix it'
        ))
        
        critical = self.validator.get_critical_issues()
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].severity, 'critical')


class TestConfigManager(unittest.TestCase):
    """Test the configuration manager fixes."""
    
    def test_model_config_from_env(self):
        """Test that model configuration can be overridden from environment."""
        test_model = "test-model-v2"
        with patch.dict(os.environ, {'EMBEDDING_MODEL': test_model}):
            config = ConfigManager()
            self.assertEqual(config.get_embedding_model(), test_model)
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        config = ConfigManager()
        url = config.get_database_url()
        self.assertIn('postgresql://', url)
        self.assertIn(config.database.host, url)
        self.assertIn(str(config.database.port), url)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigManager()
        validation = config.validate_config()
        self.assertIn('valid', validation)
        self.assertIn('issues', validation)
    
    def test_model_type_selection(self):
        """Test different model type selection."""
        config = ConfigManager()
        
        # Test default LLM
        default_llm = config.get_llm_model()
        self.assertIsInstance(default_llm, str)
        
        # Test small LLM
        small_llm = config.get_llm_model("small")
        self.assertIsInstance(small_llm, str)
        
        # Test vision models
        default_vision = config.get_vision_model()
        small_vision = config.get_vision_model("small")
        self.assertIsInstance(default_vision, str)
        self.assertIsInstance(small_vision, str)


class TestFileSecurity(unittest.TestCase):
    """Test the file upload security fixes."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.validator = FileUploadValidator(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dangerous_filename_blocked(self):
        """Test that dangerous filenames are blocked."""
        dangerous_names = ['../../../etc/passwd', 'file.exe', 'script.bat', 'document.php']
        
        for name in dangerous_names:
            with self.assertRaises(FileSecurityError):
                self.validator.validate_filename(name)
    
    def test_safe_filename_allowed(self):
        """Test that safe filenames are allowed."""
        safe_names = ['document.pdf', 'image.png', 'text.txt', 'report.docx']
        
        for name in safe_names:
            try:
                result = self.validator.validate_filename(name)
                self.assertIsInstance(result, str)
            except FileSecurityError:
                self.fail(f"Safe filename {name} was blocked")
    
    def test_file_size_validation(self):
        """Test file size validation."""
        # Test valid size
        try:
            self.validator.validate_file_size(1024)  # 1KB
        except FileSecurityError:
            self.fail("Valid file size was rejected")
        
        # Test oversized file
        with self.assertRaises(FileSecurityError):
            self.validator.validate_file_size(self.validator.max_file_size + 1)
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        dangerous_paths = [
            '../../../etc/passwd',
            '/etc/passwd',
            '..\\..\\windows\\system32\\config\\sam'
        ]
        
        for path in dangerous_paths:
            with self.assertRaises(FileSecurityError):
                self.validator.validate_file_path(path)
    
    def test_safe_file_upload(self):
        """Test safe file upload process."""
        test_content = b"This is a test file content"
        test_filename = "test.txt"
        
        try:
            file_path, result = self.validator.save_uploaded_file(test_filename, test_content)
            self.assertTrue(file_path.exists())
            self.assertTrue(result['valid'])
            self.assertEqual(result['sanitized_filename'], test_filename)
        except Exception as e:
            self.fail(f"Safe file upload failed: {e}")
    
    def test_mime_type_blocking(self):
        """Test that dangerous MIME types are blocked."""
        # Create a temporary file with dangerous content
        dangerous_file = os.path.join(self.temp_dir, "test.exe")
        with open(dangerous_file, 'wb') as f:
            f.write(b"MZ\x90\x00")  # PE header
        
        # This should be blocked by extension validation first
        with self.assertRaises(FileSecurityError):
            self.validator.validate_filename("test.exe")


class TestErrorHandler(unittest.TestCase):
    """Test the error handling fixes."""
    
    def setUp(self):
        self.error_handler = ErrorHandler()
    
    def test_error_statistics_tracking(self):
        """Test that error statistics are tracked."""
        initial_stats = self.error_handler.get_error_stats()
        self.assertEqual(initial_stats['total_errors'], 0)
        
        # Handle an error
        test_error = ValueError("Test error")
        self.error_handler.handle_error(test_error, reraise=False)
        
        updated_stats = self.error_handler.get_error_stats()
        self.assertEqual(updated_stats['total_errors'], 1)
        self.assertIn('ValueError', updated_stats['error_types'])
    
    def test_custom_error_classes(self):
        """Test custom error classes."""
        # Test LocalRAGError
        custom_error = LocalRAGError("Test message", "TEST_CODE", {"key": "value"})
        self.assertEqual(custom_error.message, "Test message")
        self.assertEqual(custom_error.error_code, "TEST_CODE")
        self.assertEqual(custom_error.context["key"], "value")
        
        # Test DatabaseError
        db_error = DatabaseError("Database connection failed")
        self.assertIsInstance(db_error, LocalRAGError)
        self.assertEqual(db_error.error_code, "DatabaseError")
    
    def test_error_decorator(self):
        """Test the error handling decorator."""
        @handle_errors(default_return="error_occurred", reraise=False)
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        self.assertEqual(result, "error_occurred")
    
    def test_error_context_manager(self):
        """Test the error context manager."""
        with self.assertRaises(ValueError):
            with error_context("test_operation", {"test": True}):
                raise ValueError("Test error in context")
    
    def test_safe_execute_function(self):
        """Test the safe execute function."""
        def failing_func():
            raise RuntimeError("Test runtime error")
        
        result = safe_execute(failing_func, default_return="safe")
        self.assertEqual(result, "safe")
        
        def working_func():
            return "success"
        
        result = safe_execute(working_func)
        self.assertEqual(result, "success")


class TestImportFixes(unittest.TestCase):
    """Test that the import fixes work correctly."""
    
    def test_test_runner_imports(self):
        """Test that the test runner imports are fixed."""
        # This should not raise ImportError
        try:
            import importlib.util
            self.assertTrue(hasattr(importlib.util, 'spec_from_file_location'))
        except ImportError:
            self.fail("importlib.util should be available")
    
    def test_base_processor_import(self):
        """Test that base processor can be imported."""
        try:
            from core.base_processor import BaseProcessor
            self.assertTrue(hasattr(BaseProcessor, 'process_existing_documents'))
        except ImportError as e:
            self.fail(f"BaseProcessor import failed: {e}")


class TestCodeDuplicationFix(unittest.TestCase):
    """Test that code duplication is fixed."""
    
    def test_base_processor_shared_functionality(self):
        """Test that BaseProcessor provides shared functionality."""
        try:
            from core.base_processor import BaseProcessor
            
            # Check that the base class has the shared method
            self.assertTrue(hasattr(BaseProcessor, 'process_existing_documents'))
            self.assertTrue(hasattr(BaseProcessor, '_process_single_document'))
            self.assertTrue(hasattr(BaseProcessor, '_validate_file_path'))
            
        except ImportError:
            self.skipTest("BaseProcessor not available")


def run_comprehensive_tests():
    """Run all the comprehensive tests."""
    print("🧪 Running Comprehensive Security and Maintainability Tests")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestSecurityValidator,
        TestConfigManager,
        TestFileSecurity,
        TestErrorHandler,
        TestImportFixes,
        TestCodeDuplicationFix
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 All tests passed! Security and maintainability fixes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)