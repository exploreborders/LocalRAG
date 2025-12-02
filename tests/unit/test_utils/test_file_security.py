"""
Unit tests for file security validation utilities.
"""

import tempfile
from pathlib import Path

import pytest

from src.utils.file_security import (
    FileSecurityError,
    FileUploadValidator,
)


class TestFileUploadValidator:
    """Test the FileUploadValidator class."""

    def test_init(self):
        """Test FileUploadValidator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)
            assert validator is not None
            assert validator.upload_dir.exists()
            assert validator.upload_dir.is_dir()

    def test_validate_filename_safe(self):
        """Test filename validation with safe names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test safe filenames
            safe_names = [
                "document.pdf",
                "report.docx",
                "data.xlsx",
                "presentation.pptx",
                "notes.txt",
            ]

            for filename in safe_names:
                result = validator.validate_filename(filename)
                assert isinstance(result, str)
                assert result.endswith(Path(filename).suffix)

    def test_validate_filename_dangerous(self):
        """Test filename validation with dangerous names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test dangerous filenames
            dangerous_names = [
                "malicious.exe",
                "script.bat",
                "dangerous.scr",
                "../../../etc/passwd",
                "file;rm -rf /",
                "test|cat /etc/passwd",
            ]

            for filename in dangerous_names:
                with pytest.raises(FileSecurityError):
                    validator.validate_filename(filename)

    def test_validate_file_size_normal(self):
        """Test file size validation with normal size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test normal size (1MB)
            normal_size = 1024 * 1024
            validator.validate_file_size(normal_size)  # Should not raise

    def test_validate_file_size_oversized(self):
        """Test file size validation with oversized file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test oversized content (200MB)
            oversized = 200 * 1024 * 1024
            with pytest.raises(FileSecurityError):
                validator.validate_file_size(oversized)

    def test_generate_safe_filename(self):
        """Test safe filename generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test normal filename
            result = validator.generate_safe_filename("test document.pdf")
            assert result == "test document.pdf"  # Should be sanitized

    def test_validate_uploaded_file_valid(self):
        """Test validation of valid uploaded file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test valid file
            filename = "document.pdf"
            file_size = 1024
            content = b"PDF content"

            result = validator.validate_uploaded_file(filename, file_size, content)
            assert isinstance(result, dict)
            assert "valid" in result
            assert result["valid"] is True

    def test_validate_uploaded_file_invalid_size(self):
        """Test validation of oversized uploaded file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test oversized file
            filename = "document.pdf"
            file_size = 200 * 1024 * 1024  # 200MB
            content = b"x" * 1024

            with pytest.raises(FileSecurityError):
                validator.validate_uploaded_file(filename, file_size, content)

    def test_validate_uploaded_file_invalid_name(self):
        """Test validation of file with dangerous name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test dangerous filename
            filename = "../../../etc/passwd"
            file_size = 1024
            content = b"content"

            with pytest.raises(FileSecurityError):
                validator.validate_uploaded_file(filename, file_size, content)

    def test_validate_mime_type_allowed(self):
        """Test MIME type validation with allowed types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test allowed MIME types
            allowed_files = [
                "document.pdf",
                "report.docx",
                "data.xlsx",
                "notes.txt",
            ]

            for filename in allowed_files:
                # Create a temporary file
                file_path = Path(temp_dir) / filename
                file_path.write_text("test content")

                # Should not raise an exception
                validator.validate_mime_type(str(file_path))

    def test_validate_mime_type_unknown_extension(self):
        """Test MIME type validation with unknown extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test unknown extension - should not raise (defaults to application/octet-stream)
            unknown_filename = "file.unknown"
            file_path = Path(temp_dir) / unknown_filename
            file_path.write_text("test content")

            # Should not raise for unknown extensions
            validator.validate_mime_type(str(file_path))

    def test_validate_file_path_valid(self):
        """Test file path validation with valid paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test valid file path within upload directory
            file_path = Path(temp_dir) / "test.pdf"
            file_path.write_text("test content")

            result = validator.validate_file_path(str(file_path))
            assert isinstance(result, Path)
            assert result == file_path.resolve()

    def test_validate_file_path_outside_upload_dir(self):
        """Test file path validation with paths outside upload directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            # Test path outside upload directory
            outside_path = Path(temp_dir).parent / "outside.txt"

            with pytest.raises(FileSecurityError):
                validator.validate_file_path(str(outside_path))

    def test_save_uploaded_file_success(self):
        """Test successful file saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = FileUploadValidator(temp_dir)

            filename = "test.pdf"
            content = b"PDF content here"

            saved_path, file_info = validator.save_uploaded_file(filename, content)

            assert saved_path.exists()
            assert saved_path.is_file()
            assert saved_path.suffix == ".pdf"
            assert saved_path.parent == validator.upload_dir

            assert isinstance(file_info, dict)
            assert "sanitized_filename" in file_info
            assert "file_size" in file_info
            assert file_info["valid"] is True

            # Verify content was saved
            assert saved_path.read_bytes() == content


class TestFileSecurityError:
    """Test the custom FileSecurityError exception."""

    def test_file_security_error_creation(self):
        """Test FileSecurityError creation."""
        error = FileSecurityError("Security violation detected")
        assert str(error) == "Security violation detected"
        assert isinstance(error, Exception)
