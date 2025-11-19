"""
File upload security utilities for LocalRAG system.
Provides secure file handling and validation to prevent directory traversal
and other file-based attacks.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# MIME type detection based on file extensions
EXTENSION_MIME_MAP = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.rst': 'text/x-rst',
    '.org': 'text/org',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.odt': 'application/vnd.oasis.opendocument.text',
    '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
    '.odp': 'application/vnd.oasis.opendocument.presentation',
    '.pdf': 'application/pdf',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.webp': 'image/webp',
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.xml': 'application/xml',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.epub': 'application/epub+zip',
    '.mobi': 'application/x-mobipocket-ebook',
}

logger = logging.getLogger(__name__)


class FileSecurityError(Exception):
    """Exception raised for file security violations."""
    pass


class FileUploadValidator:
    """Validates file uploads for security compliance."""
    
    # Allowed file extensions for document processing
    ALLOWED_EXTENSIONS = {
        # Text documents
        '.txt', '.md', '.rst', '.org',
        # Microsoft Office
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # OpenDocument
        '.odt', '.ods', '.odp',
        # PDF
        '.pdf',
        # Images (for OCR/vision processing)
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp',
        # Other formats
        '.csv', '.json', '.xml', '.html', '.htm',
        # E-books
        '.epub', '.mobi',
    }
    
    # Maximum file size (100MB by default)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
    
    # Dangerous file extensions to block
    BLOCKED_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
        '.app', '.deb', '.pkg', '.dmg', '.rpm', '.msi', '.msm', '.msp',
        '.php', '.asp', '.aspx', '.jsp', '.py', '.pl', '.rb', '.sh',
        '.ps1', '.psm1', '.psd1', '.ps1xml', '.psc1', '.cdxml', '.acl',
        '.reg', '.inf', '.sys', '.dll', '.ocx', '.cpl', '.drv', '.scf',
        '.lnk', '.url', '.xnk', '.wsh', '.wsf', '.vbe', '.jse', '.vbscript',
    }
    
    # MIME types that should be blocked
    BLOCKED_MIME_TYPES = {
        'application/x-executable',
        'application/x-msdownload',
        'application/x-msdos-program',
        'application/x-msi',
        'application/x-sh',
        'application/x-shellscript',
        'application/x-python',
        'application/x-perl',
        'application/x-ruby',
        'application/x-php',
        'application/x-javascript',
        'text/javascript',
        'application/javascript',
    }
    
    def __init__(self, upload_dir: str, max_file_size: Optional[int] = None):
        """
        Initialize the file upload validator.
        
        Args:
            upload_dir: Directory where files will be uploaded
            max_file_size: Maximum allowed file size in bytes
        """
        self.upload_dir = Path(upload_dir).resolve()
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize a filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
            
        Raises:
            FileSecurityError: If filename is invalid
        """
        if not filename:
            raise FileSecurityError("Filename cannot be empty")
        
        # Remove path components to prevent directory traversal
        filename = os.path.basename(filename)
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            if char in filename:
                raise FileSecurityError(f"Filename contains dangerous character: {char}")
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        name_without_ext = os.path.splitext(filename)[0].upper()
        if name_without_ext in reserved_names:
            raise FileSecurityError(f"Filename is reserved: {name_without_ext}")
        
        # Check extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in self.BLOCKED_EXTENSIONS:
            raise FileSecurityError(f"File extension is not allowed: {file_ext}")
        
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise FileSecurityError(f"File extension is not supported: {file_ext}")
        
        # Check filename length
        if len(filename) > 255:
            raise FileSecurityError("Filename is too long (max 255 characters)")
        
        return filename
    
    def validate_file_size(self, file_size: int) -> None:
        """
        Validate file size.
        
        Args:
            file_size: File size in bytes
            
        Raises:
            FileSecurityError: If file size is too large
        """
        if file_size > self.max_file_size:
            raise FileSecurityError(
                f"File size {file_size} bytes exceeds maximum allowed size {self.max_file_size} bytes"
            )
        
        if file_size <= 0:
            raise FileSecurityError("File size must be greater than 0")
    
    def validate_mime_type(self, file_path: str) -> None:
        """
        Validate file MIME type.
        
        Args:
            file_path: Path to the file to check
            
        Raises:
            FileSecurityError: If MIME type is blocked
        """
        try:
            # Get MIME type from file extension
            ext = Path(file_path).suffix.lower()
            mime_type = EXTENSION_MIME_MAP.get(ext, 'application/octet-stream')
            
            if mime_type in self.BLOCKED_MIME_TYPES:
                raise FileSecurityError(f"File MIME type is not allowed: {mime_type}")
                
        except Exception as e:
            logger.warning(f"Could not determine MIME type for {file_path}: {e}")
            # Don't fail the upload if we can't determine MIME type,
            # but log it for security monitoring
    
    def validate_file_path(self, file_path: str) -> Path:
        """
        Validate and resolve a file path to ensure it's within the upload directory.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            FileSecurityError: If path is outside upload directory
        """
        try:
            # Convert to Path object and resolve
            path = Path(file_path).resolve()
            
            # Check if path is within upload directory
            if not str(path).startswith(str(self.upload_dir)):
                raise FileSecurityError(f"File path is outside upload directory: {path}")
            
            # Check for directory traversal in the resolved path
            if '..' in path.parts:
                raise FileSecurityError(f"Directory traversal detected in path: {path}")
            
            return path
            
        except Exception as e:
            if isinstance(e, FileSecurityError):
                raise
            raise FileSecurityError(f"Invalid file path: {e}")
    
    def generate_safe_filename(self, original_filename: str) -> str:
        """
        Generate a safe filename by adding a hash if needed.
        
        Args:
            original_filename: Original filename
            
        Returns:
            Safe filename
        """
        # Validate the original filename first
        safe_name = self.validate_filename(original_filename)
        
        # If file exists, add a hash to make it unique
        file_path = self.upload_dir / safe_name
        if file_path.exists():
            name, ext = os.path.splitext(safe_name)
            hash_suffix = hashlib.md5(original_filename.encode()).hexdigest()[:8]
            safe_name = f"{name}_{hash_suffix}{ext}"
        
        return safe_name
    
    def validate_uploaded_file(self, filename: str, file_size: int, file_content: bytes) -> Dict[str, Any]:
        """
        Perform comprehensive validation of an uploaded file.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            file_content: File content as bytes
            
        Returns:
            Dictionary with validation results
            
        Raises:
            FileSecurityError: If validation fails
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'sanitized_filename': None,
            'file_size': file_size,
        }
        
        try:
            # Validate filename
            safe_filename = self.validate_filename(filename)
            validation_result['sanitized_filename'] = safe_filename
            
            # Validate file size
            self.validate_file_size(file_size)
            
            # Create temporary file to validate MIME type
            temp_path = self.upload_dir / f"temp_{hashlib.md5(file_content).hexdigest()}"
            try:
                with open(temp_path, 'wb') as f:
                    f.write(file_content)
                
                self.validate_mime_type(str(temp_path))
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
            
            # Additional content-based checks
            self._validate_file_content(file_content, validation_result)
            
        except FileSecurityError as e:
            validation_result['valid'] = False
            validation_result['error'] = str(e)
            raise
        except Exception as e:
            validation_result['valid'] = False
            validation_result['error'] = f"Validation error: {e}"
            raise FileSecurityError(f"Validation error: {e}")
        
        return validation_result
    
    def _validate_file_content(self, content: bytes, result: Dict[str, Any]) -> None:
        """
        Perform content-based validation.
        
        Args:
            content: File content as bytes
            result: Validation result dictionary to update
        """
        # Check for common malware signatures (simplified)
        suspicious_patterns = [
            b'eval(base64_decode',
            b'shell_exec',
            b'passthru',
            b'system(',
            b'exec(',
            b'<script',
            b'javascript:',
            b'vbscript:',
        ]
        
        content_lower = content.lower()
        for pattern in suspicious_patterns:
            if pattern in content_lower:
                result['warnings'].append(f"Suspicious pattern detected: {pattern.decode()}")
        
        # Check for encrypted/zip files that might contain malware
        zip_signatures = [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08']
        for sig in zip_signatures:
            if content.startswith(sig):
                result['warnings'].append("File appears to be a compressed archive")
                break
    
    def save_uploaded_file(self, filename: str, file_content: bytes) -> Tuple[Path, Dict[str, Any]]:
        """
        Save an uploaded file with security validation.
        
        Args:
            filename: Original filename
            file_content: File content as bytes
            
        Returns:
            Tuple of (saved_file_path, validation_result)
            
        Raises:
            FileSecurityError: If validation fails
        """
        # Validate the file
        validation_result = self.validate_uploaded_file(filename, len(file_content), file_content)
        
        # Generate safe filename
        safe_filename = self.generate_safe_filename(filename)
        file_path = self.upload_dir / safe_filename
        
        # Save the file
        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Validate the saved file path
            validated_path = self.validate_file_path(str(file_path))
            
            logger.info(f"File saved successfully: {validated_path}")
            return validated_path, validation_result
            
        except Exception as e:
            # Clean up if save failed
            if file_path.exists():
                file_path.unlink()
            raise FileSecurityError(f"Failed to save file: {e}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = self.validate_file_path(file_path)
            
            stat = path.stat()
            # Get MIME type from extension
            ext = path.suffix.lower()
            mime_type = EXTENSION_MIME_MAP.get(ext, 'application/octet-stream')
            
            return {
                'path': str(path),
                'filename': path.name,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'mime_type': mime_type,
                'extension': path.suffix.lower(),
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {'error': str(e)}


def create_secure_upload_validator(upload_dir: str, max_file_size_mb: int = 100) -> FileUploadValidator:
    """
    Create a file upload validator with sensible defaults.
    
    Args:
        upload_dir: Directory for uploads
        max_file_size_mb: Maximum file size in MB
        
    Returns:
        Configured FileUploadValidator instance
    """
    max_bytes = max_file_size_mb * 1024 * 1024
    return FileUploadValidator(upload_dir, max_bytes)