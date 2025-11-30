"""
Unit tests for SecurityValidator class.

Tests security configuration validation, password checks, network settings,
and file security validation.
"""

import os
from unittest.mock import patch

import pytest

from src.utils.security_validator import SecurityValidator, validate_security_on_startup


class TestSecurityValidator:
    """Test the SecurityValidator class functionality."""

    def test_init(self):
        """Test SecurityValidator initialization."""
        validator = SecurityValidator()
        assert validator.issues == []
        assert isinstance(validator.issues, list)

    def test_validate_passwords_postgres_empty(self):
        """Test PostgreSQL password validation with empty password."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_PASSWORD": "",
                "OPENSEARCH_PASSWORD": "secure",
                "REDIS_PASSWORD": "secure",
            },
        ):
            validator = SecurityValidator()
            validator.validate_passwords()

            postgres_issues = [i for i in validator.issues if "PostgreSQL" in i.description]
            assert len(postgres_issues) == 1
            assert postgres_issues[0].severity == "high"
            assert postgres_issues[0].category == "password"
            assert "PostgreSQL password is empty" in postgres_issues[0].description

    def test_validate_passwords_postgres_too_short(self):
        """Test PostgreSQL password validation with short password."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_PASSWORD": "123",
                "OPENSEARCH_PASSWORD": "secure",
                "REDIS_PASSWORD": "secure",
            },
        ):
            validator = SecurityValidator()
            validator.validate_passwords()

            postgres_issues = [i for i in validator.issues if "PostgreSQL" in i.description]
            assert len(postgres_issues) == 1
            assert postgres_issues[0].severity == "medium"
            assert "PostgreSQL password is too short" in postgres_issues[0].description

    def test_validate_passwords_opensearch_default(self):
        """Test OpenSearch password validation with default value."""
        with patch.dict(os.environ, {"OPENSEARCH_PASSWORD": "changeme"}):
            validator = SecurityValidator()
            validator.validate_passwords()

            critical_issues = [i for i in validator.issues if i.severity == "critical"]
            assert len(critical_issues) == 1
            assert "OpenSearch password is using default value" in critical_issues[0].description

    def test_validate_passwords_opensearch_empty(self):
        """Test OpenSearch password validation with empty password."""
        with patch.dict(os.environ, {"OPENSEARCH_PASSWORD": ""}):
            validator = SecurityValidator()
            validator.validate_passwords()

            critical_issues = [i for i in validator.issues if i.severity == "critical"]
            assert len(critical_issues) == 1
            assert (
                "OpenSearch password is using default value or empty"
                in critical_issues[0].description
            )

    def test_validate_passwords_redis_empty(self):
        """Test Redis password validation with empty password."""
        with patch.dict(os.environ, {"REDIS_PASSWORD": ""}):
            validator = SecurityValidator()
            validator.validate_passwords()

            medium_issues = [i for i in validator.issues if i.severity == "medium"]
            assert len(medium_issues) == 1
            assert "Redis password is empty" in medium_issues[0].description

    def test_validate_passwords_all_good(self):
        """Test password validation with all good passwords."""
        env_vars = {
            "POSTGRES_PASSWORD": "strongpassword123",
            "OPENSEARCH_PASSWORD": "anotherstrongpass",
            "REDIS_PASSWORD": "redissecurepass",
        }
        with patch.dict(os.environ, env_vars):
            validator = SecurityValidator()
            validator.validate_passwords()

            # Should have no password-related issues
            password_issues = [i for i in validator.issues if i.category == "password"]
            assert len(password_issues) == 0

    def test_validate_network_settings_production_mode(self):
        """Test network settings validation in production mode."""
        env_vars = {"REQUIRE_SECURE_PASSWORDS": "true", "POSTGRES_HOST": "localhost"}
        with patch.dict(os.environ, env_vars):
            validator = SecurityValidator()
            validator.validate_network_settings()

            assert len(validator.issues) == 1
            assert validator.issues[0].severity == "medium"
            assert (
                "PostgreSQL bound to localhost in production mode"
                in validator.issues[0].description
            )

    def test_validate_network_settings_development_mode(self):
        """Test network settings validation in development mode."""
        env_vars = {"REQUIRE_SECURE_PASSWORDS": "false", "POSTGRES_HOST": "localhost"}
        with patch.dict(os.environ, env_vars):
            validator = SecurityValidator()
            validator.validate_network_settings()

            # Should have no network issues in development mode
            network_issues = [i for i in validator.issues if i.category == "network"]
            assert len(network_issues) == 0

    def test_validate_file_security_disabled(self):
        """Test file security validation when disabled."""
        with patch.dict(os.environ, {"ENABLE_SECURITY_VALIDATION": "false"}):
            validator = SecurityValidator()
            validator.validate_file_security()

            assert len(validator.issues) == 1
            assert validator.issues[0].severity == "medium"
            assert "File upload security validation is disabled" in validator.issues[0].description

    def test_validate_file_security_enabled(self):
        """Test file security validation when enabled."""
        with patch.dict(os.environ, {"ENABLE_SECURITY_VALIDATION": "true"}):
            validator = SecurityValidator()
            validator.validate_file_security()

            # Should have no file security issues when enabled
            file_issues = [i for i in validator.issues if i.category == "file"]
            assert len(file_issues) == 0

    def test_validate_all_comprehensive(self):
        """Test comprehensive validation with multiple issues."""
        env_vars = {
            "POSTGRES_PASSWORD": "",  # Empty
            "OPENSEARCH_PASSWORD": "changeme",  # Default
            "REDIS_PASSWORD": "",  # Empty
            "REQUIRE_SECURE_PASSWORDS": "true",
            "POSTGRES_HOST": "localhost",  # Localhost in production
            "ENABLE_SECURITY_VALIDATION": "false",  # Disabled
        }
        with patch.dict(os.environ, env_vars):
            validator = SecurityValidator()
            result = validator.validate_all()

            assert len(result) >= 4  # Should have multiple issues
            critical_issues = validator.get_critical_issues()
            assert len(critical_issues) >= 1  # At least OpenSearch issue

    def test_get_critical_issues(self):
        """Test getting only critical security issues."""
        from src.utils.security_validator import SecurityIssue

        # Create validator with mixed issues
        validator = SecurityValidator()
        validator.issues = [
            SecurityIssue(
                severity="critical",
                category="password",
                description="Critical issue",
                recommendation="Fix it",
            ),
            SecurityIssue(
                severity="high",
                category="password",
                description="High issue",
                recommendation="Fix it",
            ),
            SecurityIssue(
                severity="critical",
                category="password",
                description="Another critical",
                recommendation="Fix it",
            ),
        ]

        critical_issues = validator.get_critical_issues()
        assert len(critical_issues) == 2
        assert all(issue.severity == "critical" for issue in critical_issues)

    def test_print_report_no_issues(self, capsys):
        """Test printing report when no issues found."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_PASSWORD": "securepass123",
                "OPENSEARCH_PASSWORD": "securepass123",
                "REDIS_PASSWORD": "securepass123",
                "REQUIRE_SECURE_PASSWORDS": "false",
                "ENABLE_SECURITY_VALIDATION": "true",
            },
        ):
            validator = SecurityValidator()
            validator.validate_all()

            validator.print_report()

            captured = capsys.readouterr()
            assert "✅ No security issues found" in captured.out

    def test_print_report_with_issues(self, capsys):
        """Test printing report with various issues."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_PASSWORD": "",
                "OPENSEARCH_PASSWORD": "changeme",
                "REDIS_PASSWORD": "",
                "REQUIRE_SECURE_PASSWORDS": "true",
                "POSTGRES_HOST": "localhost",
                "ENABLE_SECURITY_VALIDATION": "false",
            },
        ):
            validator = SecurityValidator()
            validator.validate_all()

            validator.print_report()

            captured = capsys.readouterr()
            assert "CRITICAL ISSUES:" in captured.out
            assert "OpenSearch password" in captured.out

    def test_validate_security_on_startup_success(self):
        """Test successful security validation on startup."""
        env_vars = {
            "POSTGRES_PASSWORD": "strongpassword123",
            "OPENSEARCH_PASSWORD": "anotherstrongpass",
            "REDIS_PASSWORD": "redissecurepass",
        }
        with patch.dict(os.environ, env_vars):
            result = validate_security_on_startup()
            assert result is True

    def test_validate_security_on_startup_failure(self):
        """Test failed security validation on startup."""
        with patch.dict(os.environ, {"OPENSEARCH_PASSWORD": "changeme"}):
            result = validate_security_on_startup()
            assert result is False
