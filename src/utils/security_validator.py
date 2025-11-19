"""
Security configuration validator for LocalRAG system.
Validates environment variables and security settings.
"""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security configuration issue."""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'password', 'authentication', 'network', 'file'
    description: str
    recommendation: str


class SecurityValidator:
    """Validates security configuration and identifies potential issues."""
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
    
    def validate_passwords(self) -> None:
        """Validate password security settings."""
        # Check PostgreSQL password
        postgres_password = os.getenv('POSTGRES_PASSWORD', '')
        if not postgres_password:
            self.issues.append(SecurityIssue(
                severity='high',
                category='password',
                description='PostgreSQL password is empty',
                recommendation='Set a strong password for POSTGRES_PASSWORD in production'
            ))
        elif len(postgres_password) < 8:
            self.issues.append(SecurityIssue(
                severity='medium',
                category='password',
                description='PostgreSQL password is too short',
                recommendation='Use a password with at least 8 characters'
            ))
        
        # Check OpenSearch/Elasticsearch password
        opensearch_password = os.getenv('OPENSEARCH_PASSWORD', '')
        if not opensearch_password or opensearch_password == 'changeme':
            self.issues.append(SecurityIssue(
                severity='critical',
                category='password',
                description='OpenSearch password is using default value or empty',
                recommendation='Set a strong, unique password for OPENSEARCH_PASSWORD'
            ))
        
        # Check Redis password
        redis_password = os.getenv('REDIS_PASSWORD', '')
        if not redis_password:
            self.issues.append(SecurityIssue(
                severity='medium',
                category='password',
                description='Redis password is empty',
                recommendation='Set a password for Redis in production environments'
            ))
    
    def validate_network_settings(self) -> None:
        """Validate network security settings."""
        # Check if services are bound to localhost only in development
        require_secure = os.getenv('REQUIRE_SECURE_PASSWORDS', 'false').lower() == 'true'
        
        if require_secure:
            # In production, check for secure bindings
            postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
            opensearch_host = os.getenv('OPENSEARCH_HOST', 'localhost')
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            
            if postgres_host == 'localhost':
                self.issues.append(SecurityIssue(
                    severity='medium',
                    category='network',
                    description='PostgreSQL bound to localhost in production mode',
                    recommendation='Use proper network configuration for production'
                ))
    
    def validate_file_security(self) -> None:
        """Validate file upload and path security."""
        # Check for dangerous file upload settings
        enable_validation = os.getenv('ENABLE_SECURITY_VALIDATION', 'false').lower() == 'true'
        
        if not enable_validation:
            self.issues.append(SecurityIssue(
                severity='medium',
                category='file',
                description='File upload security validation is disabled',
                recommendation='Enable ENABLE_SECURITY_VALIDATION=true in production'
            ))
    
    def validate_all(self) -> List[SecurityIssue]:
        """Run all security validations."""
        self.issues.clear()
        
        self.validate_passwords()
        self.validate_network_settings()
        self.validate_file_security()
        
        return self.issues
    
    def get_critical_issues(self) -> List[SecurityIssue]:
        """Get only critical security issues."""
        return [issue for issue in self.issues if issue.severity == 'critical']
    
    def print_report(self) -> None:
        """Print a security report to console."""
        issues = self.validate_all()
        
        if not issues:
            print("✅ No security issues found")
            return
        
        print(f"\n🔒 Security Report - {len(issues)} issue(s) found")
        print("=" * 50)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = [i for i in issues if i.severity == severity]
            if severity_issues:
                print(f"\n{severity.upper()} ISSUES:")
                for issue in severity_issues:
                    print(f"  • {issue.description}")
                    print(f"    Recommendation: {issue.recommendation}")
        
        critical_count = len(self.get_critical_issues())
        if critical_count > 0:
            print(f"\n⚠️  {critical_count} critical issue(s) must be resolved before production use")


def validate_security_on_startup() -> bool:
    """Validate security settings on application startup."""
    validator = SecurityValidator()
    issues = validator.validate_all()
    
    critical_issues = validator.get_critical_issues()
    
    if critical_issues:
        logger.error("Critical security issues detected:")
        for issue in critical_issues:
            logger.error(f"  - {issue.description}: {issue.recommendation}")
        return False
    
    if issues:
        logger.warning(f"Security issues detected: {len(issues)} total")
        for issue in issues:
            logger.warning(f"  - {issue.description}")
    
    return True


if __name__ == "__main__":
    # Run security validation when called directly
    validator = SecurityValidator()
    validator.print_report()