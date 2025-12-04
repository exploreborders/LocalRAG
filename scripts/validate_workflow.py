#!/usr/bin/env python3
"""
GitHub Actions Workflow Validator
Validates the CI/CD workflow configuration
"""

from pathlib import Path

import yaml


def validate_workflow():
    """Validate the GitHub Actions workflow file."""
    workflow_path = Path(".github/workflows/ci.yml")

    if not workflow_path.exists():
        print("❌ Workflow file not found")
        return False

    try:
        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        # Basic validation - check essential structure
        if "name" not in workflow:
            print("❌ Missing workflow name")
            return False

        if "jobs" not in workflow:
            print("❌ Missing jobs section")
            return False

        jobs = workflow["jobs"]
        expected_jobs = ["test", "quality", "security", "docs", "validate", "codeql"]
        missing_jobs = [job for job in expected_jobs if job not in jobs]
        if missing_jobs:
            print(f"❌ Missing jobs: {missing_jobs}")
            return False

        # Check job dependencies
        validate_job = jobs.get("validate", {})
        validate_needs = validate_job.get("needs", [])
        required_deps = ["test", "quality", "security", "docs"]
        missing_deps = [dep for dep in required_deps if dep not in validate_needs]
        if missing_deps:
            print(f"❌ Validate job missing dependencies: {missing_deps}")
            return False

        codeql_job = jobs.get("codeql", {})
        codeql_needs = codeql_job.get("needs", [])
        if "validate" not in codeql_needs:
            print("❌ CodeQL job should depend on validate")
            return False

        print(f"✅ Found {len(jobs)} jobs with proper dependencies")
        return True

        codeql_job = jobs.get("codeql", {})
        codeql_needs = codeql_job.get("needs", [])
        if "validate" not in codeql_needs:
            print("❌ CodeQL job should depend on validate")
            return False

        if "jobs" not in workflow:
            print("❌ Missing jobs section")
            return False

        jobs = workflow["jobs"]
        expected_jobs = ["test", "quality", "security", "docs", "validate", "codeql"]
        missing_jobs = [job for job in expected_jobs if job not in jobs]
        if missing_jobs:
            print(f"❌ Missing jobs: {missing_jobs}")
            return False

        # Check for triggers (may be stored as boolean True due to YAML parsing quirk)
        on_section = None
        if "on" in workflow:
            on_section = workflow["on"]
        elif True in workflow:  # YAML parser quirk with 'on' key
            on_section = workflow[True]

        if not on_section:
            print("❌ No trigger section ('on') found")
            return False

        # Check for push or pull_request triggers
        has_push = "push" in on_section
        has_pr = "pull_request" in on_section
        if not (has_push or has_pr):
            print("❌ No push or pull_request triggers found")
            return False

        # Validate 'on' section has proper triggers
        on_section = workflow.get("on", {})
        if not on_section:
            print("❌ 'on' section is empty")
            return False

        # Check for push or pull_request triggers
        has_push = "push" in on_section
        has_pr = "pull_request" in on_section
        if not (has_push or has_pr):
            print("❌ No push or pull_request triggers found")
            return False

        # Validate jobs
        jobs = workflow.get("jobs", {})
        required_jobs = ["test", "quality", "security", "docs", "validate", "codeql"]

        for job in required_jobs:
            if job not in jobs:
                print(f"❌ Missing required job: {job}")
                return False

        # Validate job dependencies
        validate_job = jobs.get("validate", {})
        needs = validate_job.get("needs", [])
        expected_needs = ["test", "quality", "security", "docs"]
        if not all(need in needs for need in expected_needs):
            print(
                f"❌ Validate job missing dependencies. Expected: {expected_needs}, Got: {needs}"
            )
            return False

        codeql_job = jobs.get("codeql", {})
        codeql_needs = codeql_job.get("needs", [])
        if "validate" not in codeql_needs:
            print("❌ CodeQL job should depend on validate job")
            return False

        # Validate permissions
        permissions = workflow.get("permissions", {})
        required_permissions = ["contents", "security-events"]
        for perm in required_permissions:
            if perm not in permissions:
                print(f"❌ Missing required permission: {perm}")
                return False

        print("✅ Workflow validation passed")
        print(f"📊 Found {len(jobs)} jobs: {list(jobs.keys())}")
        return True

    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    success = validate_workflow()
    exit(0 if success else 1)
