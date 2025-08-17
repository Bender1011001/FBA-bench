#!/usr/bin/env python3
"""
Test Runner Script - Executes the system integration tests.

This script provides a convenient way to run the system integration tests
for the FBA-Bench project.
"""

import argparse
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, cwd=None):
    """
    Run a command and return the result.
    
    Args:
        command: Command to run
        cwd: Working directory
        
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    logger.info(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")
        
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return 1, "", str(e)


def install_dependencies():
    """
    Install the required dependencies for testing.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Installing test dependencies...")
    
    # Install pytest and related packages
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-mock"
    ])
    
    if returncode != 0:
        logger.error("Failed to install test dependencies")
        return False
    
    logger.info("Test dependencies installed successfully")
    return True


def run_unit_tests():
    """
    Run the unit tests.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running unit tests...")
    
    # Find the project root
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests"
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return False
    
    # Run pytest
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "pytest", str(test_dir), "-v"
    ], cwd=project_root)
    
    if returncode != 0:
        logger.error("Unit tests failed")
        return False
    
    logger.info("Unit tests passed successfully")
    return True


def run_integration_tests():
    """
    Run the integration tests.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running integration tests...")
    
    # Find the project root
    project_root = Path(__file__).parent.parent
    test_file = project_root / "tests" / "test_system_integration.py"
    
    if not test_file.exists():
        logger.error(f"Integration test file not found: {test_file}")
        return False
    
    # Run pytest on the integration test file
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "pytest", str(test_file), "-v"
    ], cwd=project_root)
    
    if returncode != 0:
        logger.error("Integration tests failed")
        return False
    
    logger.info("Integration tests passed successfully")
    return True


def run_linting():
    """
    Run code linting checks.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running code linting checks...")
    
    # Find the project root
    project_root = Path(__file__).parent.parent
    
    # Try to run flake8 if available
    try:
        returncode, stdout, stderr = run_command([
            sys.executable, "-m", "flake8", "."
        ], cwd=project_root)
        
        if returncode != 0:
            logger.warning("Linting issues found with flake8")
            # Don't fail the build for linting issues
    except Exception as e:
        logger.warning(f"Could not run flake8: {e}")
    
    # Try to run black if available
    try:
        returncode, stdout, stderr = run_command([
            sys.executable, "-m", "black", "--check", "."
        ], cwd=project_root)
        
        if returncode != 0:
            logger.warning("Code formatting issues found with black")
            # Don't fail the build for formatting issues
    except Exception as e:
        logger.warning(f"Could not run black: {e}")
    
    logger.info("Code linting checks completed")
    return True


def run_type_checking():
    """
    Run type checking.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running type checking...")
    
    # Find the project root
    project_root = Path(__file__).parent.parent
    
    # Try to run mypy if available
    try:
        returncode, stdout, stderr = run_command([
            sys.executable, "-m", "mypy", "."
        ], cwd=project_root)
        
        if returncode != 0:
            logger.warning("Type checking issues found with mypy")
            # Don't fail the build for type checking issues
    except Exception as e:
        logger.warning(f"Could not run mypy: {e}")
    
    logger.info("Type checking completed")
    return True


def main():
    """
    Main function to run all tests.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Run FBA-Bench tests")
    parser.add_argument(
        "--unit-only", action="store_true", help="Run only unit tests"
    )
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--lint-only", action="store_true", help="Run only linting checks"
    )
    parser.add_argument(
        "--type-only", action="store_true", help="Run only type checking"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip installing dependencies"
    )
    parser.add_argument(
        "--skip-lint", action="store_true", help="Skip linting checks"
    )
    parser.add_argument(
        "--skip-type", action="store_true", help="Skip type checking"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if needed
    if not args.skip_deps:
        if not install_dependencies():
            return 1
    
    success = True
    
    # Run tests based on arguments
    if args.unit_only:
        if not run_unit_tests():
            success = False
    elif args.integration_only:
        if not run_integration_tests():
            success = False
    elif args.lint_only:
        if not run_linting():
            success = False
    elif args.type_only:
        if not run_type_checking():
            success = False
    else:
        # Run all tests
        if not run_unit_tests():
            success = False
        
        if not run_integration_tests():
            success = False
        
        if not args.skip_lint:
            if not run_linting():
                success = False
        
        if not args.skip_type:
            if not run_type_checking():
                success = False
    
    if success:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())