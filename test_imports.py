#!/usr/bin/env python3
"""
Test script to identify all import errors in the benchmarking module.
"""

try:
    from benchmarking.core.engine import BenchmarkEngine
    print("✓ Successfully imported BenchmarkEngine")
except ImportError as e:
    print(f"✗ Import error: {e}")

try:
    from benchmarking import BenchmarkEngine
    print("✓ Successfully imported from benchmarking")
except ImportError as e:
    print(f"✗ Import error: {e}")