"""
FBA-Bench core package metadata.
Centralized version and build metadata for API and tooling.
"""

from __future__ import annotations

import os
from typing import Dict

__all__ = ["__version__", "get_build_metadata"]

# Semantic version for this release
__version__ = "3.0.0"


def get_build_metadata() -> Dict[str, str]:
    """
    Return build metadata for diagnostics and health endpoints.

    Keys:
      - version: semantic version string
      - git_sha: short or full commit SHA if provided at build time
      - build_time: ISO-8601 or RFC3339 build timestamp if provided at build time

    Values fall back to "unknown" when not set to avoid leaking build system details.
    """
    return {
        "version": __version__,
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "build_time": os.getenv("BUILD_TIME", "unknown"),
    }