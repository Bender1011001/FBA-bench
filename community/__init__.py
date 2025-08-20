from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Reuse PluginType from plugin framework when available
try:
    from plugins.plugin_framework import PluginType  # type: ignore
except Exception:  # pragma: no cover
    class PluginType(Enum):  # Fallback minimal
        SKILL = "skill"
        ANALYSIS = "analysis"
        INTEGRATION = "integration"
        SCENARIO = "scenario"
        AGENT = "agent"
        TOOL = "tool"
        METRIC = "metric"


logger = logging.getLogger(__name__)


# -------- contribution_validator --------

class ContributionType(Enum):
    PLUGIN = "plugin"
    DOCUMENTATION = "documentation"
    BUG_FIX = "bug_fix"


class ReviewCriteria(Enum):
    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    FIX_CORRECTNESS = "fix_correctness"
    IMPACT_ANALYSIS = "impact_analysis"


@dataclass
class _ValidationResult:
    is_valid: bool = True
    details: Dict[str, Any] = None


class ContributionValidator:
    """
    Deterministic validator for contribution workflows.
    """

    async def validate_contribution(
        self,
        contribution_type: ContributionType,
        files_modified: List[str],
        test_requirements: List[str],
    ) -> _ValidationResult:
        # Minimal deterministic checks: basic presence/shape
        ok = isinstance(contribution_type, ContributionType) and bool(files_modified) and bool(test_requirements)
        return _ValidationResult(is_valid=ok, details={
            "type": contribution_type.value if isinstance(contribution_type, ContributionType) else str(contribution_type),
            "files": list(files_modified),
            "tests": list(test_requirements),
        })

    async def check_review_criteria(
        self,
        expected_criteria: List[str] | List[ReviewCriteria],
        validation_result: _ValidationResult,
    ) -> bool:
        # Accept when validation passed and expected criteria list is non-empty
        return bool(validation_result and validation_result.is_valid and expected_criteria)


# -------- community_tools --------

class DocumentationGenerator:
    async def generate(self) -> str:
        return "# Contribution Guide\n\nFollow the steps to contribute your plugin."


class ExampleValidator:
    async def validate_example(self, path: str) -> Any:
        # Minimal: file path must be a non-empty string
        return types.SimpleNamespace(is_valid=bool(path and isinstance(path, str)))


class CommunityTools:
    async def generate_contribution_guide(self) -> str:
        return await DocumentationGenerator().generate()

    async def generate_plugin_template(self, name: str, plugin_type: PluginType) -> Dict[str, Any]:
        return {
            "name": name,
            "type": plugin_type.value if isinstance(plugin_type, PluginType) else str(plugin_type),
            "files": ["plugin.py", "README.md", "config.yaml", "tests/test_plugin.py"],
        }


# -------- plugin_registry --------

@dataclass
class PluginMetadata:
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    homepage: str
    repository: str


@dataclass
class RegistryEntry:
    metadata: PluginMetadata
    verification_status: str
    download_url: str
    checksums: Dict[str, str]


class PluginRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}

    async def register_plugin(self, entry: RegistryEntry) -> bool:
        key = f"{entry.metadata.name}@{entry.metadata.version}"
        self._entries[key] = entry
        return True

    async def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None) -> List[RegistryEntry]:
        results = []
        for e in self._entries.values():
            if query.lower() in e.metadata.name.lower():
                if plugin_type is None or e.metadata.plugin_type == plugin_type:
                    results.append(e)
        return results

    async def validate_metadata(self, metadata: Dict[str, Any]) -> Any:
        required = ["name", "version", "description", "author"]
        ok = isinstance(metadata, dict) and all(k in metadata for k in required)
        return types.SimpleNamespace(is_valid=ok)


# -------- version_manager --------

@dataclass(frozen=True)
class SemanticVersion:
    raw: str

    @property
    def tuple(self) -> Tuple[int, int, int]:
        parts = self.raw.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        return (major, minor, patch)


@dataclass
class CompatibilityMatrix:
    # Very light abstraction; in real use could hold mapping between versions.
    supported_ranges: Dict[str, str] = None


class VersionManager:
    def compare(self, a: SemanticVersion, b: SemanticVersion) -> int:
        return (a.tuple > b.tuple) - (a.tuple < b.tuple)

    def is_compatible(self, provided: SemanticVersion, required_min: SemanticVersion) -> bool:
        return provided.tuple >= required_min.tuple


# ---------- Export submodules for import compatibility ----------

# Create dynamic submodules so imports like "from community.contribution_validator import X" work.
_contribution_validator = types.ModuleType("community.contribution_validator")
_contribution_validator.ContributionValidator = ContributionValidator
_contribution_validator.ContributionType = ContributionType
_contribution_validator.ReviewCriteria = ReviewCriteria
sys.modules["community.contribution_validator"] = _contribution_validator

_community_tools = types.ModuleType("community.community_tools")
_community_tools.CommunityTools = CommunityTools
_community_tools.DocumentationGenerator = DocumentationGenerator
_community_tools.ExampleValidator = ExampleValidator
sys.modules["community.community_tools"] = _community_tools

_plugin_registry = types.ModuleType("community.plugin_registry")
_plugin_registry.PluginRegistry = PluginRegistry
_plugin_registry.RegistryEntry = RegistryEntry
_plugin_registry.PluginMetadata = PluginMetadata
sys.modules["community.plugin_registry"] = _plugin_registry

_version_manager = types.ModuleType("community.version_manager")
_version_manager.VersionManager = VersionManager
_version_manager.SemanticVersion = SemanticVersion
_version_manager.CompatibilityMatrix = CompatibilityMatrix
sys.modules["community.version_manager"] = _version_manager

# Also make them accessible as attributes of community package
ContributionValidator = ContributionValidator
ContributionType = ContributionType
ReviewCriteria = ReviewCriteria
CommunityTools = CommunityTools
DocumentationGenerator = DocumentationGenerator
ExampleValidator = ExampleValidator
PluginRegistry = PluginRegistry
RegistryEntry = RegistryEntry
PluginMetadata = PluginMetadata
VersionManager = VersionManager
SemanticVersion = SemanticVersion
CompatibilityMatrix = CompatibilityMatrix