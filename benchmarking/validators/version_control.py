"""
Version control management for benchmarking components.

This module provides tools for tracking versions of models, datasets, code,
and other components to ensure reproducibility of benchmark results.
"""

import os
import json
import hashlib
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)


@dataclass
class ComponentVersion:
    """Version information for a component."""
    name: str
    version: str
    type: str  # model, dataset, code, config, etc.
    path: str
    hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentVersion':
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class VersionManifest:
    """Manifest of all component versions for a benchmark run."""
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    components: List[ComponentVersion] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    git_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "components": [comp.to_dict() for comp in self.components],
            "environment": self.environment,
            "git_info": self.git_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionManifest':
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        if "components" in data:
            data["components"] = [
                ComponentVersion.from_dict(comp) 
                for comp in data["components"]
            ]
        
        return cls(**data)


class VersionControlManager:
    """
    Manages version control for benchmarking components.
    
    This class provides tools for tracking versions of all components
    used in benchmark runs to ensure reproducibility.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the version control manager.
        
        Args:
            storage_path: Path to store version manifests
        """
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / "version_manifests"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._current_manifest = None
        self._component_cache = {}
        
        logger.info(f"Initialized VersionControlManager with storage at: {self.storage_path}")
    
    def create_manifest(self, run_id: str) -> VersionManifest:
        """
        Create a new version manifest.
        
        Args:
            run_id: Unique identifier for the benchmark run
            
        Returns:
            New version manifest
        """
        self._current_manifest = VersionManifest(run_id=run_id)
        
        # Capture environment information
        self._current_manifest.environment = self._capture_environment_info()
        
        # Capture git information
        self._current_manifest.git_info = self._capture_git_info()
        
        logger.info(f"Created version manifest for run: {run_id}")
        return self._current_manifest
    
    def get_current_manifest(self) -> Optional[VersionManifest]:
        """Get the current manifest."""
        return self._current_manifest
    
    def add_component(
        self, 
        name: str, 
        component_type: str, 
        path: str, 
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ComponentVersion:
        """
        Add a component to the current manifest.
        
        Args:
            name: Name of the component
            component_type: Type of the component
            path: Path to the component
            version: Version string (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Component version information
        """
        if self._current_manifest is None:
            raise RuntimeError("No active manifest. Call create_manifest() first.")
        
        # Calculate hash if not cached
        cache_key = f"{path}:{version}"
        if cache_key not in self._component_cache:
            self._component_cache[cache_key] = self._calculate_component_hash(path)
        
        # Create component version
        component = ComponentVersion(
            name=name,
            version=version or "unknown",
            type=component_type,
            path=path,
            hash=self._component_cache[cache_key],
            metadata=metadata or {}
        )
        
        # Add to manifest
        self._current_manifest.components.append(component)
        
        logger.debug(f"Added component to manifest: {name} ({component_type})")
        return component
    
    def add_python_module(self, module_name: str, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersion:
        """
        Add a Python module to the manifest.
        
        Args:
            module_name: Name of the Python module
            metadata: Additional metadata (optional)
            
        Returns:
            Component version information
        """
        try:
            # Get module information
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                raise ImportError(f"Module not found: {module_name}")
            
            module_path = spec.origin
            if module_path is None:
                raise ValueError(f"Cannot determine path for module: {module_name}")
            
            # Get version if available
            version = "unknown"
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "__version__"):
                    version = module.__version__
            except ImportError:
                pass
            
            # Add module metadata
            module_metadata = {
                "module_name": module_name,
                "spec_name": spec.name,
                "submodule_search_locations": spec.submodule_search_locations
            }
            if metadata:
                module_metadata.update(metadata)
            
            return self.add_component(
                name=module_name,
                component_type="python_module",
                path=module_path,
                version=version,
                metadata=module_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to add Python module {module_name}: {e}")
            raise
    
    def add_model(self, model_path: str, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersion:
        """
        Add a model to the manifest.
        
        Args:
            model_path: Path to the model
            model_name: Name of the model
            metadata: Additional metadata (optional)
            
        Returns:
            Component version information
        """
        return self.add_component(
            name=model_name,
            component_type="model",
            path=model_path,
            metadata=metadata
        )
    
    def add_dataset(self, dataset_path: str, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersion:
        """
        Add a dataset to the manifest.
        
        Args:
            dataset_path: Path to the dataset
            dataset_name: Name of the dataset
            metadata: Additional metadata (optional)
            
        Returns:
            Component version information
        """
        return self.add_component(
            name=dataset_name,
            component_type="dataset",
            path=dataset_path,
            metadata=metadata
        )
    
    def add_configuration(self, config_path: str, config_name: str, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersion:
        """
        Add a configuration file to the manifest.
        
        Args:
            config_path: Path to the configuration file
            config_name: Name of the configuration
            metadata: Additional metadata (optional)
            
        Returns:
            Component version information
        """
        return self.add_component(
            name=config_name,
            component_type="config",
            path=config_path,
            metadata=metadata
        )
    
    def save_manifest(self, filename: Optional[str] = None) -> str:
        """
        Save the current manifest to disk.
        
        Args:
            filename: Filename to save (optional)
            
        Returns:
            Path to the saved manifest
        """
        if self._current_manifest is None:
            raise RuntimeError("No active manifest to save.")
        
        if filename is None:
            filename = f"manifest_{self._current_manifest.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        manifest_path = self.storage_path / filename
        
        with open(manifest_path, 'w') as f:
            json.dump(self._current_manifest.to_dict(), f, indent=2)
        
        logger.info(f"Saved manifest to: {manifest_path}")
        return str(manifest_path)
    
    def load_manifest(self, filename: str) -> VersionManifest:
        """
        Load a manifest from disk.
        
        Args:
            filename: Filename of the manifest
            
        Returns:
            Loaded version manifest
        """
        manifest_path = self.storage_path / filename
        
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        manifest = VersionManifest.from_dict(data)
        self._current_manifest = manifest
        
        logger.info(f"Loaded manifest from: {manifest_path}")
        return manifest
    
    def list_manifests(self) -> List[str]:
        """
        List all available manifests.
        
        Returns:
            List of manifest filenames
        """
        return [f.name for f in self.storage_path.glob("*.json")]
    
    def compare_manifests(self, manifest1: VersionManifest, manifest2: VersionManifest) -> Dict[str, Any]:
        """
        Compare two manifests and identify differences.
        
        Args:
            manifest1: First manifest
            manifest2: Second manifest
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "run_ids": [manifest1.run_id, manifest2.run_id],
            "timestamps": [manifest1.timestamp.isoformat(), manifest2.timestamp.isoformat()],
            "differences": {
                "components": {
                    "added": [],
                    "removed": [],
                    "changed": [],
                    "unchanged": []
                },
                "environment": {},
                "git_info": {}
            }
        }
        
        # Compare components
        components1 = {comp.name: comp for comp in manifest1.components}
        components2 = {comp.name: comp for comp in manifest2.components}
        
        all_names = set(components1.keys()) | set(components2.keys())
        
        for name in all_names:
            if name in components1 and name in components2:
                if components1[name].hash == components2[name].hash:
                    comparison["differences"]["components"]["unchanged"].append(name)
                else:
                    comparison["differences"]["components"]["changed"].append({
                        "name": name,
                        "old_version": components1[name].version,
                        "new_version": components2[name].version,
                        "old_hash": components1[name].hash,
                        "new_hash": components2[name].hash
                    })
            elif name in components1:
                comparison["differences"]["components"]["removed"].append(name)
            else:
                comparison["differences"]["components"]["added"].append(name)
        
        # Compare environment
        env1 = manifest1.environment
        env2 = manifest2.environment
        
        all_env_keys = set(env1.keys()) | set(env2.keys())
        for key in all_env_keys:
            if key in env1 and key in env2:
                if env1[key] != env2[key]:
                    comparison["differences"]["environment"][key] = {
                        "old": env1[key],
                        "new": env2[key]
                    }
            elif key in env1:
                comparison["differences"]["environment"][key] = {"old": env1[key], "new": None}
            else:
                comparison["differences"]["environment"][key] = {"old": None, "new": env2[key]}
        
        # Compare git info
        git1 = manifest1.git_info
        git2 = manifest2.git_info
        
        all_git_keys = set(git1.keys()) | set(git2.keys())
        for key in all_git_keys:
            if key in git1 and key in git2:
                if git1[key] != git2[key]:
                    comparison["differences"]["git_info"][key] = {
                        "old": git1[key],
                        "new": git2[key]
                    }
            elif key in git1:
                comparison["differences"]["git_info"][key] = {"old": git1[key], "new": None}
            else:
                comparison["differences"]["git_info"][key] = {"old": None, "new": git2[key]}
        
        return comparison
    
    def verify_reproducibility(self, reference_manifest: VersionManifest) -> Dict[str, Any]:
        """
        Verify if the current environment matches a reference manifest.
        
        Args:
            reference_manifest: Reference manifest to verify against
            
        Returns:
            Dictionary with verification results
        """
        verification = {
            "reference_run_id": reference_manifest.run_id,
            "timestamp": datetime.now().isoformat(),
            "results": {
                "components": {},
                "environment": {},
                "git_info": {}
            },
            "overall_reproducible": True
        }
        
        # Create current manifest for comparison
        current_manifest = self.create_manifest(f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Verify components
        for ref_component in reference_manifest.components:
            if os.path.exists(ref_component.path):
                current_hash = self._calculate_component_hash(ref_component.path)
                
                if current_hash == ref_component.hash:
                    verification["results"]["components"][ref_component.name] = {
                        "status": "reproducible",
                        "hash_match": True
                    }
                else:
                    verification["results"]["components"][ref_component.name] = {
                        "status": "modified",
                        "hash_match": False,
                        "reference_hash": ref_component.hash,
                        "current_hash": current_hash
                    }
                    verification["overall_reproducible"] = False
            else:
                verification["results"]["components"][ref_component.name] = {
                    "status": "missing",
                    "hash_match": False,
                    "error": "Component not found"
                }
                verification["overall_reproducible"] = False
        
        # Verify environment
        for key, value in reference_manifest.environment.items():
            current_value = current_manifest.environment.get(key)
            if current_value == value:
                verification["results"]["environment"][key] = {
                    "status": "reproducible",
                    "match": True
                }
            else:
                verification["results"]["environment"][key] = {
                    "status": "different",
                    "match": False,
                    "reference": value,
                    "current": current_value
                }
                verification["overall_reproducible"] = False
        
        # Verify git info
        for key, value in reference_manifest.git_info.items():
            current_value = current_manifest.git_info.get(key)
            if current_value == value:
                verification["results"]["git_info"][key] = {
                    "status": "reproducible",
                    "match": True
                }
            else:
                verification["results"]["git_info"][key] = {
                    "status": "different",
                    "match": False,
                    "reference": value,
                    "current": current_value
                }
                verification["overall_reproducible"] = False
        
        return verification
    
    def _calculate_component_hash(self, path: str) -> str:
        """Calculate hash of a component."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Component not found: {path}")
        
        hasher = hashlib.sha256()
        
        if os.path.isfile(path):
            # Hash file contents
            with open(path, 'rb') as f:
                hasher.update(f.read())
        elif os.path.isdir(path):
            # Hash directory contents recursively
            for root, dirs, files in os.walk(path):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
        else:
            raise ValueError(f"Unsupported path type: {path}")
        
        return hasher.hexdigest()
    
    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture environment information."""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "environment_variables": dict(os.environ)
        }
    
    def _capture_git_info(self) -> Dict[str, str]:
        """Capture git repository information."""
        git_info = {}
        
        try:
            # Get git root directory
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            git_info["git_root"] = git_root
            
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info["commit_hash"] = commit_hash
            
            # Get git branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info["branch"] = branch
            
            # Check if working directory is clean
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info["working_directory_clean"] = len(status) == 0
            
            # Get remote URL
            try:
                remote_url = subprocess.check_output(
                    ["git", "remote", "get-url", "origin"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                git_info["remote_url"] = remote_url
            except subprocess.CalledProcessError:
                git_info["remote_url"] = "unknown"
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Not in a git repository or git not available
            git_info["error"] = "Not in a git repository or git not available"
        
        return git_info