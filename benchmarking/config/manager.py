"""
Configuration management for benchmarking.

This module provides tools for loading, validating, and managing benchmarking
configurations with support for environment-specific settings and validation.
"""

import os
import logging
import yaml
import json
from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from .schema import (
    ConfigurationSchema, 
    ValidationResult, 
    SchemaRegistry,
    schema_registry
)

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationProfile:
    """Configuration profile for different environments."""
    name: str
    description: str
    environment: str
    config: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "environment": self.environment,
            "config": self.config,
            "overrides": self.overrides,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigurationProfile':
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)


class ConfigurationManager:
    """
    Configuration manager for benchmarking.
    
    This class provides tools for loading, validating, and managing
    benchmarking configurations with support for environment-specific settings.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.config_dir / "profiles").mkdir(exist_ok=True)
        (self.config_dir / "templates").mkdir(exist_ok=True)
        (self.config_dir / "environments").mkdir(exist_ok=True)
        
        # Schema registry
        self.schema_registry = schema_registry
        
        # Active configurations
        self._active_configs: Dict[str, Dict[str, Any]] = {}
        self._profiles: Dict[str, ConfigurationProfile] = {}
        
        # Environment variables
        self._env_prefix = "FBA_BENCH_"
        
        logger.info(f"Initialized ConfigurationManager with config directory: {self.config_dir}")
    
    def load_config(self, config_path: str, config_type: str = "benchmark") -> Dict[str, Any]:
        """
        Load a configuration from file.
        
        Args:
            config_path: Path to configuration file
            config_type: Type of configuration (for validation)
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            # Load configuration based on file extension
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
            
            # Validate configuration
            validation_result = self.schema_registry.validate(config_type, config)
            if not validation_result.is_valid:
                error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))
            
            # Apply environment overrides
            config = self._apply_environment_overrides(config)
            
            # Store as active configuration
            config_id = config.get("benchmark_id", config_file.stem)
            self._active_configs[config_id] = config
            
            logger.info(f"Loaded configuration from {config_path} with ID: {config_id}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], config_path: str, config_type: str = "benchmark") -> None:
        """
        Save a configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration file
            config_type: Type of configuration (for validation)
        """
        # Validate configuration before saving
        validation_result = self.schema_registry.validate(config_type, config)
        if not validation_result.is_valid:
            error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Add metadata
            if "metadata" not in config:
                config["metadata"] = {}
            
            config["metadata"]["last_updated"] = datetime.now().isoformat()
            config["metadata"]["saved_by"] = "ConfigurationManager"
            
            # Save configuration based on file extension
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def create_profile(
        self, 
        name: str, 
        environment: str, 
        description: str = "",
        base_config: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ConfigurationProfile:
        """
        Create a configuration profile.
        
        Args:
            name: Name of the profile
            environment: Environment (development, testing, production)
            description: Description of the profile
            base_config: Base configuration
            overrides: Configuration overrides
            
        Returns:
            Created configuration profile
        """
        profile = ConfigurationProfile(
            name=name,
            environment=environment,
            description=description,
            config=base_config or {},
            overrides=overrides or {}
        )
        
        self._profiles[name] = profile
        
        # Save profile to disk
        self._save_profile(profile)
        
        logger.info(f"Created configuration profile: {name} for environment: {environment}")
        return profile
    
    def load_profile(self, name: str) -> Optional[ConfigurationProfile]:
        """
        Load a configuration profile.
        
        Args:
            name: Name of the profile
            
        Returns:
            Loaded profile or None if not found
        """
        # Check cache first
        if name in self._profiles:
            return self._profiles[name]
        
        # Try to load from disk
        profile_file = self.config_dir / "profiles" / f"{name}.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                
                profile = ConfigurationProfile.from_dict(data)
                self._profiles[name] = profile
                
                logger.info(f"Loaded configuration profile: {name}")
                return profile
                
            except Exception as e:
                logger.error(f"Failed to load profile {name}: {e}")
                return None
        
        return None
    
    def get_config_from_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Get configuration from a profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            ValueError: If profile is not found
        """
        profile = self.load_profile(profile_name)
        if profile is None:
            raise ValueError(f"Profile not found: {profile_name}")
        
        # Start with base config
        config = profile.config.copy()
        
        # Apply overrides
        config = self._deep_merge(config, profile.overrides)
        
        # Apply environment overrides
        config = self._apply_environment_overrides(config)
        
        logger.info(f"Generated configuration from profile: {profile_name}")
        return config
    
    def list_profiles(self) -> List[str]:
        """
        List all available configuration profiles.
        
        Returns:
            List of profile names
        """
        # Update from disk
        profiles_dir = self.config_dir / "profiles"
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.json"):
                name = profile_file.stem
                if name not in self._profiles:
                    self.load_profile(name)
        
        return list(self._profiles.keys())
    
    def delete_profile(self, name: str) -> bool:
        """
        Delete a configuration profile.
        
        Args:
            name: Name of the profile
            
        Returns:
            True if profile was deleted, False if not found
        """
        if name in self._profiles:
            del self._profiles[name]
        
        profile_file = self.config_dir / "profiles" / f"{name}.json"
        if profile_file.exists():
            try:
                profile_file.unlink()
                logger.info(f"Deleted configuration profile: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete profile {name}: {e}")
                return False
        
        return False
    
    def create_template(self, config_type: str, template_name: str, template_config: Dict[str, Any]) -> None:
        """
        Create a configuration template.
        
        Args:
            config_type: Type of configuration
            template_name: Name of the template
            template_config: Template configuration
        """
        # Validate template configuration
        validation_result = self.schema_registry.validate(config_type, template_config)
        if not validation_result.is_valid:
            error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
            raise ValueError(f"Template validation failed:\n" + "\n".join(error_messages))
        
        template_file = self.config_dir / "templates" / f"{config_type}_{template_name}.yaml"
        template_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(template_file, 'w') as f:
                yaml.dump(template_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created template: {template_name} for type: {config_type}")
            
        except Exception as e:
            logger.error(f"Failed to create template {template_name}: {e}")
            raise
    
    def list_templates(self, config_type: Optional[str] = None) -> List[str]:
        """
        List available configuration templates.
        
        Args:
            config_type: Filter by configuration type
            
        Returns:
            List of template names
        """
        templates_dir = self.config_dir / "templates"
        if not templates_dir.exists():
            return []
        
        templates = []
        for template_file in templates_dir.glob("*.yaml"):
            if config_type is None:
                templates.append(template_file.stem)
            else:
                prefix = f"{config_type}_"
                if template_file.stem.startswith(prefix):
                    templates.append(template_file.stem[len(prefix):])
        
        return templates
    
    def load_template(self, config_type: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration template.
        
        Args:
            config_type: Type of configuration
            template_name: Name of the template
            
        Returns:
            Template configuration or None if not found
        """
        template_file = self.config_dir / "templates" / f"{config_type}_{template_name}.yaml"
        if template_file.exists():
            try:
                with open(template_file, 'r') as f:
                    template = yaml.safe_load(f)
                
                logger.info(f"Loaded template: {template_name} for type: {config_type}")
                return template
                
            except Exception as e:
                logger.error(f"Failed to load template {template_name}: {e}")
                return None
        
        return None
    
    def validate_config(self, config: Dict[str, Any], config_type: str = "benchmark") -> ValidationResult:
        """
        Validate a configuration.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration
            
        Returns:
            ValidationResult with validation results
        """
        return self.schema_registry.validate(config_type, config)
    
    def get_active_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an active configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            Configuration dictionary or None if not found
        """
        return self._active_configs.get(config_id)
    
    def list_active_configs(self) -> List[str]:
        """
        List all active configuration IDs.
        
        Returns:
            List of configuration IDs
        """
        return list(self._active_configs.keys())
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Base configuration
            
        Returns:
            Configuration with environment overrides applied
        """
        result = config.copy()
        
        # Get environment variables with prefix
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(self._env_prefix)}
        
        for env_var, env_value in env_vars:
            # Convert environment variable name to configuration path
            config_path = env_var[len(self._env_prefix):].lower().split('_')
            
            # Navigate to the target location in the configuration
            current = result
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value with type conversion
            final_key = config_path[-1]
            if final_key in current:
                # Try to maintain the original type
                original_value = current[final_key]
                if isinstance(original_value, bool):
                    current[final_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(original_value, int):
                    current[final_key] = int(env_value)
                elif isinstance(original_value, float):
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value
            else:
                # Default to string
                current[final_key] = env_value
        
        return result
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _save_profile(self, profile: ConfigurationProfile) -> None:
        """
        Save a configuration profile to disk.
        
        Args:
            profile: Profile to save
        """
        profile_file = self.config_dir / "profiles" / f"{profile.name}.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save profile {profile.name}: {e}")
            raise
    
    def create_environment_config(self, environment: str, config: Dict[str, Any]) -> None:
        """
        Create environment-specific configuration.
        
        Args:
            environment: Environment name
            config: Environment configuration
        """
        env_file = self.config_dir / "environments" / f"{environment}.yaml"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(env_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created environment configuration: {environment}")
            
        except Exception as e:
            logger.error(f"Failed to create environment config {environment}: {e}")
            raise
    
    def load_environment_config(self, environment: str) -> Optional[Dict[str, Any]]:
        """
        Load environment-specific configuration.
        
        Args:
            environment: Environment name
            
        Returns:
            Environment configuration or None if not found
        """
        env_file = self.config_dir / "environments" / f"{environment}.yaml"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                logger.info(f"Loaded environment configuration: {environment}")
                return config
                
            except Exception as e:
                logger.error(f"Failed to load environment config {environment}: {e}")
                return None
        
        return None
    
    def list_environments(self) -> List[str]:
        """
        List available environment configurations.
        
        Returns:
            List of environment names
        """
        env_dir = self.config_dir / "environments"
        if not env_dir.exists():
            return []
        
        return [f.stem for f in env_dir.glob("*.yaml")]


# Global configuration manager instance
config_manager = ConfigurationManager()