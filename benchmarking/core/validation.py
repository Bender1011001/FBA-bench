"""
Validation-related data structures and exceptions for the FBA-Bench framework.

This module defines classes for representing validation outcomes and specific
exceptions for schema validation failures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ValidationError:
    """Represents a single validation error encountered."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    field_path: Optional[str] = None

@dataclass
class ValidationResult:
    """
    Encapsulates the outcome of a validation operation.
    """
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, code: str, message: str, details: Optional[Dict[str, Any]] = None, field_path: Optional[str] = None):
        """Adds an error to the validation result."""
        self.is_valid = False
        self.errors.append(ValidationError(code, message, details, field_path))

    def add_warning(self, code: str, message: str, details: Optional[Dict[str, Any]] = None, field_path: Optional[str] = None):
        """Adds a warning to the validation result."""
        self.warnings.append(ValidationError(code, message, details, field_path))

    def to_dict(self) -> Dict[str, Any]:
        """Converts the validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [err.__dict__ for err in self.errors],
            "warnings": [warn.__dict__ for warn in self.warnings],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Creates a ValidationResult from a dictionary."""
        errors = [ValidationError(**err) for err in data.get("errors", [])]
        warnings = [ValidationError(**warn) for warn in data.get("warnings", [])]
        return cls(
            is_valid=data.get("is_valid", False),
            errors=errors,
            warnings=warnings,
            metadata=data.get("metadata", {}),
        )

class SchemaValidationError(ValueError):
    """
    Custom exception for errors encountered during schema validation.
    """
    def __init__(self, message: str, schema_path: Optional[str] = None, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.schema_path = schema_path
        self.errors = errors if errors is not None else []

    def __str__(self):
        err_str = super().__str__()
        if self.schema_path:
            err_str = f"{err_str} in schema '{self.schema_path}'"
        if self.errors:
            err_str = f"{err_str}. Details: {self.errors}"
        return err_str