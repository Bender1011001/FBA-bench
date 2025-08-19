"""
Validators framework package.

Exports:
- Function-style registry helpers: register_validator, get_validator, list_validators
- Auto-imports built-in validators to trigger registration on package import
"""

from .registry import register_validator, get_validator, list_validators  # function-style helpers

# Import legacy utilities for backward-compat public API
from .deterministic import DeterministicEnvironment
from .version_control import VersionControlManager
from .statistical_validator import StatisticalValidator
from .audit_trail import AuditTrailManager
from .reproducibility_validator import ReproducibilityValidator

# Auto-import built-in function-style validators (each module calls register_validator on import)
from . import structural_consistency  # noqa: F401
from . import determinism_check  # noqa: F401
from . import reproducibility_metadata  # noqa: F401
# The following will be added and auto-registered:
# - schema_adherence
# - outlier_detection
# - fairness_balance
try:
    from . import schema_adherence  # noqa: F401
    from . import outlier_detection  # noqa: F401
    from . import fairness_balance  # noqa: F401
except Exception:
    # Optional during partial installs; tests that need them will import directly
    pass

__all__ = [
    "register_validator",
    "get_validator",
    "list_validators",
    "DeterministicEnvironment",
    "VersionControlManager",
    "StatisticalValidator",
    "AuditTrailManager",
    "ReproducibilityValidator",
]