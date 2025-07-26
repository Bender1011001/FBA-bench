"""
FBA-Bench Dashboard Package

Provides real-time dashboard capabilities for FBA-Bench simulation analysis.
"""

from .models import DashboardState, ExecutiveSummary, FinancialDeepDive
from .api import run_dashboard_server, dashboard_api
from .secure_api import secure_data_provider, security_manager

__version__ = "1.0.0"
__all__ = [
    "DashboardState",
    "ExecutiveSummary",
    "FinancialDeepDive",
    "run_dashboard_server",
    "dashboard_api",
    "secure_data_provider",
    "security_manager"
]