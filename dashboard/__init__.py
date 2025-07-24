"""
FBA-Bench Dashboard Package

Provides real-time dashboard capabilities for FBA-Bench simulation analysis.
"""

from .models import DashboardState, ExecutiveSummary, FinancialDeepDive
from .data_exporter import DashboardDataExporter
from .api import run_dashboard_server, dashboard_api

__version__ = "1.0.0"
__all__ = [
    "DashboardState",
    "ExecutiveSummary", 
    "FinancialDeepDive",
    "DashboardDataExporter",
    "run_dashboard_server",
    "dashboard_api"
]