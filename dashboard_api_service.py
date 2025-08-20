# Compatibility shim for legacy imports used by tests
# Re-exports the production service from services.dashboard_api_service

from services.dashboard_api_service import *  # noqa: F401,F403