from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from benchmarking.config.pydantic_config import UnifiedAgentRunnerConfig

class AgentConfigurationResponse(BaseModel):
    agent_framework: str
    agent_type: str
    description: str
    example_config: Dict[str, Any] | None = None

class AgentValidationRequest(BaseModel):
    agent_config: UnifiedAgentRunnerConfig

class AgentValidationResponse(BaseModel):
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class FrameworksResponse(BaseModel):
    frameworks: List[str]