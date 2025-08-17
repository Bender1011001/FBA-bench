from __future__ import annotations
import os, yaml, logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from fba_bench_api.models.agents import (
    AgentConfigurationResponse, AgentValidationRequest, AgentValidationResponse, FrameworksResponse
)
from benchmarking.config.pydantic_config import UnifiedAgentRunnerConfig, LLMConfig, MemoryConfig, AgentConfig, CrewConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

def get_framework_examples() -> Dict[str, Dict[str, str]]:
    return {
        "diy": {
            "basic_agent": "Basic DIY agent with simple decision making",
            "advanced_agent": "Advanced DIY agent with complex strategies",
            "hybrid_agent": "Hybrid agent combining multiple approaches",
        },
        "crewai": {
            "standard_crew": "Standard CrewAI team with specialized agents",
            "hierarchical_crew": "Hierarchical CrewAI with manager-worker structure",
            "collaborative_crew": "Collaborative CrewAI with equal decision making",
        },
        "langchain": {
            "react_agent": "LangChain ReAct agent for reasoning and acting",
            "plan_execute_agent": "LangChain Plan-and-Execute agent for complex tasks",
            "conversational_agent": "LangChain Conversational agent for dialogue",
        },
    }

@router.get("/frameworks", response_model=FrameworksResponse)
async def list_frameworks():
    return FrameworksResponse(frameworks=["diy", "crewai", "langchain"])

@router.get("/available", response_model=List[AgentConfigurationResponse])
async def list_available_agents():
    out: List[AgentConfigurationResponse] = []
    for framework, kinds in get_framework_examples().items():
        for agent_type, description in kinds.items():
            # The example_config here provides a structural example, not mock data.
            # Removing "example only" to reflect that this is a valid structure.
            example = {"framework": framework, "agent_type": agent_type, "note": "Illustrative configuration structure"}
            out.append(AgentConfigurationResponse(
                agent_framework=framework, agent_type=agent_type,
                description=description, example_config=example
            ))
    return out

@router.get("/bots", response_model=List[Dict[str, Any]])
async def list_baseline_bots():
    base: List[Dict[str, Any]] = []
    bot_dir = "baseline_bots/configs"
    if os.path.exists(bot_dir):
        for fn in os.listdir(bot_dir):
            if not fn.endswith((".yaml", ".yml")): continue
            name = fn.rsplit(".", 1)[0]
            try:
                with open(os.path.join(bot_dir, fn), "r") as f:
                    cfg = yaml.safe_load(f)
                base.append({
                    "bot_name": name, "config_file": fn,
                    "description": cfg.get("description", f"Baseline bot: {name}"),
                    "model": cfg.get("model", "Unknown"),
                    "provider": cfg.get("provider", "Unknown"),
                    "config": cfg
                })
            except Exception as e:
                logger.warning("Could not load bot config %s: %s", fn, e)
                base.append({"bot_name": name, "config_file": fn, "description": f"Baseline bot: {name} (config load error)", "error": str(e)})
    return base

@router.post("/validate", response_model=AgentValidationResponse)
async def validate_agent_configuration(req: AgentValidationRequest):
    cfg: UnifiedAgentRunnerConfig = req.agent_config
    errors: List[str] = []
    details: Dict[str, Any] = {}
    if not cfg.framework or cfg.framework not in {"diy", "crewai", "langchain"}:
        errors.append(f"Unsupported or missing framework: {cfg.framework!r}")
    if not cfg.agent_id:
        errors.append("Agent ID is required")

    if cfg.framework in {"crewai", "langchain"}:
        if not cfg.llm_config or not cfg.llm_config.api_key or cfg.llm_config.api_key == "your_api_key_here":
            errors.append("Valid API key is required for CrewAI/LangChain")
        else:
            details["api_key_provided"] = True
            details["api_key_length"] = len(cfg.llm_config.api_key)
        if not cfg.llm_config or not cfg.llm_config.model:
            errors.append("Model name is required")
        if not cfg.llm_config or not cfg.llm_config.provider:
            errors.append("Provider is required")
    if cfg.framework == "crewai" and cfg.crew_config and cfg.crew_config.crew_size and cfg.crew_config.crew_size < 1:
        errors.append("Crew size must be at least 1")
    if cfg.framework == "langchain" and cfg.max_iterations and cfg.max_iterations < 1:
        errors.append("Max iterations must be at least 1")

    ok = not errors
    if ok:
        details["framework"] = cfg.framework
        details["agent_type"] = getattr(cfg.agent_config, "agent_type", "unknown") if cfg.agent_config else "unknown"
        msg = f"Agent configuration for {cfg.framework} is valid"
    else:
        msg = f"Agent configuration has {len(errors)} validation errors"
        details["errors"] = errors
    return AgentValidationResponse(is_valid=ok, message=msg, details=details)