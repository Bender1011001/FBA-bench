"""
FBA-Bench v3 Research Toolkit API Server

FastAPI-based REST API and WebSocket server that provides real-time access
to simulation state for research tools and dashboards.

Core Endpoints:
- GET /api/v1/simulation/snapshot - Complete simulation state snapshot
- GET /api/v1/simulation/events - Recent events with filtering
- WebSocket /ws/events - Real-time event streaming

The API is read-only and cannot influence the simulation.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
import uuid
import yaml
import os
import logging
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from event_bus import EventBus
from services.dashboard_api_service import DashboardAPIService
from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
from agent_runners.configs.framework_configs import get_framework_examples, create_example_config
from agent_runners.configs.config_schema import AgentRunnerConfig, LLMConfig, MemoryConfig, AgentConfig, CrewConfig
from experiment_cli import ExperimentConfig as CLIExperimentConfig # To avoid name collision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for configurations and experiments (for demonstration)
simulation_configs_db: Dict[str, Dict[str, Any]] = {}  # {config_id: config_data}
experiment_configs_db: Dict[str, Dict[str, Any]] = {}  # {experiment_id: config_data}
templates_db: Dict[str, Dict[str, Any]] = {} # {template_name: config_data}

# Global state for simulation orchestrator instances
active_simulations: Dict[str, SimulationOrchestrator] = {} # {simulation_id: orchestrator_instance}
active_experiments: Dict[str, Any] = {} # {experiment_id: experiment_manager_instance}

class SimulationSnapshot(BaseModel):
    """Pydantic model for simulation snapshot response."""
    current_tick: Optional[int] = None
    simulation_time: Optional[str] = None
    last_update: Optional[str] = None
    uptime_seconds: Optional[int] = None
    products: Optional[Dict[str, Any]] = None
    competitors: Optional[Dict[str, Any]] = None
    market_summary: Optional[Dict[str, Any]] = None
    financial_summary: Optional[Dict[str, Any]] = None
    agents: Optional[Dict[str, Any]] = None
    command_stats: Optional[Dict[str, Any]] = None
    event_stats: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class EventFilter(BaseModel):
    """Pydantic model for event filtering parameters."""
    event_type: Optional[str] = None
    limit: int = 20
    since_tick: Optional[int] = None

# 1. Configuration Management API Models
class SimulationConfigCreate(BaseModel):
    """Request model for creating a new simulation configuration."""
    name: str = Field(..., description="A unique name for the simulation configuration")
    description: Optional[str] = Field(None, description="A brief description of the configuration")
    tick_interval_seconds: float = Field(1.0, description="Time interval between simulation ticks in seconds")
    max_ticks: Optional[int] = Field(None, description="Maximum number of ticks before simulation stops")
    start_time: Optional[datetime] = Field(None, description="Simulation start time, defaults to now")
    time_acceleration: float = Field(1.0, description="Simulation time acceleration factor")
    seed: Optional[int] = Field(None, description="Master seed for deterministic simulation runs")
    # Additional parameters that might be required by ExperimentConfig or SimulationRunner
    base_parameters: Optional[Dict[str, Any]] = Field({}, description="Base parameters for the simulation setup, including product, market, competitor config etc.")
    
    def to_simulation_config(self) -> SimulationConfig:
        return SimulationConfig(
            tick_interval_seconds=self.tick_interval_seconds,
            max_ticks=self.max_ticks,
            start_time=self.start_time,
            time_acceleration=self.time_acceleration,
            seed=self.seed
        )

class SimulationConfigUpdate(SimulationConfigCreate):
    """Request model for updating an existing simulation configuration."""
    # All fields are optional for update, allowing partial updates
    name: Optional[str] = None
    
class SimulationConfigResponse(BaseModel):
    """Response model for a simulation configuration."""
    config_id: str = Field(..., description="Unique identifier for the simulation configuration")
    name: str = Field(..., description="Name of the simulation configuration")
    description: Optional[str] = Field(None, description="Description of the configuration")
    tick_interval_seconds: float
    max_ticks: Optional[int]
    start_time: Optional[datetime]
    time_acceleration: float
    seed: Optional[int]
    base_parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class ConfigTemplateSave(BaseModel):
    """Request model to save a simulation configuration as a template."""
    config_id: str = Field(..., description="The ID of the configuration to save as a template")
    template_name: str = Field(..., description="Name for the new template")
    description: Optional[str] = Field(None, description="Description of the template")

class ConfigTemplateResponse(BaseModel):
    """Response model for a configuration template."""
    template_name: str
    description: Optional[str]
    config_data: Dict[str, Any] # Full config data of the template
    created_at: datetime


# 2. Simulation Control API Models
class SimulationStartRequest(BaseModel):
    """Request model for starting a simulation."""
    config_id: str = Field(..., description="ID of the simulation configuration to use")
    simulation_id: Optional[str] = Field(None, description="Optional: Specific ID for this simulation run. If not provided, a UUID will be generated.")

class SimulationControlResponse(BaseModel):
    """Generic response for simulation control actions."""
    success: bool
    message: str
    simulation_id: Optional[str] = None
    status: Optional[Dict[str, Any]] = None

class SimulationStatusResponse(BaseModel):
    """Response model for detailed simulation status."""
    is_running: bool
    is_paused: bool
    current_tick: int
    real_time: str
    simulation_time: str
    config: Dict[str, Any]
    statistics: Dict[str, Any]
    simulation_id: Optional[str] = None
    message: Optional[str] = None


# 3. Agent/Bot Management API Models
class AgentConfigurationResponse(BaseModel):
    """Response model for an available agent configuration."""
    agent_framework: str = Field(..., description="The framework this agent belongs to (e.g., 'diy', 'crewai', 'langchain')")
    agent_type: str = Field(..., description="The specific type/model of the agent within its framework (e.g., 'advanced_agent', 'standard_crew')")
    description: str = Field(..., description="A brief description of this agent configuration")
    example_config: Dict[str, Any] = Field(None, description="An example configuration JSON for this agent, including default parameters")

class AgentValidationRequest(BaseModel):
    """Request model for validating agent configuration."""
    agent_config: AgentRunnerConfig = Field(..., description="The full agent runner configuration to validate")

class AgentValidationResponse(BaseModel):
    """Response model for agent configuration validation."""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class FrameworksResponse(BaseModel):
    """Response model for listing available frameworks."""
    frameworks: List[str]

# 4. Experiment Management API Models
class ExperimentCreateRequest(BaseModel):
    """Request model for creating and starting a parameter sweep experiment."""
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Description of the experiment")
    base_parameters: Dict[str, Any] = Field(..., description="Base parameters for each simulation run")
    parameter_sweep: Dict[str, List[Any]] = Field(..., description="Parameters to sweep over and their values")
    output_config: Optional[Dict[str, Any]] = Field({}, description="Output configuration for results")
    parallel_workers: int = Field(1, ge=1, description="Number of parallel processes for running simulations")
    max_runs: Optional[int] = Field(None, ge=1, description="Maximum number of runs to execute (for testing/limits)")

class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status and progress."""
    experiment_id: str
    status: str = Field(..., description="Current status of the experiment (e.g., 'running', 'completed', 'stopped', 'failed')")
    total_runs: int
    completed_runs: int
    successful_runs: int
    failed_runs: int
    progress_percentage: float
    start_time: datetime
    end_time: Optional[datetime] = None
    current_run_details: Optional[Dict[str, Any]] = None # Details of the run currently in progress
    message: Optional[str] = None

class ExperimentResultsResponse(BaseModel):
    """Response model for experiment results."""
    experiment_id: str
    status: str
    results_summary: Dict[str, Any] # Aggregated metrics, analysis etc.
    individual_run_results: List[Dict[str, Any]] # List of results for each run (or a sample)
    results_uri: Optional[str] = Field(None, description="URI or path to the full results directory/archive")


class ConnectionManager:
    """Manages WebSocket connections for real-time event streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📡 WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"📡 WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast_event(self, event_data: Dict[str, Any]):
        """Broadcast event to all connected WebSocket clients."""
        if not self.active_connections:
            return
            
        message = json.dumps(event_data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global state
dashboard_service: Optional[DashboardAPIService] = None
active_event_bus: Optional[EventBus] = None # New global to hold the event bus
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage API server lifecycle."""
    global dashboard_service, active_event_bus
    
    logger.info("Starting FBA-Bench Research Toolkit API Server...")
    
    # Initialize EventBus and DashboardAPIService
    active_event_bus = EventBus()
    dashboard_service = DashboardAPIService(active_event_bus) # Pass the active_event_bus
    
    # Start event bus
    await active_event_bus.start()

    # Start services (but don't connect to actual simulation yet)
    # In production, this would connect to the running simulation
    logger.info("API Server ready (dashboard service initialized, event bus started)")
    
    yield
    
    # Cleanup
    if dashboard_service:
        await dashboard_service.stop()
    if active_event_bus:
        await active_event_bus.stop()
    logger.info("API Server stopped")

# Create FastAPI app
app = FastAPI(
    title="FBA-Bench Research Toolkit API",
    description="Real-time simulation data API for research and analysis, now with full control capabilities.",
    version="2.0.0", # Updated version since we're adding control
    lifespan=lifespan
)


# Removed duplicate lifespan function and app initialization

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React dev server
        "http://127.0.0.1:5173",  # Alternative localhost format
        "http://127.0.0.1:3000",  # Alternative localhost format
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"], # Added PATCH
    allow_headers=["*"],
)

# Dependency for simulation orchestrator (placeholder for actual simulation instance)
class SimulationManager:
    """Manages the lifecycle and state of simulation orchestrator instances."""
    def __init__(self):
        self.orchestrators: Dict[str, SimulationOrchestrator] = {}
        self.orchestrator_lock = asyncio.Lock() # For thread safety

    async def get_orchestrator(self, sim_id: str) -> SimulationOrchestrator:
        async with self.orchestrator_lock:
            orchestrator = self.orchestrators.get(sim_id)
            if not orchestrator:
                raise HTTPException(status_code=404, detail=f"Simulation with ID '{sim_id}' not found.")
            return orchestrator

    async def add_orchestrator(self, sim_id: str, orchestrator: SimulationOrchestrator):
        async with self.orchestrator_lock:
            if sim_id in self.orchestrators:
                logger.warning(f"Simulation with ID '{sim_id}' already exists. Overwriting.")
            self.orchestrators[sim_id] = orchestrator
            logger.info(f"Added simulation orchestrator with ID: {sim_id}")

    async def remove_orchestrator(self, sim_id: str):
        async with self.orchestrator_lock:
            if sim_id in self.orchestrators:
                del self.orchestrators[sim_id]
                logger.info(f"Removed simulation orchestrator with ID: {sim_id}")

    def get_simulation_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        orchestrator = self.orchestrators.get(sim_id)
        if orchestrator:
            return orchestrator.get_status()
        return None

    def get_all_simulation_ids(self) -> List[str]:
        return list(self.orchestrators.keys())

simulation_manager = SimulationManager()


# Configuration Management API Endpoints
@app.post("/api/v1/config/simulation", response_model=SimulationConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation_configuration(config_data: SimulationConfigCreate):
    """
    Create a new simulation configuration.
    
    A unique `config_id` will be generated for the configuration.
    """
    config_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    try:
        # Create a dictionary that combines all fields, including base_parameters
        new_config_entry = config_data.model_dump(by_alias=True, exclude_unset=True)
        new_config_entry['config_id'] = config_id
        new_config_entry['created_at'] = current_time
        new_config_entry['updated_at'] = current_time

        # Store the complete configuration
        simulation_configs_db[config_id] = new_config_entry
        
        logger.info(f"Created simulation configuration: {config_id} - {config_data.name}")
        return SimulationConfigResponse(**new_config_entry)
    except ValidationError as e:
        logger.error(f"Validation error creating config: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Error creating simulation configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/api/v1/config/simulation/{config_id}", response_model=SimulationConfigResponse)
async def get_simulation_configuration(config_id: str):
    """
    Get a simulation configuration by its ID.
    """
    config = simulation_configs_db.get(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return SimulationConfigResponse(**config)

@app.put("/api/v1/config/simulation/{config_id}", response_model=SimulationConfigResponse)
async def update_simulation_configuration(
    config_id: str,
    config_data: SimulationConfigUpdate
):
    """
    Update an existing simulation configuration.
    
    Allows partial updates of the configuration fields.
    """
    if config_id not in simulation_configs_db:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    current_config = simulation_configs_db[config_id]
    update_data = config_data.model_dump(by_alias=True, exclude_unset=True)
    
    # Merge existing config with update data
    for key, value in update_data.items():
        if key == "base_parameters" and isinstance(value, dict) and isinstance(current_config.get("base_parameters"), dict):
            current_config["base_parameters"].update(value)
        else:
            current_config[key] = value
            
    current_config['updated_at'] = datetime.now()
    simulation_configs_db[config_id] = current_config # Ensure the in-memory DB is updated with the modified dictionary
    
    logger.info(f"Updated simulation configuration: {config_id} - {current_config.get('name')}")
    return SimulationConfigResponse(**current_config)

@app.delete("/api/v1/config/simulation/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_simulation_configuration(config_id: str):
    """
    Delete a simulation configuration by its ID.
    
    Returns 204 No Content on successful deletion.
    """
    if config_id not in simulation_configs_db:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    del simulation_configs_db[config_id]
    logger.info(f"Deleted simulation configuration: {config_id}")
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={"message": "Configuration deleted successfully"})
    # return {"message": "Configuration deleted successfully"} # FastAPI handles 204 for no content in response

@app.get("/api/v1/config/templates", response_model=List[ConfigTemplateResponse])
async def list_configuration_templates():
    """
    List all available configuration templates.
    """
    templates = []
    for template_name, data in templates_db.items():
        # Ensure 'config_data' containing the full config is also part of the response
        templates.append(ConfigTemplateResponse(
            template_name=template_name,
            description=data.get('description'),
            config_data=data.get('config_data', {}),
            created_at=data.get('created_at', datetime.min) # Use datetime.min as a fallback
        ))
    logger.info("Listed all configuration templates.")
    return templates

@app.post("/api/v1/config/templates", response_model=ConfigTemplateResponse, status_code=status.HTTP_201_CREATED)
async def save_configuration_as_template(template_data: ConfigTemplateSave):
    """
    Save an existing simulation configuration as a reusable template.
    """
    config_id = template_data.config_id
    template_name = template_data.template_name
    
    if config_id not in simulation_configs_db:
        raise HTTPException(status_code=404, detail=f"Configuration with ID '{config_id}' not found.")
    
    if template_name in templates_db:
        raise HTTPException(status_code=409, detail=f"Template with name '{template_name}' already exists. Use PUT to update.")
        
    config_to_save = simulation_configs_db[config_id].copy()
    config_to_save['template_description'] = template_data.description # Add template-specific description
    
    templates_db[template_name] = {
        "template_name": template_name,
        "description": template_data.description,
        "config_data": config_to_save, # Store the full config
        "created_at": datetime.now()
    }
    logger.info(f"Saved configuration {config_id} as template: {template_name}")
    return ConfigTemplateResponse(**templates_db[template_name])


# 2. Simulation Control API Endpoints
@app.post("/api/v1/simulation/start", response_model=SimulationControlResponse)
async def start_simulation(start_request: SimulationStartRequest):
    """
    Start a new simulation with a given configuration.
    
    If `simulation_id` is provided and a simulation with that ID is already running,
    it will return an error. Otherwise, a new simulation instance is created.
    """
    config_id = start_request.config_id
    sim_id = start_request.simulation_id or str(uuid.uuid4())

    if sim_id in simulation_manager.get_all_simulation_ids():
        raise HTTPException(status_code=409, detail=f"Simulation with ID '{sim_id}' is already running.")

    config_data = simulation_configs_db.get(config_id)
    if not config_data:
        raise HTTPException(status_code=404, detail=f"Configuration with ID '{config_id}' not found.")
    
    try:
        # Create SimulationConfig from stored config data
        sim_config = SimulationConfig(
            tick_interval_seconds=config_data.get('tick_interval_seconds', 1.0),
            max_ticks=config_data.get('max_ticks'),
            start_time=config_data.get('start_time'),
            time_acceleration=config_data.get('time_acceleration', 1.0),
            seed=config_data.get('seed')
        )
        # Create new orchestrator instance (using the global event bus)
        orchestrator = SimulationOrchestrator(sim_config)
        
        # Add to manager and start
        await simulation_manager.add_orchestrator(sim_id, orchestrator)
        await orchestrator.start(active_event_bus) # Use the global active_event_bus
        
        logger.info(f"Started simulation '{sim_id}' with config '{config_id}'")
        return SimulationControlResponse(
            success=True,
            message=f"Simulation '{sim_id}' started successfully with configuration '{config_id}'.",
            simulation_id=sim_id,
            status=orchestrator.get_status()
        )
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        # Clean up if partially started
        await simulation_manager.remove_orchestrator(sim_id)
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {e}")

@app.post("/api/v1/simulation/stop/{simulation_id}", response_model=SimulationControlResponse)
async def stop_simulation(simulation_id: str):
    """
    Stop a running simulation by its ID.
    """
    orchestrator = await simulation_manager.get_orchestrator(simulation_id)
    if not orchestrator.is_running:
        raise HTTPException(status_code=400, detail=f"Simulation '{simulation_id}' is not running.")
    
    try:
        await orchestrator.stop()
        await simulation_manager.remove_orchestrator(simulation_id)
        logger.info(f"Stopped simulation: {simulation_id}")
        return SimulationControlResponse(
            success=True,
            message=f"Simulation '{simulation_id}' stopped.",
            simulation_id=simulation_id
        )
    except Exception as e:
        logger.error(f"Error stopping simulation '{simulation_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop simulation: {e}")

@app.post("/api/v1/simulation/pause/{simulation_id}", response_model=SimulationControlResponse)
async def pause_simulation(simulation_id: str):
    """
    Pause a running simulation by its ID.
    """
    orchestrator = await simulation_manager.get_orchestrator(simulation_id)
    if not orchestrator.is_running:
        raise HTTPException(status_code=400, detail=f"Simulation '{simulation_id}' is not running.")
    if orchestrator.is_paused:
        raise HTTPException(status_code=400, detail=f"Simulation '{simulation_id}' is already paused.")
    
    try:
        await orchestrator.pause()
        logger.info(f"Paused simulation: {simulation_id}")
        return SimulationControlResponse(
            success=True,
            message=f"Simulation '{simulation_id}' paused.",
            simulation_id=simulation_id,
            status=orchestrator.get_status()
        )
    except Exception as e:
        logger.error(f"Error pausing simulation '{simulation_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause simulation: {e}")

@app.post("/api/v1/simulation/resume/{simulation_id}", response_model=SimulationControlResponse)
async def resume_simulation(simulation_id: str):
    """
    Resume a paused simulation by its ID.
    """
    orchestrator = await simulation_manager.get_orchestrator(simulation_id)
    if not orchestrator.is_running:
        raise HTTPException(status_code=400, detail=f"Simulation '{simulation_id}' is not running.")
    if not orchestrator.is_paused:
        raise HTTPException(status_code=400, detail=f"Simulation '{simulation_id}' is not paused.")
    
    try:
        await orchestrator.resume()
        logger.info(f"Resumed simulation: {simulation_id}")
        return SimulationControlResponse(
            success=True,
            message=f"Simulation '{simulation_id}' resumed.",
            simulation_id=simulation_id,
            status=orchestrator.get_status()
        )
    except Exception as e:
        logger.error(f"Error resuming simulation '{simulation_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume simulation: {e}")

@app.get("/api/v1/simulation/status/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str):
    """
    Get detailed status of a running or paused simulation.
    """
    orchestrator = await simulation_manager.get_orchestrator(simulation_id)
    logger.info(f"Retrieved status for simulation: {simulation_id}")
    return SimulationStatusResponse(simulation_id=simulation_id, **orchestrator.get_status())


# 3. Agent/Bot Management API Endpoints
@app.get("/api/v1/agents/available", response_model=List[AgentConfigurationResponse])
async def list_available_agents():
    """
    List all available agent types and configurations from different frameworks.
    """
    available_agents = []
    
    try:
        # Get framework examples
        framework_examples = get_framework_examples()
        
        for framework, agent_types in framework_examples.items():
            for agent_type, description in agent_types.items():
                # Create example config for documentation
                try:
                    example_config = create_example_config(
                        framework=framework,
                        config_type=agent_type,
                        agent_id="example_agent",
                        api_key="your_api_key_here"
                    )
                    example_dict = example_config.model_dump(by_alias=True, exclude_unset=True)
                    # Remove sensitive fields for display
                    if 'api_key' in str(example_dict):
                        example_dict = str(example_dict).replace("your_api_key_here", "[API_KEY_REQUIRED]")
                except Exception:
                    example_dict = {"error": "Could not generate example config"}
                
                available_agents.append(AgentConfigurationResponse(
                    agent_framework=framework,
                    agent_type=agent_type,
                    description=description,
                    example_config=example_dict
                ))
        
        logger.info(f"Listed {len(available_agents)} available agent configurations")
        return available_agents
        
    except Exception as e:
        logger.error(f"Error listing available agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available agents: {e}")

@app.get("/api/v1/agents/bots", response_model=List[Dict[str, Any]])
async def list_baseline_bots():
    """
    List available baseline bots with their configurations.
    """
    baseline_bots = []
    
    try:
        # List bot config files from baseline_bots/configs/
        bot_configs_dir = "baseline_bots/configs"
        
        if os.path.exists(bot_configs_dir):
            for filename in os.listdir(bot_configs_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    bot_name = filename.replace('.yaml', '').replace('.yml', '')
                    config_path = os.path.join(bot_configs_dir, filename)
                    
                    try:
                        with open(config_path, 'r') as f:
                            bot_config = yaml.safe_load(f)
                        
                        baseline_bots.append({
                            "bot_name": bot_name,
                            "config_file": filename,
                            "description": bot_config.get('description', f"Baseline bot: {bot_name}"),
                            "model": bot_config.get('model', 'Unknown'),
                            "provider": bot_config.get('provider', 'Unknown'),
                            "config": bot_config
                        })
                    except Exception as e:
                        logger.warning(f"Could not load bot config {filename}: {e}")
                        baseline_bots.append({
                            "bot_name": bot_name,
                            "config_file": filename,
                            "description": f"Baseline bot: {bot_name} (config load error)",
                            "error": str(e)
                        })
        
        logger.info(f"Listed {len(baseline_bots)} baseline bots")
        return baseline_bots
        
    except Exception as e:
        logger.error(f"Error listing baseline bots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve baseline bots: {e}")

@app.post("/api/v1/agents/validate", response_model=AgentValidationResponse)
async def validate_agent_configuration(validation_request: AgentValidationRequest):
    """
    Validate an agent configuration.
    
    This includes checking required fields, API key validity, and configuration consistency.
    """
    agent_config = validation_request.agent_config
    
    try:
        # Basic validation - check required fields
        validation_errors = []
        validation_details = {}
        
        # Check framework
        if not agent_config.framework:
            validation_errors.append("Framework is required")
        elif agent_config.framework not in ["diy", "crewai", "langchain"]:
            validation_errors.append(f"Unsupported framework: {agent_config.framework}")
        
        # Check agent_id
        if not agent_config.agent_id:
            validation_errors.append("Agent ID is required")
        
        # Framework-specific validation
        if agent_config.framework == "crewai" or agent_config.framework == "langchain":
            if not agent_config.llm_config:
                validation_errors.append("LLM config is required for CrewAI and LangChain frameworks")
            else:
                # Check API key presence (but don't validate actual key for security)
                if not agent_config.llm_config.api_key or agent_config.llm_config.api_key == "your_api_key_here":
                    validation_errors.append("Valid API key is required")
                else:
                    validation_details["api_key_provided"] = True
                    validation_details["api_key_length"] = len(agent_config.llm_config.api_key)
                
                # Check model and provider
                if not agent_config.llm_config.model:
                    validation_errors.append("Model name is required")
                if not agent_config.llm_config.provider:
                    validation_errors.append("Provider is required")
        
        # CrewAI specific validation
        if agent_config.framework == "crewai":
            if agent_config.crew_config:
                if agent_config.crew_config.crew_size and agent_config.crew_config.crew_size < 1:
                    validation_errors.append("Crew size must be at least 1")
        
        # LangChain specific validation
        if agent_config.framework == "langchain":
            if agent_config.max_iterations and agent_config.max_iterations < 1:
                validation_errors.append("Max iterations must be at least 1")
        
        # Overall validation result
        is_valid = len(validation_errors) == 0
        
        if is_valid:
            message = f"Agent configuration for {agent_config.framework} framework is valid"
            validation_details["framework"] = agent_config.framework
            validation_details["agent_type"] = getattr(agent_config.agent_config, 'agent_type', 'unknown') if agent_config.agent_config else 'unknown'
        else:
            message = f"Agent configuration has {len(validation_errors)} validation errors"
            validation_details["errors"] = validation_errors
        
        logger.info(f"Validated agent config for {agent_config.agent_id}: {'valid' if is_valid else 'invalid'}")
        
        return AgentValidationResponse(
            is_valid=is_valid,
            message=message,
            details=validation_details
        )
        
    except Exception as e:
        logger.error(f"Error validating agent configuration: {e}")
        return AgentValidationResponse(
            is_valid=False,
            message=f"Validation failed due to error: {e}",
            details={"error": str(e)}
        )

@app.get("/api/v1/agents/frameworks", response_model=FrameworksResponse)
async def list_available_frameworks():
    """
    List all available agent frameworks.
    """
    try:
        frameworks = ["diy", "crewai", "langchain"]
        logger.info("Listed available frameworks")
        return FrameworksResponse(frameworks=frameworks)
    except Exception as e:
        logger.error(f"Error listing frameworks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve frameworks: {e}")


# 4. Experiment Management API Endpoints
@app.post("/api/v1/experiments", response_model=ExperimentStatusResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(experiment_request: ExperimentCreateRequest):
    """
    Create and start a parameter sweep experiment.
    
    This creates an experiment configuration similar to sweep.yaml and starts running it.
    """
    experiment_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    try:
        # Calculate total parameter combinations
        total_combinations = 1
        for param_values in experiment_request.parameter_sweep.values():
            total_combinations *= len(param_values)
        
        # Apply max_runs limit if specified
        total_runs = min(total_combinations, experiment_request.max_runs) if experiment_request.max_runs else total_combinations
        
        # Create experiment configuration similar to ExperimentConfig from CLI
        experiment_config = {
            "experiment_name": experiment_request.experiment_name,
            "description": experiment_request.description,
            "base_parameters": experiment_request.base_parameters,
            "parameter_sweep": experiment_request.parameter_sweep,
            "output_config": experiment_request.output_config,
            "parallel_workers": experiment_request.parallel_workers,
            "max_runs": experiment_request.max_runs
        }
        
        # Store experiment in database
        experiment_data = {
            "experiment_id": experiment_id,
            "config": experiment_config,
            "status": "running",
            "total_runs": total_runs,
            "completed_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "progress_percentage": 0.0,
            "start_time": current_time,
            "end_time": None,
            "current_run_details": None,
            "message": f"Experiment started with {total_runs} total runs"
        }
        
        experiment_configs_db[experiment_id] = experiment_data
        
        # In a real implementation, you would start the experiment runner here
        # For now, we'll just mark it as running
        logger.info(f"Created experiment {experiment_id}: {experiment_request.experiment_name} with {total_runs} runs")
        
        return ExperimentStatusResponse(**experiment_data)
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {e}")

@app.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentStatusResponse)
async def get_experiment_status(experiment_id: str):
    """
    Get the current status and progress of an experiment.
    """
    if experiment_id not in experiment_configs_db:
        raise HTTPException(status_code=404, detail=f"Experiment with ID '{experiment_id}' not found.")
    
    experiment_data = experiment_configs_db[experiment_id]
    logger.info(f"Retrieved status for experiment: {experiment_id}")
    return ExperimentStatusResponse(**experiment_data)

@app.post("/api/v1/experiments/{experiment_id}/stop", response_model=ExperimentStatusResponse)
async def stop_experiment(experiment_id: str):
    """
    Stop a running experiment.
    """
    if experiment_id not in experiment_configs_db:
        raise HTTPException(status_code=404, detail=f"Experiment with ID '{experiment_id}' not found.")
    
    experiment_data = experiment_configs_db[experiment_id]
    
    if experiment_data["status"] not in ["running", "paused"]:
        raise HTTPException(status_code=400, detail=f"Experiment '{experiment_id}' is not running and cannot be stopped.")
    
    try:
        # Update experiment status
        experiment_data["status"] = "stopped"
        experiment_data["end_time"] = datetime.now()
        experiment_data["message"] = "Experiment stopped by user request"
        
        # In a real implementation, you would stop the actual experiment runner here
        experiment_configs_db[experiment_id] = experiment_data
        
        logger.info(f"Stopped experiment: {experiment_id}")
        return ExperimentStatusResponse(**experiment_data)
        
    except Exception as e:
        logger.error(f"Error stopping experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop experiment: {e}")

@app.get("/api/v1/experiments/{experiment_id}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(experiment_id: str):
    """
    Get the results of a completed or stopped experiment.
    """
    if experiment_id not in experiment_configs_db:
        raise HTTPException(status_code=404, detail=f"Experiment with ID '{experiment_id}' not found.")
    
    experiment_data = experiment_configs_db[experiment_id]
    
    try:
        # In a real implementation, you would load actual results from the results directory
        # For now, we'll return a placeholder structure
        results_summary = {
            "experiment_name": experiment_data["config"]["experiment_name"],
            "total_runs": experiment_data["total_runs"],
            "successful_runs": experiment_data["successful_runs"],
            "failed_runs": experiment_data["failed_runs"],
            "average_execution_time": 0.0,  # Would be calculated from actual results
            "parameter_sweep_summary": experiment_data["config"]["parameter_sweep"]
        }
        
        # Placeholder for individual run results
        individual_results = []
        
        # In a real implementation, you would scan the results directory for run files
        # and load the actual simulation results
        
        logger.info(f"Retrieved results for experiment: {experiment_id}")
        
        return ExperimentResultsResponse(
            experiment_id=experiment_id,
            status=experiment_data["status"],
            results_summary=results_summary,
            individual_run_results=individual_results,
            results_uri=f"results/{experiment_data['config']['experiment_name']}_{experiment_id}"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving experiment results {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve experiment results: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with comprehensive API documentation."""
    return """
    <html>
        <head>
            <title>FBA-Bench Research Toolkit API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .endpoint { background: #f5f5f5; padding: 8px; margin: 5px 0; border-radius: 5px; }
                .method { color: #007bff; font-weight: bold; }
                .section { margin: 20px 0; }
                h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
                h3 { color: #555; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>🚀 FBA-Bench Research Toolkit API</h1>
            <p>Complete simulation control and research data API for FBA-Bench.</p>
            
            <div class="section">
                <h2>🔧 Configuration Management</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/config/simulation/{config_name}</code><br>
                    Create new simulation configuration
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/config/simulation/{config_name}</code><br>
                    Get simulation configuration by name
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/config/simulation/</code><br>
                    List all simulation configurations
                </div>
                <div class="endpoint">
                    <span class="method">PUT</span> <code>/api/v1/config/simulation/{config_name}</code><br>
                    Update simulation configuration
                </div>
                <div class="endpoint">
                    <span class="method">DELETE</span> <code>/api/v1/config/simulation/{config_name}</code><br>
                    Delete simulation configuration
                </div>
            </div>
            
            <div class="section">
                <h2>🎮 Simulation Control</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/simulation/start</code><br>
                    Start simulation with configuration
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/simulation/stop</code><br>
                    Stop current simulation
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/simulation/pause</code><br>
                    Pause current simulation
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/simulation/resume</code><br>
                    Resume paused simulation
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/simulation/status</code><br>
                    Get current simulation status
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 Agent & Bot Management</h2>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/agents/available_bots</code><br>
                    List available baseline bots
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/agents/frameworks</code><br>
                    List supported agent frameworks
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/agents/validate</code><br>
                    Validate agent configuration
                </div>
            </div>
            
            <div class="section">
                <h2>🧪 Experiment Management</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/experiments</code><br>
                    Create and start parameter sweep experiment
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/experiments/{experiment_id}</code><br>
                    Get experiment status and progress
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/v1/experiments/{experiment_id}/stop</code><br>
                    Stop running experiment
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/experiments/{experiment_id}/results</code><br>
                    Get experiment results and analysis
                </div>
            </div>
            
            <div class="section">
                <h2>📡 Real-time Data API</h2>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/simulation/snapshot</code><br>
                    Get complete simulation state snapshot
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/v1/simulation/events</code><br>
                    Get recent events with optional filtering
                </div>
                <div class="endpoint">
                    <span class="method">WebSocket</span> <code>/ws/events</code><br>
                    Real-time event streaming
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/dashboard</code><br>
                    Real-time simulation dashboard
                </div>
            </div>
            
            <div class="section">
                <h2>🔗 Quick Links</h2>
                <ul>
                    <li><a href="/docs">📖 Complete API Documentation (Swagger UI)</a></li>
                    <li><a href="/redoc">📋 API Documentation (ReDoc)</a></li>
                    <li><a href="/dashboard">📊 Live Dashboard</a></li>
                    <li><a href="/api/v1/simulation/snapshot">📸 Simulation Snapshot</a></li>
                </ul>
            </div>
        </body>
    </html>
    """


@app.get("/api/v1/simulation/snapshot", response_model=SimulationSnapshot)
async def get_simulation_snapshot():
    """
    Get complete simulation state snapshot.
    
    Returns comprehensive real-time simulation state including:
    - Current tick and timing information
    - Product prices and inventory
    - Competitor market landscape  
    - Sales and financial metrics
    - Agent activity and command history
    - System performance stats
    """
    if not dashboard_service:
        raise HTTPException(status_code=503, detail="Dashboard service not available")
    
    snapshot = dashboard_service.get_simulation_snapshot()
    return SimulationSnapshot(**snapshot)


@app.get("/api/v1/simulation/events")
async def get_recent_events(
    event_type: Optional[str] = Query(None, description="Filter by event type (sales, commands)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of events to return"),
    since_tick: Optional[int] = Query(None, description="Only return events since this tick")
):
    """
    Get recent events with optional filtering.
    
    Parameters:
    - event_type: Filter by 'sales' or 'commands' (optional)
    - limit: Maximum events to return (1-100, default: 20)
    - since_tick: Only return events from this tick onwards, inclusive (optional)
    
    Returns:
    A JSON response containing:
    - events: List of filtered events with tick_number, timestamp, and event details
    - event_type: The event type filter applied (if any)
    - limit: The limit parameter used
    - total_returned: Number of events in the response
    - filtered: Boolean indicating if tick filtering was applied
    - since_tick: The tick filter value (if filtering was applied)
    - timestamp: Response generation timestamp
    
    Note: Events are stored with tick_number to enable time-based filtering.
    Each event is tagged with the tick during which it occurred.
    """
    if not dashboard_service:
        raise HTTPException(status_code=503, detail="Dashboard service not available")
    
    events = dashboard_service.get_recent_events(
        event_type=event_type,
        limit=limit,
        since_tick=since_tick
    )
    
    response = {
        "events": events,
        "event_type": event_type,
        "limit": limit,
        "total_returned": len(events),
        "timestamp": datetime.now().isoformat()
    }
    
    # Include filtering info in response if tick filtering was applied
    if since_tick is not None:
        response["since_tick"] = since_tick
        response["filtered"] = True
    else:
        response["filtered"] = False
    
    return response


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FBA-Bench Research Toolkit API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "dashboard_service_running": dashboard_service is not None and dashboard_service.is_running
    }


# WebSocket Endpoints

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    Clients can connect to receive real-time updates about:
    - Sales transactions
    - Price changes  
    - Agent commands
    - Market updates
    - System events
    """
    await connection_manager.connect(websocket)
    
    try:
        # Send initial snapshot
        if dashboard_service:
            snapshot = dashboard_service.get_simulation_snapshot()
            await websocket.send_text(json.dumps({
                "type": "snapshot",
                "data": snapshot,
                "timestamp": datetime.now().isoformat()
            }))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message (ping, filter updates, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client commands
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# Dashboard Endpoints

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the real-time dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FBA-Bench Real-Time Dashboard</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            .header { 
                border-bottom: 1px solid #eee; 
                padding-bottom: 20px; 
                margin-bottom: 20px; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .metric-card { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 6px; 
                border-left: 4px solid #007bff; 
            }
            .metric-title { 
                font-size: 12px; 
                color: #666; 
                text-transform: uppercase; 
                letter-spacing: 0.5px; 
            }
            .metric-value { 
                font-size: 24px; 
                font-weight: bold; 
                color: #333; 
                margin-top: 5px; 
            }
            .status { 
                display: inline-block; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold; 
            }
            .status.connected { 
                background: #d4edda; 
                color: #155724; 
            }
            .status.disconnected { 
                background: #f8d7da; 
                color: #721c24; 
            }
            .log { 
                background: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 4px; 
                padding: 15px; 
                height: 200px; 
                overflow-y: auto; 
                font-family: 'Courier New', monospace; 
                font-size: 12px; 
            }
            .log-entry { 
                margin-bottom: 5px; 
                word-wrap: break-word; 
            }
            .timestamp { 
                color: #666; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 FBA-Bench Real-Time Dashboard</h1>
                <p>Live simulation monitoring and analytics</p>
                <span id="connectionStatus" class="status disconnected">Disconnected</span>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Current Tick</div>
                    <div class="metric-value" id="currentTick">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Revenue</div>
                    <div class="metric-value" id="totalRevenue">$0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Profit</div>
                    <div class="metric-value" id="totalProfit">$0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Units Sold</div>
                    <div class="metric-value" id="unitsSold">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Active Agents</div>
                    <div class="metric-value" id="activeAgents">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Events/sec</div>
                    <div class="metric-value" id="eventsPerSec">0</div>
                </div>
            </div>
            
            <h3>📡 Real-Time Event Stream</h3>
            <div id="eventLog" class="log">
                <div class="log-entry">Connecting to simulation...</div>
            </div>
        </div>
        
        <script>
            let websocket = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 10;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/ws/events';
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function() {
                    console.log('WebSocket connected');
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = 'status connected';
                    reconnectAttempts = 0;
                    addLogEntry('✅ Connected to simulation', 'success');
                };
                
                websocket.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    handleWebSocketMessage(message);
                };
                
                websocket.onclose = function() {
                    console.log('WebSocket disconnected');
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    document.getElementById('connectionStatus').className = 'status disconnected';
                    addLogEntry('❌ Disconnected from simulation', 'error');
                    
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 2000 * reconnectAttempts);
                        addLogEntry(`🔄 Reconnecting... (attempt ${reconnectAttempts})`, 'info');
                    }
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    addLogEntry('⚠️ Connection error', 'error');
                };
            }
            
            function handleWebSocketMessage(message) {
                if (message.type === 'snapshot') {
                    updateDashboard(message.data);
                    addLogEntry('📸 Snapshot received', 'info');
                } else if (message.type === 'event') {
                    addLogEntry(`📝 ${message.event_type}: ${JSON.stringify(message.data)}`, 'event');
                } else if (message.type === 'heartbeat') {
                    // Silent heartbeat
                } else {
                    addLogEntry(`📨 ${message.type}`, 'info');
                }
            }
            
            function updateDashboard(data) {
                document.getElementById('currentTick').textContent = data.current_tick || 0;
                
                const financial = data.financial_summary || {};
                document.getElementById('totalRevenue').textContent = 
                    '$' + (financial.total_revenue / 100 || 0).toFixed(2);
                document.getElementById('totalProfit').textContent = 
                    '$' + (financial.total_profit / 100 || 0).toFixed(2);
                document.getElementById('unitsSold').textContent = 
                    financial.total_units_sold || 0;
                
                const agentCount = Object.keys(data.agents || {}).length;
                document.getElementById('activeAgents').textContent = agentCount;
                
                const eventStats = data.event_stats || {};
                document.getElementById('eventsPerSec').textContent = 
                    eventStats.events_per_second || 0;
            }
            
            function addLogEntry(message, type = 'info') {
                const log = document.getElementById('eventLog');
                const timestamp = new Date().toLocaleTimeString();
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
                log.appendChild(entry);
                log.scrollTop = log.scrollHeight;
                
                // Keep only last 100 entries
                while (log.children.length > 100) {
                    log.removeChild(log.firstChild);
                }
            }
            
            // Initial connection
            connectWebSocket();
            
            // Periodic ping to keep connection alive
            setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """


# Utility function to broadcast events (called by external services)
async def broadcast_event_to_clients(event_type: str, event_data: Dict[str, Any]):
    """Broadcast event to all connected WebSocket clients."""
    await connection_manager.broadcast_event({
        "type": "event",
        "event_type": event_type,
        "data": event_data,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FBA-Bench Research Toolkit API Server...")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📖 API Docs: http://localhost:8000/docs")
    print("📡 WebSocket: ws://localhost:8000/ws/events")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )