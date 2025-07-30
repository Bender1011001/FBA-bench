export interface SimulationSnapshot {
    current_tick: number;
    simulation_time: string;
    last_update: string;
    uptime_seconds: number;
    products: Record<string, any>;
    competitors: Record<string, any>;
    market_summary: Record<string, any>;
    financial_summary: Record<string, any>;
    agents: Record<string, any>;
    command_stats: Record<string, any>;
    event_stats: Record<string, any>;
    metadata: Record<string, any>;
}

export interface ApiResponse<T> {
    success: boolean;
    message: string;
    data: T;
}

// 1. Configuration Management API Types
export interface SimulationSettings {
    simulationName: string;
    description: string;
    duration: number; // in ticks
    randomSeed: number;
    metricsInterval: number;
    snapshotInterval: number;
}

// Existing LLMConfig updated with max_tokens
export interface LLMConfig {
    provider?: string;
    model?: string;
    temperature?: number;
    api_key?: string;
    max_tokens?: number; // Added for template consistency
}

export interface TemplateAgentConfig {
  agentName: string;
  agentType: string; // Maps to framework in AgentRunnerConfig
  model: string; // Maps to llm_config.model
  max_tokens: number; // Maps to llm_config.max_tokens
  temperature: number; // Maps to llm_config.temperature
  llmInterface: string; // Maps to llm_config.provider
  role: string;
  behavior: string;
  // Add other common agent config fields if necessary that are not covered by AgentRunnerConfig directly
}

export interface Constraints {
    maxBudget: number;
    maxTime: number; // in milliseconds
    tokenLimits: Record<string, number>; // e.g., { grok4: 1000000 }
}

export interface ExperimentParameter {
  name: string;
  values: (string | number)[];
}

// Existing ExperimentConfig updated to use ExperimentParameter
export interface ExperimentConfig {
    experimentName?: string; // Changed from experiment_name for consistency if needed
    description?: string;
    baseParameters?: Record<string, unknown>; // Changed from base_parameters
    parameters?: ExperimentParameter[]; // Using new interface
    outputConfig?: { // Changed from output_config
        saveEvents?: boolean; // Changed from save_events
        saveSnapshots?: boolean; // Changed from save_snapshots
        snapshotIntervalHours?: number; // Changed from snapshot_interval_hours
        metricsToTrack?: string[]; // Changed from metrics_to_track
        [key: string]: unknown;
    };
    parallelRuns?: number; // Changed from parallel_workers
    iterations?: number; // Changed from max_runs as per template usage
    [key: string]: unknown;
}

export interface Configuration {
    simulationSettings: SimulationSettings;
    agentConfigs: TemplateAgentConfig[];
    llmSettings: LLMConfig;
    constraints: Constraints;
    experimentSettings: ExperimentConfig;
}

// Remaining existing interfaces below this line
export interface SimulationConfig {
    config_id?: string;
    name?: string;
    description?: string;
    tick_interval_seconds?: number;
    max_ticks?: number;
    start_time?: string; // Represent as string for simplicity in form, convert to Date when sending
    time_acceleration?: number;
    seed?: number;
    // Common simulation parameters
    duration_hours?: number;
    initial_price?: number;
    cost_basis?: number;
    initial_inventory?: number;
    base_parameters?: Record<string, unknown>;
    created_at?: string;
    updated_at?: string;
}

export interface Template {
  id: string;
  name: string;
  description: string;
  useCase: string;
  configuration: Configuration;
}

export interface AgentRunnerConfig {
    agent_id?: string;
    framework?: string;
    agent_config?: AgentConfig;
    llm_config?: LLMConfig;
    memory_config?: MemoryConfig;
    max_iterations?: number;
    timeout_seconds?: number;
    verbose?: boolean;
    crew_config?: CrewConfig; // Specific to CrewAI
    custom_config?: CustomAgentConfig; // For DIY framework custom settings
    [key: string]: unknown; // Allow for arbitrary top-level fields
}

export interface AgentConfig {
    agent_type?: string;
    target_asin?: string;
    strategy?: string;
    price_sensitivity?: number;
    reaction_speed?: number;
    [key: string]: unknown; // Allow for arbitrary agent config fields
}

export interface CrewConfig {
    process?: string;
    crew_size?: number;
    roles?: string[];
    collaboration_mode?: string;
    allow_delegation?: boolean;
}

export interface CustomAgentConfig {
    llm_type?: string;
    model_name?: string;
    api_key?: string;
    bot_type?: string;
    reorder_threshold?: number;
    reorder_quantity?: number;
    [key: string]: unknown; // Allow for arbitrary custom config fields
}

export interface MemoryConfig {
    type?: string;
    window_size?: number;
}

export interface FrameworkData {
    frameworks: string[];
}

export interface AgentConfigurationResponse {
    agent_framework: string;
    agent_type: string;
    description: string;
    example_config: AgentRunnerConfig; // Using AgentRunnerConfig as example type
}
