interface Money {
  amount: string;
  currency: string;
}

export interface ProductState {
    id: string;
    name: string;
    current_price: Money;
    // Add other product-related fields as needed
}

export interface CompetitorState {
    id: string;
    name: string;
    current_price: Money;
    // Add other competitor-related fields as needed
}

export interface FinancialSummary {
    total_revenue: Money;
    total_costs: Money;
    total_profit: Money;
    total_sales: Money; // Added for use in metrics
}

export interface MarketSummary {
    trust_score: number;
    // Add other market-related fields as needed
}

export interface AgentSnapshot {
    id: string;
    name: string;
    status: 'active' | 'paused' | 'error' | 'idle';
    // Add other agent-related fields as needed for the snapshot
}

export interface RecentSale {
    sale_id: string;
    product_id: string;
    quantity: number;
    sale_price: Money;
    timestamp: string;
}

export interface SimulationMetadata {
    simulation_id?: string;
    status?: 'running' | 'paused' | 'stopped' | 'error' | 'starting' | 'idle';
    total_ticks?: number;
    [key: string]: unknown; // For flexibility
}

export interface SimulationSnapshot {
    current_tick: number;
    simulation_time: string;
    last_update: string;
    uptime_seconds: number;
    products: Record<string, ProductState>;
    competitor_states: CompetitorState[];
    market_summary: MarketSummary;
    financial_summary: FinancialSummary;
    agents: Record<string, AgentSnapshot>;
    command_stats: Record<string, unknown>; // Keep as unknown for now if structure is unclear
    event_stats: { ticks_per_second?: number; [key: string]: unknown; }; // Added ticks_per_second
    metadata: SimulationMetadata;
    recent_sales: RecentSale[]; // Added recent sales array
    timestamp: string; // Add timestamp to snapshot for last update time
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
    duration: number; // in ticks (maps to 'duration' in validation)
    tickInterval: number; // New: in seconds
    randomSeed: number;
    metricsInterval: number;
    snapshotInterval: number;
    initialPrice: number; // New: initial trading price
    inventory: number; // New: initial agent inventory
    [key: string]: string | number | boolean; // Add index signature for form flexibility
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

export interface ConstraintConfig {
  tier: 'T0' | 'T1' | 'T2' | 'T3';
  budgetLimitUSD: number;
  tokenLimit: number; // This will map to max_total_tokens from backend
  rateLimitPerMinute: number;
  memoryLimitMB: number;
  dailyBudgetLimitUSD?: number;
  monthlyBudgetLimitUSD?: number;
}

export interface TierConfig {
  name: string;
  budgetLimit: number; // Total budget in USD
  tokenLimit: number; // Max total tokens
  rateLimit: number; // API calls per minute
  memoryLimit: number; // Agent memory size in MB
  features: string[];
  description: string;
}

export interface ConstraintUsage {
  budgetUsedUSD: number;
  budgetLimitUSD: number;
  tokenUsed: number;
  tokenLimit: number;
  rateUsed: number;
  rateLimit: number;
  memoryUsedMB: number;
  memoryLimitMB: number;
  estimatedTimeUntilBudgetReached?: string;
  estimatedTimeUntilTokensReached?: string;
  budgetHealth: 'safe' | 'warning' | 'critical';
  tokenHealth: 'safe' | 'warning' | 'critical';
  rateHealth: 'safe' | 'warning' | 'critical';
  memoryHealth: 'safe' | 'warning' | 'critical';
}

export interface ExperimentParameter {
  name: string;
  type: 'discrete' | 'range';
  values?: (string | number)[]; // For discrete values
  min?: number; // For range
  max?: number; // For range
  step?: number; // For range
}

// Existing ExperimentConfig updated to use ExperimentParameter
export interface ExperimentConfig {
    experimentName?: string; // Changed from experiment_name for consistency if needed
    description?: string;
    baseParameters?: Record<string, (string | number)[]>; // Changed from base_parameters to align with new structure
    parameters?: ExperimentParameter[]; // Using new interface
    outputConfig?: { // Changed from output_config
        saveEvents?: boolean; // Changed from save_events
        saveSnapshots?: boolean; // Changed from save_snapshots
        snapshotIntervalHours?: number; // Changed from snapshot_interval_hours
        metricsToTrack?: string[]; // Changed from metrics_to_track
        [key: string]: unknown;
    };
    batchSize?: number; // Renamed from parallelRuns for clarity
    iterations?: number; // Changed from max_runs as per template usage
    [key: string]: unknown;
}

export type ExperimentStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';

export interface ExperimentProgressUpdate {
    experimentId: string;
    status?: ExperimentStatus;
    progress?: number;
    estimatedCompletionTime?: string;
    // Add other fields that might come from WebSocket updates
    [key: string]: unknown;
}

export interface ExperimentRunResult {
    experimentId: string;
    message: string;
    // Potentially other data like initial status, start time
}

export interface ExperimentExecution {
    id: string;
    experimentName: string;
    description?: string;
    config: ExperimentConfig;
    status: ExperimentStatus;
    progress?: number; // 0-100
    startTime?: string;
    endTime?: string;
    estimatedCompletionTime?: string;
    lastUpdated?: string; // Timestamp of the last update
    resultsSummary?: Record<string, unknown>; // Basic summary of results
    // Detailed results will be fetched by ExperimentResults component
}

export interface ExperimentResultSummary {
    id: string;
    experimentName: string;
    description: string;
    status: ExperimentStatus;
    startTime: string;
    endTime: string;
    totalCombinations: number;
    completedCombinations: number;
    // Add key metrics/results here that can be displayed in a table
    keyMetrics?: Record<string, number | string>;
}

export interface DetailedExperimentResult {
    experimentId: string;
    config: ExperimentConfig; // The full configuration used
    summary: ExperimentResultSummary;
    // Potentially full result data, e.g.,
    // simulationRuns: SimulationRunData[];
    // rawData: any; // More detailed raw output
    // For now, keep it simple
    resultsData: Record<string, unknown>; // Placeholder for actual detailed results
}

// Add a type for metric data if we want to visualize it
export interface MetricDataPoint {
    label: string;
    value: number;
}

export interface Configuration {
    simulationSettings: SimulationSettings;
    agentConfigs: TemplateAgentConfig[];
    llmSettings: LLMConfig;
    constraints: ConstraintConfig;
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

export interface ConnectionStatus {
  connected: boolean;
  reconnectAttempts: number;
  lastHeartbeat?: string;
}
// Specific Event Payloads
export interface TickEventPayload {
    tick_number: number;
}

export interface SaleEventPayload {
    sale: {
        quantity: number;
        product_asin: string;
        sale_price: Money;
        buyer_id: string;
        competitor_id?: string;
    };
}

export interface SetPriceCommandPayload {
    product_asin: string;
    new_price: Money;
}

export interface ProductPriceUpdatedPayload {
    product_asin: string;
    old_price: Money;
    new_price: Money;
}

export interface CompetitorPricesUpdatedPayload {
    competitor_states: CompetitorState[];
}

export type SimulationEvent =
    | { type: 'tick'; timestamp: string; payload: TickEventPayload }
    | { type: 'sale_occurred'; timestamp: string; payload: SaleEventPayload }
    | { type: 'set_price_command'; timestamp: string; payload: SetPriceCommandPayload }
    | { type: 'product_price_updated'; timestamp: string; payload: ProductPriceUpdatedPayload }
    | { type: 'competitor_prices_updated'; timestamp: string; payload: CompetitorPricesUpdatedPayload }
    | { type: string; timestamp: string; payload: Record<string, unknown> }; // Fallback for other, unknown event types

export interface SimulationStatus {
  id: string;
  status: 'running' | 'paused' | 'stopped' | 'error' | 'starting' | 'idle';
  currentTick: number;
  totalTicks: number;
  simulationTime: string;
  realTime: string;
  ticksPerSecond: number;
  revenue: number;
  costs: number;
  profit: number;
  activeAgentCount: number;
  totalAgentCount?: number; // Added for clarity, might be in snapshot metadata
}

export interface AgentStatus {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'error' | 'idle';
  profit: number;
  decisions: number;
  lastAction: string;
  lastActionTime: string;
}

export interface SystemHealth {
  apiResponseTime: number;
  wsConnectionStatus: 'connected' | 'disconnected' | 'error';
  memoryUsage: number;
  cpuUsage: number;
  dbConnectionStatus: 'connected' | 'disconnected';
  queueLength: number;
}

export interface DashboardMetric {
  label: string;
  value: string | number;
  formatType?: 'currency' | 'number' | 'percentage' | 'time' | 'string'; // Added 'string'
  trend?: 'up' | 'down' | 'neutral';
  unit?: string;
  description?: string;
  color?: string; // For status indicators
}

// New Interfaces for Results Visualization and Analysis Dashboard
export interface ResultsData {
  experimentId: string;
  simulationResults: SimulationResult[];
  aggregatedMetrics: AggregatedMetrics;
  agentPerformance: AgentPerformanceData[];
  financialMetrics: FinancialMetrics;
  timeSeriesData: TimeSeriesData[];
}

export interface SimulationResult {
  timestamp: string;
  tick: number;
  revenue: number;
  costs: number;
  profit: number;
  agentMetrics: Record<string, number>;
  marketMetrics: MarketMetrics;
}

export interface AggregatedMetrics {
  totalRevenue: number;
  totalCosts: number;
  totalProfit: number;
  averageTicksPerSecond: number;
  topPerformingAgent: string;
  experimentDuration: number;
}

// Placeholder for AgentPerformanceData - define based on backend structure
export interface AgentPerformanceData {
  agentId: string;
  profit: number;
  decisionsMade: number;
  accuracy: number;
  // Add other agent-specific metrics
}

// Placeholder for FinancialMetrics - define based on backend structure
export interface FinancialMetrics {
  totalRevenue: number;
  totalCosts: number;
  totalProfit: number;
  // Add other key financial metrics
}

// Placeholder for TimeSeriesData - define based on backend structure
export interface TimeSeriesData {
  timestamp: string;
  tick: number;
  revenue: number;
  costs: number;
  profit: number;
  [key: string]: number | string; // For other metrics over time
}

// Placeholder for MarketMetrics - define based on backend structure
export interface MarketMetrics {
  priceTrend: number;
  inventoryLevels: number;
  // Add other market-specific metrics
}

export interface ChartConfiguration {
  chartType: 'line' | 'bar' | 'area' | 'scatter' | 'pie' | 'gauge' | 'heatmap' | 'correlation' | 'radar' | 'treemap' | 'funnel';
  dataSource: string; // e.g., 'financialMetrics', 'agentPerformance', 'timeSeriesData'
  timeRange?: { start: string; end: string }; // Optional for non-time-series charts
  metrics: string[]; // Metrics to display on the chart
  groupBy?: string; // For grouping data, e.g., by agent type
  filters?: Record<string, unknown>; // Additional filters
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  colorScheme?: string[]; // Custom color scheme for charts
  showLegend?: boolean;
  showGrid?: boolean;
  interactive?: boolean; // Enable drill-down and hover interactions
}

// Benchmarking Framework Types
export interface BenchmarkConfig {
  benchmark_id: string;
  name: string;
  description?: string;
  version?: string;
  environment: {
    deterministic: boolean;
    random_seed?: number;
    parallel_execution: boolean;
    max_workers: number;
  };
  scenarios: ScenarioConfig[];
  agents: AgentConfig[];
  metrics: MetricsConfig;
  execution: ExecutionConfig;
  output: OutputConfig;
  validation: ValidationConfig;
}

export interface ScenarioConfig {
  id: string;
  name?: string;
  type: string;
  description?: string;
  config: Record<string, unknown>;
  enabled?: boolean;
  priority?: number;
  parameters: {
    duration?: number;
    complexity?: 'low' | 'medium' | 'high';
    domain?: string;
    difficulty?: 'easy' | 'medium' | 'hard' | 'expert';
  };
  metadata?: {
    author?: string;
    version?: string;
    tags?: string[];
  };
}

export interface AgentConfig {
  id: string;
  name?: string;
  type: string;
  description?: string;
  framework_config: {
    framework: string;
    model?: string;
    parameters: Record<string, unknown>;
  };
  capabilities?: string[];
  constraints?: {
    max_memory?: number;
    max_cpu?: number;
    timeout?: number;
  };
  enabled?: boolean;
}

export interface MetricsConfig {
  categories: string[];
  custom_metrics?: CustomMetricConfig[];
}

export interface CustomMetricConfig {
  name: string;
  type: string;
  config: Record<string, unknown>;
}

export interface ExecutionConfig {
  runs_per_scenario?: number;
  max_duration?: number;
  timeout?: number;
  retry_on_failure?: boolean;
  max_retries?: number;
}

export interface OutputConfig {
  format?: 'json' | 'csv' | 'yaml';
  path?: string;
  include_detailed_logs?: boolean;
  include_audit_trail?: boolean;
}

export interface ValidationConfig {
  enabled?: boolean;
  statistical_significance?: boolean;
  confidence_level?: number;
  reproducibility_check?: boolean;
}

// Benchmark Results Types
export interface BenchmarkResult {
  benchmark_name: string;
  config_hash: string;
  start_time: string;
  end_time: string;
  duration_seconds: number;
  scenario_results: ScenarioResult[];
  metadata: Record<string, unknown>;
}

export interface ScenarioResult {
  scenario_name: string;
  start_time: string;
  end_time: string;
  duration_seconds: number;
  agent_results: AgentRunResult[];
}

export interface AgentRunResult {
  agent_id: string;
  scenario_name: string;
  run_number: number;
  start_time: string;
  end_time: string;
  duration_seconds: number;
  metrics: MetricResult[];
  errors: string[];
  success: boolean;
}

export interface MetricResult {
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

// Multi-dimensional Metrics Types
export interface MultiDimensionalMetric {
  name: string;
  category: string;
  dimensions: string[];
  values: number[];
  unit: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface CapabilityAssessment {
  agent_id: string;
  scenario_name: string;
  capabilities: {
    cognitive: number;
    business: number;
    technical: number;
    ethical: number;
    [key: string]: number;
  };
  overall_score: number;
  timestamp: string;
}

export interface PerformanceHeatmap {
  agents: string[];
  scenarios: string[];
  metrics: string[];
  data: number[][][]; // 3D array: [agent][scenario][metric]
  timestamp: string;
}

export interface ExecutionProgress {
  benchmark_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number; // 0-100
  current_scenario?: string;
  current_agent?: string;
  current_run?: number;
  estimated_completion_time?: string;
  start_time?: string;
  end_time?: string;
  errors?: string[];
  metadata?: Record<string, unknown>;
}

// WebSocket Event Types for Benchmarking
export type BenchmarkWebSocketEvent =
  | { type: 'benchmark_started'; payload: { benchmark_id: string; timestamp: string } }
  | { type: 'benchmark_progress'; payload: ExecutionProgress }
  | { type: 'scenario_completed'; payload: { scenario_name: string; duration_seconds: number } }
  | { type: 'agent_run_completed'; payload: AgentRunResult }
  | { type: 'benchmark_completed'; payload: BenchmarkResult }
  | { type: 'benchmark_error'; payload: { error: string; timestamp: string } }
  | { type: 'metrics_update'; payload: MultiDimensionalMetric[] }
  | { type: 'real_time_metrics'; payload: { agent_id: string; scenario_name: string; tick: number; metrics: MetricResult[] } };

// Configuration Template Types
export interface ConfigurationTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  config: BenchmarkConfig;
  tags?: string[];
  is_default?: boolean;
  created_at?: string;
  updated_at?: string;
}

// Report Types
export interface BenchmarkReport {
  id: string;
  benchmark_id: string;
  title: string;
  description?: string;
  template_id?: string;
  format: 'pdf' | 'html' | 'json' | 'csv';
  sections: ReportSection[];
  generated_at: string;
  metadata?: Record<string, unknown>;
}

export interface ReportSection {
  id: string;
  title: string;
  type: 'summary' | 'charts' | 'tables' | 'raw_data' | 'analysis';
  content: Record<string, unknown>;
  order: number;
  visible?: boolean;
}

// Export Options
export interface ExportOptions {
  format: 'json' | 'csv' | 'pdf' | 'html' | 'yaml';
  include_metadata?: boolean;
  include_raw_data?: boolean;
  include_charts?: boolean;
  include_analysis?: boolean;
  date_range?: { start: string; end: string };
  filters?: Record<string, unknown>;
}
