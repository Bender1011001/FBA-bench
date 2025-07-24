/**
 * TypeScript types for FBA-Bench Dashboard
 * Mirrors the Pydantic models from the backend API
 */

export type DistressProtocolStatus = 'ok' | 'warning' | 'active';
export type SupplierStatus = 'active' | 'blacklisted' | 'suspended' | 'bankrupt';
export type SupplierType = 'international' | 'domestic';
export type EventType = 'adversarial' | 'agent_action' | 'market_event' | 'customer_event';

// Core Dashboard Data Types

export interface KPIMetrics {
  resilient_net_worth: number;
  daily_profit: number;
  total_profit: number;
  cash_balance: number;
  seller_trust_score: number;
  distress_protocol_status: DistressProtocolStatus;
  simulation_day: number;
}

export interface TimeSeriesPoint {
  day: number;
  timestamp: string;
  value: number;
}

export interface PerformanceMetrics {
  cash_balance: TimeSeriesPoint[];
  revenue: TimeSeriesPoint[];
  profit: TimeSeriesPoint[];
  inventory_value: TimeSeriesPoint[];
  total_sales: TimeSeriesPoint[];
  best_seller_rank: TimeSeriesPoint[];
}

export interface AgentStatus {
  current_goal: string;
  compute_budget_used: number;
  api_budget_used: number;
  strategic_plan_coherence: number;
}

export interface EventLogEntry {
  timestamp: string;
  event_type: EventType;
  title: string;
  description: string;
  details?: Record<string, any>;
}

export interface ExecutiveSummary {
  kpis: KPIMetrics;
  performance_metrics: PerformanceMetrics;
  agent_status: AgentStatus;
  event_log: EventLogEntry[];
}

// Financial Deep Dive Types

export interface ProfitLossStatement {
  revenue: Record<string, number>;
  cost_of_goods_sold: Record<string, number>;
  gross_profit: Record<string, number>;
  operating_expenses: Record<string, Record<string, number>>;
  net_profit: Record<string, number>;
}

export interface FeeBreakdown {
  referral_fees: number;
  fba_fulfillment_fees: number;
  storage_fees: number;
  ancillary_fees: number;
  penalty_fees: number;
}

export interface BalanceSheet {
  assets: Record<string, number>;
  liabilities: Record<string, number>;
  equity: Record<string, number>;
}

export interface FinancialDeepDive {
  profit_loss: ProfitLossStatement;
  fee_breakdown: FeeBreakdown;
  balance_sheet: BalanceSheet;
}

// Product & Market Analysis Types

export interface BSRComponents {
  ema_sales_velocity: TimeSeriesPoint[];
  ema_conversion: TimeSeriesPoint[];
  rel_sales_index: TimeSeriesPoint[];
  rel_price_index: TimeSeriesPoint[];
}

export interface ProductInfo {
  asin: string;
  category: string;
  cost: number;
  price: number;
  current_quantity: number;
  days_of_supply: number;
  inventory_turnover: number;
}

export interface CompetitorData {
  asin: string;
  price: number;
  sales_velocity: number;
  bsr: number;
  strategy: string;
  is_agent: boolean;
}

export interface ProductMarketAnalysis {
  product_info: ProductInfo;
  bsr_components: BSRComponents;
  competitors: CompetitorData[];
}

// Supply Chain Types

export interface SupplierInfo {
  supplier_id: string;
  name: string;
  supplier_type: SupplierType;
  status: SupplierStatus;
  reputation_score: number;
  moq: number;
  lead_time: number;
}

export interface OrderInfo {
  order_id: string;
  supplier_id: string;
  quantity: number;
  status: string;
  expected_delivery: string;
}

export interface SupplyChainOperations {
  suppliers: SupplierInfo[];
  active_orders: OrderInfo[];
}

// Agent Cognition Types

export interface GoalStackItem {
  goal_id: string;
  description: string;
  priority: number;
  status: string;
}

export interface MemoryEntry {
  memory_type: string;
  content: string;
  timestamp: string;
  relevance_score: number;
}

export interface StrategicPlan {
  mission: string;
  objectives: string[];
  coherence_score: number;
}

export interface AgentCognition {
  goal_stack: GoalStackItem[];
  memory_entries: MemoryEntry[];
  strategic_plan: StrategicPlan;
}

// WebSocket Event Types

export interface WebSocketEvent {
  event_type: string;
  data: Record<string, any>;
  timestamp: string;
}

// Complete Dashboard State

export interface DashboardState {
  executive_summary: ExecutiveSummary;
  financial_deep_dive: FinancialDeepDive;
  product_market_analysis: ProductMarketAnalysis;
  supply_chain_operations: SupplyChainOperations;
  agent_cognition: AgentCognition;
  last_updated: string;
}

// UI State Types

export interface TabState {
  activeTab: string;
  loading: boolean;
  error: string | null;
}

export interface ChartConfig {
  type: 'line' | 'bar' | 'pie' | 'area';
  title: string;
  xAxis?: string;
  yAxis?: string;
  series: string[];
  colors?: string[];
}

export interface FilterState {
  timeRange: {
    start: string;
    end: string;
  };
  eventTypes: EventType[];
  showOnlyAgent: boolean;
}

// API Response Types

export interface APIResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp: string;
}

export interface APIError {
  error: string;
  detail?: string;
  timestamp: string;
}

// Chart Data Types for ECharts

export interface EChartsOption {
  title?: {
    text: string;
    left?: string;
  };
  tooltip?: {
    trigger: string;
    axisPointer?: {
      type: string;
    };
  };
  legend?: {
    data: string[];
    bottom?: number;
  };
  grid?: {
    left: string;
    right: string;
    bottom: string;
    containLabel: boolean;
  };
  xAxis?: {
    type: string;
    data?: string[];
    name?: string;
  };
  yAxis?: {
    type: string;
    name?: string;
  };
  series: Array<{
    name: string;
    type: string;
    data: number[] | Array<{ value: number; name: string }>;
    smooth?: boolean;
    areaStyle?: object;
    itemStyle?: {
      color: string;
    };
  }>;
  color?: string[];
}

// Utility Types

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}