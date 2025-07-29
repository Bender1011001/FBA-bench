// Core simulation types matching backend Pydantic models

export interface Money {
  amount: string; // String representation from Python Money class
}

export interface CompetitorState {
  competitor_id: string;
  current_price: Money;
  last_updated: string; // ISO timestamp
}

export interface SimulationSnapshot {
  current_tick: number;
  total_sales: Money;
  our_product_price: Money;
  competitor_states: CompetitorState[];
  recent_sales: SalesResult[];
  trust_score: number;
  timestamp: string; // ISO timestamp
}

export interface Product {
  asin: string;
  title: string;
  category: string;
  brand: string;
  price: Money;
  rating: number;
  review_count: number;
  is_prime: boolean;
  seller_name: string;
  listing_url: string;
}

export interface Competitor {
  competitor_id: string;
  name: string;
  products: Product[];
  trust_score: number;
}

export interface SalesResult {
  timestamp: string; // ISO timestamp
  product_asin: string;
  sale_price: Money;
  quantity: number;
  buyer_id: string;
  competitor_id?: string;
}

// Event types matching backend events
export interface TickEvent {
  type: 'tick';
  tick_number: number;
  timestamp: string;
}

export interface SaleOccurred {
  type: 'sale_occurred';
  sale: SalesResult;
  timestamp: string;
}

export interface SetPriceCommand {
  type: 'set_price_command';
  product_asin: string;
  new_price: Money;
  timestamp: string;
}

export interface ProductPriceUpdated {
  type: 'product_price_updated';
  product_asin: string;
  old_price: Money;
  new_price: Money;
  timestamp: string;
}

export interface CompetitorPricesUpdated {
  type: 'competitor_prices_updated';
  competitor_states: CompetitorState[];
  timestamp: string;
}

// Union type for all possible events
export type SimulationEvent = 
  | TickEvent
  | SaleOccurred
  | SetPriceCommand
  | ProductPriceUpdated
  | CompetitorPricesUpdated;

// WebSocket message types
export interface WebSocketMessage {
  type: string;
  data: SimulationEvent;
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// Dashboard-specific types
export interface DashboardMetric {
  label: string;
  value: string | number;
  trend?: 'up' | 'down' | 'neutral';
  formatType?: 'currency' | 'percentage' | 'number';
}

export interface ConnectionStatus {
  connected: boolean;
  lastHeartbeat?: string;
  reconnectAttempts: number;
}