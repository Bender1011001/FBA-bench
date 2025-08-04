import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { FixedSizeList as List } from 'react-window';
import { useSimulationStore } from '../store/simulationStore';
import { useTheme } from '../contexts/ThemeContext';
import type {
  SimulationEvent,
  TickEventPayload,
  SaleEventPayload,
  SetPriceCommandPayload,
  ProductPriceUpdatedPayload,
  CompetitorPricesUpdatedPayload
} from '../types';
import {
  sanitizeText,
  sanitizeTimestamp,
  sanitizeInteger,
  sanitizeExternalData
} from '../utils/sanitization';

// Debounce utility function
const debounce = <T extends (...args: Parameters<T>) => ReturnType<T>>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) & { cancel: () => void } => {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  
  const debounced = (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout);
    }
    
    timeout = setTimeout(() => {
      func(...args);
    }, wait);
  };
  
  debounced.cancel = () => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }
  };
  
  return debounced;
};

interface EventLogProps {
  className?: string;
  maxEvents?: number;
  autoScroll?: boolean;
  showFilters?: boolean;
}


// Format timestamp using centralized sanitization utility
const formatTimestamp = (timestamp: string): string => {
  return sanitizeTimestamp(timestamp);
};

// Type guards for discriminated union
function isTickEvent(event: SimulationEvent): event is (SimulationEvent & { type: 'tick', payload: TickEventPayload }) {
  return event.type === 'tick';
}

function isSaleOccurredEvent(event: SimulationEvent): event is (SimulationEvent & { type: 'sale_occurred', payload: SaleEventPayload }) {
  return event.type === 'sale_occurred';
}

function isSetPriceCommandEvent(event: SimulationEvent): event is (SimulationEvent & { type: 'set_price_command', payload: SetPriceCommandPayload }) {
  return event.type === 'set_price_command';
}

function isProductPriceUpdatedEvent(event: SimulationEvent): event is (SimulationEvent & { type: 'product_price_updated', payload: ProductPriceUpdatedPayload }) {
  return event.type === 'product_price_updated';
}

function isCompetitorPricesUpdatedEvent(event: SimulationEvent): event is (SimulationEvent & { type: 'competitor_prices_updated', payload: CompetitorPricesUpdatedPayload }) {
  return event.type === 'competitor_prices_updated';
}

const getEventIcon = (eventType: string) => {
  switch (eventType) {
    case 'tick':
      return (
        <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
        </svg>
      );
    case 'sale_occurred':
      return (
        <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path d="M8.433 7.418c.155-.103.346-.196.567-.267v1.698a2.305 2.305 0 01-.567-.267C8.07 8.34 8 8.114 8 8c0-.114.07-.34.433-.582zM11 12.849v-1.698c.22.071.412.164.567.267.364.243.433.468.433.582 0 .114-.07.34-.433.582a2.305 2.305 0 01-.567.267z" />
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-3-8a1 1 0 011-1v-1a1 1 0 112 0v1a2.93 2.93 0 01.817.244l.853-.853a1 1 0 111.414 1.414l-.853.853c.218.251.369.54.369.842 0 1.33-1.5 2-3 2v1a1 1 0 11-2 0v-1a2.93 2.93 0 01-.817-.244l-.853.853a1 1 0 11-1.414-1.414l.853-.853C7.131 11.46 7 11.17 7 10.999c0-1.33 1.5-2 3-2z" clipRule="evenodd" />
        </svg>
      );
    case 'set_price_command':
      return (
        <svg className="w-4 h-4 text-purple-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
        </svg>
      );
    case 'product_price_updated':
      return (
        <svg className="w-4 h-4 text-orange-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
        </svg>
      );
    case 'competitor_prices_updated':
      return (
        <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
          <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    default:
      return (
        <svg className="w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
        </svg>
      );
  }
};

const getEventDescription = (event: SimulationEvent): string => {
  try {
    // Sanitize the entire event payload to ensure all data is clean
    const sanitizedEvent = sanitizeExternalData(event);
    
    if (isTickEvent(sanitizedEvent)) {
      const tickNumber = sanitizeInteger(sanitizedEvent.payload.tick_number, 0);
      return `Simulation advanced to tick ${tickNumber}`;
    }
    if (isSaleOccurredEvent(sanitizedEvent)) {
      const quantity = sanitizeInteger(sanitizedEvent.payload.sale.quantity, 0);
      const productAsin = sanitizeText(sanitizedEvent.payload.sale.product_asin) || 'unknown';
      const amount = sanitizeText(sanitizedEvent.payload.sale.sale_price.amount) || '?';
      return `Sale: ${quantity}x ${productAsin} for ${amount}`;
    }
    if (isSetPriceCommandEvent(sanitizedEvent)) {
      const productAsin = sanitizeText(sanitizedEvent.payload.product_asin) || 'unknown';
      const newPrice = sanitizeText(sanitizedEvent.payload.new_price.amount) || '?';
      return `Price command: Set ${productAsin} to ${newPrice}`;
    }
    if (isProductPriceUpdatedEvent(sanitizedEvent)) {
      const productAsin = sanitizeText(sanitizedEvent.payload.product_asin) || 'unknown';
      const oldPrice = sanitizeText(sanitizedEvent.payload.old_price.amount) || '?';
      const newPrice = sanitizeText(sanitizedEvent.payload.new_price.amount) || '?';
      return `Price updated: ${productAsin} from ${oldPrice} to ${newPrice}`;
    }
    if (isCompetitorPricesUpdatedEvent(sanitizedEvent)) {
      const competitorStates = Array.isArray(sanitizedEvent.payload.competitor_states)
        ? sanitizedEvent.payload.competitor_states
        : [];
      const competitorCount = sanitizeInteger(competitorStates.length, 0);
      return `Competitor prices updated (${competitorCount} competitors)`;
    }
    return `Unknown event: ${sanitizeText(sanitizedEvent.type)}`;
  } catch (error) {
    console.error('Error generating event description:', error);
    return '[Event Display Error]';
  }
};

const getEventTypeLabel = (eventType: string): string => {
  switch (eventType) {
    case 'tick':
      return 'Tick';
    case 'sale_occurred':
      return 'Sale';
    case 'set_price_command':
      return 'Command';
    case 'product_price_updated':
      return 'Price Update';
    case 'competitor_prices_updated':
      return 'Competitor Update';
    default:
      return 'Unknown';
  }
};

export const EventLog: React.FC<EventLogProps> = React.memo(({
  className = '',
  maxEvents = 1000,
  autoScroll = true,
  showFilters = true
}) => {
  const eventLog = useSimulationStore((state) => state.simulation.eventLog);
  const { isDarkMode } = useTheme();
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('');
  const listRef = useRef<List>(null);

  // Create debounced search function
  const debouncedSearch = useMemo(
    () => debounce((value: string) => {
      setDebouncedSearchTerm(value);
    }, 300),
    []
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      // Clear any pending debounced calls
      debouncedSearch.cancel?.();
    };
  }, [debouncedSearch]);

  // Get unique event types for filtering
  const eventTypes = useMemo(() =>
    Array.from(new Set(eventLog.map(event => event.type))),
    [eventLog]
  );

  // Filter events based on selected types and search term with memoization
  const filteredEvents = useMemo(() => {
    const filtered = eventLog
      .filter(event => {
        if (selectedTypes.size > 0 && !selectedTypes.has(event.type)) {
          return false;
        }
        if (debouncedSearchTerm && !getEventDescription(event).toLowerCase().includes(debouncedSearchTerm.toLowerCase())) {
          return false;
        }
        return true;
      })
      .slice(0, maxEvents);
    
    return filtered;
  }, [eventLog, selectedTypes, debouncedSearchTerm, maxEvents]);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && listRef.current && filteredEvents.length > 0) {
      listRef.current.scrollToItem(filteredEvents.length - 1);
    }
  }, [filteredEvents.length, autoScroll]);

  // Cleanup function to prevent memory leaks
  useEffect(() => {
    return () => {
      // Clear any pending timeouts or intervals
    };
  }, []);

  const toggleEventType = (eventType: string) => {
    const newSelected = new Set(selectedTypes);
    if (newSelected.has(eventType)) {
      newSelected.delete(eventType);
    } else {
      newSelected.add(eventType);
    }
    setSelectedTypes(newSelected);
  };

  const clearFilters = useCallback(() => {
    setSelectedTypes(new Set());
    setSearchTerm('');
    setDebouncedSearchTerm('');
  }, []);

  return (
    <div className={`${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-lg shadow-sm border ${className}`}>
      {/* Header */}
      <div className={`p-4 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="flex items-center justify-between">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Event Log</h3>
          <div className={`flex items-center space-x-2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            <span>{filteredEvents.length} of {eventLog.length} events</span>
            {(selectedTypes.size > 0 || searchTerm) && (
              <button
                onClick={clearFilters}
                className={isDarkMode ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-800"}
              >
                Clear filters
              </button>
            )}
          </div>
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="mt-4 space-y-3">
            {/* Search */}
            <div>
              <input
                type="text"
                placeholder="Search events..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  debouncedSearch(e.target.value);
                }}
                className={`w-full px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                  isDarkMode
                    ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                    : 'border-gray-300 text-gray-900 placeholder-gray-500'
                }`}
              />
            </div>

            {/* Event type filters */}
            {eventTypes.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {eventTypes.map(eventType => (
                  <button
                    key={eventType}
                    onClick={() => toggleEventType(eventType)}
                    className={`
                      px-3 py-1 text-xs font-medium rounded-full border transition-colors
                      ${selectedTypes.has(eventType)
                        ? isDarkMode
                          ? 'bg-blue-900 text-blue-200 border-blue-700'
                          : 'bg-blue-100 text-blue-800 border-blue-200'
                        : isDarkMode
                          ? 'bg-gray-700 text-gray-300 border-gray-600 hover:bg-gray-600'
                          : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100'
                      }
                    `}
                  >
                    {getEventTypeLabel(eventType)}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Event list with virtual scrolling */}
      <div className="h-96">
        {filteredEvents.length === 0 ? (
          <div className={`flex items-center justify-center h-full ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            <div className="text-center">
              <svg className={`w-12 h-12 mx-auto mb-4 ${isDarkMode ? 'text-gray-600' : 'text-gray-300'}`} fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" clipRule="evenodd" />
                <path fillRule="evenodd" d="M4 5a2 2 0 012-2v1a1 1 0 001 1h6a1 1 0 001-1V3a2 2 0 012 2v6a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
              </svg>
              <p>No events to display</p>
              {(selectedTypes.size > 0 || searchTerm) && (
                <p className="text-sm mt-1">Try adjusting your filters</p>
              )}
            </div>
          </div>
        ) : (
          <List
            ref={listRef}
            width="100%" // Full width of container
            height={384} // h-96 = 24rem = 384px
            itemCount={filteredEvents.length}
            itemSize={120} // Approximate height of each event item
            itemData={{ events: filteredEvents, isDarkMode }}
            className={`divide-y ${isDarkMode ? 'divide-gray-700' : 'divide-gray-100'}`}
            overscanCount={5} // Render 5 items above and below the visible area for smoother scrolling
          >
            {EventItem}
          </List>
        )}
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function to prevent unnecessary re-renders
  return (
    prevProps.className === nextProps.className &&
    prevProps.maxEvents === nextProps.maxEvents &&
    prevProps.autoScroll === nextProps.autoScroll &&
    prevProps.showFilters === nextProps.showFilters
  );
});

// Virtualized event item component
const EventItem: React.FC<{
  index: number;
  style: React.CSSProperties;
  data: { events: SimulationEvent[]; isDarkMode: boolean };
}> = React.memo(({ index, style, data }) => {
  const event = data.events[index];
  const { isDarkMode } = data;
  
  return (
    <div style={style} className={`p-4 ${isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-1">
          {getEventIcon(event.type)}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <p className={`text-sm font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {getEventTypeLabel(event.type)}
            </p>
            <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {formatTimestamp(event.timestamp)}
            </p>
          </div>
          
          <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            {getEventDescription(event)}
          </p>
          
          {/* Additional event details */}
          {isSaleOccurredEvent(event) && (
            <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Buyer: {sanitizeText(event.payload.sale.buyer_id) || 'unknown'}
              {event.payload.sale.competitor_id && ` • Competitor: ${sanitizeText(event.payload.sale.competitor_id) || 'unknown'}`}
            </div>
          )}
          
          {isCompetitorPricesUpdatedEvent(event) && (
            <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {Array.isArray(event.payload.competitor_states) && event.payload.competitor_states.map((comp) => (
                <div key={sanitizeText(comp.id) || Math.random().toString(36).substr(2, 9)}>
                  {sanitizeText(comp.name) || 'Unknown'}: {sanitizeText(comp.current_price?.amount) || '?'}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function to prevent unnecessary re-renders
  const prevEvent = prevProps.data.events[prevProps.index];
  const nextEvent = nextProps.data.events[nextProps.index];
  
  return (
    prevProps.index === nextProps.index &&
    prevProps.style === nextProps.style &&
    prevProps.data.isDarkMode === nextProps.data.isDarkMode &&
    prevEvent?.timestamp === nextEvent?.timestamp &&
    prevEvent?.type === nextEvent?.type
  );
});

export default EventLog;