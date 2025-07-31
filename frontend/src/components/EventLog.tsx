import React, { useEffect, useRef, useState } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import type {
  SimulationEvent,
  TickEventPayload,
  SaleEventPayload,
  SetPriceCommandPayload,
  ProductPriceUpdatedPayload,
  CompetitorPricesUpdatedPayload
} from '../types';

interface EventLogProps {
  className?: string;
  maxEvents?: number;
  autoScroll?: boolean;
  showFilters?: boolean;
}

const formatTimestamp = (timestamp: string): string => {
  try {
    return new Date(timestamp).toLocaleTimeString();
  } catch {
    return 'Invalid time';
  }
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
  if (isTickEvent(event)) {
    return `Simulation advanced to tick ${event.payload.tick_number}`;
  }
  if (isSaleOccurredEvent(event)) {
    return `Sale: ${event.payload.sale.quantity}x ${event.payload.sale.product_asin} for ${event.payload.sale.sale_price.amount}`;
  }
  if (isSetPriceCommandEvent(event)) {
    return `Price command: Set ${event.payload.product_asin} to ${event.payload.new_price.amount}`;
  }
  if (isProductPriceUpdatedEvent(event)) {
    return `Price updated: ${event.payload.product_asin} from ${event.payload.old_price.amount} to ${event.payload.new_price.amount}`;
  }
  if (isCompetitorPricesUpdatedEvent(event)) {
    return `Competitor prices updated (${event.payload.competitor_states.length} competitors)`;
  }
  return `Unknown event: ${event.type}`;
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

export const EventLog: React.FC<EventLogProps> = ({
  className = '',
  maxEvents = 100,
  autoScroll = true,
  showFilters = true
}) => {
  const eventLog = useSimulationStore((state) => state.simulation.eventLog);
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Get unique event types for filtering
  const eventTypes = Array.from(new Set(eventLog.map(event => event.type)));

  // Filter events based on selected types and search term
  const filteredEvents = eventLog
    .filter(event => {
      if (selectedTypes.size > 0 && !selectedTypes.has(event.type)) {
        return false;
      }
      if (searchTerm && !getEventDescription(event).toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      return true;
    })
    .slice(0, maxEvents);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [eventLog.length, autoScroll]);

  const toggleEventType = (eventType: string) => {
    const newSelected = new Set(selectedTypes);
    if (newSelected.has(eventType)) {
      newSelected.delete(eventType);
    } else {
      newSelected.add(eventType);
    }
    setSelectedTypes(newSelected);
  };

  const clearFilters = () => {
    setSelectedTypes(new Set());
    setSearchTerm('');
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Event Log</h3>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <span>{filteredEvents.length} of {eventLog.length} events</span>
            {(selectedTypes.size > 0 || searchTerm) && (
              <button
                onClick={clearFilters}
                className="text-blue-600 hover:text-blue-800"
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
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
                        ? 'bg-blue-100 text-blue-800 border-blue-200'
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

      {/* Event list */}
      <div
        ref={scrollRef}
        className="h-96 overflow-y-auto"
      >
        {filteredEvents.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
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
          <div className="divide-y divide-gray-100">
            {filteredEvents.map((event, index) => (
              <div key={`${event.timestamp}-${index}`} className="p-4 hover:bg-gray-50">
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {getEventIcon(event.type)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-gray-900">
                        {getEventTypeLabel(event.type)}
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatTimestamp(event.timestamp)}
                      </p>
                    </div>
                    
                    <p className="text-sm text-gray-600 mt-1">
                      {getEventDescription(event)}
                    </p>
                    
                    {/* Additional event details */}
                    {isSaleOccurredEvent(event) && (
                      <div className="mt-2 text-xs text-gray-500">
                        Buyer: {event.payload.sale.buyer_id}
                        {event.payload.sale.competitor_id && ` • Competitor: ${event.payload.sale.competitor_id}`}
                      </div>
                    )}
                    
                    {isCompetitorPricesUpdatedEvent(event) && (
                      <div className="mt-2 text-xs text-gray-500">
                        {event.payload.competitor_states.map((comp) => (
                          <div key={comp.id}>
                            {comp.name}: {comp.current_price.amount}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default EventLog;