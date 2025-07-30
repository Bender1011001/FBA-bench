"""
WorldStore service for FBA-Bench v3 multi-agent platform.

Provides centralized, authoritative state management with command arbitration
and conflict resolution. All canonical market state is managed here.
"""

import asyncio
import logging
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass, field

from money import Money
from events import BaseEvent, SetPriceCommand, ProductPriceUpdated, InventoryUpdate
from event_bus import EventBus, get_event_bus


logger = logging.getLogger(__name__)


@dataclass
class ProductState:
    """
    Canonical product state managed by WorldStore.
    
    Contains the authoritative values for all product attributes
    that can be modified by agents.
    """
    asin: str
    price: Money
    last_updated: datetime
    inventory_quantity: int = 0 # New: Current inventory level
    cost_basis: Money = field(default_factory=Money.zero) # New: Average cost basis of existing inventory
    last_agent_id: Optional[str] = None
    last_command_id: Optional[str] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'asin': self.asin,
            'price': str(self.price),
            'inventory_quantity': self.inventory_quantity,
            'cost_basis': str(self.cost_basis),
            'last_updated': self.last_updated.isoformat(),
            'last_agent_id': self.last_agent_id,
            'last_command_id': self.last_command_id,
            'version': self.version,
            'metadata': self.metadata
        }


@dataclass
class CommandArbitrationResult:
    """Result of command arbitration process."""
    accepted: bool
    reason: str
    final_price: Optional[Money] = None
    arbitration_notes: Optional[str] = None


class WorldStore:
    """
    Centralized, authoritative state management service.
    
    The WorldStore is the single source of truth for all canonical market state.
    It processes commands from agents, arbitrates conflicts, and publishes
    authoritative state updates that all other services must respect.
    
    Key Responsibilities:
    - Maintain canonical product state (prices, inventory, etc.)
    - Process SetPriceCommand events from agents
    - Arbitrate conflicts when multiple agents target the same resource
    - Publish ProductPriceUpdated events for state changes
    - Ensure data consistency and integrity
    
    Multi-Agent Principles:
    - No service except WorldStore can modify canonical state
    - All agent actions flow through command-arbitration-event pattern
    - Conflict resolution is transparent and auditable
    - State changes are atomic and immediately propagated
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize WorldStore with empty state.
        
        Args:
            event_bus: EventBus instance for pub/sub communication
        """
        self.event_bus = event_bus or get_event_bus()
        
        # Canonical state storage
        self._product_state: Dict[str, ProductState] = {}
        
        # Command processing state
        self._pending_commands: List[SetPriceCommand] = []
        self._command_history: List[SetPriceCommand] = []
        
        # Arbitration configuration
        self.max_price_change_per_tick = 0.50  # Max 50% price change per tick
        self.min_price_threshold = Money(100)  # Minimum price $1.00
        self.max_price_threshold = Money(100000)  # Maximum price $1000.00
        
        # Statistics
        self.commands_processed = 0
        self.commands_rejected = 0
        self.conflicts_arbitrated = 0
        
        logger.info("WorldStore initialized - ready for multi-agent command processing")
    
    async def start(self):
        """Start the WorldStore service and subscribe to events."""
        await self.event_bus.subscribe('SetPriceCommand', self.handle_set_price_command)
        logger.info("WorldStore started - subscribed to SetPriceCommand events")
    
    async def stop(self):
        """Stop the WorldStore service."""
        logger.info("WorldStore stopped")
    
    # Command Processing
    
    async def handle_set_price_command(self, event: SetPriceCommand):
        """
        Process SetPriceCommand from agents.
        
        Performs arbitration, validation, and state updates.
        Publishes ProductPriceUpdated on successful changes.
        """
        try:
            logger.debug(f"Processing SetPriceCommand: agent={event.agent_id}, asin={event.asin}, price={event.new_price}")
            
            # Arbitrate the command
            result = await self._arbitrate_price_command(event)
            
            if result.accepted:
                # Apply the state change
                await self._apply_price_change(event, result)
                self.commands_processed += 1
                logger.info(f"SetPriceCommand accepted: agent={event.agent_id}, asin={event.asin}, new_price={result.final_price}")
            else:
                # Reject the command
                self.commands_rejected += 1
                logger.warning(f"SetPriceCommand rejected: agent={event.agent_id}, asin={event.asin}, reason={result.reason}")
            
            # Record in history
            self._command_history.append(event)
            
        except Exception as e:
            logger.error(f"Error processing SetPriceCommand: {e}", exc_info=True)
            self.commands_rejected += 1
    
    async def _arbitrate_price_command(self, command: SetPriceCommand) -> CommandArbitrationResult:
        """
        Arbitrate a price change command.
        
        Validation rules:
        1. Price must be within global min/max thresholds
        2. Price change cannot exceed max_price_change_per_tick
        3. Product must exist or be initializable
        4. Agent must be authorized (future enhancement)
        
        Conflict resolution:
        - For now, use simple timestamp-based ordering (first command wins per tick)
        - Future: More sophisticated arbitration based on agent reputation, etc.
        """
        
        # Rule 1: Validate price bounds
        if command.new_price < self.min_price_threshold:
            return CommandArbitrationResult(
                accepted=False,
                reason=f"Price {command.new_price} below minimum threshold {self.min_price_threshold}"
            )
        
        if command.new_price > self.max_price_threshold:
            return CommandArbitrationResult(
                accepted=False,
                reason=f"Price {command.new_price} above maximum threshold {self.max_price_threshold}"
            )
        
        # Get current product state
        current_state = self._product_state.get(command.asin)
        
        if current_state:
            # Rule 2: Validate price change magnitude
            current_price = current_state.price
            price_change_ratio = abs((command.new_price.cents / current_price.cents) - 1.0)
            
            if price_change_ratio > self.max_price_change_per_tick:
                return CommandArbitrationResult(
                    accepted=False,
                    reason=f"Price change {price_change_ratio:.2%} exceeds maximum {self.max_price_change_per_tick:.2%} per tick"
                )
            
            # Rule 3: Check for conflicts (multiple commands for same ASIN in current tick)
            # For now, simple implementation - always accept if passes validation
            # Future: More sophisticated conflict resolution
            
        else:
            # Product doesn't exist - initialize with command price
            logger.info(f"Initializing new product state: asin={command.asin}, price={command.new_price}")
        
        # Command accepted
        return CommandArbitrationResult(
            accepted=True,
            reason="Command validated and accepted",
            final_price=command.new_price,
            arbitration_notes=f"Processed by WorldStore at {datetime.now().isoformat()}"
        )
    
    async def _apply_price_change(self, command: SetPriceCommand, result: CommandArbitrationResult):
        """
        Apply validated price change to canonical state and publish update event.
        """
        asin = command.asin
        new_price = result.final_price
        current_state = self._product_state.get(asin)
        
        # Determine previous price
        previous_price = current_state.price if current_state else Money(2000)  # Default $20.00
        
        # Update canonical state
        if current_state:
            current_state.price = new_price
            current_state.last_updated = datetime.now()
            current_state.last_agent_id = command.agent_id
            current_state.last_command_id = command.event_id
            current_state.version += 1
        else:
            # Create new product state
            self._product_state[asin] = ProductState(
                asin=asin,
                price=new_price,
                last_updated=datetime.now(),
                last_agent_id=command.agent_id,
                last_command_id=command.event_id,
                version=1
            )
        
        # Publish ProductPriceUpdated event
        update_event = ProductPriceUpdated(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            asin=asin,
            new_price=new_price,
            previous_price=previous_price,
            agent_id=command.agent_id,
            command_id=command.event_id,
            arbitration_notes=result.arbitration_notes
        )
        
        await self.event_bus.publish(update_event)
        logger.info(f"Published ProductPriceUpdated: asin={asin}, price={new_price}")
    
    # State Query Interface
    
    def get_product_price(self, asin: str) -> Optional[Money]:
        """Get canonical price for a product."""
        state = self._product_state.get(asin)
        return state.price if state else None
    
    def get_product_state(self, asin: str) -> Optional[ProductState]:
        """Get complete product state."""
        return self._product_state.get(asin)
    
    def get_all_product_states(self) -> Dict[str, ProductState]:
        """Get all product states (read-only copy)."""
        return self._product_state.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WorldStore operational statistics."""
        return {
            'products_managed': len(self._product_state),
            'commands_processed': self.commands_processed,
            'commands_rejected': self.commands_rejected,
            'conflicts_arbitrated': self.conflicts_arbitrated,
            'command_history_size': len(self._command_history)
        }
    
    # Administrative Interface
    
    def initialize_product(self, asin: str, initial_price: Money) -> bool:
        """
        Initialize a product with starting state.
        
        Used during simulation setup to establish baseline state.
        Returns False if product already exists.
        """
        if asin in self._product_state:
            return False
        


@dataclass
class CommandArbitrationResult:
    """Result of command arbitration process."""
    accepted: bool
    reason: str
    final_price: Optional[Money] = None
    arbitration_notes: Optional[str] = None


class WorldStore:
    """
    Centralized, authoritative state management service.
    
    The WorldStore is the single source of truth for all canonical market state.
    It processes commands from agents, arbitrates conflicts, and publishes
    authoritative state updates that all other services must respect.
    
    Key Responsibilities:
    - Maintain canonical product state (prices, inventory, etc.)
    - Process SetPriceCommand events from agents
    - Arbitrate conflicts when multiple agents target the same resource
    - Publish ProductPriceUpdated events for state changes
    - Ensure data consistency and integrity
    
    Multi-Agent Principles:
    - No service except WorldStore can modify canonical state
    - All agent actions flow through command-arbitration-event pattern
    - Conflict resolution is transparent and auditable
    - State changes are atomic and immediately propagated
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize WorldStore with empty state.
        
        Args:
            event_bus: EventBus instance for pub/sub communication
        """
        self.event_bus = event_bus or get_event_bus()
        
        # Canonical state storage
        self._product_state: Dict[str, ProductState] = {}
        
        # Command processing state
        self._pending_commands: List[SetPriceCommand] = []
        self._command_history: List[SetPriceCommand] = []
        
        # Arbitration configuration
        self.max_price_change_per_tick = 0.50  # Max 50% price change per tick
        self.min_price_threshold = Money(100)  # Minimum price $1.00
        self.max_price_threshold = Money(100000)  # Maximum price $1000.00
        
        # Statistics
        self.commands_processed = 0
        self.commands_rejected = 0
        self.conflicts_arbitrated = 0
        
        logger.info("WorldStore initialized - ready for multi-agent command processing")
    
    async def start(self):
        """Start the WorldStore service and subscribe to events."""
        await self.event_bus.subscribe('SetPriceCommand', self.handle_set_price_command)
        await self.event_bus.subscribe('InventoryUpdate', self._handle_inventory_update) # Subscribe to inventory
        logger.info("WorldStore started - subscribed to SetPriceCommand events")
    
    async def stop(self):
        """Stop the WorldStore service."""
        logger.info("WorldStore stopped")
    
    # Command Processing
    
    async def handle_set_price_command(self, event: SetPriceCommand):
        """
        Process SetPriceCommand from agents.
        
        Performs arbitration, validation, and state updates.
        Publishes ProductPriceUpdated on successful changes.
        """
        try:
            logger.debug(f"Processing SetPriceCommand: agent={event.agent_id}, asin={event.asin}, price={event.new_price}")
            
            # Arbitrate the command
            result = await self._arbitrate_price_command(event)
            
            if result.accepted:
                # Apply the state change
                await self._apply_price_change(event, result)
                self.commands_processed += 1
                logger.info(f"SetPriceCommand accepted: agent={event.agent_id}, asin={event.asin}, new_price={result.final_price}")
            else:
                # Reject the command
                self.commands_rejected += 1
                logger.warning(f"SetPriceCommand rejected: agent={event.agent_id}, asin={event.asin}, reason={result.reason}")
            
            # Record in history
            self._command_history.append(event)
            
        except Exception as e:
            logger.error(f"Error processing SetPriceCommand: {e}", exc_info=True)
            self.commands_rejected += 1
    
    async def _handle_inventory_update(self, event: InventoryUpdate):
        """Handle InventoryUpdate events to update canonical inventory state."""
        try:
            asin = event.asin
            # Assume event provides new quantity and possibly updated cost basis
            new_quantity = event.new_quantity
            cost_basis = event.cost_basis # This should be provided by the inventory update source
            
            current_state = self._product_state.get(asin)
            if current_state:
                current_state.inventory_quantity = new_quantity
                current_state.cost_basis = cost_basis if cost_basis else current_state.cost_basis
                current_state.last_updated = datetime.now()
                logger.debug(f"Updated inventory for {asin}: new_quantity={new_quantity}, cost_basis={cost_basis}")
            else:
                # If product state doesn't exist, initialize it with inventory info
                # This depends on whether inventory can arrive before a price is set.
                # For now, if no product state, we create a minimal one.
                self._product_state[asin] = ProductState(
                    asin=asin,
                    price=Money.zero(), # Default price if not set yet
                    inventory_quantity=new_quantity,
                    cost_basis=cost_basis if cost_basis else Money.zero(),
                    last_updated=datetime.now(),
                    last_agent_id="system_inventory",
                    last_command_id=event.event_id,
                    version=1
                )
                logger.info(f"Initialized product state with inventory: asin={asin}, quantity={new_quantity}, cost={cost_basis}")

        except Exception as e:
            logger.error(f"Error handling InventoryUpdate event {event.event_id}: {e}", exc_info=True)

    async def _arbitrate_price_command(self, command: SetPriceCommand) -> CommandArbitrationResult:
        """
        Arbitrate a price change command.
        
        Validation rules:
        1. Price must be within global min/max thresholds
        2. Price change cannot exceed max_price_change_per_tick
        3. Product must exist or be initializable
        4. Agent must be authorized (future enhancement)
        
        Conflict resolution:
        - For now, use simple timestamp-based ordering (first command wins per tick)
        - Future: More sophisticated arbitration based on agent reputation, etc.
        """
        
        # Rule 1: Validate price bounds
        if command.new_price < self.min_price_threshold:
            return CommandArbitrationResult(
                accepted=False,
                reason=f"Price {command.new_price} below minimum threshold {self.min_price_threshold}"
            )
        
        if command.new_price > self.max_price_threshold:
            return CommandArbitrationResult(
                accepted=False,
                reason=f"Price {command.new_price} above maximum threshold {self.max_price_threshold}"
            )
        
        # Get current product state
        current_state = self._product_state.get(command.asin)
        
        if current_state:
            # Rule 2: Validate price change magnitude
            current_price = current_state.price
            price_change_ratio = abs((command.new_price.cents / current_price.cents) - 1.0)
            
            if price_change_ratio > self.max_price_change_per_tick:
                return CommandArbitrationResult(
                    accepted=False,
                    reason=f"Price change {price_change_ratio:.2%} exceeds maximum {self.max_price_change_per_tick:.2%} per tick"
                )
            
            # Rule 3: Check for conflicts (multiple commands for same ASIN in current tick)
            # For now, simple implementation - always accept if passes validation
            # Future: More sophisticated conflict resolution
            
        else:
            # Product doesn't exist - initialize with command price
            logger.info(f"Initializing new product state: asin={command.asin}, price={command.new_price}")
        
        # Command accepted
        return CommandArbitrationResult(
            accepted=True,
            reason="Command validated and accepted",
            final_price=command.new_price,
            arbitration_notes=f"Processed by WorldStore at {datetime.now().isoformat()}"
        )
    
    async def _apply_price_change(self, command: SetPriceCommand, result: CommandArbitrationResult):
        """
        Apply validated price change to canonical state and publish update event.
        """
        asin = command.asin
        new_price = result.final_price
        current_state = self._product_state.get(asin)
        
        # Determine previous price
        previous_price = current_state.price if current_state else Money(2000)  # Default $20.00
        
        # Update canonical state
        if current_state:
            current_state.price = new_price
            current_state.last_updated = datetime.now()
            current_state.last_agent_id = command.agent_id
            current_state.last_command_id = command.event_id
            current_state.version += 1
        else:
            # Create new product state, inheriting inventory/cost if it was already initialized via InventoryUpdate
            existing_inventory = 0
            existing_cost_basis = Money.zero()
            if asin in self._product_state: # Check if a stub was created by inventory update
                existing_inventory = self._product_state[asin].inventory_quantity
                existing_cost_basis = self._product_state[asin].cost_basis

            self._product_state[asin] = ProductState(
                asin=asin,
                price=new_price,
                inventory_quantity=existing_inventory, # Preserve existing inventory
                cost_basis=existing_cost_basis, # Preserve existing cost basis
                last_updated=datetime.now(),
                last_agent_id=command.agent_id,
                last_command_id=command.event_id,
                version=1
            )
        
        # Publish ProductPriceUpdated event
        update_event = ProductPriceUpdated(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            asin=asin,
            new_price=new_price,
            previous_price=previous_price,
            agent_id=command.agent_id,
            command_id=command.event_id,
            arbitration_notes=result.arbitration_notes
        )
        
        await self.event_bus.publish(update_event)
        logger.info(f"Published ProductPriceUpdated: asin={asin}, price={new_price}")
    
    # State Query Interface
    
    def get_product_price(self, asin: str) -> Optional[Money]:
        """Get canonical price for a product."""
        state = self._product_state.get(asin)
        return state.price if state else None
    
    def get_product_state(self, asin: str) -> Optional[ProductState]:
        """Get complete product state."""
        return self._product_state.get(asin)
    
    def get_all_product_states(self) -> Dict[str, ProductState]:
        """Get all product states (read-only copy)."""
        return self._product_state.copy()

    def get_product_inventory_quantity(self, asin: str) -> int:
        """Get current inventory quantity for a product."""
        state = self._product_state.get(asin)
        return state.inventory_quantity if state else 0

    def get_product_cost_basis(self, asin: str) -> Money:
        """Get current cost basis for a product."""
        state = self._product_state.get(asin)
        return state.cost_basis if state else Money.zero()

    def get_statistics(self) -> Dict[str, Any]:
        """Get WorldStore operational statistics."""
        return {
            'products_managed': len(self._product_state),
            'commands_processed': self.commands_processed,
            'commands_rejected': self.commands_rejected,
            'conflicts_arbitrated': self.conflicts_arbitrated,
            'command_history_size': len(self._command_history)
        }
    
    # Administrative Interface
    
    def initialize_product(self, asin: str, initial_price: Money, initial_inventory: int = 0, initial_cost_basis: Money = Money.zero()) -> bool:
        """
        Initialize a product with starting state.
        
        Used during simulation setup to establish baseline state.
        Returns False if product already exists.
        """
        if asin in self._product_state:
            return False
        
        self._product_state[asin] = ProductState(
            asin=asin,
            price=initial_price,
            inventory_quantity=initial_inventory,
            cost_basis=initial_cost_basis,
            last_updated=datetime.now(),
            last_agent_id="system",
            last_command_id="initialization",
            version=1
        )
        
        logger.info(f"Initialized product state: asin={asin}, price={initial_price}, inventory={initial_inventory}, cost={initial_cost_basis}")
        return True
    
    def reset_state(self):
        """Reset all state - used for testing."""
        self._product_state.clear()
        self._pending_commands.clear()
        self._command_history.clear()
        self.commands_processed = 0
        self.commands_rejected = 0
        self.conflicts_arbitrated = 0
        logger.info("WorldStore state reset")


# Global instance management
_world_store_instance: Optional[WorldStore] = None


def get_world_store() -> WorldStore:
    """Get the global WorldStore instance."""
    global _world_store_instance
    if _world_store_instance is None:
        _world_store_instance = WorldStore()
    return _world_store_instance


def set_world_store(world_store: WorldStore):
    """Set the global WorldStore instance."""
    global _world_store_instance
    _world_store_instance = world_store