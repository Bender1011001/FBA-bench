"""
WorldStore service for FBA-Bench v3 multi-agent platform.

Provides centralized, authoritative state management with command arbitration
and conflict resolution. All canonical market state is managed here.
"""

import asyncio
import logging
import json
import os
import uuid
from typing import Dict, Optional, Any, List, Protocol
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from money import Money
from events import BaseEvent, SetPriceCommand, ProductPriceUpdated, InventoryUpdate, WorldStateSnapshotEvent
from event_bus import EventBus, get_event_bus


logger = logging.getLogger(__name__)


# --- Persistence Layer for WorldStore ---

class PersistenceBackend(Protocol):
    """
    Abstract interface for WorldStore persistence backends.
    Allows saving and loading the canonical state.
    """
    async def save_state(self, state: Dict[str, Any], timestamp: datetime, tick: Optional[int] = None) -> str:
        """Saves a snapshot of the current world state."""
        ...

    async def load_latest_state(self) -> Optional[Dict[str, Any]]:
        """Loads the most recent world state snapshot."""
        ...

    async def load_state_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Loads a specific world state snapshot by its ID."""
        ...

    async def initialize(self):
        """Initializes the persistence backend (e.g., connects to DB)."""
        ...
    
    async def shutdown(self):
        """Shuts down the persistence backend."""
        ...

class InMemoryStorageBackend:
    """
    A simple in-memory storage backend for WorldStore state snapshots.
    NOT FOR PRODUCTION USE - primarily for testing and demonstration.
    """
    def __init__(self):
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._latest_snapshot_id: Optional[str] = None
        logger.info("InMemoryStorageBackend initialized.")

    async def initialize(self):
        logger.info("InMemoryStorageBackend initialized - no external connection needed.")
        pass

    async def shutdown(self):
        logger.info("InMemoryStorageBackend shut down.")
        pass

    async def save_state(self, state: Dict[str, Any], timestamp: datetime, tick: Optional[int] = None) -> str:
        snapshot_id = f"snapshot_{str(uuid.uuid4())}"
        snapshot_data = {
            "id": snapshot_id,
            "timestamp": timestamp.isoformat(),
            "tick": tick,
            "state": state
        }
        self._snapshots[snapshot_id] = snapshot_data
        self._latest_snapshot_id = snapshot_id
        logger.info(f"Saved in-memory state snapshot: {snapshot_id} at tick {tick}")
        return snapshot_id

    async def load_latest_state(self) -> Optional[Dict[str, Any]]:
        if self._latest_snapshot_id and self._latest_snapshot_id in self._snapshots:
            logger.info(f"Loading latest in-memory state snapshot: {self._latest_snapshot_id}")
            return self._snapshots[self._latest_snapshot_id]["state"]
        logger.info("No latest in-memory state snapshot found.")
        return None

    async def load_state_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        snapshot_data = self._snapshots.get(snapshot_id)
        if snapshot_data:
            logger.info(f"Loading in-memory state snapshot by ID: {snapshot_id}")
            return snapshot_data["state"]
        logger.warning(f"In-memory state snapshot {snapshot_id} not found.")
        return None

class JsonFileStorageBackend:
    """
    A file-based storage backend for WorldStore state snapshots using JSON files.
    More suitable for production than InMemoryStorageBackend for single-instance setups.
    """
    def __init__(self, snapshot_dir: str = "world_store_snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self._latest_snapshot_id: Optional[str] = None
        logger.info(f"JsonFileStorageBackend initialized with directory: {self.snapshot_dir}")

    async def initialize(self):
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"JsonFileStorageBackend ensured directory exists: {self.snapshot_dir}")
            # Attempt to find the latest snapshot ID on startup
            await self._find_latest_snapshot_id()
        except Exception as e:
            logger.error(f"Failed to initialize JsonFileStorageBackend directory: {e}", exc_info=True)
            raise

    async def shutdown(self):
        logger.info("JsonFileStorageBackend shut down.")
        pass

    def _get_snapshot_path(self, snapshot_id: str) -> Path:
        return self.snapshot_dir / f"{snapshot_id}.json"

    async def _find_latest_snapshot_id(self):
        """Finds the ID of the latest snapshot based on modification time."""
        latest_mtime = 0
        latest_id = None
        try:
            for file_path in self.snapshot_dir.glob("*.json"):
                if file_path.is_file():
                    mtime = file_path.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_id = file_path.stem # filename without extension
            self._latest_snapshot_id = latest_id
            if latest_id:
                logger.info(f"JsonFileStorageBackend identified latest snapshot as: {latest_id}")
            else:
                logger.info("JsonFileStorageBackend found no existing snapshots.")
        except Exception as e:
            logger.error(f"Error finding latest snapshot ID in JsonFileStorageBackend: {e}", exc_info=True)


    async def save_state(self, state: Dict[str, Any], timestamp: datetime, tick: Optional[int] = None) -> str:
        snapshot_id = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{tick or 'N'}"
        snapshot_data = {
            "id": snapshot_id,
            "timestamp": timestamp.isoformat(),
            "tick": tick,
            "state": state
        }
        snapshot_path = self._get_snapshot_path(snapshot_id)
        try:
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=4)
            self._latest_snapshot_id = snapshot_id
            logger.info(f"Saved JSON state snapshot: {snapshot_id} to {snapshot_path}")
            return snapshot_id
        except Exception as e:
            logger.error(f"Failed to save JSON state snapshot {snapshot_id}: {e}", exc_info=True)
            raise

    async def load_latest_state(self) -> Optional[Dict[str, Any]]:
        if not self._latest_snapshot_id:
            await self._find_latest_snapshot_id() # Ensure we have tried to find it
        
        if self._latest_snapshot_id:
            return await self.load_state_by_id(self._latest_snapshot_id)
        logger.info("No latest JSON state snapshot found.")
        return None

    async def load_state_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        snapshot_path = self._get_snapshot_path(snapshot_id)
        try:
            if not snapshot_path.exists():
                logger.warning(f"JSON state snapshot {snapshot_id} not found at {snapshot_path}.")
                return None
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            logger.info(f"Loaded JSON state snapshot by ID: {snapshot_id} from {snapshot_path}")
            return snapshot_data.get("state")
        except Exception as e:
            logger.error(f"Failed to load JSON state snapshot {snapshot_id}: {e}", exc_info=True)
            return None

# --- End Persistence Layer ---


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
    inventory_quantity: int = 0 # Current inventory level
    cost_basis: Money = field(default_factory=Money.zero) # Average cost basis of existing inventory
    last_agent_id: Optional[str] = None
    last_command_id: Optional[str] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging/serialization."""
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductState":
        """Create ProductState from a dictionary."""
        return cls(
            asin=data['asin'],
            price=Money(data['price']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            inventory_quantity=data.get('inventory_quantity', 0),
            cost_basis=Money(data.get('cost_basis', "$0.00")),
            last_agent_id=data.get('last_agent_id'),
            last_command_id=data.get('last_command_id'),
            version=data.get('version', 1),
            metadata=data.get('metadata', {})
        )


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
    - Publish ProductPriceUpdated events for state changes, and WorldStateSnapshotEvent
    - Ensure data consistency and integrity
    - Integrate with a persistence backend for long-horizon simulations.
    
    Multi-Agent Principles:
    - No service except WorldStore can modify canonical state
    - All agent actions flow through command-arbitration-event pattern
    - Conflict resolution is transparent and auditable
    - State changes are atomic and immediately propagated
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, storage_backend: Optional[PersistenceBackend] = None):
        """
        Initialize WorldStore with empty state.
        
        Args:
            event_bus: EventBus instance for pub/sub communication
            storage_backend: Optional backend for persisting state snapshots.
                            If None, defaults to InMemoryStorageBackend.
        """
        self.event_bus = event_bus or get_event_bus()
        self.storage_backend = storage_backend if storage_backend is not None else InMemoryStorageBackend()
        
        # Canonical state storage
        self._product_state: Dict[str, ProductState] = {}
        
        # Command processing state for enhanced conflict resolution
        self._pending_commands_by_asin_this_tick: Dict[str, List[SetPriceCommand]] = {}
        self._processed_command_ids_this_tick: set[str] = set() # To avoid re-processing exact duplicates within a tick
        self._command_history: List[SetPriceCommand] = []
        
        # Arbitration configuration
        self.max_price_change_per_tick = 0.50  # Max 50% price change per tick
        self.min_price_threshold = Money(100)  # Minimum price $1.00
        self.max_price_threshold = Money(100000)  # Maximum price $1000.00
        
        # Statistics
        self.commands_processed = 0
        self.commands_rejected = 0
        self.conflicts_arbitrated = 0 # Counts when a command is rejected due to a prior accepted one for same ASIN in the tick
        self.snapshots_saved = 0
        
        logger.info("WorldStore initialized - ready for multi-agent command processing")
    
    async def start(self):
        """Start the WorldStore service, subscribe to events, and initialize storage."""
        await self.event_bus.subscribe('SetPriceCommand', self.handle_set_price_command)
        await self.event_bus.subscribe('InventoryUpdate', self._handle_inventory_update)
        await self.event_bus.subscribe('TickEvent', self._handle_tick_event_for_snapshot) # To clear per-tick state
        await self.event_bus.subscribe('TickEndEvent', self._handle_tick_end_event) # To clear per-tick state

        if self.storage_backend:
            await self.storage_backend.initialize()
            loaded_state = await self.storage_backend.load_latest_state()
            if loaded_state:
                self._load_state_from_dict(loaded_state)
                logger.info(f"WorldStore loaded state from persistence backend. Current products: {len(self._product_state)}")
            else:
                logger.info("No existing state found in persistence backend.")
        
        logger.info("WorldStore started - subscribed to SetPriceCommand, InventoryUpdate, TickEvent, TickEndEvent events")
    
    async def stop(self):
        """Stop the WorldStore service and shut down storage backend."""
        if self.storage_backend:
            await self.storage_backend.shutdown()
        logger.info("WorldStore stopped")

    async def _handle_tick_event_for_snapshot(self, event: Any): # Using Any to avoid circular import
        """Handles TickEvents to trigger periodic state snapshots."""
        if event.tick_number % 100 == 0: # Example: save every 100 ticks
            logger.info(f"Tick {event.tick_number}: Triggering WorldStore state snapshot.")
            await self.save_state_snapshot(tick=event.tick_number)

    async def _handle_tick_end_event(self, event: Any): # Using Any to avoid circular import
        """Handles TickEndEvents to clear per-tick command tracking."""
        logger.debug(f"Tick {event.tick_number} ended. Clearing per-tick command tracking.")
        self._pending_commands_by_asin_this_tick.clear()
        self._processed_command_ids_this_tick.clear()
    
    # Command Processing
    
    async def handle_set_price_command(self, event: SetPriceCommand):
        """
        Process SetPriceCommand from agents.
        
        Performs arbitration, validation, and state updates.
        Publishes ProductPriceUpdated on successful changes.
        """
        try:
            logger.debug(f"Processing SetPriceCommand: agent={event.agent_id}, asin={event.asin}, price={event.new_price}, command_id={event.event_id}")

            if event.event_id in self._processed_command_ids_this_tick:
                logger.warning(f"Duplicate SetPriceCommand ignored: command_id={event.event_id} from agent={event.agent_id} for asin={event.asin}")
                self.commands_rejected += 1
                return

            # Arbitrate the command
            result = await self._arbitrate_price_command(event)
            
            if result.accepted:
                # Apply the state change
                await self._apply_price_change(event, result)
                self.commands_processed += 1
                # Mark this command as processed for this tick
                self._processed_command_ids_this_tick.add(event.event_id)
                # Add to pending commands for this ASIN for this tick to block subsequent ones
                if event.asin not in self._pending_commands_by_asin_this_tick:
                    self._pending_commands_by_asin_this_tick[event.asin] = []
                self._pending_commands_by_asin_this_tick[event.asin].append(event)

                logger.info(f"SetPriceCommand accepted: agent={event.agent_id}, asin={event.asin}, new_price={result.final_price}, command_id={event.event_id}")
            else:
                # Reject the command
                self.commands_rejected += 1
                if "already accepted for this ASIN in the current tick" in result.reason:
                    self.conflicts_arbitrated +=1
                logger.warning(f"SetPriceCommand rejected: agent={event.agent_id}, asin={event.asin}, reason={result.reason}, command_id={event.event_id}")
            
            # Record in history
            self._command_history.append(event)
            
        except Exception as e:
            logger.error(f"Error processing SetPriceCommand {event.event_id}: {e}", exc_info=True)
            self.commands_rejected += 1
    
    async def _handle_inventory_update(self, event: InventoryUpdate):
        """Handle InventoryUpdate events to update canonical inventory state."""
        try:
            asin = event.asin
            new_quantity = event.new_quantity
            cost_basis = event.cost_basis 

            current_state = self._product_state.get(asin)
            if current_state:
                current_state.inventory_quantity = new_quantity
                current_state.cost_basis = cost_basis if cost_basis else current_state.cost_basis
                current_state.last_updated = datetime.now()
                logger.debug(f"Updated inventory for {asin}: new_quantity={new_quantity}, cost_basis={cost_basis}")
            else:
                self._product_state[asin] = ProductState(
                    asin=asin,
                    price=Money.zero(), 
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
        Arbitrate a price change command with enhanced conflict resolution.
        
        Validation rules:
        1. Price must be within global min/max thresholds
        2. Price change cannot exceed max_price_change_per_tick
        3. Product must exist or be initializable
        
        Conflict resolution:
        - If another command for the same ASIN has already been accepted in this tick, reject.
        - Commands for the same ASIN within the same tick are processed in order of arrival (timestamp/event_id).
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
        
        current_state = self._product_state.get(command.asin)
        
        if current_state:
            # Rule 2: Validate price change magnitude
            current_price = current_state.price
            # Avoid division by zero if current_price is zero, though unlikely for Money type unless explicitly set.
            if current_price.cents > 0:
                price_change_ratio = abs((command.new_price.cents / current_price.cents) - 1.0)
                if price_change_ratio > self.max_price_change_per_tick:
                    return CommandArbitrationResult(
                        accepted=False,
                        reason=f"Price change {price_change_ratio:.2%} exceeds maximum {self.max_price_change_per_tick:.2%} per tick"
                    )
            
            # Rule 3: Enhanced Conflict Resolution
            # Check if a command for this ASIN has already been processed and accepted in this tick
            if command.asin in self._pending_commands_by_asin_this_tick and \
               self._pending_commands_by_asin_this_tick[command.asin]:
                # There's at least one pending/accepted command for this ASIN this tick.
                # The current design processes sequentially, so if it's in the list, it's been accepted.
                # We reject subsequent ones for the same ASIN in the same tick.
                # The `handle_set_price_command` adds to this list *after* successful arbitration and application.
                # So, if it's already here, it means another command for this ASIN was processed first.
                # This check effectively means "only one price change per ASIN per tick".
                # If more fine-grained ordering (e.g., by timestamp) is needed beyond "first come, first served"
                # by the async event loop, the logic here would need to queue and sort.
                # For now, this simple check prevents multiple updates to the same ASIN within one tick.
                # The `handle_set_price_command` adds to `_pending_commands_by_asin_this_tick` *after* success.
                # So, if we find it here during a new call, it means a previous one succeeded.
                 # This check is now effectively: "has a command for this ASIN already been accepted in this tick?"
                # The list `_pending_commands_by_asin_this_tick[command.asin]` holds commands that were accepted.
                if self._pending_commands_by_asin_this_tick[command.asin]: # If the list is not empty
                    return CommandArbitrationResult(
                        accepted=False,
                        reason=f"Another price command for ASIN {command.asin} was already accepted for this tick."
                    )
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
        
        previous_price = current_state.price if current_state else Money(2000)  # Default $20.00
        
        if current_state:
            current_state.price = new_price
            current_state.last_updated = datetime.now()
            current_state.last_agent_id = command.agent_id
            current_state.last_command_id = command.event_id
            current_state.version += 1
        else:
            existing_inventory = 0
            existing_cost_basis = Money.zero()
            # Check if a stub was created by inventory update before price was set
            if asin in self._product_state: 
                existing_inventory = self._product_state[asin].inventory_quantity
                existing_cost_basis = self._product_state[asin].cost_basis

            self._product_state[asin] = ProductState(
                asin=asin,
                price=new_price,
                inventory_quantity=existing_inventory, 
                cost_basis=existing_cost_basis, 
                last_updated=datetime.now(),
                last_agent_id=command.agent_id,
                last_command_id=command.event_id,
                version=1
            )
        
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

    # --- Marketing visibility (effect) helpers ---

    def get_marketing_visibility(self, asin: str) -> float:
        """
        Return current marketing visibility multiplier for an ASIN.
        1.0 = neutral baseline, >1.0 increases demand proportionally.
        """
        state = self._product_state.get(asin)
        if not state:
            return 1.0
        try:
            vis = float(state.metadata.get("marketing_visibility", 1.0))
        except Exception:
            vis = 1.0
        # Bound visibility to a reasonable range [0.1, 5.0] to avoid instabilities
        return max(0.1, min(5.0, vis))

    def set_marketing_visibility(self, asin: str, visibility: float) -> None:
        """
        Set marketing visibility multiplier for an ASIN in canonical state.
        Bounds value to [0.1, 5.0] and updates metadata.
        """
        v = max(0.1, min(5.0, float(visibility)))
        state = self._product_state.get(asin)
        if not state:
            # Initialize a stub product with zero price/inventory if it doesn't exist yet
            self._product_state[asin] = ProductState(
                asin=asin,
                price=Money.zero(),
                inventory_quantity=0,
                cost_basis=Money.zero(),
                last_updated=datetime.now(),
                last_agent_id="system_marketing",
                last_command_id="marketing_visibility_init",
                version=1,
                metadata={"marketing_visibility": v},
            )
            return
        state.metadata["marketing_visibility"] = v
        state.last_updated = datetime.now()
    def get_statistics(self) -> Dict[str, Any]:
        """Get WorldStore operational statistics."""
        return {
            'products_managed': len(self._product_state),
            'commands_processed': self.commands_processed,
            'commands_rejected': self.commands_rejected,
            'conflicts_arbitrated': self.conflicts_arbitrated,
            'command_history_size': len(self._command_history),
            'snapshots_saved': self.snapshots_saved,
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
    
    async def save_state_snapshot(self, tick: Optional[int] = None) -> Optional[str]:
        """
        Saves a snapshot of the current WorldStore state using the configured backend.
        Publishes a WorldStateSnapshotEvent if successful.
        """
        if not self.storage_backend:
            logger.warning("No storage backend configured for WorldStore, cannot save snapshot.")
            return None
        
        serializable_state = {
            asin: product_state.to_dict()
            for asin, product_state in self._product_state.items()
        }
        
        timestamp = datetime.now()
        snapshot_id = await self.storage_backend.save_state(serializable_state, timestamp, tick)
        self.snapshots_saved += 1
        
        snapshot_event = WorldStateSnapshotEvent(
            event_id=f"world_state_snapshot_{snapshot_id}",
            timestamp=timestamp,
            snapshot_id=snapshot_id,
            tick_number=tick,
            product_count=len(self._product_state),
            summary_metrics={
                "total_products": len(self._product_state)
            }
        )
        await self.event_bus.publish(snapshot_event)
        logger.info(f"WorldStore state snapshot '{snapshot_id}' saved successfully.")
        return snapshot_id

    async def load_state_snapshot(self, snapshot_id: Optional[str] = None) -> bool:
        """
        Loads a world state snapshot from the configured backend.
        If snapshot_id is None, loads the latest state.
        Returns True on success, False on failure.
        """
        if not self.storage_backend:
            logger.warning("No storage backend configured for WorldStore, cannot load snapshot.")
            return False

        loaded_data = None
        if snapshot_id:
            loaded_data = await self.storage_backend.load_state_by_id(snapshot_id)
        else:
            loaded_data = await self.storage_backend.load_latest_state()

        if loaded_data:
            self._load_state_from_dict(loaded_data)
            logger.info(f"WorldStore state loaded successfully from snapshot {snapshot_id or 'latest'}."
                        f" Current products: {len(self._product_state)}")
            return True
        logger.warning(f"Failed to load WorldStore state snapshot {snapshot_id or 'latest'}.")
        return False

    def _load_state_from_dict(self, state_data: Dict[str, Any]):
        """Internal helper to populate _product_state from a dictionary of serializable states."""
        self._product_state.clear()
        for asin, product_data in state_data.items():
            self._product_state[asin] = ProductState.from_dict(product_data)
        logger.info(f"Populated WorldStore state with {len(self._product_state)} products from dictionary.")
        
    def reset_state(self):
        """Reset all state - used for testing."""
        self._product_state.clear()
        self._pending_commands_by_asin_this_tick.clear()
        self._processed_command_ids_this_tick.clear()
        self._command_history.clear()
        self.commands_processed = 0
        self.commands_rejected = 0
        self.conflicts_arbitrated = 0
        self.snapshots_saved = 0
        logger.info("WorldStore state reset")


# Global instance management
_world_store_instance: Optional[WorldStore] = None


def get_world_store(storage_backend: Optional[PersistenceBackend] = None) -> WorldStore:
    """
    Get the global WorldStore instance.
    Allows specifying a storage_backend on first call.
    If no backend is provided on the first call, defaults to JsonFileStorageBackend.
    """
    global _world_store_instance
    if _world_store_instance is None:
        # If a storage_backend is provided, use it; otherwise, default to JsonFileStorageBackend.
        # This makes JsonFileStorageBackend the default for production-readiness.
        backend_to_use = storage_backend if storage_backend is not None else JsonFileStorageBackend()
        _world_store_instance = WorldStore(storage_backend=backend_to_use)
        logger.info(f"Global WorldStore instance created with backend: {type(backend_to_use).__name__}")
    elif storage_backend is not None:
        logger.warning("Global WorldStore instance already exists. Provided storage_backend is ignored.")
    return _world_store_instance


def set_world_store(world_store: WorldStore):
    """Set the global WorldStore instance."""
    global _world_store_instance
    _world_store_instance = world_store
    logger.info("Global WorldStore instance has been set.")