"""SimulationOrchestrator for FBA-Bench v3 event-driven architecture."""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from events import TickEvent
from event_bus import EventBus


logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation orchestrator."""
    tick_interval_seconds: float = 1.0
    max_ticks: Optional[int] = None
    start_time: Optional[datetime] = None
    time_acceleration: float = 1.0  # 1.0 = real time, 2.0 = 2x speed, etc.
    auto_start: bool = False


class SimulationOrchestrator:
    """
    Event-driven simulation orchestrator for FBA-Bench v3.
    
    This is the heartbeat of the simulation - its only responsibility is to
    run the main time loop and publish TickEvent to the EventBus at each time step.
    All other simulation logic has been moved to event-driven services.
    
    Key principles:
    - Single responsibility: only manages time progression
    - Stateless: no business logic, only timing control
    - Event-driven: communicates only through events
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation orchestrator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.event_bus: Optional[EventBus] = None
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.current_tick = 0
        self.start_time = config.start_time or datetime.now()
        self.simulation_start_time = datetime.now()
        
        # Control variables
        self._orchestrator_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            'total_ticks': 0,
            'total_runtime_seconds': 0.0,
            'average_tick_duration': 0.0,
            'events_published': 0,
            'last_tick_timestamp': None
        }
        
        logger.info(f"SimulationOrchestrator initialized with {config.tick_interval_seconds}s interval")
    
    async def start(self, event_bus: EventBus) -> None:
        """Start the simulation orchestrator."""
        if self.is_running:
            logger.warning("SimulationOrchestrator already running")
            return
        
        self.event_bus = event_bus
        self.is_running = True
        self.simulation_start_time = datetime.now()
        self._stop_event.clear()
        self._pause_event.set()  # Start unpaused
        
        # Start the main simulation loop
        self._orchestrator_task = asyncio.create_task(self._simulation_loop())
        
        logger.info("SimulationOrchestrator started")
    
    async def stop(self) -> None:
        """Stop the simulation orchestrator."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        # Cancel the orchestrator task
        if self._orchestrator_task:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
        
        # Update final statistics
        self._update_final_stats()
        
        logger.info(f"SimulationOrchestrator stopped after {self.current_tick} ticks")
    
    async def pause(self) -> None:
        """Pause the simulation."""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            self._pause_event.clear()
            logger.info("SimulationOrchestrator paused")
    
    async def resume(self) -> None:
        """Resume the simulation."""
        if self.is_running and self.is_paused:
            self.is_paused = False
            self._pause_event.set()
            logger.info("SimulationOrchestrator resumed")
    
    async def _simulation_loop(self) -> None:
        """Main simulation loop that publishes tick events."""
        logger.info("Simulation loop started")
        
        try:
            while self.is_running and not self._stop_event.is_set():
                # Check if we should stop due to max ticks
                if self.config.max_ticks and self.current_tick >= self.config.max_ticks:
                    logger.info(f"Reached maximum ticks ({self.config.max_ticks}), stopping simulation")
                    break
                
                # Wait for resume if paused
                await self._pause_event.wait()
                
                # Check if we were stopped while paused
                if not self.is_running:
                    break
                
                # Process the tick
                tick_start_time = datetime.now()
                await self._process_tick()
                tick_duration = (datetime.now() - tick_start_time).total_seconds()
                
                # Update statistics
                self._update_tick_stats(tick_duration)
                
                # Wait for the next tick interval
                await self._wait_for_next_tick(tick_duration)
                
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            self.is_running = False
        
        logger.info("Simulation loop ended")
    
    async def _process_tick(self) -> None:
        """Process a single simulation tick."""
        # Calculate simulation time
        simulation_time = self._calculate_simulation_time()
        
        # Generate tick metadata
        tick_metadata = self._generate_tick_metadata()
        
        # Create tick event
        tick_event = TickEvent(
            event_id=f"tick_{self.current_tick}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            tick_number=self.current_tick,
            simulation_time=simulation_time,
            metadata=tick_metadata
        )
        
        # Publish tick event
        if self.event_bus:
            try:
                await self.event_bus.publish(tick_event)
                self.stats['events_published'] += 1
                logger.debug(f"Published TickEvent {self.current_tick}")
                
            except Exception as e:
                logger.error(f"Error publishing TickEvent {self.current_tick}: {e}")
        else:
            logger.warning("EventBus not available, cannot publish TickEvent")
        
        # Increment tick counter
        self.current_tick += 1
    
    def _calculate_simulation_time(self) -> datetime:
        """Calculate the current simulation time based on time acceleration."""
        real_elapsed = datetime.now() - self.simulation_start_time
        accelerated_elapsed = timedelta(seconds=real_elapsed.total_seconds() * self.config.time_acceleration)
        return self.start_time + accelerated_elapsed
    
    def _generate_tick_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the tick event."""
        simulation_time = self._calculate_simulation_time()
        
        # Basic metadata
        metadata = {
            'real_time': datetime.now().isoformat(),
            'simulation_time': simulation_time.isoformat(),
            'time_acceleration': self.config.time_acceleration,
            'tick_interval': self.config.tick_interval_seconds,
            'total_runtime_seconds': self.stats['total_runtime_seconds']
        }
        
        # Add seasonal factors based on simulation month
        month = simulation_time.month
        seasonal_factors = {
            1: 0.8, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.05,
            7: 1.1, 8: 1.05, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.4
        }
        metadata['seasonal_factor'] = seasonal_factors.get(month, 1.0)
        
        # Add day-of-week factor
        weekday = simulation_time.weekday()  # 0=Monday, 6=Sunday
        weekday_factors = {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,  # Mon-Fri
            5: 1.2, 6: 1.1  # Sat-Sun (higher demand)
        }
        metadata['weekday_factor'] = weekday_factors.get(weekday, 1.0)
        
        # Add hour-of-day factor
        hour = simulation_time.hour
        if 6 <= hour <= 10:  # Morning peak
            metadata['hour_factor'] = 1.3
        elif 11 <= hour <= 14:  # Lunch peak
            metadata['hour_factor'] = 1.1
        elif 18 <= hour <= 22:  # Evening peak
            metadata['hour_factor'] = 1.4
        elif 23 <= hour or hour <= 5:  # Night
            metadata['hour_factor'] = 0.3
        else:  # Regular hours
            metadata['hour_factor'] = 1.0
        
        # Add simulation-specific metadata
        metadata['is_peak_season'] = month in [11, 12]  # Nov-Dec holiday season
        metadata['is_weekend'] = weekday >= 5
        metadata['is_business_hours'] = 9 <= hour <= 17
        
        return metadata
    
    async def _wait_for_next_tick(self, tick_duration: float) -> None:
        """Wait for the next tick interval, accounting for processing time."""
        # Calculate how long to wait
        target_interval = self.config.tick_interval_seconds
        wait_time = max(0, target_interval - tick_duration)
        
        if wait_time > 0:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_time)
                # If we get here, stop was requested during wait
                return
            except asyncio.TimeoutError:
                # Normal case - wait time elapsed
                pass
        elif tick_duration > target_interval:
            # Tick took longer than interval - log warning
            logger.warning(f"Tick {self.current_tick} took {tick_duration:.3f}s, longer than interval {target_interval}s")
    
    def _update_tick_stats(self, tick_duration: float) -> None:
        """Update tick statistics."""
        self.stats['total_ticks'] += 1
        self.stats['total_runtime_seconds'] = (datetime.now() - self.simulation_start_time).total_seconds()
        self.stats['last_tick_timestamp'] = datetime.now().isoformat()
        
        # Calculate average tick duration
        if self.stats['total_ticks'] > 0:
            total_tick_time = self.stats.get('total_tick_time', 0.0) + tick_duration
            self.stats['total_tick_time'] = total_tick_time
            self.stats['average_tick_duration'] = total_tick_time / self.stats['total_ticks']
    
    def _update_final_stats(self) -> None:
        """Update final statistics when stopping."""
        self.stats['total_runtime_seconds'] = (datetime.now() - self.simulation_start_time).total_seconds()
        
        if self.stats['total_runtime_seconds'] > 0:
            self.stats['ticks_per_second'] = self.stats['total_ticks'] / self.stats['total_runtime_seconds']
        else:
            self.stats['ticks_per_second'] = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        current_time = datetime.now()
        simulation_time = self._calculate_simulation_time()
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_tick': self.current_tick,
            'real_time': current_time.isoformat(),
            'simulation_time': simulation_time.isoformat(),
            'config': {
                'tick_interval_seconds': self.config.tick_interval_seconds,
                'max_ticks': self.config.max_ticks,
                'time_acceleration': self.config.time_acceleration
            },
            'statistics': self.stats.copy()
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.config.auto_start and self.event_bus:
            await self.start(self.event_bus)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()