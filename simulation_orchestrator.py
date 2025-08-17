"""SimulationOrchestrator for FBA-Bench v3 event-driven architecture."""

import asyncio
import logging
import random
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from fba_events import TickEvent
from event_bus import get_event_bus # Only import the getter
from reproducibility.sim_seed import SimSeed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from event_bus import EventBus # For type hinting only

# OpenTelemetry Imports
from opentelemetry import trace
from instrumentation.tracer import setup_tracing
from instrumentation.simulation_tracer import SimulationTracer

logger = logging.getLogger(__name__)

# Initialize tracer for SimulationOrchestrator module
orchestrator_tracer_provider = setup_tracing(service_name="fba-bench-simulation-orchestrator")
simulation_tracer = SimulationTracer(orchestrator_tracer_provider)


@dataclass
class SimulationConfig:
    """Configuration for simulation orchestrator."""
    tick_interval_seconds: float = 1.0
    max_ticks: Optional[int] = None
    start_time: Optional[datetime] = None
    time_acceleration: float = 1.0  # 1.0 = real time, 2.0 = 2x speed, etc.
    auto_start: bool = False
    seed: Optional[int] = None  # New: Master seed for deterministic runs


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
    - Deterministic: uses SimSeed for reproducibility
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation orchestrator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.event_bus: Optional['EventBus'] = None # Use forward reference
        
        # Initialize seed system if provided
        if self.config.seed is not None:
            self._rng_tick = random.Random(SimSeed.derive_seed("tick_processing_rng"))
            self._rng_metadata = random.Random(SimSeed.derive_seed("metadata_generation_rng"))
            self._rng_general = random.Random(SimSeed.derive_seed("general_orchestrator_rng"))
        else:
            # Fallback to non-deterministic RNGs if no seed provided
            self._rng_tick = random.Random()
            self._rng_metadata = random.Random()
            self._rng_general = random.Random()

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
        if self.config.seed is not None:
            logger.info(f"Deterministic mode enabled with master seed: {self.config.seed}")
    
    async def start(self, event_bus: 'EventBus') -> None: # Use forward reference
        """Start the simulation orchestrator."""
        if self.is_running:
            logger.warning("SimulationOrchestrator already running")
            return
        
        self.event_bus = event_bus
        self.is_running = True
        self.simulation_start_time = datetime.now()
        self._stop_event.clear()
        self._pause_event.set()  # Start unpaused
        
        # Temporarily comment out tracing during run start to bypass AttributeError
        # with simulation_tracer.trace_simulation_run(
        #     simulation_id=f"sim_run_{self.start_time.strftime('%Y%m%d%H%M%S')}",
        #     scenario_name="default", # Placeholder
        #     total_ticks=self.config.max_ticks or -1
        # ) as simulation_span: # Use a name that implies it's the actual span
        #     if hasattr(simulation_span, 'set_attribute'): # Check if method exists
        #         simulation_span.set_attribute("config.seed", self.config.seed)

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
                # Temporarily comment out tracing for tick progression
                # with simulation_tracer.trace_tick_progression(
                #     tick=self.current_tick,
                #     timestamp=self._calculate_simulation_time().isoformat()
                # ) as tick_span_ctx: 
                await self._process_tick()
                # if hasattr(tick_span_ctx, 'set_attribute'):
                #     tick_span_ctx.set_attribute("tick.duration_seconds", (datetime.now() - tick_start_time).total_seconds())

                tick_duration = (datetime.now() - tick_start_time).total_seconds()
                
                # Update statistics
                self._update_tick_stats(tick_duration)
                
                # Wait for the next tick interval
                await self._wait_for_next_tick(tick_duration)
                
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            # temporarily disable setting error status on span
            # if hasattr(trace.get_current_span(), 'set_status'):
            #     trace.get_current_span().set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
            self.is_running = False
        
        logger.info("Simulation loop ended")
    
    async def _process_tick(self) -> None:
        """Process a single simulation tick using a deterministic RNG for event IDs."""
        # Calculate simulation time
        simulation_time = self._calculate_simulation_time()
        
        # Generate tick metadata
        tick_metadata = self._generate_tick_metadata()
        
        # Create tick event with a deterministically generated event_id
        # Use _rng_tick for ID generation to ensure determinism
        deterministic_id = self._rng_tick.getrandbits(64) # Get a deterministic random ID
        tick_event = TickEvent(
            event_id=f"tick_{self.current_tick}_{deterministic_id}",
            timestamp=datetime.now(), # Keep real timestamp, but ensure events are otherwise deterministic
            tick_number=self.current_tick,
            simulation_time=simulation_time,
            metadata=tick_metadata
        )
        
        # Publish tick event
        if self.event_bus:
            try:
                # Temporarily disable tracing for service execution
                # with simulation_tracer.trace_service_execution(
                #     service_name="event_bus_publish",
                #     tick=self.current_tick
                # ) as service_span_ctx: # This will be the _SpanContextManager instance
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
        """Calculate the current simulation time based purely on tick number for determinism."""
        # This makes simulation time independent of real-world elapsed time, crucial for determinism.
        elapsed_ticks_seconds = self.current_tick * self.config.tick_interval_seconds
        accelerated_elapsed = timedelta(seconds=elapsed_ticks_seconds * self.config.time_acceleration)
        return self.start_time + accelerated_elapsed
    
    def _generate_tick_metadata(self) -> Dict[str, Any]:
        """
        Generate metadata for the tick event.
        Use self._rng_metadata for any probabilistic elements to ensure determinism.
        """
        simulation_time = self._calculate_simulation_time()
        
        # Basic metadata
        metadata = {
            # Use deterministic timestamp or a fixed value for full reproducibility
            # For audit, real_time might still be useful for non-reproducible aspects
            'real_time_capture': datetime.now().isoformat(), # Capture real time for logs, not core logic
            'simulation_time': simulation_time.isoformat(),
            'time_acceleration': self.config.time_acceleration,
            'tick_interval': self.config.tick_interval_seconds,
            'total_runtime_seconds': self.stats['total_runtime_seconds']
        }
        
        # Add seasonal factors based on simulation month
        month = simulation_time.month
        # These factors are fixed, so they do not need RNG
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
        # These factors are fixed, so they do not need RNG
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
        
        # Example of using a random variable within metadata, derived deterministically
        # This could be for e.g., simulating minor daily external shocks
        # metadata['stochastic_factor'] = self._rng_metadata.uniform(0.9, 1.1)
        
        # Add simulation-specific metadata
        metadata['is_peak_season'] = month in [11, 12]  # Nov-Dec holiday season
        metadata['is_weekend'] = weekday >= 5
        metadata['is_business_hours'] = 9 <= hour <= 17
        
        return metadata
    
    async def _wait_for_next_tick(self, tick_duration: float) -> None:
        """Wait for the next tick interval, accounting for processing time."""
        # Calculate how long to wait
        target_interval = self.config.tick_interval_seconds
        # This logic remains as it controls the *real-time* progression, not simulation determinism
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
        """Update tick statistics. Uses self._rng_general for any random components."""
        self.stats['total_ticks'] += 1
        # Use self._rng_general if any randomness is introduced here
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
        
        # This part does not need RNG as it's purely calculative
        if self.stats['total_runtime_seconds'] > 0:
            self.stats['ticks_per_second'] = self.stats['total_ticks'] / self.stats['total_runtime_seconds']
        else:
            self.stats['ticks_per_second'] = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        current_time_for_status = datetime.now() # This can remain non-deterministic for live status
        simulation_time = self._calculate_simulation_time()
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_tick': self.current_tick,
            'real_time': current_time_for_status.isoformat(),
            'simulation_time': simulation_time.isoformat(),
            'config': {
                'tick_interval_seconds': self.config.tick_interval_seconds,
                'max_ticks': self.config.max_ticks,
                'time_acceleration': self.config.time_acceleration,
                'seed': self.config.seed # Include seed in status for debugging
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
        # Ensure to reset the master seed after simulation run, especially if not testing
        if self.config.seed is not None:
            SimSeed.reset_master_seed()