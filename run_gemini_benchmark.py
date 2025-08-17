#!/usr/bin/env python3
"""
Simple script to run a benchmark test with Gemini Flash agent.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarking.core.engine import BenchmarkEngine
from benchmarking.config import ConfigurationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the benchmark."""
    try:
        logger.info("Starting Gemini Flash benchmark test...")
        
        # Load benchmark configuration from YAML file
        config_manager = ConfigurationManager()
        config = config_manager.load_config("benchmark_gemini_flash.yaml", "benchmark")
        
        logger.info("Benchmark configuration loaded successfully")
        
        # Create and initialize the benchmark engine
        engine = BenchmarkEngine(config)
        
        # Initialize the engine
        await engine.initialize()
        
        logger.info("Running benchmark...")
        
        # Run the benchmark
        result = await engine.run_benchmark()
        
        logger.info("Benchmark completed successfully!")
        
        # Save results
        results_path = await engine.save_results()
        logger.info(f"Results saved to: {results_path}")
        
        # Print summary
        summary = engine.get_summary()
        logger.info("Benchmark Summary:")
        logger.info(f"  - Total duration: {summary.get('total_duration_seconds', 0):.2f} seconds")
        logger.info(f"  - Scenarios completed: {len(summary.get('scenario_results', []))}")
        logger.info(f"  - Agents tested: {len(summary.get('agents_tested', []))}")
        logger.info(f"  - Success rate: {summary.get('success_rate', 0):.2f}%")
        
        # Clean up
        await engine.cleanup()
        
        logger.info("Gemini Flash benchmark test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())