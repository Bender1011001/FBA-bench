"""
Integration tests for database and API integration.

This module contains comprehensive integration tests that verify the interaction
between the FBA-Bench system components and database/API systems, including
data persistence, retrieval, updates, and API endpoint functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import tempfile
import os
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
import sqlite3
import pymongo
import requests
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
import redis
from elasticsearch import Elasticsearch

# Import FBA-Bench components
from benchmarking.core.engine import BenchmarkEngine, BenchmarkConfig, BenchmarkResult
from benchmarking.scenarios.base import ScenarioConfig, ScenarioResult, BaseScenario
from benchmarking.scenarios.registry import ScenarioRegistry
from benchmarking.metrics.base import BaseMetric, MetricConfig, MetricResult
from benchmarking.metrics.registry import MetricRegistry
from benchmarking.validators.base import BaseValidator, ValidationResult
from benchmarking.validators.registry import ValidatorRegistry
from agent_runners.base_runner import BaseAgentRunner, AgentConfig


# SQLAlchemy Models
Base = declarative_base()


class BenchmarkModel(Base):
    """SQLAlchemy model for benchmark results."""
    __tablename__ = "benchmarks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scenario_name = Column(String, nullable=False)
    agent_ids = Column(Text, nullable=False)  # JSON string
    metric_names = Column(Text, nullable=False)  # JSON string
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    errors = Column(Text, nullable=True)  # JSON string
    results = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    metrics = relationship("BenchmarkMetricModel", back_populates="benchmark", cascade="all, delete-orphan")
    validations = relationship("BenchmarkValidationModel", back_populates="benchmark", cascade="all, delete-orphan")


class BenchmarkMetricModel(Base):
    """SQLAlchemy model for benchmark metrics."""
    __tablename__ = "benchmark_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    benchmark_id = Column(String, ForeignKey("benchmarks.id"), nullable=False)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    benchmark = relationship("BenchmarkModel", back_populates="metrics")


class BenchmarkValidationModel(Base):
    """SQLAlchemy model for benchmark validations."""
    __tablename__ = "benchmark_validations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    benchmark_id = Column(String, ForeignKey("benchmarks.id"), nullable=False)
    validator_name = Column(String, nullable=False)
    validation_type = Column(String, nullable=False)
    result = Column(Boolean, nullable=False)
    score = Column(Float, nullable=True)
    details = Column(Text, nullable=True)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    benchmark = relationship("BenchmarkModel", back_populates="validations")


class ScenarioModel(Base):
    """SQLAlchemy model for scenarios."""
    __tablename__ = "scenarios"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=False)
    domain = Column(String, nullable=False)
    duration_ticks = Column(Integer, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentModel(Base):
    """SQLAlchemy model for agents."""
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, nullable=False, unique=True)
    agent_type = Column(String, nullable=False)
    agent_class = Column(String, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MetricModel(Base):
    """SQLAlchemy model for metrics."""
    __tablename__ = "metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=False)
    unit = Column(String, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    target_value = Column(Float, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_db(self) -> Session:
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def save_benchmark(self, benchmark_result: BenchmarkResult) -> str:
        """Save benchmark result to database."""
        db = next(self.get_db())
        
        try:
            # Create benchmark model
            benchmark_model = BenchmarkModel(
                id=str(uuid.uuid4()),
                scenario_name=benchmark_result.scenario_name,
                agent_ids=json.dumps(benchmark_result.agent_ids),
                metric_names=json.dumps(benchmark_result.metric_names),
                start_time=benchmark_result.start_time,
                end_time=benchmark_result.end_time,
                duration_seconds=benchmark_result.duration_seconds,
                success=benchmark_result.success,
                errors=json.dumps(benchmark_result.errors) if benchmark_result.errors else None,
                results=json.dumps(benchmark_result.results) if benchmark_result.results else None
            )
            
            db.add(benchmark_model)
            db.commit()
            db.refresh(benchmark_model)
            
            return benchmark_model.id
        
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_benchmark(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """Get benchmark result from database."""
        db = next(self.get_db())
        
        try:
            benchmark_model = db.query(BenchmarkModel).filter(BenchmarkModel.id == benchmark_id).first()
            
            if not benchmark_model:
                return None
            
            # Convert to dictionary
            benchmark_dict = {
                "id": benchmark_model.id,
                "scenario_name": benchmark_model.scenario_name,
                "agent_ids": json.loads(benchmark_model.agent_ids),
                "metric_names": json.loads(benchmark_model.metric_names),
                "start_time": benchmark_model.start_time.isoformat() if benchmark_model.start_time else None,
                "end_time": benchmark_model.end_time.isoformat() if benchmark_model.end_time else None,
                "duration_seconds": benchmark_model.duration_seconds,
                "success": benchmark_model.success,
                "errors": json.loads(benchmark_model.errors) if benchmark_model.errors else [],
                "results": json.loads(benchmark_model.results) if benchmark_model.results else {},
                "created_at": benchmark_model.created_at.isoformat(),
                "updated_at": benchmark_model.updated_at.isoformat()
            }
            
            return benchmark_dict
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def get_benchmarks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get benchmark results from database."""
        db = next(self.get_db())
        
        try:
            benchmark_models = db.query(BenchmarkModel).order_by(BenchmarkModel.created_at.desc()).offset(offset).limit(limit).all()
            
            benchmarks = []
            for benchmark_model in benchmark_models:
                benchmark_dict = {
                    "id": benchmark_model.id,
                    "scenario_name": benchmark_model.scenario_name,
                    "agent_ids": json.loads(benchmark_model.agent_ids),
                    "metric_names": json.loads(benchmark_model.metric_names),
                    "start_time": benchmark_model.start_time.isoformat() if benchmark_model.start_time else None,
                    "end_time": benchmark_model.end_time.isoformat() if benchmark_model.end_time else None,
                    "duration_seconds": benchmark_model.duration_seconds,
                    "success": benchmark_model.success,
                    "created_at": benchmark_model.created_at.isoformat(),
                    "updated_at": benchmark_model.updated_at.isoformat()
                }
                benchmarks.append(benchmark_dict)
            
            return benchmarks
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def save_scenario(self, scenario_config: ScenarioConfig) -> str:
        """Save scenario configuration to database."""
        db = next(self.get_db())
        
        try:
            # Check if scenario already exists
            existing_scenario = db.query(ScenarioModel).filter(ScenarioModel.name == scenario_config.name).first()
            
            if existing_scenario:
                # Update existing scenario
                existing_scenario.description = scenario_config.description
                existing_scenario.domain = scenario_config.domain
                existing_scenario.duration_ticks = scenario_config.duration_ticks
                existing_scenario.parameters = json.dumps(scenario_config.parameters) if scenario_config.parameters else None
                existing_scenario.enabled = scenario_config.enabled
                existing_scenario.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(existing_scenario)
                
                return existing_scenario.id
            else:
                # Create new scenario
                scenario_model = ScenarioModel(
                    id=str(uuid.uuid4()),
                    name=scenario_config.name,
                    description=scenario_config.description,
                    domain=scenario_config.domain,
                    duration_ticks=scenario_config.duration_ticks,
                    parameters=json.dumps(scenario_config.parameters) if scenario_config.parameters else None,
                    enabled=scenario_config.enabled
                )
                
                db.add(scenario_model)
                db.commit()
                db.refresh(scenario_model)
                
                return scenario_model.id
        
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get scenario configuration from database."""
        db = next(self.get_db())
        
        try:
            scenario_model = db.query(ScenarioModel).filter(ScenarioModel.name == scenario_name).first()
            
            if not scenario_model:
                return None
            
            # Convert to dictionary
            scenario_dict = {
                "id": scenario_model.id,
                "name": scenario_model.name,
                "description": scenario_model.description,
                "domain": scenario_model.domain,
                "duration_ticks": scenario_model.duration_ticks,
                "parameters": json.loads(scenario_model.parameters) if scenario_model.parameters else {},
                "enabled": scenario_model.enabled,
                "created_at": scenario_model.created_at.isoformat(),
                "updated_at": scenario_model.updated_at.isoformat()
            }
            
            return scenario_dict
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Get all scenario configurations from database."""
        db = next(self.get_db())
        
        try:
            scenario_models = db.query(ScenarioModel).all()
            
            scenarios = []
            for scenario_model in scenario_models:
                scenario_dict = {
                    "id": scenario_model.id,
                    "name": scenario_model.name,
                    "description": scenario_model.description,
                    "domain": scenario_model.domain,
                    "duration_ticks": scenario_model.duration_ticks,
                    "parameters": json.loads(scenario_model.parameters) if scenario_model.parameters else {},
                    "enabled": scenario_model.enabled,
                    "created_at": scenario_model.created_at.isoformat(),
                    "updated_at": scenario_model.updated_at.isoformat()
                }
                scenarios.append(scenario_dict)
            
            return scenarios
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def save_agent(self, agent_config: AgentConfig) -> str:
        """Save agent configuration to database."""
        db = next(self.get_db())
        
        try:
            # Check if agent already exists
            existing_agent = db.query(AgentModel).filter(AgentModel.agent_id == agent_config.agent_id).first()
            
            if existing_agent:
                # Update existing agent
                existing_agent.agent_type = agent_config.agent_type
                existing_agent.agent_class = agent_config.agent_class
                existing_agent.parameters = json.dumps(agent_config.parameters) if agent_config.parameters else None
                existing_agent.enabled = True
                existing_agent.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(existing_agent)
                
                return existing_agent.id
            else:
                # Create new agent
                agent_model = AgentModel(
                    id=str(uuid.uuid4()),
                    agent_id=agent_config.agent_id,
                    agent_type=agent_config.agent_type,
                    agent_class=agent_config.agent_class,
                    parameters=json.dumps(agent_config.parameters) if agent_config.parameters else None,
                    enabled=True
                )
                
                db.add(agent_model)
                db.commit()
                db.refresh(agent_model)
                
                return agent_model.id
        
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration from database."""
        db = next(self.get_db())
        
        try:
            agent_model = db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
            
            if not agent_model:
                return None
            
            # Convert to dictionary
            agent_dict = {
                "id": agent_model.id,
                "agent_id": agent_model.agent_id,
                "agent_type": agent_model.agent_type,
                "agent_class": agent_model.agent_class,
                "parameters": json.loads(agent_model.parameters) if agent_model.parameters else {},
                "enabled": agent_model.enabled,
                "created_at": agent_model.created_at.isoformat(),
                "updated_at": agent_model.updated_at.isoformat()
            }
            
            return agent_dict
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get all agent configurations from database."""
        db = next(self.get_db())
        
        try:
            agent_models = db.query(AgentModel).all()
            
            agents = []
            for agent_model in agent_models:
                agent_dict = {
                    "id": agent_model.id,
                    "agent_id": agent_model.agent_id,
                    "agent_type": agent_model.agent_type,
                    "agent_class": agent_model.agent_class,
                    "parameters": json.loads(agent_model.parameters) if agent_model.parameters else {},
                    "enabled": agent_model.enabled,
                    "created_at": agent_model.created_at.isoformat(),
                    "updated_at": agent_model.updated_at.isoformat()
                }
                agents.append(agent_dict)
            
            return agents
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def save_metric(self, metric_config: MetricConfig) -> str:
        """Save metric configuration to database."""
        db = next(self.get_db())
        
        try:
            # Check if metric already exists
            existing_metric = db.query(MetricModel).filter(MetricModel.name == metric_config.name).first()
            
            if existing_metric:
                # Update existing metric
                existing_metric.description = metric_config.description
                existing_metric.unit = metric_config.unit
                existing_metric.min_value = metric_config.min_value
                existing_metric.max_value = metric_config.max_value
                existing_metric.target_value = metric_config.target_value
                existing_metric.enabled = True
                existing_metric.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(existing_metric)
                
                return existing_metric.id
            else:
                # Create new metric
                metric_model = MetricModel(
                    id=str(uuid.uuid4()),
                    name=metric_config.name,
                    description=metric_config.description,
                    unit=metric_config.unit,
                    min_value=metric_config.min_value,
                    max_value=metric_config.max_value,
                    target_value=metric_config.target_value,
                    enabled=True
                )
                
                db.add(metric_model)
                db.commit()
                db.refresh(metric_model)
                
                return metric_model.id
        
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get metric configuration from database."""
        db = next(self.get_db())
        
        try:
            metric_model = db.query(MetricModel).filter(MetricModel.name == metric_name).first()
            
            if not metric_model:
                return None
            
            # Convert to dictionary
            metric_dict = {
                "id": metric_model.id,
                "name": metric_model.name,
                "description": metric_model.description,
                "unit": metric_model.unit,
                "min_value": metric_model.min_value,
                "max_value": metric_model.max_value,
                "target_value": metric_model.target_value,
                "enabled": metric_model.enabled,
                "created_at": metric_model.created_at.isoformat(),
                "updated_at": metric_model.updated_at.isoformat()
            }
            
            return metric_dict
        
        except Exception as e:
            raise e
        finally:
            db.close()
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all metric configurations from database."""
        db = next(self.get_db())
        
        try:
            metric_models = db.query(MetricModel).all()
            
            metrics = []
            for metric_model in metric_models:
                metric_dict = {
                    "id": metric_model.id,
                    "name": metric_model.name,
                    "description": metric_model.description,
                    "unit": metric_model.unit,
                    "min_value": metric_model.min_value,
                    "max_value": metric_model.max_value,
                    "target_value": metric_model.target_value,
                    "enabled": metric_model.enabled,
                    "created_at": metric_model.created_at.isoformat(),
                    "updated_at": metric_model.updated_at.isoformat()
                }
                metrics.append(metric_dict)
            
            return metrics
        
        except Exception as e:
            raise e
        finally:
            db.close()


class APIService:
    """Service for API operations."""
    
    def __init__(self, database_service: DatabaseService):
        self.database_service = database_service
        self.app = FastAPI(title="FBA-Bench API", description="API for FBA-Bench system")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/scenarios", response_model=List[Dict[str, Any]])
        async def get_scenarios():
            """Get all scenarios."""
            return self.database_service.get_scenarios()
        
        @self.app.get("/scenarios/{scenario_name}", response_model=Dict[str, Any])
        async def get_scenario(scenario_name: str):
            """Get a specific scenario."""
            scenario = self.database_service.get_scenario(scenario_name)
            if not scenario:
                raise HTTPException(status_code=404, detail="Scenario not found")
            return scenario
        
        @self.app.post("/scenarios", response_model=Dict[str, str])
        async def create_scenario(scenario_config: Dict[str, Any]):
            """Create or update a scenario."""
            try:
                config = ScenarioConfig(
                    name=scenario_config.get("name"),
                    description=scenario_config.get("description"),
                    domain=scenario_config.get("domain"),
                    duration_ticks=scenario_config.get("duration_ticks"),
                    parameters=scenario_config.get("parameters", {}),
                    enabled=scenario_config.get("enabled", True)
                )
                scenario_id = self.database_service.save_scenario(config)
                return {"id": scenario_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/agents", response_model=List[Dict[str, Any]])
        async def get_agents():
            """Get all agents."""
            return self.database_service.get_agents()
        
        @self.app.get("/agents/{agent_id}", response_model=Dict[str, Any])
        async def get_agent(agent_id: str):
            """Get a specific agent."""
            agent = self.database_service.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return agent
        
        @self.app.post("/agents", response_model=Dict[str, str])
        async def create_agent(agent_config: Dict[str, Any]):
            """Create or update an agent."""
            try:
                config = AgentConfig(
                    agent_id=agent_config.get("agent_id"),
                    agent_type=agent_config.get("agent_type"),
                    agent_class=agent_config.get("agent_class"),
                    parameters=agent_config.get("parameters", {})
                )
                agent_id = self.database_service.save_agent(config)
                return {"id": agent_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/metrics", response_model=List[Dict[str, Any]])
        async def get_metrics():
            """Get all metrics."""
            return self.database_service.get_metrics()
        
        @self.app.get("/metrics/{metric_name}", response_model=Dict[str, Any])
        async def get_metric(metric_name: str):
            """Get a specific metric."""
            metric = self.database_service.get_metric(metric_name)
            if not metric:
                raise HTTPException(status_code=404, detail="Metric not found")
            return metric
        
        @self.app.post("/metrics", response_model=Dict[str, str])
        async def create_metric(metric_config: Dict[str, Any]):
            """Create or update a metric."""
            try:
                config = MetricConfig(
                    name=metric_config.get("name"),
                    description=metric_config.get("description"),
                    unit=metric_config.get("unit"),
                    min_value=metric_config.get("min_value"),
                    max_value=metric_config.get("max_value"),
                    target_value=metric_config.get("target_value")
                )
                metric_id = self.database_service.save_metric(config)
                return {"id": metric_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/benchmarks", response_model=List[Dict[str, Any]])
        async def get_benchmarks(limit: int = 100, offset: int = 0):
            """Get all benchmarks."""
            return self.database_service.get_benchmarks(limit=limit, offset=offset)
        
        @self.app.get("/benchmarks/{benchmark_id}", response_model=Dict[str, Any])
        async def get_benchmark(benchmark_id: str):
            """Get a specific benchmark."""
            benchmark = self.database_service.get_benchmark(benchmark_id)
            if not benchmark:
                raise HTTPException(status_code=404, detail="Benchmark not found")
            return benchmark
        
        @self.app.post("/benchmarks", response_model=Dict[str, str])
        async def create_benchmark(benchmark_result: Dict[str, Any]):
            """Create a benchmark result."""
            try:
                result = BenchmarkResult(
                    scenario_name=benchmark_result.get("scenario_name"),
                    agent_ids=benchmark_result.get("agent_ids", []),
                    metric_names=benchmark_result.get("metric_names", []),
                    start_time=datetime.fromisoformat(benchmark_result.get("start_time")) if benchmark_result.get("start_time") else None,
                    end_time=datetime.fromisoformat(benchmark_result.get("end_time")) if benchmark_result.get("end_time") else None,
                    duration_seconds=benchmark_result.get("duration_seconds"),
                    success=benchmark_result.get("success", True),
                    errors=benchmark_result.get("errors", []),
                    results=benchmark_result.get("results", {})
                )
                benchmark_id = self.database_service.save_benchmark(result)
                return {"id": benchmark_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))


class MockAgentForDBAPI(BaseAgentRunner):
    """Mock agent implementation for database and API integration testing."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.responses = []
        self.actions_taken = []
    
    async def initialize(self) -> None:
        """Initialize the mock agent."""
        self.is_initialized = True
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return a response."""
        response = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "response": f"Mock response to: {input_data.get('content', '')}",
            "confidence": np.random.uniform(0.7, 0.95),
            "processing_time": np.random.uniform(0.01, 0.1)
        }
        
        self.responses.append(response)
        return response
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        success = np.random.random() > 0.1  # 90% success rate
        
        result = {
            "agent_id": self.config.agent_id,
            "action": action.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "result": f"Executed action: {action.get('type', 'unknown')}",
            "execution_time": np.random.uniform(0.01, 0.2)
        }
        
        self.actions_taken.append(result)
        return result
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the agent."""
        metrics = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "responses_count": len(self.responses),
            "actions_count": len(self.actions_taken),
            "avg_response_time": np.mean([r["processing_time"] for r in self.responses]) if self.responses else 0,
            "success_rate": len([a for a in self.actions_taken if a["status"] == "completed"]) / len(self.actions_taken) if self.actions_taken else 0
        }
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        self.is_initialized = False


class TestScenarioForDBAPI(BaseScenario):
    """Test scenario implementation for database and API integration testing."""
    
    def _validate_domain_parameters(self) -> List[str]:
        """Validate domain-specific parameters."""
        return []
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the test scenario."""
        await super().initialize(parameters)
        self.test_data = parameters.get("test_data", {})
    
    async def setup_for_agent(self, agent_id: str) -> None:
        """Setup the scenario for a specific agent."""
        await super().setup_for_agent(agent_id)
        self.agent_states[agent_id]["test_data"] = self.test_data
    
    async def update_tick(self, tick: int, state) -> None:
        """Update the scenario for a specific tick."""
        await super().update_tick(tick, state)
        
        # Simulate some scenario state changes
        for agent_id in self.agent_states:
            agent_state = self.agent_states[agent_id]
            agent_state["progress"] = tick / self.duration_ticks
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Evaluate agent performance in the scenario."""
        base_metrics = await super().evaluate_agent_performance(agent_id)
        
        agent_state = self.agent_states[agent_id]
        progress = agent_state.get("progress", 0.0)
        
        scenario_metrics = {
            "progress": progress,
            "efficiency_score": progress * 0.9,
            "task_completion_rate": progress * 0.85
        }
        
        return {**base_metrics, **scenario_metrics}


class TestMetricForDBAPI(BaseMetric):
    """Test metric implementation for database and API integration testing."""
    
    def __init__(self, config: MetricConfig):
        super().__init__(config)
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """Calculate the metric value."""
        # Simple calculation for testing
        return np.random.uniform(70.0, 95.0)


class TestDatabaseAPIIntegration:
    """Test cases for database and API integration."""
    
    @pytest.fixture
    def temp_database_file(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        yield temp_filename
        
        # Clean up
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
    
    @pytest.fixture
    def database_service(self, temp_database_file):
        """Create a database service instance."""
        database_url = f"sqlite:///{temp_database_file}"
        return DatabaseService(database_url)
    
    @pytest.fixture
    def api_service(self, database_service):
        """Create an API service instance."""
        return APIService(database_service)
    
    @pytest.fixture
    def test_client(self, api_service):
        """Create a test client for the API."""
        return TestClient(api_service.app)
    
    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            agent_id="test_agent",
            agent_type="mock_for_db_api",
            agent_class="MockAgentForDBAPI",
            parameters={"test_param": "test_value"}
        )
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_scenario_for_db_api",
            description="Test scenario for database and API integration",
            domain="test",
            duration_ticks=20,
            parameters={"test_data": {"key": "value"}}
        )
    
    @pytest.fixture
    def metric_config(self):
        """Create a test metric configuration."""
        return MetricConfig(
            name="test_metric_for_db_api",
            description="Test metric for database and API integration",
            unit="score",
            min_value=0.0,
            max_value=100.0,
            target_value=85.0
        )
    
    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create a mock agent for database and API integration testing."""
        return MockAgentForDBAPI(agent_config)
    
    @pytest.fixture
    def test_scenario(self, scenario_config):
        """Create a test scenario for database and API integration testing."""
        return TestScenarioForDBAPI(scenario_config)
    
    @pytest.fixture
    def test_metric(self, metric_config):
        """Create a test metric for database and API integration testing."""
        return TestMetricForDBAPI(metric_config)
    
    @pytest.fixture
    def benchmark_config(self):
        """Create a test benchmark configuration."""
        return BenchmarkConfig(
            name="database_api_integration_test",
            description="Test database and API integration",
            max_duration=300,
            tick_interval=0.1,
            metrics_collection_interval=1.0
        )
    
    @pytest.fixture
    def benchmark_engine(self, benchmark_config):
        """Create a benchmark engine instance."""
        return BenchmarkEngine(benchmark_config)
    
    def test_database_service_save_and_get_scenario(self, database_service, scenario_config):
        """Test database service save and get scenario."""
        # Save scenario
        scenario_id = database_service.save_scenario(scenario_config)
        assert scenario_id is not None
        
        # Get scenario
        scenario_dict = database_service.get_scenario(scenario_config.name)
        assert scenario_dict is not None
        assert scenario_dict["name"] == scenario_config.name
        assert scenario_dict["description"] == scenario_config.description
        assert scenario_dict["domain"] == scenario_config.domain
        assert scenario_dict["duration_ticks"] == scenario_config.duration_ticks
        assert scenario_dict["parameters"] == scenario_config.parameters
        assert scenario_dict["enabled"] == scenario_config.enabled
    
    def test_database_service_get_scenarios(self, database_service, scenario_config):
        """Test database service get scenarios."""
        # Save scenario
        database_service.save_scenario(scenario_config)
        
        # Get all scenarios
        scenarios = database_service.get_scenarios()
        assert len(scenarios) > 0
        
        # Check that our test scenario is in the list
        scenario_names = [s["name"] for s in scenarios]
        assert scenario_config.name in scenario_names
    
    def test_database_service_save_and_get_agent(self, database_service, agent_config):
        """Test database service save and get agent."""
        # Save agent
        agent_id = database_service.save_agent(agent_config)
        assert agent_id is not None
        
        # Get agent
        agent_dict = database_service.get_agent(agent_config.agent_id)
        assert agent_dict is not None
        assert agent_dict["agent_id"] == agent_config.agent_id
        assert agent_dict["agent_type"] == agent_config.agent_type
        assert agent_dict["agent_class"] == agent_config.agent_class
        assert agent_dict["parameters"] == agent_config.parameters
        assert agent_dict["enabled"] is True
    
    def test_database_service_get_agents(self, database_service, agent_config):
        """Test database service get agents."""
        # Save agent
        database_service.save_agent(agent_config)
        
        # Get all agents
        agents = database_service.get_agents()
        assert len(agents) > 0
        
        # Check that our test agent is in the list
        agent_ids = [a["agent_id"] for a in agents]
        assert agent_config.agent_id in agent_ids
    
    def test_database_service_save_and_get_metric(self, database_service, metric_config):
        """Test database service save and get metric."""
        # Save metric
        metric_id = database_service.save_metric(metric_config)
        assert metric_id is not None
        
        # Get metric
        metric_dict = database_service.get_metric(metric_config.name)
        assert metric_dict is not None
        assert metric_dict["name"] == metric_config.name
        assert metric_dict["description"] == metric_config.description
        assert metric_dict["unit"] == metric_config.unit
        assert metric_dict["min_value"] == metric_config.min_value
        assert metric_dict["max_value"] == metric_config.max_value
        assert metric_dict["target_value"] == metric_config.target_value
        assert metric_dict["enabled"] is True
    
    def test_database_service_get_metrics(self, database_service, metric_config):
        """Test database service get metrics."""
        # Save metric
        database_service.save_metric(metric_config)
        
        # Get all metrics
        metrics = database_service.get_metrics()
        assert len(metrics) > 0
        
        # Check that our test metric is in the list
        metric_names = [m["name"] for m in metrics]
        assert metric_config.name in metric_names
    
    def test_database_service_save_and_get_benchmark(self, database_service):
        """Test database service save and get benchmark."""
        # Create a benchmark result
        benchmark_result = BenchmarkResult(
            scenario_name="test_scenario",
            agent_ids=["test_agent"],
            metric_names=["test_metric"],
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=10),
            duration_seconds=10.0,
            success=True,
            errors=[],
            results={"test_metric": {"value": 85.0}}
        )
        
        # Save benchmark
        benchmark_id = database_service.save_benchmark(benchmark_result)
        assert benchmark_id is not None
        
        # Get benchmark
        benchmark_dict = database_service.get_benchmark(benchmark_id)
        assert benchmark_dict is not None
        assert benchmark_dict["id"] == benchmark_id
        assert benchmark_dict["scenario_name"] == benchmark_result.scenario_name
        assert benchmark_dict["agent_ids"] == benchmark_result.agent_ids
        assert benchmark_dict["metric_names"] == benchmark_result.metric_names
        assert benchmark_dict["success"] == benchmark_result.success
        assert benchmark_dict["results"] == benchmark_result.results
    
    def test_database_service_get_benchmarks(self, database_service):
        """Test database service get benchmarks."""
        # Create and save multiple benchmark results
        for i in range(3):
            benchmark_result = BenchmarkResult(
                scenario_name=f"test_scenario_{i}",
                agent_ids=[f"test_agent_{i}"],
                metric_names=[f"test_metric_{i}"],
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=10),
                duration_seconds=10.0,
                success=True,
                errors=[],
                results={f"test_metric_{i}": {"value": 85.0}}
            )
            database_service.save_benchmark(benchmark_result)
        
        # Get all benchmarks
        benchmarks = database_service.get_benchmarks()
        assert len(benchmarks) >= 3
        
        # Check that our test benchmarks are in the list
        scenario_names = [b["scenario_name"] for b in benchmarks]
        for i in range(3):
            assert f"test_scenario_{i}" in scenario_names
    
    def test_database_service_get_benchmarks_with_pagination(self, database_service):
        """Test database service get benchmarks with pagination."""
        # Create and save multiple benchmark results
        for i in range(10):
            benchmark_result = BenchmarkResult(
                scenario_name=f"test_scenario_{i}",
                agent_ids=[f"test_agent_{i}"],
                metric_names=[f"test_metric_{i}"],
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=10),
                duration_seconds=10.0,
                success=True,
                errors=[],
                results={f"test_metric_{i}": {"value": 85.0}}
            )
            database_service.save_benchmark(benchmark_result)
        
        # Get first page of benchmarks
        benchmarks_page1 = database_service.get_benchmarks(limit=5, offset=0)
        assert len(benchmarks_page1) == 5
        
        # Get second page of benchmarks
        benchmarks_page2 = database_service.get_benchmarks(limit=5, offset=5)
        assert len(benchmarks_page2) == 5
        
        # Verify that the pages are different
        page1_ids = [b["id"] for b in benchmarks_page1]
        page2_ids = [b["id"] for b in benchmarks_page2]
        assert set(page1_ids).isdisjoint(set(page2_ids))
    
    def test_api_service_health_check(self, test_client):
        """Test API service health check."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_api_service_create_and_get_scenario(self, test_client, scenario_config):
        """Test API service create and get scenario."""
        # Create scenario
        scenario_data = {
            "name": scenario_config.name,
            "description": scenario_config.description,
            "domain": scenario_config.domain,
            "duration_ticks": scenario_config.duration_ticks,
            "parameters": scenario_config.parameters,
            "enabled": scenario_config.enabled
        }
        
        response = test_client.post("/scenarios", json=scenario_data)
        assert response.status_code == 200
        create_response = response.json()
        assert "id" in create_response
        
        # Get scenario
        response = test_client.get(f"/scenarios/{scenario_config.name}")
        assert response.status_code == 200
        scenario_response = response.json()
        assert scenario_response["name"] == scenario_config.name
        assert scenario_response["description"] == scenario_config.description
        assert scenario_response["domain"] == scenario_config.domain
        assert scenario_response["duration_ticks"] == scenario_config.duration_ticks
        assert scenario_response["parameters"] == scenario_config.parameters
        assert scenario_response["enabled"] == scenario_config.enabled
    
    def test_api_service_get_scenarios(self, test_client, scenario_config):
        """Test API service get scenarios."""
        # Create scenario
        scenario_data = {
            "name": scenario_config.name,
            "description": scenario_config.description,
            "domain": scenario_config.domain,
            "duration_ticks": scenario_config.duration_ticks,
            "parameters": scenario_config.parameters,
            "enabled": scenario_config.enabled
        }
        
        test_client.post("/scenarios", json=scenario_data)
        
        # Get all scenarios
        response = test_client.get("/scenarios")
        assert response.status_code == 200
        scenarios_response = response.json()
        assert len(scenarios_response) > 0
        
        # Check that our test scenario is in the list
        scenario_names = [s["name"] for s in scenarios_response]
        assert scenario_config.name in scenario_names
    
    def test_api_service_get_nonexistent_scenario(self, test_client):
        """Test API service get non-existent scenario."""
        response = test_client.get("/scenarios/nonexistent_scenario")
        assert response.status_code == 404
    
    def test_api_service_create_and_get_agent(self, test_client, agent_config):
        """Test API service create and get agent."""
        # Create agent
        agent_data = {
            "agent_id": agent_config.agent_id,
            "agent_type": agent_config.agent_type,
            "agent_class": agent_config.agent_class,
            "parameters": agent_config.parameters
        }
        
        response = test_client.post("/agents", json=agent_data)
        assert response.status_code == 200
        create_response = response.json()
        assert "id" in create_response
        
        # Get agent
        response = test_client.get(f"/agents/{agent_config.agent_id}")
        assert response.status_code == 200
        agent_response = response.json()
        assert agent_response["agent_id"] == agent_config.agent_id
        assert agent_response["agent_type"] == agent_config.agent_type
        assert agent_response["agent_class"] == agent_config.agent_class
        assert agent_response["parameters"] == agent_config.parameters
        assert agent_response["enabled"] is True
    
    def test_api_service_get_agents(self, test_client, agent_config):
        """Test API service get agents."""
        # Create agent
        agent_data = {
            "agent_id": agent_config.agent_id,
            "agent_type": agent_config.agent_type,
            "agent_class": agent_config.agent_class,
            "parameters": agent_config.parameters
        }
        
        test_client.post("/agents", json=agent_data)
        
        # Get all agents
        response = test_client.get("/agents")
        assert response.status_code == 200
        agents_response = response.json()
        assert len(agents_response) > 0
        
        # Check that our test agent is in the list
        agent_ids = [a["agent_id"] for a in agents_response]
        assert agent_config.agent_id in agent_ids
    
    def test_api_service_get_nonexistent_agent(self, test_client):
        """Test API service get non-existent agent."""
        response = test_client.get("/agents/nonexistent_agent")
        assert response.status_code == 404
    
    def test_api_service_create_and_get_metric(self, test_client, metric_config):
        """Test API service create and get metric."""
        # Create metric
        metric_data = {
            "name": metric_config.name,
            "description": metric_config.description,
            "unit": metric_config.unit,
            "min_value": metric_config.min_value,
            "max_value": metric_config.max_value,
            "target_value": metric_config.target_value
        }
        
        response = test_client.post("/metrics", json=metric_data)
        assert response.status_code == 200
        create_response = response.json()
        assert "id" in create_response
        
        # Get metric
        response = test_client.get(f"/metrics/{metric_config.name}")
        assert response.status_code == 200
        metric_response = response.json()
        assert metric_response["name"] == metric_config.name
        assert metric_response["description"] == metric_config.description
        assert metric_response["unit"] == metric_config.unit
        assert metric_response["min_value"] == metric_config.min_value
        assert metric_response["max_value"] == metric_config.max_value
        assert metric_response["target_value"] == metric_config.target_value
        assert metric_response["enabled"] is True
    
    def test_api_service_get_metrics(self, test_client, metric_config):
        """Test API service get metrics."""