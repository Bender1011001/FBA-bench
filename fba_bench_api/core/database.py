from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.types import TypeDecorator, TEXT
import json

DATABASE_URL = "sqlite:///./fba_bench.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class JSONEncodedDict(TypeDecorator):
    """Enables proper storage and retrieval of JSON metadata."""
    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value

class ExperimentConfigDB(Base):
    __tablename__ = "experiment_configs"
    
    experiment_id = Column(String, primary_key=True, index=True)
    experiment_name = Column(String, index=True)
    description = Column(String, nullable=True)
    config_data = Column(JSONEncodedDict, nullable=False)
    status = Column(String, default="created")
    total_runs = Column(Integer)
    completed_runs = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    current_run_details = Column(JSONEncodedDict, nullable=True)
    message = Column(String, nullable=True)

class SimulationConfigDB(Base):
    __tablename__ = "simulation_configs"
    
    config_id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    tick_interval_seconds = Column(Float)
    max_ticks = Column(Integer, nullable=True)
    start_time = Column(DateTime, nullable=True)
    time_acceleration = Column(Float)
    seed = Column(Integer, nullable=True)
    base_parameters = Column(JSONEncodedDict, nullable=False)
    agent_configs = Column(JSONEncodedDict, nullable=False) # Store as JSON
    llm_settings = Column(JSONEncodedDict, nullable=False)   # Store as JSON
    constraints = Column(JSONEncodedDict, nullable=False)    # Store as JSON
    experiment_settings = Column(JSONEncodedDict, nullable=True) # Store as JSON
    original_frontend_config = Column(JSONEncodedDict, nullable=True) # Store as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class TemplateDB(Base):
    __tablename__ = "templates"
    
    template_name = Column(String, primary_key=True, index=True)
    description = Column(String, nullable=True)
    config_data = Column(JSONEncodedDict, nullable=False) # Store the full config JSON
    created_at = Column(DateTime, default=datetime.utcnow)

def create_db_tables():
    Base.metadata.create_all(bind=engine)