"""
Integration tests for production environment setups.

This module contains comprehensive integration tests that verify the FBA-Bench system
can operate in environments that mimic production setups, including different
deployment configurations, infrastructure components, and operational conditions.
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
import subprocess
import docker
import kubernetes
from kubernetes import client, config
import redis
import pymongo
from elasticsearch import Elasticsearch
import prometheus_client
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import yaml
import socket
import time
import threading
import requests
import psutil
import logging

# Import FBA-Bench components
from benchmarking.core.engine import BenchmarkEngine, BenchmarkConfig, BenchmarkResult
from benchmarking.scenarios.base import ScenarioConfig, ScenarioResult, BaseScenario
from benchmarking.scenarios.registry import ScenarioRegistry
from benchmarking.metrics.base import BaseMetric, MetricConfig, MetricResult
from benchmarking.metrics.registry import MetricRegistry
from benchmarking.validators.base import BaseValidator, ValidationResult
from benchmarking.validators.registry import ValidatorRegistry
from agent_runners.base_runner import BaseAgentRunner, AgentConfig
from infrastructure.distributed_coordinator import DistributedCoordinator
from infrastructure.distributed_event_bus import DistributedEventBus
from infrastructure.performance_monitor import PerformanceMonitor
from infrastructure.resource_manager import ResourceManager
from infrastructure.scalability_config import ScalabilityConfig


class ProductionEnvironmentConfig:
    """Configuration for production environment testing."""
    
    def __init__(self, environment_type: str):
        self.environment_type = environment_type
        self.config = self._load_environment_config(environment_type)
    
    def _load_environment_config(self, environment_type: str) -> Dict[str, Any]:
        """Load configuration for the specified environment type."""
        if environment_type == "docker_compose":
            return {
                "deployment_type": "docker_compose",
                "services": [
                    {
                        "name": "fba_bench_api",
                        "image": "fba_bench:latest",
                        "ports": ["8000:8000"],
                        "environment": {
                            "DATABASE_URL": "postgresql://user:password@db:5432/fba_bench",
                            "REDIS_URL": "redis://redis:6379",
                            "ELASTICSEARCH_URL": "http://elasticsearch:9200"
                        },
                        "depends_on": ["db", "redis", "elasticsearch"]
                    },
                    {
                        "name": "db",
                        "image": "postgres:13",
                        "environment": {
                            "POSTGRES_USER": "user",
                            "POSTGRES_PASSWORD": "password",
                            "POSTGRES_DB": "fba_bench"
                        },
                        "volumes": ["postgres_data:/var/lib/postgresql/data"]
                    },
                    {
                        "name": "redis",
                        "image": "redis:6-alpine"
                    },
                    {
                        "name": "elasticsearch",
                        "image": "elasticsearch:7.10.1",
                        "environment": {
                            "discovery.type": "single-node",
                            "ES_JAVA_OPTS": "-Xms512m -Xmx512m"
                        },
                        "volumes": ["elasticsearch_data:/usr/share/elasticsearch/data"]
                    }
                ],
                "volumes": {
                    "postgres_data": {},
                    "elasticsearch_data": {}
                }
            }
        elif environment_type == "kubernetes":
            return {
                "deployment_type": "kubernetes",
                "namespace": "fba-bench",
                "replicas": 3,
                "resources": {
                    "requests": {
                        "cpu": "500m",
                        "memory": "512Mi"
                    },
                    "limits": {
                        "cpu": "1000m",
                        "memory": "1Gi"
                    }
                },
                "services": [
                    {
                        "name": "fba-bench-api",
                        "type": "LoadBalancer",
                        "port": 80,
                        "target_port": 8000
                    },
                    {
                        "name": "fba-bench-db",
                        "type": "ClusterIP",
                        "port": 5432
                    },
                    {
                        "name": "fba-bench-redis",
                        "type": "ClusterIP",
                        "port": 6379
                    },
                    {
                        "name": "fba-bench-elasticsearch",
                        "type": "ClusterIP",
                        "port": 9200
                    }
                ],
                "configmaps": {
                    "fba-bench-config": {
                        "DATABASE_URL": "postgresql://user:password@fba-bench-db:5432/fba_bench",
                        "REDIS_URL": "redis://fba-bench-redis:6379",
                        "ELASTICSEARCH_URL": "http://fba-bench-elasticsearch:9200"
                    }
                }
            }
        elif environment_type == "distributed":
            return {
                "deployment_type": "distributed",
                "nodes": [
                    {
                        "name": "api_node",
                        "role": "api",
                        "host": "api.fba-bench.local",
                        "port": 8000
                    },
                    {
                        "name": "worker_node_1",
                        "role": "worker",
                        "host": "worker1.fba-bench.local",
                        "port": 8001
                    },
                    {
                        "name": "worker_node_2",
                        "role": "worker",
                        "host": "worker2.fba-bench.local",
                        "port": 8002
                    },
                    {
                        "name": "db_node",
                        "role": "database",
                        "host": "db.fba-bench.local",
                        "port": 5432
                    },
                    {
                        "name": "cache_node",
                        "role": "cache",
                        "host": "cache.fba-bench.local",
                        "port": 6379
                    },
                    {
                        "name": "search_node",
                        "role": "search",
                        "host": "search.fba-bench.local",
                        "port": 9200
                    }
                ],
                "load_balancer": {
                    "type": "round_robin",
                    "health_check_interval": 10,
                    "health_check_timeout": 5
                },
                "message_queue": {
                    "type": "redis",
                    "host": "cache.fba-bench.local",
                    "port": 6379
                }
            }
        else:
            raise ValueError(f"Unknown environment type: {environment_type}")


class ProductionEnvironmentManager:
    """Manager for production environment testing."""
    
    def __init__(self, environment_config: ProductionEnvironmentConfig):
        self.environment_config = environment_config
        self.config = environment_config.config
        self.deployment_type = self.config["deployment_type"]
        self.services = {}
        self.containers = {}
        self.pods = {}
        self.processes = {}
        self.networks = {}
        self.volumes = {}
        self.is_running = False
    
    def setup_environment(self) -> None:
        """Setup the production environment based on the configuration."""
        if self.deployment_type == "docker_compose":
            self._setup_docker_compose_environment()
        elif self.deployment_type == "kubernetes":
            self._setup_kubernetes_environment()
        elif self.deployment_type == "distributed":
            self._setup_distributed_environment()
        else:
            raise ValueError(f"Unknown deployment type: {self.deployment_type}")
        
        self.is_running = True
    
    def teardown_environment(self) -> None:
        """Teardown the production environment."""
        if self.deployment_type == "docker_compose":
            self._teardown_docker_compose_environment()
        elif self.deployment_type == "kubernetes":
            self._teardown_kubernetes_environment()
        elif self.deployment_type == "distributed":
            self._teardown_distributed_environment()
        else:
            raise ValueError(f"Unknown deployment type: {self.deployment_type}")
        
        self.is_running = False
    
    def _setup_docker_compose_environment(self) -> None:
        """Setup Docker Compose environment."""
        # Create docker-compose.yml file
        docker_compose_content = {
            "version": "3.8",
            "services": {}
        }
        
        for service in self.config["services"]:
            service_config = {
                "image": service["image"]
            }
            
            if "ports" in service:
                service_config["ports"] = service["ports"]
            
            if "environment" in service:
                service_config["environment"] = service["environment"]
            
            if "depends_on" in service:
                service_config["depends_on"] = service["depends_on"]
            
            if "volumes" in service:
                service_config["volumes"] = service["volumes"]
            
            docker_compose_content["services"][service["name"]] = service_config
        
        if "volumes" in self.config:
            docker_compose_content["volumes"] = self.config["volumes"]
        
        # Write docker-compose.yml to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml.dump(docker_compose_content, temp_file)
            docker_compose_file = temp_file.name
        
        try:
            # Start Docker Compose
            subprocess.run(
                ["docker-compose", "-f", docker_compose_file, "up", "-d"],
                check=True,
                capture_output=True
            )
            
            # Get container IDs
            result = subprocess.run(
                ["docker-compose", "-f", docker_compose_file, "ps", "-q"],
                check=True,
                capture_output=True,
                text=True
            )
            
            container_ids = result.stdout.strip().split('\n')
            
            # Store container information
            for i, service in enumerate(self.config["services"]):
                if i < len(container_ids):
                    self.containers[service["name"]] = container_ids[i]
            
            # Wait for services to be ready
            self._wait_for_services_ready()
            
        finally:
            # Clean up temporary file
            os.unlink(docker_compose_file)
    
    def _teardown_docker_compose_environment(self) -> None:
        """Teardown Docker Compose environment."""
        # Create docker-compose.yml file
        docker_compose_content = {
            "version": "3.8",
            "services": {}
        }
        
        for service in self.config["services"]:
            service_config = {
                "image": service["image"]
            }
            
            if "ports" in service:
                service_config["ports"] = service["ports"]
            
            if "environment" in service:
                service_config["environment"] = service["environment"]
            
            if "depends_on" in service:
                service_config["depends_on"] = service["depends_on"]
            
            if "volumes" in service:
                service_config["volumes"] = service["volumes"]
            
            docker_compose_content["services"][service["name"]] = service_config
        
        if "volumes" in self.config:
            docker_compose_content["volumes"] = self.config["volumes"]
        
        # Write docker-compose.yml to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml.dump(docker_compose_content, temp_file)
            docker_compose_file = temp_file.name
        
        try:
            # Stop and remove Docker Compose
            subprocess.run(
                ["docker-compose", "-f", docker_compose_file, "down", "-v"],
                check=True,
                capture_output=True
            )
            
            # Clear container information
            self.containers = {}
            
        finally:
            # Clean up temporary file
            os.unlink(docker_compose_file)
    
    def _setup_kubernetes_environment(self) -> None:
        """Setup Kubernetes environment."""
        try:
            # Load Kubernetes configuration
            config.load_kube_config()
            
            # Create Kubernetes API client
            k8s_client = client.CoreV1Api()
            apps_client = client.AppsV1Api()
            
            # Create namespace
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=self.config["namespace"])
            )
            
            try:
                k8s_client.create_namespace(namespace)
            except client.rest.ApiException as e:
                if e.status != 409:  # 409 means namespace already exists
                    raise
            
            # Create ConfigMap
            config_map_name = "fba-bench-config"
            config_map_data = {}
            
            for config_name, config_value in self.config["configmaps"][config_map_name].items():
                config_map_data[config_name] = config_value
            
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=config_map_name,
                    namespace=self.config["namespace"]
                ),
                data=config_map_data
            )
            
            try:
                k8s_client.create_namespaced_config_map(
                    namespace=self.config["namespace"],
                    body=config_map
                )
            except client.rest.ApiException as e:
                if e.status != 409:  # 409 means ConfigMap already exists
                    raise
            
            # Create Services
            for service in self.config["services"]:
                service_spec = client.V1ServiceSpec(
                    type=service["type"],
                    ports=[client.V1ServicePort(
                        port=service["port"],
                        target_port=client.IntOrString(service["target_port"])
                    )],
                    selector={"app": service["name"]}
                )
                
                service_obj = client.V1Service(
                    metadata=client.V1ObjectMeta(
                        name=service["name"],
                        namespace=self.config["namespace"]
                    ),
                    spec=service_spec
                )
                
                try:
                    k8s_client.create_namespaced_service(
                        namespace=self.config["namespace"],
                        body=service_obj
                    )
                except client.rest.ApiException as e:
                    if e.status != 409:  # 409 means Service already exists
                        raise
            
            # Create Deployments
            for service in self.config["services"]:
                if service["name"] == "fba-bench-api":
                    # Create API deployment
                    container = client.V1Container(
                        name="fba-bench-api",
                        image="fba_bench:latest",
                        ports=[client.V1ContainerPort(container_port=8000)],
                        env_from=[client.V1EnvFromSource(
                            config_map_ref=client.V1ConfigMapEnvSource(
                                name=config_map_name
                            )
                        )],
                        resources=client.V1ResourceRequirements(
                            requests=self.config["resources"]["requests"],
                            limits=self.config["resources"]["limits"]
                        )
                    )
                    
                    template = client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": service["name"]}
                        ),
                        spec=client.V1PodSpec(containers=[container])
                    )
                    
                    spec = client.V1DeploymentSpec(
                        replicas=self.config["replicas"],
                        template=template,
                        selector={"matchLabels": {"app": service["name"]}}
                    )
                    
                    deployment = client.V1Deployment(
                        metadata=client.V1ObjectMeta(
                            name=service["name"],
                            namespace=self.config["namespace"]
                        ),
                        spec=spec
                    )
                    
                    try:
                        apps_client.create_namespaced_deployment(
                            namespace=self.config["namespace"],
                            body=deployment
                        )
                    except client.rest.ApiException as e:
                        if e.status != 409:  # 409 means Deployment already exists
                            raise
            
            # Wait for pods to be ready
            self._wait_for_pods_ready()
            
        except Exception as e:
            logging.error(f"Error setting up Kubernetes environment: {e}")
            raise
    
    def _teardown_kubernetes_environment(self) -> None:
        """Teardown Kubernetes environment."""
        try:
            # Load Kubernetes configuration
            config.load_kube_config()
            
            # Create Kubernetes API client
            k8s_client = client.CoreV1Api()
            apps_client = client.AppsV1Api()
            
            # Delete Deployments
            for service in self.config["services"]:
                if service["name"] == "fba-bench-api":
                    try:
                        apps_client.delete_namespaced_deployment(
                            name=service["name"],
                            namespace=self.config["namespace"]
                        )
                    except client.rest.ApiException as e:
                        if e.status != 404:  # 404 means Deployment doesn't exist
                            raise
            
            # Delete Services
            for service in self.config["services"]:
                try:
                    k8s_client.delete_namespaced_service(
                        name=service["name"],
                        namespace=self.config["namespace"]
                    )
                except client.rest.ApiException as e:
                    if e.status != 404:  # 404 means Service doesn't exist
                        raise
            
            # Delete ConfigMap
            config_map_name = "fba-bench-config"
            try:
                k8s_client.delete_namespaced_config_map(
                    name=config_map_name,
                    namespace=self.config["namespace"]
                )
            except client.rest.ApiException as e:
                if e.status != 404:  # 404 means ConfigMap doesn't exist
                    raise
            
            # Delete namespace
            try:
                k8s_client.delete_namespace(name=self.config["namespace"])
            except client.rest.ApiException as e:
                if e.status != 404:  # 404 means namespace doesn't exist
                    raise
            
            # Clear pod information
            self.pods = {}
            
        except Exception as e:
            logging.error(f"Error tearing down Kubernetes environment: {e}")
            raise
    
    def _setup_distributed_environment(self) -> None:
        """Setup distributed environment."""
        # Create mock processes for each node
        for node in self.config["nodes"]:
            # Create a mock process for the node
            process = subprocess.Popen(
                ["python", "-c", f"import time; print('Node {node['name']} running on {node['host']}:{node['port']}'); time.sleep(60)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes[node["name"]] = process
        
        # Wait for processes to start
        time.sleep(2)
        
        # Check if processes are running
        for node_name, process in self.processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"Process for node {node_name} failed to start")
    
    def _teardown_distributed_environment(self) -> None:
        """Teardown distributed environment."""
        # Terminate all processes
        for node_name, process in self.processes.items():
            if process.poll() is None:
                process.terminate()
                process.wait()
        
        # Clear process information
        self.processes = {}
    
    def _wait_for_services_ready(self) -> None:
        """Wait for services to be ready."""
        # Wait for services to be ready
        for service in self.config["services"]:
            if service["name"] == "fba_bench_api":
                # Wait for API service to be ready
                url = "http://localhost:8000/health"
                self._wait_for_url_ready(url)
            elif service["name"] == "db":
                # Wait for database to be ready
                # This is a simplified check - in a real environment, you would check the database connection
                time.sleep(5)
            elif service["name"] == "redis":
                # Wait for Redis to be ready
                # This is a simplified check - in a real environment, you would check the Redis connection
                time.sleep(2)
            elif service["name"] == "elasticsearch":
                # Wait for Elasticsearch to be ready
                # This is a simplified check - in a real environment, you would check the Elasticsearch connection
                time.sleep(10)
    
    def _wait_for_pods_ready(self) -> None:
        """Wait for pods to be ready."""
        try:
            # Load Kubernetes configuration
            config.load_kube_config()
            
            # Create Kubernetes API client
            k8s_client = client.CoreV1Api()
            
            # Wait for pods to be ready
            timeout = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                pods = k8s_client.list_namespaced_pod(namespace=self.config["namespace"])
                
                all_ready = True
                for pod in pods.items:
                    if pod.status.phase != "Running":
                        all_ready = False
                        break
                    
                    for container_status in pod.status.container_statuses:
                        if not container_status.ready:
                            all_ready = False
                            break
                    
                    if not all_ready:
                        break
                
                if all_ready:
                    # Store pod information
                    for pod in pods.items:
                        self.pods[pod.metadata.name] = pod
                    
                    return
                
                time.sleep(5)
            
            raise TimeoutError("Timeout waiting for pods to be ready")
            
        except Exception as e:
            logging.error(f"Error waiting for pods to be ready: {e}")
            raise
    
    def _wait_for_url_ready(self, url: str, timeout: int = 60) -> None:
        """Wait for a URL to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        raise TimeoutError(f"Timeout waiting for URL {url} to be ready")
    
    def get_service_url(self, service_name: str) -> str:
        """Get the URL for a service."""
        if self.deployment_type == "docker_compose":
            if service_name == "fba_bench_api":
                return "http://localhost:8000"
            elif service_name == "db":
                return "postgresql://user:password@localhost:5432/fba_bench"
            elif service_name == "redis":
                return "redis://localhost:6379"
            elif service_name == "elasticsearch":
                return "http://localhost:9200"
        elif self.deployment_type == "kubernetes":
            if service_name == "fba-bench-api":
                # In a real environment, you would get the service URL from Kubernetes
                return "http://localhost:8000"
            elif service_name == "fba-bench-db":
                return "postgresql://user:password@localhost:5432/fba_bench"
            elif service_name == "fba-bench-redis":
                return "redis://localhost:6379"
            elif service_name == "fba-bench-elasticsearch":
                return "http://localhost:9200"
        elif self.deployment_type == "distributed":
            node = next((n for n in self.config["nodes"] if n["name"] == service_name), None)
            if node:
                return f"http://{node['host']}:{node['port']}"
        
        raise ValueError(f"Unknown service name: {service_name}")
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get the health status of a service."""
        url = self.get_service_url(service_name)
        
        if service_name in ["fba_bench_api", "fba-bench-api"]:
            health_url = f"{url}/health"
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
            except requests.exceptions.RequestException as e:
                return {"status": "unhealthy", "error": str(e)}
