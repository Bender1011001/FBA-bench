#!/bin/bash

# FBA-Bench Docker Setup Script
# This script helps users set up and run FBA-Bench using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    print_status "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose first."
    print_status "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# Create configuration file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    print_status "Creating configuration file..."
    cat > config.yaml << EOF
# FBA-Bench Configuration
# This file contains default settings for the benchmarking tool

# API Keys (add your own keys here)
api_keys:
  openai: ""
  anthropic: ""
  google: ""
  cohere: ""
  openrouter: ""

# Default settings
defaults:
  llm: "gpt-4"
  scenario: "standard"
  agent: "basic"
  metrics: ["revenue", "profit", "costs"]
  auto_save: true
  notifications: true

# UI settings
ui:
  theme: "system"
  language: "en"
  timezone: "UTC"

# Server settings
server:
  host: "localhost"
  port: 8000
  debug: false

# Data storage
storage:
  results_dir: "scenario_results"
  logs_dir: "logs"
  max_log_size: "10MB"
  backup_count: 5
EOF
    print_success "Configuration file created."
fi

# Create necessary directories
print_status "Creating data directories..."
mkdir -p scenario_results logs ssl

# Function to start the services
start_services() {
    print_status "Starting FBA-Bench services..."
    
    # Build and start the services
    docker-compose up --build -d
    
    # Wait for the services to be healthy
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if the main service is healthy
    if docker-compose ps fba-bench | grep -q "healthy"; then
        print_success "FBA-Bench is running successfully!"
        echo ""
        print_status "Access the application at: http://localhost:8000"
        print_status "API documentation at: http://localhost:8000/docs"
        echo ""
        print_status "To view logs, run: docker-compose logs -f"
        print_status "To stop the services, run: $0 stop"
    else
        print_error "FBA-Bench failed to start. Check the logs with: docker-compose logs"
        exit 1
    fi
}

# Function to stop the services
stop_services() {
    print_status "Stopping FBA-Bench services..."
    docker-compose down
    print_success "Services stopped."
}

# Function to show logs
show_logs() {
    if [ -n "$2" ]; then
        docker-compose logs -f "$2"
    else
        docker-compose logs -f
    fi
}

# Function to reset the installation
reset_installation() {
    print_warning "This will remove all containers, networks, and volumes."
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing all containers, networks, and volumes..."
        docker-compose down -v
        print_success "Installation reset."
    else
        print_status "Reset cancelled."
    fi
}

# Function to update the installation
update_installation() {
    print_status "Updating FBA-Bench..."
    git pull
    docker-compose build --no-cache
    docker-compose up -d
    print_success "FBA-Bench updated successfully!"
}

# Main script logic
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    logs)
        show_logs "$@"
        ;;
    reset)
        reset_installation
        ;;
    update)
        update_installation
        ;;
    status)
        docker-compose ps
        ;;
    *)
        echo "FBA-Bench Docker Setup Script"
        echo ""
        echo "Usage: $0 {start|stop|restart|logs|reset|update|status}"
        echo ""
        echo "Commands:"
        echo "  start     Start FBA-Bench services (default)"
        echo "  stop      Stop FBA-Bench services"
        echo "  restart   Restart FBA-Bench services"
        echo "  logs      Show logs (optionally specify service name)"
        echo "  reset     Reset the installation (removes all data)"
        echo "  update    Update FBA-Bench to the latest version"
        echo "  status    Show status of all services"
        echo ""
        exit 1
        ;;
esac