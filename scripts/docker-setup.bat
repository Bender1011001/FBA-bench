@echo off
REM FBA-Bench Docker Setup Script for Windows
REM This script helps users set up and run FBA-Bench using Docker

setlocal enabledelayedexpansion

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker first.
    echo [INFO] Visit https://docs.docker.com/get-docker/ for installation instructions.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker.
    pause
    exit /b 1
)

REM Check if docker-compose is installed
docker-compose --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] docker-compose is not installed. Please install docker-compose first.
    echo [INFO] Visit https://docs.docker.com/compose/install/ for installation instructions.
    pause
    exit /b 1
)

REM Create configuration file if it doesn't exist
if not exist "config.yaml" (
    echo [INFO] Creating configuration file...
    (
    echo # FBA-Bench Configuration
    echo # This file contains default settings for the benchmarking tool
    echo.
    echo # API Keys ^(add your own keys here^)
    echo api_keys:
    echo   openai: ""
    echo   anthropic: ""
    echo   google: ""
    echo   cohere: ""
    echo   openrouter: ""
    echo.
    echo # Default settings
    echo defaults:
    echo   llm: "gpt-4"
    echo   scenario: "standard"
    echo   agent: "basic"
    echo   metrics: ["revenue", "profit", "costs"]
    echo   auto_save: true
    echo   notifications: true
    echo.
    echo # UI settings
    echo ui:
    echo   theme: "system"
    echo   language: "en"
    echo   timezone: "UTC"
    echo.
    echo # Server settings
    echo server:
    echo   host: "localhost"
    echo   port: 8000
    echo   debug: false
    echo.
    echo # Data storage
    echo storage:
    echo   results_dir: "scenario_results"
    echo   logs_dir: "logs"
    echo   max_log_size: "10MB"
    echo   backup_count: 5
    ) > config.yaml
    echo [SUCCESS] Configuration file created.
)

REM Create necessary directories
echo [INFO] Creating data directories...
if not exist "scenario_results" mkdir scenario_results
if not exist "logs" mkdir logs
if not exist "ssl" mkdir ssl

REM Function to start the services
if "%1"=="start" goto :start_services
if "%1"=="stop" goto :stop_services
if "%1"=="restart" goto :restart_services
if "%1"=="logs" goto :show_logs
if "%1"=="reset" goto :reset_installation
if "%1"=="update" goto :update_installation
if "%1"=="status" goto :show_status

REM If no argument provided, default to start
if "%1"=="" goto :start_services

REM Show help if unknown command
echo FBA-Bench Docker Setup Script
echo.
echo Usage: %~nx0 {start^|stop^|restart^|logs^|reset^|update^|status}
echo.
echo Commands:
echo   start     Start FBA-Bench services ^(default^)
echo   stop      Stop FBA-Bench services
echo   restart   Restart FBA-Bench services
echo   logs      Show logs ^(optionally specify service name^)
echo   reset     Reset the installation ^(removes all data^)
echo   update    Update FBA-Bench to the latest version
echo   status    Show status of all services
echo.
pause
exit /b 0

:start_services
echo [INFO] Starting FBA-Bench services...

REM Build and start the services
docker-compose up --build -d

REM Wait for the services to be healthy
echo [INFO] Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check if the main service is healthy
docker-compose ps fba-bench | findstr "healthy" >nul
if %errorLevel% equ 0 (
    echo [SUCCESS] FBA-Bench is running successfully!
    echo.
    echo [INFO] Access the application at: http://localhost:8000
    echo [INFO] API documentation at: http://localhost:8000/docs
    echo.
    echo [INFO] To view logs, run: %~nx0 logs
    echo [INFO] To stop the services, run: %~nx0 stop
) else (
    echo [ERROR] FBA-Bench failed to start. Check the logs with: %~nx0 logs
    pause
    exit /b 1
)
pause
exit /b 0

:stop_services
echo [INFO] Stopping FBA-Bench services...
docker-compose down
echo [SUCCESS] Services stopped.
pause
exit /b 0

:restart_services
call :stop_services
call :start_services
exit /b 0

:show_logs
if "%2"=="" (
    docker-compose logs -f
) else (
    docker-compose logs -f %2
)
exit /b 0

:reset_installation
echo [WARNING] This will remove all containers, networks, and volumes.
set /p confirm="Are you sure you want to continue? (y/N): "
if /i "!confirm!"=="y" (
    echo [INFO] Removing all containers, networks, and volumes...
    docker-compose down -v
    echo [SUCCESS] Installation reset.
) else (
    echo [INFO] Reset cancelled.
)
pause
exit /b 0

:update_installation
echo [INFO] Updating FBA-Bench...
git pull
docker-compose build --no-cache
docker-compose up -d
echo [SUCCESS] FBA-Bench updated successfully!
pause
exit /b 0

:show_status
docker-compose ps
pause
exit /b 0