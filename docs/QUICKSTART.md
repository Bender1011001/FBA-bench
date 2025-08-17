# FBA-Bench Quick Start Guide

This guide will help you get up and running with FBA-Bench, the user-friendly LLM benchmarking tool.

## Prerequisites

Before installing FBA-Bench, ensure you have the following prerequisites:

- **Python 3.8 or higher**: [Download Python](https://python.org)
- **Node.js 16 or higher**: [Download Node.js](https://nodejs.org)
- **Git**: [Download Git](https://git-scm.com)
- **Docker and Docker Compose (optional)**: [Download Docker](https://docker.com)

## Installation Options

### Option 1: Native Installation (Recommended for Development)

#### Linux/macOS

1. Download and run the installation script:
   ```bash
   chmod +x scripts/install.sh
   ./scripts/install.sh
   ```

2. Follow the on-screen instructions to complete the installation.

#### Windows

1. Run the installation script:
   ```cmd
   scripts\install.bat
   ```

2. Follow the on-screen instructions to complete the installation.

### Option 2: Docker Installation (Recommended for Production)

#### Linux/macOS

1. Run the Docker setup script:
   ```bash
   chmod +x scripts/docker-setup.sh
   ./scripts/docker-setup.sh
   ```

#### Windows

1. Run the Docker setup script:
   ```cmd
   scripts\docker-setup.bat
   ```

## First-Time Setup

### 1. Configure API Keys

After installation, you need to configure your API keys:

1. Open the configuration file:
   - **Native installation**: `config.yaml` in the installation directory
   - **Docker installation**: `config.yaml` in the project root

2. Add your API keys for the LLM providers you want to use:
   ```yaml
   api_keys:
     openai: "your-openai-api-key"
     anthropic: "your-anthropic-api-key"
     google: "your-google-api-key"
     cohere: "your-cohere-api-key"
     openrouter: "your-openrouter-api-key"
   ```

### 2. Start the Application

#### Native Installation

- **Linux/macOS**: Run `./start.sh` from the installation directory
- **Windows**: Run `start.bat` from the installation directory

#### Docker Installation

- **Linux/macOS**: Run `./scripts/docker-setup.sh start`
- **Windows**: Run `scripts\docker-setup.bat start`

### 3. Access the Application

Open your web browser and navigate to:
- **Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Basic Usage

### 1. Dashboard

The Dashboard provides an overview of recent benchmark runs and quick access to features:
- View recent benchmark runs
- Access quick actions for common tasks
- Monitor system health and status

### 2. Configure a Benchmark

1. Navigate to the **Benchmark Configuration** section
2. Follow the step-by-step wizard:
   - **Step 1**: Select LLM models to benchmark
   - **Step 2**: Choose or create a scenario
   - **Step 3**: Configure agent settings
   - **Step 4**: Select metrics to measure
   - **Step 5**: Set execution parameters
3. Save your configuration

### 3. Run a Benchmark

1. Go to the **Benchmark Runner** section
2. Select a saved configuration
3. Click "Start Benchmark"
4. Monitor progress in real-time
5. View logs and control the execution

### 4. Analyze Results

1. Navigate to the **Results Analysis & Visualization** section
2. Select a completed benchmark run
3. Explore visualizations and metrics
4. Compare results across different runs
5. Export results for further analysis

### 5. Manage Settings

1. Go to the **Settings/Admin** section
2. Manage API keys
3. Configure default settings
4. Customize UI preferences

## Troubleshooting

### Common Issues

#### Installation Fails

- Ensure all prerequisites are installed
- Check that you have sufficient disk space
- Run the installation script with appropriate permissions

#### Application Won't Start

- Check that all required services are running
- Verify your API keys are correctly configured
- Check the logs for error messages

#### Docker Issues

- Ensure Docker is running
- Check that you have sufficient resources allocated to Docker
- Verify Docker Compose is installed

#### API Key Errors

- Verify your API keys are valid and active
- Check that you have sufficient credits/quotas
- Ensure the API keys are correctly formatted in the configuration file

### Getting Help

If you encounter issues not covered here:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review the logs in the `logs` directory
3. Check the [GitHub Issues](https://github.com/your-org/fba-bench/issues)
4. Contact support at support@fba-bench.com

## Next Steps

- Explore the [User Manual](USER_MANUAL.md) for detailed features
- Learn about [Advanced Configuration](ADVANCED_CONFIG.md)
- Contribute to the project on [GitHub](https://github.com/your-org/fba-bench)

---

Happy benchmarking! 🚀