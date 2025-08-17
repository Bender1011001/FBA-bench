# FBA-Bench Dockerfile
# This Dockerfile creates a container for the FBA-Bench LLM benchmarking tool

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and package-lock.json for frontend
COPY frontend/package*.json ./frontend/

# Install frontend dependencies
RUN cd frontend && npm install

# Copy the entire project
COPY . .

# Build the frontend
RUN cd frontend && npm run build:skip-tests

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Create directories for data
RUN mkdir -p /app/scenario_results /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the application
CMD ["python", "api_server.py"]