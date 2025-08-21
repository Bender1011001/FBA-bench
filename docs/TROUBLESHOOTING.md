# FBA-Bench Troubleshooting Guide

This guide provides solutions to common issues you might encounter while using FBA-Bench.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Startup Issues](#startup-issues)
- [Docker Issues](#docker-issues)
- [API Key Issues](#api-key-issues)
- [Performance Issues](#performance-issues)
- [Benchmark Issues](#benchmark-issues)
- [Frontend Issues](#frontend-issues)
- [Database Issues](#database-issues)
- [Network Issues](#network-issues)
- [Logging and Debugging](#logging-and-debugging)

## Installation Issues

### Python Installation Fails

**Symptoms**: Error messages during Python dependency installation.

**Solutions**:

1. **Check Python version**:
   ```bash
   python3 --version
   ```
   Ensure you have Python 3.8 or higher.

2. **Upgrade pip**:
   ```bash
   python3 -m pip install --upgrade pip
   ```

3. **Install in a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

4. **Check for missing system dependencies**:
   - **Ubuntu/Debian**: `sudo apt-get install build-essential python3-dev`
   - **CentOS/RHEL**: `sudo yum groupinstall "Development Tools"`
   - **macOS**: `xcode-select --install`

### Node.js Installation Fails

**Symptoms**: Error messages during frontend dependency installation.

**Solutions**:

1. **Check Node.js version**:
   ```bash
   node --version
   ```
   Ensure you have Node.js 16 or higher.

2. **Clear npm cache**:
   ```bash
   npm cache clean --force
   ```

3. **Delete node_modules and reinstall**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Install with legacy peer deps**:
   ```bash
   npm install --legacy-peer-deps
   ```

### Git Clone Fails

**Symptoms**: Unable to clone the repository.

**Solutions**:

1. **Check Git installation**:
   ```bash
   git --version
   ```

2. **Check network connectivity**:
   ```bash
   ping github.com
   ```

3. **Use HTTPS instead of SSH**:
   ```bash
   git clone https://github.com/your-org/fba-bench.git
   ```

4. **Configure Git credentials**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## Startup Issues

### Application Won't Start

**Symptoms**: Application fails to start or crashes immediately.

**Solutions**:

1. **Check configuration file**:
   - Verify `config.yaml` exists and is properly formatted
   - Check for syntax errors in the configuration file

2. **Check logs**:
   ```bash
   tail -f logs/fba-bench.log
   ```

3. **Run in debug mode**:
    ```bash
    poetry run uvicorn fba_bench_api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
    ```

4. **Check port availability**:
   ```bash
   # Linux/macOS
   netstat -tulpn | grep :8000
   # Windows
   netstat -ano | findstr :8000
   ```

5. **Check Python environment**:
   ```bash
   which python3  # Linux/macOS
   where python   # Windows
   ```

### Port Already in Use

**Symptoms**: Error message indicating port 8000 is already in use.

**Solutions**:

1. **Find the process using the port**:
   ```bash
   # Linux/macOS
   lsof -i :8000
   # Windows
   netstat -ano | findstr :8000
   ```

2. **Kill the process**:
   ```bash
   # Linux/macOS
   kill -9 <PID>
   # Windows
   taskkill /PID <PID> /F
   ```

3. **Change the port in configuration**:
   ```yaml
   server:
     port: 8001  # or any other available port
   ```

## Docker Issues

### Docker Daemon Not Running

**Symptoms**: Error message "Cannot connect to the Docker daemon".

**Solutions**:

1. **Start Docker service**:
   ```bash
   # Linux (systemd)
   sudo systemctl start docker
   sudo systemctl enable docker
   # macOS/Windows
   # Start Docker Desktop application
   ```

2. **Check Docker status**:
   ```bash
   docker info
   ```

### Docker Build Fails

**Symptoms**: Docker build process fails with errors.

**Solutions**:

1. **Check Dockerfile syntax**:
   - Verify all commands are properly formatted
   - Check for missing dependencies

2. **Clean build cache**:
   ```bash
   docker builder prune
   ```

3. **Build with no cache**:
   ```bash
   docker-compose build --no-cache
   ```

4. **Check disk space**:
   ```bash
   df -h
   ```

### Docker Container Keeps Restarting

**Symptoms**: Container starts but immediately restarts.

**Solutions**:

1. **Check container logs**:
   ```bash
   docker-compose logs fba-bench
   ```

2. **Inspect container**:
   ```bash
   docker inspect fba-bench
   ```

3. **Run container interactively**:
   ```bash
   docker-compose run fba-bench bash
   ```

## API Key Issues

### Invalid API Key

**Symptoms**: Error messages about invalid API keys.

**Solutions**:

1. **Verify API key format**:
   - Check for extra spaces or special characters
   - Ensure the key is complete and not truncated

2. **Test API key directly**:
   ```bash
   curl -H "Authorization: Bearer YOUR_API_KEY" https://api.openai.com/v1/models
   ```

3. **Check API key permissions**:
   - Ensure the key has the necessary permissions
   - Verify the key is not expired

### API Key Not Found

**Symptoms**: Error message indicating API key is not configured.

**Solutions**:

1. **Check configuration file**:
   ```yaml
   api_keys:
     openai: "your-api-key-here"
   ```

2. **Verify configuration file path**:
   - Ensure the application is reading from the correct configuration file
   - Check file permissions

3. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export ANTHROPIC_API_KEY="your-api-key"
   ```

## Performance Issues

### Slow Benchmark Execution

**Symptoms**: Benchmarks take longer than expected to complete.

**Solutions**:

1. **Check system resources**:
   ```bash
   # Linux/macOS
   top
   htop
   # Windows
   taskmgr
   ```

2. **Optimize benchmark configuration**:
   - Reduce the number of concurrent runs
   - Decrease the number of test iterations
   - Use smaller test datasets

3. **Check network latency**:
   ```bash
   ping api.openai.com
   ```

4. **Enable caching**:
   ```yaml
   cache:
     enabled: true
     ttl: 3600
   ```

### High Memory Usage

**Symptoms**: Application consumes excessive memory.

**Solutions**:

1. **Monitor memory usage**:
   ```bash
   # Linux/macOS
   free -h
   # Windows
   tasklist | findstr python
   ```

2. **Configure memory limits**:
   ```yaml
   performance:
     max_memory_usage: "4GB"
     gc_interval: 300
   ```

3. **Clear cache**:
   ```bash
   rm -rf .cache/*
   ```

## Benchmark Issues

### Benchmark Fails to Start

**Symptoms**: Benchmark configuration is valid but execution fails.

**Solutions**:

1. **Check benchmark configuration**:
   - Verify all required parameters are set
   - Ensure parameter values are within valid ranges

2. **Check agent compatibility**:
   - Verify the agent is compatible with the selected scenario
   - Check agent configuration

3. **Run in verbose mode**:
    ```bash
    poetry run uvicorn fba_bench_api.main:app --host 0.0.0.0 --port 8000 --log-level info
    ```

### Benchmark Results Inconsistent

**Symptoms**: Same benchmark produces different results on multiple runs.

**Solutions**:

1. **Set random seed**:
   ```yaml
   benchmark:
     random_seed: 42
   ```

2. **Check for external factors**:
   - Network latency variations
   - API rate limits
   - Model updates

3. **Increase sample size**:
   ```yaml
   benchmark:
     iterations: 10
     warmup_iterations: 3
   ```

## Frontend Issues

### Frontend Won't Load

**Symptoms**: Web interface fails to load or displays errors.

**Solutions**:

1. **Check browser console**:
   - Open Developer Tools (F12)
   - Check Console tab for error messages

2. **Clear browser cache**:
   - Press Ctrl+Shift+R (or Cmd+Shift+R on Mac)
   - Clear browser data and cookies

3. **Check API connectivity**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

4. **Rebuild frontend**:
   ```bash
   cd frontend
   npm run build
   cd ..
   ```

### WebSocket Connection Fails

**Symptoms**: Real-time updates not working, WebSocket errors.

**Solutions**:

1. **Check WebSocket endpoint**:
   ```bash
   curl -I http://localhost:8000/ws/benchmarking
   ```

2. **Verify proxy configuration**:
   - Check nginx or reverse proxy settings
   - Ensure WebSocket headers are properly forwarded

3. **Check firewall settings**:
   - Ensure WebSocket ports are open
   - Check for network restrictions

## Database Issues

### Database Connection Fails

**Symptoms**: Application unable to connect to the database.

**Solutions**:

1. **Check database service status**:
   ```bash
   docker-compose ps postgres
   ```

2. **Verify database configuration**:
   ```yaml
   database:
     host: localhost
     port: 5432
     name: fba_bench
     user: fba_bench_user
     password: fba_bench_password
   ```

3. **Test database connection**:
   ```bash
   docker-compose exec postgres psql -U fba_bench_user -d fba_bench
   ```

### Database Migration Fails

**Symptoms**: Database schema updates fail.

**Solutions**:

1. **Check migration files**:
   - Verify migration files exist and are valid
   - Check for syntax errors in migration files

2. **Reset database**:
   ```bash
   docker-compose down -v
   docker-compose up -d postgres
   ```

3. **Manual migration**:
   ```bash
   docker-compose exec fba-bench python manage.py migrate
   ```

## Network Issues

### Network Connectivity Problems

**Symptoms**: Unable to connect to external APIs or services.

**Solutions**:

1. **Check network connectivity**:
   ```bash
   ping google.com
   ping api.openai.com
   ```

2. **Check DNS resolution**:
   ```bash
   nslookup api.openai.com
   ```

3. **Check proxy settings**:
   ```bash
   echo $HTTP_PROXY
   echo $HTTPS_PROXY
   ```

4. **Configure proxy in application**:
   ```yaml
   network:
     proxy:
       http: "http://proxy.example.com:8080"
       https: "http://proxy.example.com:8080"
   ```

### SSL/TLS Certificate Issues

**Symptoms**: SSL certificate errors when connecting to APIs.

**Solutions**:

1. **Update certificate bundle**:
   ```bash
   # Update CA certificates
   sudo update-ca-certificates
   ```

2. **Disable SSL verification (not recommended for production)**:
   ```yaml
   network:
     ssl_verify: false
   ```

3. **Use custom CA bundle**:
   ```yaml
   network:
     ca_bundle: "/path/to/custom/ca-bundle.crt"
   ```

## Logging and Debugging

### Enable Debug Logging

**Solutions**:

1. **Set debug level in configuration**:
   ```yaml
   logging:
     level: "DEBUG"
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   ```

2. **Enable debug mode at runtime**:
    ```bash
    poetry run uvicorn fba_bench_api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
    ```

3. **Set environment variable**:
   ```bash
   export LOG_LEVEL=DEBUG
   ```

### Collect Diagnostic Information

**Solutions**:

1. **Generate system report**:
    ```bash
    curl -fsS http://localhost:8000/api/v1/health
    ```

2. **Collect logs**:
   ```bash
   tar -czf logs.tar.gz logs/
   ```

3. **Check system information**:
   ```bash
   uname -a
   python3 --version
   node --version
   docker --version
   docker-compose --version
   ```

### Common Error Messages

#### "ModuleNotFoundError: No module named 'xxx'"

**Cause**: Missing Python dependency.

**Solution**:
```bash
pip install xxx
```

#### "Connection refused"

**Cause**: Service not running or incorrect port.

**Solution**:
- Check if the service is running
- Verify the port number
- Check firewall settings

#### "401 Unauthorized"

**Cause**: Invalid or missing API key.

**Solution**:
- Verify API key is correct
- Check API key permissions
- Ensure API key is properly configured

#### "429 Too Many Requests"

**Cause**: API rate limit exceeded.

**Solution**:
- Wait for rate limit to reset
- Reduce request frequency
- Use API key with higher rate limits

## Getting Additional Help

If you're still experiencing issues after trying these solutions:

1. **Check the GitHub Issues**: Search for similar issues at [https://github.com/your-org/fba-bench/issues](https://github.com/your-org/fba-bench/issues)

2. **Create a New Issue**: Include the following information:
   - Operating system and version
   - Python and Node.js versions
   - FBA-Bench version
   - Complete error messages
   - Steps to reproduce the issue
   - Expected vs. actual behavior

3. **Contact Support**: Email support@fba-bench.com with detailed information about your issue.

4. **Join the Community**: Join our Discord server or Slack channel for community support.

---

Remember to always backup your configuration and data before making significant changes to your setup.