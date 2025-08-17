import pytest
import subprocess
import time
import os
import signal
import httpx

@pytest.fixture(scope="module")
def api_server():
    """
    Pytest fixture to start and stop the FBA-Bench API server.
    Ensures the server is running before tests and properly terminated afterwards.
    Handles potential port conflicts and provides server access via httpx client.
    """
    process = None
    server_url = "http://localhost:8000"
    
    # Try to terminate any existing process on port 8000 before starting
    # This helps in case of previous test run failures
    try:
        # Check if python api_server.py is running
        check_command = os.popen(f'netstat -ano | findstr :8000').read()
        if "8000" in check_command:
            pid_match = check_command.split()[-1]
            if pid_match:
                print(f"Attempting to terminate existing process on port 8000 with PID {pid_match}")
                try:
                    os.kill(int(pid_match), signal.SIGTERM)
                    time.sleep(1) # Give it a moment to terminate
                except ProcessLookupError:
                    print(f"Process with PID {pid_match} not found.")
                except Exception as e:
                    print(f"Error terminating process on port 8000: {e}")
                
    except Exception as e:
        print(f"Error checking for existing process on port 8000: {e}")

    try:
        # Start the API server
        # Use a shell command on Windows to manage the process directly from Python
        # Add creationflags for DETACHED_PROCESS to prevent the subprocess from being terminated
        # when the parent process exits in certain environments
        cmd = ["python", "api_server.py"]
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            creationflags=subprocess.DETACHED_PROCESS if os.name == 'nt' else 0 # Detach process on Windows
        )
        print(f"Started API server with PID: {process.pid}")

        # Wait for the server to start up
        # We can poll the health endpoint rather than a fixed sleep
        max_retries = 10
        for i in range(max_retries):
            try:
                # Use a new httpx.Client for the health check to avoid fixture interference
                with httpx.Client() as client:
                    response = client.get(f"{server_url}/api/v1/health", timeout=5)
                    if response.status_code == 200 and response.json().get("status") == "healthy":
                        print("API server is up and healthy.")
                        break
            except httpx.RequestError as e:
                print(f"Waiting for API server... (attempt {i+1}/{max_retries}) - {e}")
            time.sleep(2) # Wait 2 seconds before retrying
        else:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"Server failed to start. STDOUT:\n{stdout}\nSTDERR:\n{stderr}")
                raise RuntimeError("API server failed to start within the given timeout.")
            else:
                raise RuntimeError("API server did not respond within the given timeout.")

        yield server_url # Provide the server URL to tests

    finally:
        if process:
            # Attempt graceful termination first
            print(f"Attempting to terminate API server with PID: {process.pid}")
            try:
                if os.name == 'nt':  # For Windows
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/F", "/T"], check=True, capture_output=True)
                else: # For Unix/Linux/macOS
                    process.terminate()
                process.wait(timeout=5)  # Give it some time to terminate
                if process.poll() is None:
                    print(f"API server with PID {process.pid} did not terminate gracefully. Killing...")
                    if os.name == 'nt':
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/F", "/T"], check=True, capture_output=True)
                    else:
                        process.kill()
                    process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"API server with PID {process.pid} did not terminate after timeout. Killing...")
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/F", "/T"], check=True, capture_output=True)
                else:
                    process.kill()
            except Exception as e:
                print(f"Error terminating API server process: {e}")
            finally:
                if process.poll() is None:
                    print(f"Warning: API server process with PID {process.pid} might still be running.")
                else:
                    print(f"API server with PID {process.pid} terminated successfully.")
                