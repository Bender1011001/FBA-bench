@echo off
REM FBA-Bench Installation Script for Windows
REM This script installs the FBA-Bench LLM benchmarking tool with all dependencies

setlocal enabledelayedexpansion

REM Check if script is run as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [WARNING] This script is running as administrator. It's recommended to run it as a regular user.
    pause
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python version: %PYTHON_VERSION%

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Node.js is not installed. Installing Node.js...
    echo Please download and install Node.js from https://nodejs.org
    echo After installation, please run this script again.
    pause
    exit /b 1
)

REM Check Node.js version
for /f "tokens=1" %%i in ('node --version') do set NODE_VERSION=%%i
echo [INFO] Node.js version: %NODE_VERSION%

REM Check if Git is installed
git --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Git is not installed. Please install Git from https://git-scm.com
    pause
    exit /b 1
)

REM Set installation directory
set INSTALL_DIR=%USERPROFILE%\fba-bench
echo [INFO] Creating installation directory: %INSTALL_DIR%
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
)

REM Clone or update the repository
if exist "%INSTALL_DIR%\.git" (
    echo [INFO] Repository already exists. Updating...
    cd /d "%INSTALL_DIR%"
    git pull
) else (
    echo [INFO] Cloning repository...
    git clone https://github.com/your-org/fba-bench.git "%INSTALL_DIR%"
    cd /d "%INSTALL_DIR%"
)

REM Create Python virtual environment
echo [INFO] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install Python dependencies
echo [INFO] Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Install frontend dependencies
echo [INFO] Installing frontend dependencies...
cd frontend
npm install
npm run build
cd ..

REM Create configuration file
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

REM Create startup script
echo [INFO] Creating startup script...
(
echo @echo off
echo REM FBA-Bench Startup Script
echo.
echo cd /d "%%~dp0"
echo call .venv\Scripts\activate.bat
echo python api_server.py
) > start.bat

REM Create desktop shortcut
echo [INFO] Creating desktop shortcut...
set SCRIPT="%TEMP%\create_shortcut.vbs"
echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = "%USERPROFILE%\Desktop\FBA-Bench.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = "%INSTALL_DIR%\start.bat" >> %SCRIPT%
echo oLink.WorkingDirectory = "%INSTALL_DIR%" >> %SCRIPT%
echo oLink.IconLocation = "%INSTALL_DIR%\frontend\public\favicon.ico, 0" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%
cscript //nologo %SCRIPT%
del %SCRIPT%

REM Create uninstall script
echo [INFO] Creating uninstall script...
(
echo @echo off
echo REM FBA-Bench Uninstall Script
echo.
echo echo This will uninstall FBA-Bench from your system.
echo pause
echo.
echo echo Removing desktop shortcut...
echo del "%USERPROFILE%\Desktop\FBA-Bench.lnk" >nul 2>&1
echo.
echo echo Removing installation directory...
echo rmdir /s /q "%INSTALL_DIR%"
echo.
echo echo FBA-Bench has been uninstalled.
echo pause
) > uninstall.bat

echo.
echo [SUCCESS] Installation completed successfully!
echo.
echo [INFO] Next steps:
echo 1. Edit the configuration file at %INSTALL_DIR%\config.yaml
echo 2. Add your API keys to the configuration file
echo 3. Start the application by running: %INSTALL_DIR%\start.bat
echo 4. Open your browser and navigate to: http://localhost:8000
echo.
echo [INFO] For more information, see the documentation at: %INSTALL_DIR%\docs\
echo.
pause