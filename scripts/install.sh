#!/bin/bash

# FBA-Bench Installation Script for Linux/macOS
# This script installs the FBA-Bench LLM benchmarking tool with all dependencies

set -e  # Exit on any error

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

# Check if script is run as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Detect operating system
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

print_status "Detected operating system: ${MACHINE}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: ${PYTHON_VERSION}"

# Compare Python version
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.8 or higher is required. Found: ${PYTHON_VERSION}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_warning "Node.js is not installed. Installing Node.js..."
    
    if [ "$MACHINE" = "Linux" ]; then
        # Install Node.js on Linux
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [ "$MACHINE" = "Mac" ]; then
        # Install Node.js on macOS using Homebrew
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew is not installed. Please install Homebrew first."
            exit 1
        fi
        brew install node
    fi
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2)
print_status "Node.js version: ${NODE_VERSION}"

# Check if Git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/fba-bench"
print_status "Creating installation directory: ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

# Clone or update the repository
if [ -d "${INSTALL_DIR}/.git" ]; then
    print_status "Repository already exists. Updating..."
    cd "${INSTALL_DIR}"
    git pull
else
    print_status "Cloning repository..."
    git clone https://github.com/your-org/fba-bench.git "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
fi

# Create Python virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install frontend dependencies
print_status "Installing frontend dependencies..."
cd frontend
npm install
npm run build
cd ..

# Create configuration file
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

# Create desktop shortcut (Linux only)
if [ "$MACHINE" = "Linux" ]; then
    print_status "Creating desktop shortcut..."
    cat > ~/Desktop/FBA-Bench.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=FBA-Bench
Comment=LLM Benchmarking Tool
Exec=bash -c "cd ${INSTALL_DIR} && source .venv/bin/activate && python api_server.py"
Icon=${INSTALL_DIR}/frontend/public/favicon.ico
Terminal=false
Categories=Development;Science;
EOF
    chmod +x ~/Desktop/FBA-Bench.desktop
fi

# Create startup script
print_status "Creating startup script..."
cat > "${INSTALL_DIR}/start.sh" << EOF
#!/bin/bash
# FBA-Bench Startup Script

cd "\$(dirname "\$0")"
source .venv/bin/activate
python api_server.py
EOF
chmod +x "${INSTALL_DIR}/start.sh"

# Create systemd service file (Linux only)
if [ "$MACHINE" = "Linux" ]; then
    print_status "Creating systemd service file..."
    cat > /tmp/fba-bench.service << EOF
[Unit]
Description=FBA-Bench LLM Benchmarking Tool
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/.venv/bin/python api_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/fba-bench.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable fba-bench.service
fi

print_success "Installation completed successfully!"
echo ""
print_status "Next steps:"
echo "1. Edit the configuration file at ${INSTALL_DIR}/config.yaml"
echo "2. Add your API keys to the configuration file"
echo "3. Start the application by running: ${INSTALL_DIR}/start.sh"
echo "4. Open your browser and navigate to: http://localhost:8000"
echo ""
print_status "For more information, see the documentation at: ${INSTALL_DIR}/docs/"