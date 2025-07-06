#!/bin/bash

# RDB Web Interface Startup Script
# This script is designed to be run by systemd

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/home/topher/Desktop/Projects/rdb"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$PROJECT_DIR/run_web.py"

# Default configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="5000"
DEFAULT_DATA_DIR="$PROJECT_DIR/data"

# Use environment variables or defaults
RDB_HOST="${RDB_HOST:-$DEFAULT_HOST}"
RDB_PORT="${RDB_PORT:-$DEFAULT_PORT}"
RDB_DATA_DIR="${RDB_DATA_DIR:-$DEFAULT_DATA_DIR}"
RDB_LOG_LEVEL="${RDB_LOG_LEVEL:-INFO}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting RDB Web Interface..."

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    log "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    log "ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify RDB is installed
if ! python -c "import rdb" 2>/dev/null; then
    log "ERROR: RDB package not found in virtual environment"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$RDB_DATA_DIR" ]; then
    log "WARNING: Data directory not found: $RDB_DATA_DIR"
    log "Creating data directory..."
    mkdir -p "$RDB_DATA_DIR"
fi

# Check if index exists
INDEX_FILE="$RDB_DATA_DIR/index/index.faiss"
if [ ! -f "$INDEX_FILE" ]; then
    log "WARNING: Search index not found at $INDEX_FILE"
    log "Web interface will work but search functionality will be limited"
fi

# Export environment variables
export RDB_DATA_DIR
export RDB_LOG_LEVEL

log "Configuration:"
log "  Host: $RDB_HOST"
log "  Port: $RDB_PORT"
log "  Data Directory: $RDB_DATA_DIR"
log "  Log Level: $RDB_LOG_LEVEL"
log "  Working Directory: $(pwd)"

# Start the web interface
log "Starting RDB web interface on $RDB_HOST:$RDB_PORT..."

exec python "$PYTHON_SCRIPT" \
    --host "$RDB_HOST" \
    --port "$RDB_PORT" \
    --data-dir "$RDB_DATA_DIR"
