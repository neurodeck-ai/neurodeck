#!/bin/bash

# NeuroDeck Console Launch Script
# Activates virtual environment and loads config before starting TUI

# Set virtual environment directory
VENV_DIR="venv${VENV_ENVIRONMENT:+_$VENV_ENVIRONMENT}"

echo "🚀 Starting NeuroDeck Console..."

# Check if we're in the right directory
if [ ! -f "config/.env" ]; then
    echo "❌ Error: config/.env not found. Please run from neurodeck root directory."
    exit 1
fi

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: $VENV_DIR directory not found. Please run from neurodeck root directory."
    exit 1
fi

source $VENV_DIR/bin/activate

# Load environment variables
set -a  # Automatically export all variables
source config/.env
set +a  # Stop auto-exporting

# Verify token is loaded
if [ -z "$NEURODECK_TOKEN" ]; then
    echo "❌ Error: NEURODECK_TOKEN not found in config/.env"
    exit 1
fi

echo "✅ Environment loaded, starting console..."

# Start the console
python -m neurodeck.console