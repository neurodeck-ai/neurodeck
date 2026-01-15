#!/bin/bash
#
# NeuroDeck Orchestrator Startup Script
# Run from the root of the source tree
#

set -e  # Exit on any error

# Set virtual environment directory
VENV_DIR="venv${VENV_ENVIRONMENT:+_$VENV_ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting NeuroDeck Orchestrator...${NC}"

# Check we're in the right directory
if [[ ! -d "neurodeck" || ! -f "config/agents.ini" ]]; then
    echo -e "${RED}❌ Error: Must run from NeuroDeck root directory${NC}"
    echo "Expected structure:"
    echo "  neurodeck/          # Python modules"
    echo "  config/agents.ini   # Configuration"
    echo "  $VENV_DIR/         # Virtual environment"
    exit 1
fi

# Check virtual environment exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${RED}❌ Virtual environment not found at ./$VENV_DIR${NC}"
    echo "Run: python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r requirements.txt"
    exit 1
fi

# Check configuration exists
if [[ ! -f "config/agents.ini" ]]; then
    echo -e "${RED}❌ Configuration file not found: config/agents.ini${NC}"
    exit 1
fi

# Check environment file exists
if [[ ! -f "config/.env" ]]; then
    echo -e "${RED}❌ Environment file not found: config/.env${NC}"
    echo "Copy config/.env.template to config/.env and add your API keys"
    exit 1
fi

# Load environment variables
echo -e "${YELLOW}📁 Loading environment variables...${NC}"
set -a  # Automatically export all variables
source config/.env
set +a  # Stop auto-exporting

# Check required environment variables
required_vars=("NEURODECK_TOKEN" "ANTHROPIC_API_KEY" "XAI_API_KEY" "OPENAI_API_KEY" "GROQ_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo "Please add these to config/.env"
    exit 1
fi

# Check TLS certificates exist
if [[ ! -f "config/certs/server.crt" || ! -f "config/certs/server.key" ]]; then
    echo -e "${YELLOW}🔒 TLS certificates not found, they will be generated automatically${NC}"
fi

# Create logs directory
mkdir -p logs

# Show configuration summary
echo -e "${GREEN}✅ Configuration validated${NC}"
echo -e "${GREEN}✅ Environment variables loaded${NC}"
echo ""
echo -e "${BLUE}📋 Configuration Summary:${NC}"
echo "  Host: localhost:9999"
echo "  Config: config/agents.ini"
echo "  Logs: logs/orchestrator.log"
echo "  TLS Certs: config/certs/"
echo ""

# Parse agents from config file
agent_count=$(grep -c '^\[agent:' config/agents.ini || echo "0")
echo -e "${BLUE}🤖 Configured Agents: ${agent_count}${NC}"
grep '^\[agent:' config/agents.ini | sed 's/\[agent:\(.*\)\]/  - \1/' || true
echo ""

# Final confirmation
echo -e "${YELLOW}Starting orchestrator daemon...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Add Python path and run orchestrator
export PYTHONPATH="${PWD}/neurodeck:${PYTHONPATH}"

# Start orchestrator with explicit environment
env \
    NEURODECK_TOKEN="$NEURODECK_TOKEN" \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    XAI_API_KEY="$XAI_API_KEY" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    GROQ_API_KEY="$GROQ_API_KEY" \
    PYTHONPATH="${PWD}/neurodeck:${PYTHONPATH}" \
    $VENV_DIR/bin/python -m neurodeck.orchestrator \
    --config config/agents.ini \
    --host localhost \
    --port 9999