# NeuroDeck

A multi-agent AI orchestration system that lets multiple AI models chat together in a shared conversation. Support for Claude, OpenAI, Groq, and xAI.

![NeuroDeck Screenshot](screenshot.jpg)

## Features

- **Multi-Agent Chat**: Multiple AI agents in a shared conversation, they can @mention each other
- **Tool Support**: Built-in tools (filesystem, bash, TTS) + MCP server integration
- **Text-to-Speech**: Voice responses via Coqui TTS
- **Secure Communication**: TLS encryption between orchestrator and agents
- **Per-Agent Config**: Each agent can have different models, system prompts, and tool permissions

## Requirements

- Python 3.10+
- API keys from at least one provider (see below)

## Installation

```bash
# Clone and enter directory
git clone https://github.com/neurodeck-ai/neurodeck.git
cd neurodeck

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Getting API Keys

You need at least one API key. More keys = more agents in your chat.

| Provider | Get Key | Cost |
|----------|---------|------|
| **Groq** (recommended to start) | [console.groq.com](https://console.groq.com) | Free tier available |
| **Anthropic** (Claude) | [console.anthropic.com](https://console.anthropic.com) | Pay per use |
| **OpenAI** (GPT-4) | [platform.openai.com](https://platform.openai.com) | Pay per use |
| **xAI** (Grok) | [console.x.ai](https://console.x.ai) | Pay per use |

## Configuration

### Step 1: Set up environment variables

```bash
cp .env.example config/.env
```

Edit `config/.env` and add your API keys:

```ini
# Add the keys you have (you don't need all of them)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
```

> **Note:** `NEURODECK_TOKEN` is auto-generated on first run and added to your `.env` file automatically.

### Step 2: Configure agents in agents.ini

The file `config/agents.ini` controls everything. Here's how it works:

```ini
[orchestrator]
host = localhost
port = 9999
auth_token = GENERATE_ON_FIRST_RUN  # Auto-generated on first run
log_level = DEBUG
username = human
tts_enabled = true

# Each [agent:name] section creates an AI agent
[agent:claude]
provider = claude                          # claude, openai, groq, or xai
model = claude-sonnet-4-20250514           # Model to use
api_endpoint = https://api.anthropic.com/v1/messages
api_key_env = ANTHROPIC_API_KEY            # Which env var has the key
max_tokens = 4096
temperature = 0.7
tools = filesystem,chat_info,bash          # Tools this agent can use
system_prompt = You are Claude, a helpful assistant...

[agent:grok]
provider = xai
model = grok-3-mini
api_endpoint = https://api.x.ai/v1
api_key_env = XAI_API_KEY
# ... etc
```

**Key settings:**
- `provider`: Must match your API key type (claude/openai/groq/xai)
- `api_key_env`: Name of the environment variable containing your key
- `model`: The specific model to use (see provider docs for options)
- `tools`: Comma-separated list of tools the agent can use
- `system_prompt`: The agent's personality and instructions

### Available Models

**Groq** (fast, free tier):
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`

**Anthropic**:
- `claude-sonnet-4-20250514`
- `claude-3-5-haiku-20241022`

**OpenAI**:
- `gpt-4o`
- `gpt-4o-mini`

**xAI**:
- `grok-3-mini`
- `grok-2-latest`

## Running NeuroDeck

```bash
# Terminal 1: Start the orchestrator
source venv/bin/activate
source config/.env
./run_orchestrator.sh

# Terminal 2: Connect the console UI
source venv/bin/activate
source config/.env
./run_console.sh
```

## Usage

Once connected, just type messages. All configured agents will see them and can respond.

- **@mention**: Use `@agentname` to direct a message to a specific agent
- **@everyone / @all**: Mention all agents at once
- **Commands**: Type `/help` in the console to see available commands

## Admin Tools

### The Architect

Send messages to agents from external scripts (useful for debugging or automation):

```bash
# Message all agents
python architect_chat.py "@everyone What errors are you seeing?"

# Message specific agent
python architect_chat.py "@claude Describe your current state"
```

## Tools

Agents can use these built-in tools:

| Tool | Description |
|------|-------------|
| `filesystem` | Read/write files in allowed directories |
| `bash` | Execute shell commands (sandboxed) |
| `chat_info` | Get info about chat participants |
| `tts_config` | Adjust text-to-speech settings |

Tool permissions are configured per-agent in `agents.ini` and globally in `[tool:toolname]` sections.

## Troubleshooting

**"Missing API key" error**: Make sure you've sourced your .env file: `source config/.env`

**Agent not responding**: Check `logs/orchestrator.log` for errors

**TTS not working**: TTS requires additional setup - see `documentation/tts/`

## Documentation

- [Full Specification](documentation/NeuroDeck_Specification.md)
- [Adding Commands](documentation/adding-commands-cookbook.md)
- [Tool Development](documentation/built-in-tools-cookbook.md)
- [Bash Security Config](documentation/bash-tool-security-configuration.md)
- [TTS Setup](documentation/tts/)

## License

GPL-3.0 - see [LICENSE](LICENSE)