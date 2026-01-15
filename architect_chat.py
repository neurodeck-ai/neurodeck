#!/usr/bin/env python3
"""
The Architect - Direct communication with NeuroDeck agents.

This tool allows external processes (like Claude Code) to send messages
directly to the NeuroDeck chat as "the_architect". Useful for:
- Debugging agent issues by asking them directly what they're experiencing
- Injecting system announcements or instructions
- Testing agent responsiveness

Usage:
    python architect_chat.py <message>

Examples:
    # Ask all agents about errors
    python architect_chat.py "@all What tool errors are you seeing?"

    # Direct message to specific agent
    python architect_chat.py "@grok Describe your last tool timeout"

    # General announcement
    python architect_chat.py "System maintenance in 5 minutes"

Requirements:
    - Orchestrator must be running on localhost:9999
    - Auth token read from config/agents.ini [orchestrator] section
    - Or set NEURODECK_AUTH_TOKEN environment variable

The sender appears as "the_architect" in the chat.
"""

import asyncio
import configparser
import json
import os
import ssl
import sys
from datetime import datetime
from pathlib import Path


def get_auth_token():
    """Get auth token from config file."""
    config_path = Path(__file__).parent / "config" / "agents.ini"
    if config_path.exists():
        config = configparser.ConfigParser()
        config.read(config_path)
        if "orchestrator" in config:
            return config["orchestrator"].get("auth_token", "")
    return os.getenv("NEURODECK_AUTH_TOKEN", "")


async def send_message(message: str, host: str = "localhost", port: int = 9999):
    """Connect to orchestrator and send a chat message as the_architect."""

    token = get_auth_token()
    if not token:
        print("Error: Could not find auth token in config/agents.ini")
        return False

    # Create SSL context (allow self-signed certs)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        reader, writer = await asyncio.open_connection(
            host, port, ssl=ssl_context,
            limit=10 * 1024 * 1024
        )

        # Authenticate as UI client named the_architect
        auth_msg = {
            "type": "auth",
            "token": token,
            "client_id": "ui-the_architect",
            "client_type": "ui"
        }
        writer.write((json.dumps(auth_msg) + "\n").encode())
        await writer.drain()

        # Wait for auth response
        response = await reader.readline()
        auth_result = json.loads(response.decode())

        if auth_result.get("type") != "auth_result" or not auth_result.get("success"):
            print(f"Authentication failed: {auth_result}")
            return False

        print(f"✓ Connected as the_architect")

        # Send chat message
        chat_msg = {
            "type": "chat",
            "sender": "the_architect",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        writer.write((json.dumps(chat_msg) + "\n").encode())
        await writer.drain()

        print(f"✓ Message sent: {message[:50]}...")

        # Keep connection open briefly to receive any immediate responses
        print("Listening for responses (5 seconds)...")
        try:
            for _ in range(10):
                try:
                    response = await asyncio.wait_for(reader.readline(), timeout=0.5)
                    if response:
                        msg = json.loads(response.decode())
                        if msg.get("type") == "chat":
                            print(f"\n[{msg.get('sender')}]: {msg.get('content')[:200]}")
                except asyncio.TimeoutError:
                    pass
        except Exception as e:
            pass

        writer.close()
        await writer.wait_closed()
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python architect_chat.py <message>")
        print("Example: python architect_chat.py '@all What tool errors are you seeing?'")
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    asyncio.run(send_message(message))


if __name__ == "__main__":
    main()
