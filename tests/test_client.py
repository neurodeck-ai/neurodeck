#!/usr/bin/env python3
"""
Test client for NeuroDeck orchestrator.

This client connects to the orchestrator, authenticates, and allows
sending chat messages to test AI agent responses.
"""

import asyncio
import ssl
import sys
import os
from typing import Optional, Dict, Any

# Add neurodeck to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurodeck'))

from neurodeck.common.protocol import (
    MessageProtocol, MessageType, 
    auth_message, chat_message
)
from neurodeck.common.security import CertificateManager
from neurodeck.common.config import ConfigManager


class TestClient:
    """Test client for communicating with NeuroDeck orchestrator."""
    
    def __init__(self, host: str = "localhost", port: int = 9999, config_path: str = "config/agents.ini"):
        self.host = host
        self.port = port
        self.config_path = config_path
        
        # Connection state
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.authenticated = False
        
        # Load configuration to get auth token
        self.config_manager = ConfigManager(config_path)
        self.orchestrator_config = self.config_manager.load_orchestrator_config()
        
        print(f"Test client initialized - connecting to {host}:{port}")
    
    async def connect(self):
        """Connect to the orchestrator via TLS."""
        print("Connecting to orchestrator...")
        
        # Create SSL context for client
        cert_manager = CertificateManager()
        ssl_context = cert_manager.create_ssl_context(is_server=False)
        
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port, ssl=ssl_context
            )
            self.connected = True
            print("✅ Connected to orchestrator")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    async def authenticate(self):
        """Authenticate with the orchestrator."""
        print("Authenticating...")
        
        client_id = "ui-test-client"
        auth_msg = auth_message(self.orchestrator_config.auth_token, client_id)
        
        await self.send_message(auth_msg)
        
        # Wait for authentication result
        response = await self.receive_message()
        if not response or response.get("type") != MessageType.AUTH_RESULT.value:
            raise RuntimeError("Invalid authentication response")
        
        if not response.get("success"):
            error_msg = response.get("message", "Authentication failed")
            raise RuntimeError(f"Authentication failed: {error_msg}")
        
        self.authenticated = True
        print("✅ Authenticated successfully")
    
    async def send_message(self, message_dict: Dict[str, Any]):
        """Send message to orchestrator."""
        if not self.writer:
            raise RuntimeError("Not connected")
        
        try:
            encoded = MessageProtocol.encode_message(message_dict)
            self.writer.write(encoded)
            await self.writer.drain()
        except Exception as e:
            print(f"Failed to send message: {e}")
            raise
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from orchestrator."""
        if not self.reader:
            return None
        
        try:
            line = await self.reader.readline()
            if not line:
                return None
            
            return MessageProtocol.decode_message(line)
        except Exception as e:
            print(f"Failed to receive message: {e}")
            return None
    
    async def send_chat(self, content: str):
        """Send chat message and display responses."""
        print(f"\n💬 Sending: {content}")
        
        chat_msg = chat_message("user", content)
        await self.send_message(chat_msg)
        
        print("📡 Waiting for responses...")
        
        # Listen for responses for a reasonable time
        responses_received = 0
        timeout_count = 0
        max_timeout = 10  # Wait up to 10 seconds for responses
        
        while timeout_count < max_timeout:
            try:
                # Wait for message with short timeout
                message = await asyncio.wait_for(self.receive_message(), timeout=1.0)
                
                if message is None:
                    break
                
                message_type = message.get("type")
                
                if message_type == MessageType.CHAT.value:
                    sender = message.get("sender", "unknown")
                    response_content = message.get("content", "")
                    print(f"\n🤖 {sender}: {response_content}")
                    responses_received += 1
                    timeout_count = 0  # Reset timeout when we get responses
                    
                elif message_type == MessageType.SYSTEM_LOG.value:
                    # Show system logs for debugging
                    sender = message.get("sender", "system")
                    level = message.get("level", "info")
                    log_content = message.get("content", "")
                    print(f"📋 [{sender}] {level.upper()}: {log_content}")
                
            except asyncio.TimeoutError:
                timeout_count += 1
                if responses_received == 0 and timeout_count >= 3:
                    print("⏰ No responses yet, still waiting...")
        
        if responses_received == 0:
            print("😐 No responses received from any agents")
        else:
            print(f"\n✅ Received {responses_received} response(s)")
    
    async def disconnect(self):
        """Disconnect from orchestrator."""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
                print("👋 Disconnected")
            except Exception as e:
                print(f"Error disconnecting: {e}")
        
        self.connected = False
        self.authenticated = False


async def interactive_test():
    """Interactive test session with the orchestrator."""
    print("🧪 NeuroDeck Test Client")
    print("=" * 50)
    
    client = TestClient()
    
    try:
        # Connect and authenticate
        await client.connect()
        await client.authenticate()
        
        print("\n🎯 Ready to chat! Type 'quit' to exit, 'help' for test commands.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💭 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 Test Commands:")
                    print("  'hello everyone' - General greeting to all agents")
                    print("  '@claudius help me with code' - Direct mention to Claudius")
                    print("  '@grok check this for security issues' - Ask Grok about security")
                    print("  '@chatgpt explain this concept' - Ask ChatGPT for explanation") 
                    print("  '@kimi analyze this complex problem' - Ask Kimi for deep analysis")
                    print("  'who's online?' - See which agents respond")
                    print("  'quit' - Exit the test client")
                    continue
                
                # Send message and wait for responses
                await client.send_chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    finally:
        await client.disconnect()


async def automated_test():
    """Run automated tests with predefined messages."""
    print("🤖 Running Automated Tests")
    print("=" * 50)
    
    client = TestClient()
    
    try:
        await client.connect()
        await client.authenticate()
        
        # Test messages
        test_messages = [
            "Hello everyone! This is a test message.",
            "@claudius Can you help me understand how NeuroDeck works?",
            "@grok What security considerations should I be aware of?",
            "@chatgpt Explain the concept of multi-agent AI systems.",
            "@kimi Can you analyze the architecture of this chat system?",
            "Who's online right now?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{'='*20} Test {i}/{len(test_messages)} {'='*20}")
            await client.send_chat(message)
            await asyncio.sleep(2)  # Brief pause between tests
        
        print(f"\n{'='*50}")
        print("✅ Automated tests completed!")
        
    finally:
        await client.disconnect()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroDeck Test Client")
    parser.add_argument("--auto", action="store_true", help="Run automated tests")
    parser.add_argument("--host", default="localhost", help="Orchestrator host")
    parser.add_argument("--port", type=int, default=9999, help="Orchestrator port")
    parser.add_argument("--config", default="config/agents.ini", help="Configuration file")
    
    args = parser.parse_args()
    
    try:
        if args.auto:
            asyncio.run(automated_test())
        else:
            asyncio.run(interactive_test())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()