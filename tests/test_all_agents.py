#!/usr/bin/env python3
"""
Test all agents individually with 1+1 question.
"""

import asyncio
import sys
import os

# Add neurodeck to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurodeck'))

from test_client import TestClient
from neurodeck.common.protocol import chat_message

async def test_agent(agent_name):
    print(f"🧪 Testing {agent_name.upper()}")
    print("=" * 30)
    
    client = TestClient()
    
    try:
        await client.connect()
        await client.authenticate()
        
        # Send message to specific agent
        message_text = f"@{agent_name} What is 1+1? Just answer the number."
        print(f"Sending: {message_text}")
        
        chat_msg = chat_message("user", message_text)
        await client.send_message(chat_msg)
        
        # Listen for response
        responses = 0
        all_responses = []
        for i in range(20):  # Wait up to 20 seconds for slower responses
            try:
                message = await asyncio.wait_for(client.receive_message(), timeout=1.0)
                if message:
                    msg_type = message.get("type", "unknown")
                    if msg_type == "chat":
                        sender = message.get("sender", "unknown")
                        content = message.get("content", "")
                        all_responses.append(f"{sender}: {content[:100]}...")
                        if sender == agent_name:
                            print(f"✅ {agent_name}: {content}")
                            responses += 1
                            break
                        elif len(all_responses) <= 3:  # Show first few responses
                            print(f"📢 {sender}: {content[:100]}{'...' if len(content) > 100 else ''}")
                    elif msg_type == "system_log":
                        sender = message.get("sender", "system") 
                        level = message.get("level", "info")
                        log_content = message.get("message", "")
                        if sender == agent_name and level in ["error", "ERROR"]:
                            print(f"⚠️  {agent_name} error: {log_content}")
            except asyncio.TimeoutError:
                if i < 5 or i % 3 == 0:  # Show fewer "waiting" messages
                    print(f"... waiting ({i+1}/20)")
        
        if responses == 0:
            print(f"❌ No response from {agent_name}")
            if all_responses:
                print(f"   Other agents responded: {len(all_responses)} messages")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
    
    print()

async def main():
    agents = ["claudius", "grok", "chatgpt", "kimi"]
    
    for agent in agents:
        await test_agent(agent)
        await asyncio.sleep(0.5)  # Brief pause between tests

if __name__ == "__main__":
    asyncio.run(main())