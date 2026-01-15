#!/usr/bin/env python3
"""
Test individual agents standalone (without orchestrator server).

This script directly imports and tests each agent's response generation
to isolate issues in the agent code vs orchestrator infrastructure.
"""

import asyncio
import os
import sys
from typing import Dict, Any, List

# Add the project to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurodeck'))

async def test_claude_agent_standalone():
    """Test Claude agent response generation directly."""
    try:
        from neurodeck.agents.claude_agent import ClaudeAgent
        from neurodeck.common.config import load_config
        
        print("🧪 Testing Claude Agent Standalone")
        print("=" * 40)
        
        # Load environment variables
        if os.path.exists('config/.env'):
            with open('config/.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Create agent instance (won't connect to orchestrator)
        agents, orchestrator_config, mcp_tools = load_config('config/agents.ini')
        agent_config = agents.get('claudius')
        
        if not agent_config:
            print("❌ Could not find claudius agent config")
            return False
            
        # Initialize API client directly
        api_key = os.getenv(agent_config.api_key_env)
        if not api_key:
            print(f"❌ API key {agent_config.api_key_env} not found")
            return False
            
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Test conversation format
        conversation = [
            {"role": "user", "content": "@claudius What is 1+1? Just answer the number."}
        ]
        
        print("🔍 Making direct API call...")
        response = await client.messages.create(
            model=agent_config.model,
            max_tokens=agent_config.max_tokens,
            temperature=agent_config.temperature,
            messages=conversation
        )
        
        if response.content and len(response.content) > 0:
            answer = response.content[0].text.strip()
            print(f"✅ Claude Standalone: {answer}")
            return True
        else:
            print("❌ Empty response from Claude")
            return False
            
    except Exception as e:
        print(f"❌ Claude Standalone Error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

async def test_openai_agent_standalone():
    """Test OpenAI agent response generation directly."""
    try:
        from neurodeck.common.config import load_config
        
        print("🧪 Testing OpenAI Agent Standalone")
        print("=" * 40)
        
        # Load environment variables
        if os.path.exists('config/.env'):
            with open('config/.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Get agent config
        agents, orchestrator_config, mcp_tools = load_config('config/agents.ini')
        agent_config = agents.get('chatgpt')
        
        if not agent_config:
            print("❌ Could not find chatgpt agent config")
            return False
            
        # Initialize API client directly
        api_key = os.getenv(agent_config.api_key_env)
        if not api_key:
            print(f"❌ API key {agent_config.api_key_env} not found") 
            return False
            
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Test conversation with system prompt
        messages = [
            {"role": "system", "content": agent_config.system_prompt},
            {"role": "user", "content": "@chatgpt What is 1+1? Just answer the number."}
        ]
        
        print("🔍 Making direct API call...")
        response = await client.chat.completions.create(
            model=agent_config.model,
            messages=messages,
            max_tokens=agent_config.max_tokens,
            temperature=agent_config.temperature
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                answer = content.strip()
                print(f"✅ OpenAI Standalone: {answer}")
                return True
            else:
                print("❌ Empty content from OpenAI")
                return False
        else:
            print("❌ No choices from OpenAI")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI Standalone Error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

async def test_xai_agent_standalone():
    """Test xAI agent response generation directly."""
    try:
        from neurodeck.common.config import load_config
        
        print("🧪 Testing xAI Agent Standalone")
        print("=" * 40)
        
        # Load environment variables
        if os.path.exists('config/.env'):
            with open('config/.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Get agent config
        agents, orchestrator_config, mcp_tools = load_config('config/agents.ini')
        agent_config = agents.get('grok')
        
        if not agent_config:
            print("❌ Could not find grok agent config")
            return False
            
        # Initialize API client directly
        api_key = os.getenv(agent_config.api_key_env)
        if not api_key:
            print(f"❌ API key {agent_config.api_key_env} not found")
            return False
            
        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Test conversation with system prompt
        messages = [
            {"role": "system", "content": agent_config.system_prompt},
            {"role": "user", "content": "@grok What is 1+1? Just answer the number."}
        ]
        
        print("🔍 Making direct API call...")
        response = await client.chat.completions.create(
            model=agent_config.model,
            messages=messages,
            max_tokens=agent_config.max_tokens,
            temperature=agent_config.temperature
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                answer = content.strip()
                print(f"✅ xAI Standalone: {answer}")
                return True
            else:
                print("❌ Empty content from xAI")
                print(f"🔍 Response object: {response.choices[0]}")
                return False
        else:
            print("❌ No choices from xAI")
            return False
            
    except Exception as e:
        print(f"❌ xAI Standalone Error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

async def test_groq_agent_standalone():
    """Test Groq agent response generation directly."""
    try:
        from neurodeck.common.config import load_config
        
        print("🧪 Testing Groq Agent Standalone")
        print("=" * 40)
        
        # Load environment variables
        if os.path.exists('config/.env'):
            with open('config/.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Get agent config
        agents, orchestrator_config, mcp_tools = load_config('config/agents.ini')
        agent_config = agents.get('kimi')
        
        if not agent_config:
            print("❌ Could not find kimi agent config")
            return False
            
        # Initialize API client directly
        api_key = os.getenv(agent_config.api_key_env)
        if not api_key:
            print(f"❌ API key {agent_config.api_key_env} not found")
            return False
            
        import groq
        client = groq.AsyncGroq(api_key=api_key)
        
        # Test conversation with system prompt
        messages = [
            {"role": "system", "content": agent_config.system_prompt},
            {"role": "user", "content": "@kimi What is 1+1? Just answer the number."}
        ]
        
        print("🔍 Making direct API call...")
        response = await client.chat.completions.create(
            model=agent_config.model,
            messages=messages,
            max_tokens=agent_config.max_tokens,
            temperature=agent_config.temperature
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                answer = content.strip()
                print(f"✅ Groq Standalone: {answer}")
                return True
            else:
                print("❌ Empty content from Groq")
                return False
        else:
            print("❌ No choices from Groq")
            return False
            
    except Exception as e:
        print(f"❌ Groq Standalone Error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Test all agents standalone."""
    print("🧪 Testing All Agents Standalone (bypassing orchestrator)")
    print("=" * 60)
    
    results = {}
    
    print("\n" + "="*60)
    results['claude'] = await test_claude_agent_standalone()
    
    print("\n" + "="*60)
    results['openai'] = await test_openai_agent_standalone()
    
    print("\n" + "="*60)
    results['xai'] = await test_xai_agent_standalone()
    
    print("\n" + "="*60)
    results['groq'] = await test_groq_agent_standalone()
    
    print("\n" + "="*60)
    print("📊 Standalone Agent Test Results:")
    for provider, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"  {provider.upper()}: {status}")
    
    working_count = sum(results.values())
    print(f"\n🎯 {working_count}/4 agents working standalone")
    
    if working_count == 4:
        print("\n🔍 All agents work standalone - the issue is in orchestrator integration!")
    else:
        print(f"\n🔍 {4-working_count} agents have issues even standalone!")

if __name__ == "__main__":
    asyncio.run(main())