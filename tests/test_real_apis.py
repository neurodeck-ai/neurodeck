#!/usr/bin/env python3
"""
Test real API calls for each AI provider to verify they work.
"""

import asyncio
import sys
import os

# Add neurodeck to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurodeck'))

from neurodeck.common.config import ConfigManager

async def test_claude_api():
    """Test Anthropic Claude API directly."""
    print("🧪 Testing Claude API...")
    
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in environment")
            return False
        
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "What is 1+1? Just answer the number."}
            ]
        )
        
        answer = response.content[0].text.strip()
        print(f"✅ Claude response: {answer}")
        return True
        
    except Exception as e:
        print(f"❌ Claude API error: {e}")
        return False

async def test_openai_api():
    """Test OpenAI API directly."""
    print("🧪 Testing OpenAI API...")
    
    try:
        import openai
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "What is 1+1? Just answer the number."}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ OpenAI response: {answer}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

async def test_xai_api():
    """Test xAI API directly."""
    print("🧪 Testing xAI API...")
    
    try:
        import openai
        
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            print("❌ XAI_API_KEY not found in environment")
            return False
        
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        response = await client.chat.completions.create(
            model="grok-4-latest",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "What is 1+1? Just answer the number."}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ xAI response: {answer}")
        return True
        
    except Exception as e:
        print(f"❌ xAI API error: {e}")
        return False

async def test_groq_api():
    """Test Groq API directly."""
    print("🧪 Testing Groq API...")
    
    try:
        import groq
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment")
            return False
        
        client = groq.AsyncGroq(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "What is 1+1? Just answer the number."}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ Groq response: {answer}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False

async def main():
    """Test all APIs."""
    print("🔍 Testing Real API Calls")
    print("=" * 40)
    
    # Load environment variables
    if os.path.exists('config/.env'):
        print("Loading environment variables from config/.env...")
        with open('config/.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    results = {}
    
    # Test each API
    results['claude'] = await test_claude_api()
    print()
    
    results['openai'] = await test_openai_api()
    print()
    
    results['xai'] = await test_xai_api()
    print()
    
    results['groq'] = await test_groq_api()
    print()
    
    # Summary
    print("=" * 40)
    print("📋 API Test Results:")
    for provider, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"  {provider.upper()}: {status}")
    
    working_count = sum(results.values())
    print(f"\n🎯 {working_count}/4 APIs working correctly")

if __name__ == "__main__":
    asyncio.run(main())