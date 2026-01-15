#!/usr/bin/env python3
"""
Test AI APIs directly without any orchestrator/agent infrastructure.
This will isolate whether the issue is with the APIs or the agent code.
"""

import asyncio
import os
import sys

async def test_claude_direct():
    """Test Claude API directly."""
    try:
        import anthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found")
            return False
        
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": "What is 1+1? Just answer the number."}]
        )
        answer = response.content[0].text.strip()
        print(f"✅ Claude Direct: {answer}")
        return True
    except Exception as e:
        print(f"❌ Claude Direct Error: {e}")
        return False

async def test_openai_direct():
    """Test OpenAI API directly."""
    try:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return False
        
        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=50,
            messages=[{"role": "user", "content": "What is 1+1? Just answer the number."}]
        )
        answer = response.choices[0].message.content.strip()
        print(f"✅ OpenAI Direct: {answer}")
        return True
    except Exception as e:
        print(f"❌ OpenAI Direct Error: {e}")
        return False

async def test_xai_direct():
    """Test xAI API directly with enhanced debugging."""
    try:
        import openai
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            print("❌ XAI_API_KEY not found")
            return False
        
        print(f"🔍 Using xAI API key: {api_key[:10]}...")
        
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=30.0  # Add timeout
        )
        
        print("🔍 Making request to xAI API...")
        # Try different model options
        models_to_try = ["grok-4-0709", "grok-4", "grok-4-latest", "grok-3", "grok-2"]
        
        for model in models_to_try:
            print(f"🔍 Trying model: {model}")
            try:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=1000,  # Much higher token limit
                    temperature=0.1,
                    messages=[{"role": "user", "content": "What is 1+1?"}]
                )
                print(f"✅ Model {model} works!")
                break  # Exit loop if successful
            except Exception as model_error:
                print(f"❌ Model {model} failed: {model_error}")
                if model == models_to_try[-1]:  # Last model failed
                    raise model_error  # Re-raise the last error
        
        print(f"🔍 Response object: {response.choices[0] if response.choices else 'No choices'}")
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content
            print(f"🔍 Message content: '{content}' (type: {type(content)})")
            
            if content and content.strip():
                answer = content.strip()
                print(f"✅ xAI Direct: {answer}")
                return True
            else:
                print("❌ xAI returned empty content")
                return False
        else:
            print("❌ xAI returned no choices")
            return False
            
    except Exception as e:
        print(f"❌ xAI Direct Error: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

async def test_groq_direct():
    """Test Groq API directly."""
    try:
        import groq
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY not found")
            return False
        
        client = groq.AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=50,
            messages=[{"role": "user", "content": "What is 1+1? Just answer the number."}]
        )
        answer = response.choices[0].message.content.strip()
        print(f"✅ Groq Direct: {answer}")
        return True
    except Exception as e:
        print(f"❌ Groq Direct Error: {e}")
        return False

async def main():
    """Test all APIs directly."""
    print("🧪 Testing AI APIs Directly (bypassing all agent code)")
    print("=" * 60)
    
    # Load environment variables
    if os.path.exists('config/.env'):
        with open('config/.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    results = {}
    
    print("Testing Claude...")
    results['claude'] = await test_claude_direct()
    
    print("Testing OpenAI...")
    results['openai'] = await test_openai_direct()
    
    print("Testing xAI...")
    results['xai'] = await test_xai_direct()
    
    print("Testing Groq...")
    results['groq'] = await test_groq_direct()
    
    print("\n" + "=" * 60)
    print("📊 Direct API Test Results:")
    for provider, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"  {provider.upper()}: {status}")
    
    working_count = sum(results.values())
    print(f"\n🎯 {working_count}/4 APIs working directly")
    
    if working_count == 4:
        print("\n🔍 All APIs work directly - the issue is in the agent message handling code!")
    else:
        print(f"\n🔍 {4-working_count} APIs are broken - this explains the non-responsive agents!")

if __name__ == "__main__":
    asyncio.run(main())