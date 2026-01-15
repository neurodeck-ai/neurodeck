"""Test imports for NeuroDeck modules."""

def test_common_imports():
    """Test common module imports."""
    try:
        from neurodeck.common.config import load_config
        print("✅ Config import successful")
        
        from neurodeck.common.protocol import MessageProtocol
        print("✅ Protocol import successful")
        
        from neurodeck.common.database import ContextDatabase
        print("✅ Database import successful")
        
        from neurodeck.common.security import CertificateManager
        print("✅ Security import successful")
        
        from neurodeck.common.logging import get_logger
        print("✅ Logging import successful")
        
        print("🎉 All common imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Common import failed: {e}")
        return False

def test_orchestrator_imports():
    """Test orchestrator module imports."""
    try:
        from neurodeck.orchestrator.server import OrchestratorServer
        print("✅ Server import successful")
        
        from neurodeck.orchestrator.process_manager import ProcessManager
        print("✅ Process manager import successful")
        
        print("🎉 All orchestrator imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator import failed: {e}")
        return False

def test_agent_imports():
    """Test agent module imports (without AI libraries).""" 
    try:
        # Test base agent
        import sys
        import os
        
        # Mock the AI library imports temporarily
        class MockAnthropic:
            class Anthropic:
                def __init__(self, api_key):
                    pass
                    
        class MockOpenAI:
            class AsyncOpenAI:
                def __init__(self, api_key, base_url=None):
                    pass
                    
        class MockGroq:
            class AsyncGroq:
                def __init__(self, api_key):
                    pass
        
        # Temporarily add mocks to sys.modules
        sys.modules['anthropic'] = MockAnthropic()
        sys.modules['openai'] = MockOpenAI()
        sys.modules['groq'] = MockGroq()
        
        from neurodeck.agents.base_agent import BaseAgent
        print("✅ Base agent import successful")
        
        from neurodeck.agents.claude_agent import ClaudeAgent
        print("✅ Claude agent import successful")
        
        from neurodeck.agents.openai_agent import OpenAIAgent
        print("✅ OpenAI agent import successful")
        
        from neurodeck.agents.xai_agent import XAIAgent
        print("✅ xAI agent import successful")
        
        from neurodeck.agents.groq_agent import GroqAgent
        print("✅ Groq agent import successful")
        
        print("🎉 All agent imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Agent import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from neurodeck.common.config import ConfigManager
        
        # Test with actual config file
        config_manager = ConfigManager("config/agents.ini")
        orchestrator_config = config_manager.load_orchestrator_config()
        agent_configs = config_manager.load_agent_configs()
        mcp_configs = config_manager.load_mcp_tool_configs()
        
        print(f"✅ Config loaded: {len(agent_configs)} agents, {len(mcp_configs)} MCP tools")
        print(f"   Orchestrator: {orchestrator_config.host}:{orchestrator_config.port}")
        print(f"   Agents: {list(agent_configs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all import tests."""
    print("🧪 Testing NeuroDeck imports...\n")
    
    tests = [
        ("Common modules", test_common_imports),
        ("Orchestrator modules", test_orchestrator_imports), 
        ("Agent modules", test_agent_imports),
        ("Configuration loading", test_config_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*50}")
    print("📋 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! System is ready for testing.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)