"""Test script for OpenRouter integration with deepagents.

This script demonstrates how to use OpenRouter with deepagents.
Make sure to set OPENROUTER_API_KEY and optionally OPENROUTER_MODEL in your .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_openrouter_configuration():
    """Test if OpenRouter is properly configured."""
    print("=" * 60)
    print("Testing OpenRouter Configuration")
    print("=" * 60)
    
    from deepagents import is_openrouter_configured
    
    if is_openrouter_configured():
        print("✓ OpenRouter API key is configured")
        model_name = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
        print(f"✓ Using model: {model_name}")
        return True
    else:
        print("✗ OpenRouter API key is NOT configured")
        print("\nPlease set OPENROUTER_API_KEY in your .env file")
        print("Get your API key from: https://openrouter.ai/keys")
        return False


def test_openrouter_model_creation():
    """Test creating an OpenRouter model instance."""
    print("\n" + "=" * 60)
    print("Testing OpenRouter Model Creation")
    print("=" * 60)
    
    try:
        from deepagents import get_openrouter_model
        
        # Test with default model
        model = get_openrouter_model()
        print(f"✓ Successfully created OpenRouter model")
        print(f"  Model: {model.model_name}")
        print(f"  Base URL: {model.openai_api_base}")
        
        # Test with specific model (latest as of 2026)
        custom_model = get_openrouter_model("openai/gpt-5.2")
        print(f"✓ Successfully created custom OpenRouter model")
        print(f"  Model: {custom_model.model_name}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create OpenRouter model: {e}")
        return False


def test_create_deep_agent_with_openrouter():
    """Test creating a deep agent with OpenRouter."""
    print("\n" + "=" * 60)
    print("Testing Deep Agent Creation with OpenRouter")
    print("=" * 60)
    
    try:
        from deepagents import create_deep_agent
        
        # Create agent - will automatically use OpenRouter if configured
        agent = create_deep_agent(
            system_prompt="You are a helpful AI assistant."
        )
        print("✓ Successfully created deep agent with OpenRouter")
        print(f"  Agent type: {type(agent).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create deep agent: {e}")
        return False


def test_create_deep_agent_with_explicit_model():
    """Test creating a deep agent with an explicit OpenRouter model."""
    print("\n" + "=" * 60)
    print("Testing Deep Agent with Explicit OpenRouter Model")
    print("=" * 60)
    
    try:
        from deepagents import create_deep_agent, get_openrouter_model
        
        # Create specific OpenRouter model
        model = get_openrouter_model("meta-llama/llama-3.1-70b-instruct")
        
        # Create agent with explicit model
        agent = create_deep_agent(
            model=model,
            system_prompt="You are a helpful AI assistant."
        )
        print("✓ Successfully created deep agent with explicit OpenRouter model")
        print(f"  Model: meta-llama/llama-3.1-70b-instruct")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create deep agent with explicit model: {e}")
        return False


def test_simple_invocation():
    """Test a simple agent invocation (requires valid API key)."""
    print("\n" + "=" * 60)
    print("Testing Simple Agent Invocation")
    print("=" * 60)
    
    if not is_openrouter_configured():
        print("⊘ Skipping invocation test (no API key configured)")
        return None
    
    try:
        from deepagents import create_deep_agent
        
        agent = create_deep_agent(
            system_prompt="You are a helpful assistant. Keep responses brief."
        )
        
        print("Invoking agent with test message...")
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Say 'Hello from OpenRouter!' and nothing else."}]
        })
        
        response = result["messages"][-1].content
        print(f"✓ Agent responded successfully")
        print(f"  Response: {response}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to invoke agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DeepAgents OpenRouter Integration Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Configuration
    results.append(("Configuration", test_openrouter_configuration()))
    
    if results[0][1]:  # Only continue if configured
        # Test 2: Model creation
        results.append(("Model Creation", test_openrouter_model_creation()))
        
        # Test 3: Deep agent creation
        results.append(("Deep Agent Creation", test_create_deep_agent_with_openrouter()))
        
        # Test 4: Explicit model
        results.append(("Explicit Model", test_create_deep_agent_with_explicit_model()))
        
        # Test 5: Simple invocation (optional)
        invocation_result = test_simple_invocation()
        if invocation_result is not None:
            results.append(("Simple Invocation", invocation_result))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    from deepagents import is_openrouter_configured
    main()

