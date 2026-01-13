"""Example: Using DeepAgents with OpenRouter

This example demonstrates how to use DeepAgents with OpenRouter to access
various AI models through a single API.

Setup:
1. Copy .env.example to .env
2. Add your OpenRouter API key: OPENROUTER_API_KEY=your_key_here
3. Optionally set OPENROUTER_MODEL to your preferred model
4. Run: uv run python examples/openrouter_example.py
"""

from deepagents import create_deep_agent, get_openrouter_model, is_openrouter_configured


def example_1_automatic_configuration():
    """Example 1: Automatic OpenRouter configuration via environment variables."""
    print("\n" + "=" * 70)
    print("Example 1: Automatic Configuration")
    print("=" * 70)
    
    if not is_openrouter_configured():
        print("⚠️  OpenRouter not configured. Set OPENROUTER_API_KEY in .env file")
        return
    
    # When OPENROUTER_API_KEY is set, this automatically uses OpenRouter
    agent = create_deep_agent(
        system_prompt="You are a helpful AI assistant. Be concise."
    )
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one sentence."}]
    })
    
    print(f"Response: {result['messages'][-1].content}")


def example_2_explicit_model():
    """Example 2: Explicitly specify an OpenRouter model."""
    print("\n" + "=" * 70)
    print("Example 2: Explicit Model Selection")
    print("=" * 70)
    
    if not is_openrouter_configured():
        print("⚠️  OpenRouter not configured. Set OPENROUTER_API_KEY in .env file")
        return
    
    # Explicitly choose a specific model
    model = get_openrouter_model("anthropic/claude-3-haiku")
    
    agent = create_deep_agent(
        model=model,
        system_prompt="You are a helpful AI assistant. Be concise."
    )
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Name one benefit of AI. One sentence only."}]
    })
    
    print(f"Model: anthropic/claude-3-haiku")
    print(f"Response: {result['messages'][-1].content}")


def example_3_multiple_models():
    """Example 3: Use different models for different tasks."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple Models for Different Tasks")
    print("=" * 70)
    
    if not is_openrouter_configured():
        print("⚠️  OpenRouter not configured. Set OPENROUTER_API_KEY in .env file")
        return
    
    # Fast model for simple tasks
    fast_agent = create_deep_agent(
        model=get_openrouter_model("anthropic/claude-3-haiku"),
        system_prompt="You are a helpful assistant. Be very brief."
    )
    
    # Powerful model for complex tasks (latest as of 2026)
    powerful_agent = create_deep_agent(
        model=get_openrouter_model("anthropic/claude-sonnet-4.5"),
        system_prompt="You are an expert analyst. Be thorough but concise."
    )
    
    # Simple task with fast model
    simple_result = fast_agent.invoke({
        "messages": [{"role": "user", "content": "What's the capital of France?"}]
    })
    print(f"\nFast model (Haiku): {simple_result['messages'][-1].content}")
    
    # Complex task with powerful model
    complex_result = powerful_agent.invoke({
        "messages": [{"role": "user", "content": "Explain quantum computing in 2 sentences."}]
    })
    print(f"\nPowerful model (Sonnet): {complex_result['messages'][-1].content}")


def example_4_with_tools():
    """Example 4: Using OpenRouter with custom tools."""
    print("\n" + "=" * 70)
    print("Example 4: OpenRouter with Custom Tools")
    print("=" * 70)
    
    if not is_openrouter_configured():
        print("⚠️  OpenRouter not configured. Set OPENROUTER_API_KEY in .env file")
        return
    
    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle.
        
        Args:
            length: The length of the rectangle
            width: The width of the rectangle
            
        Returns:
            The area of the rectangle
        """
        return length * width
    
    agent = create_deep_agent(
        model=get_openrouter_model("anthropic/claude-sonnet-4.5"),
        tools=[calculate_area],
        system_prompt="You are a helpful assistant with access to calculation tools."
    )
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What's the area of a rectangle that is 5 meters long and 3 meters wide?"}]
    })
    
    print(f"Response: {result['messages'][-1].content}")


def example_5_with_subagents():
    """Example 5: Using different OpenRouter models for subagents."""
    print("\n" + "=" * 70)
    print("Example 5: OpenRouter with Subagents")
    print("=" * 70)
    
    if not is_openrouter_configured():
        print("⚠️  OpenRouter not configured. Set OPENROUTER_API_KEY in .env file")
        return
    
    # Define a specialized subagent with a different model
    math_subagent = {
        "name": "math-expert",
        "description": "Expert at solving mathematical problems",
        "system_prompt": "You are a mathematics expert. Solve problems step by step.",
        "model": get_openrouter_model("anthropic/claude-3-opus"),  # More powerful model
    }
    
    # Main agent uses a different model (latest as of 2026)
    agent = create_deep_agent(
        model=get_openrouter_model("anthropic/claude-sonnet-4.5"),
        subagents=[math_subagent],
        system_prompt="You are a helpful assistant. Delegate math problems to the math expert."
    )
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Calculate the sum of squares from 1 to 5."}]
    })
    
    print(f"Response: {result['messages'][-1].content}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DeepAgents + OpenRouter Examples")
    print("=" * 70)
    
    if not is_openrouter_configured():
        print("\n⚠️  OpenRouter is not configured!")
        print("\nTo run these examples:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenRouter API key: OPENROUTER_API_KEY=your_key_here")
        print("3. Get your key from: https://openrouter.ai/keys")
        print("\nRunning configuration check only...\n")
    
    # Run examples
    example_1_automatic_configuration()
    example_2_explicit_model()
    example_3_multiple_models()
    example_4_with_tools()
    example_5_with_subagents()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

