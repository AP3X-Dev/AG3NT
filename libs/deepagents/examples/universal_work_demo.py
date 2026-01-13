"""Interactive demo of the Universal Work System middleware.

This script creates an agent with the UniversalWorkMiddleware and allows
you to interact with it to test the work management features.
"""

import os
from langchain.agents import create_agent
from deepagents.middleware.universal_work import UniversalWorkMiddleware


def main():
    """Run interactive demo of Universal Work System."""
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY=your-key-here")
        return
    
    print("=" * 70)
    print("Universal Work System - Interactive Demo")
    print("=" * 70)
    print()
    print("This agent has access to the Universal Work System with these tools:")
    print("  • write_todos / read_todos - Backward-compatible todo management")
    print("  • work_item_create / work_item_get / inbox_list - Work item management")
    print("  • link_create / link_list - Link work items together")
    print("  • agent_session_start / agent_activity_log - Session tracking")
    print("  • triage_suggest - AI-powered duplicate/related item detection")
    print("  • feedback_record - Record feedback on suggestions")
    print()
    print("Data is stored in: .universal_work/")
    print()
    print("=" * 70)
    print()
    
    # Create agent with Universal Work middleware
    agent = create_agent(
        model="anthropic:claude-sonnet-4-20250514",
        middleware=[UniversalWorkMiddleware(storage_path=".universal_work")],
    )
    
    print("Agent ready! Try these example prompts:")
    print()
    print("1. 'Create a work item for fixing the login bug'")
    print("2. 'List all work items in the inbox'")
    print("3. 'Create a plan with 3 steps for implementing user authentication'")
    print("4. 'Show me the current todos'")
    print("5. 'Create another work item about login issues, then check for duplicates'")
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70)
    print()
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            print("\n🤖 Agent: ", end="", flush=True)
            
            # Invoke agent
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            
            # Get the last AI message
            ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
            if ai_messages:
                response = ai_messages[-1].content
                print(response)
            else:
                print("(No response)")
            
            # Show any tool calls for transparency
            tool_calls = [msg for msg in result["messages"] if msg.type == "tool"]
            if tool_calls and len(tool_calls) > 0:
                print(f"\n   [Used {len(tool_calls)} tool(s)]")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()

