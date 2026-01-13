# OpenRouter Quick Start Guide

Get started with OpenRouter in DeepAgents in 3 simple steps.

## Step 1: Get Your API Key

1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up or log in
3. Create a new API key
4. Copy your API key

## Step 2: Configure Your Environment

Create a `.env` file in your project root:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional - specify which model to use (defaults to anthropic/claude-sonnet-4.5)
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
```

## Step 3: Use DeepAgents

```python
from deepagents import create_deep_agent

# That's it! DeepAgents will automatically use OpenRouter
agent = create_deep_agent(
    system_prompt="You are a helpful AI assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})

print(result["messages"][-1].content)
```

## Popular Models (Latest as of January 2026)

Try different models by changing `OPENROUTER_MODEL` in your `.env` file:

```bash
# Anthropic (Latest - Recommended)
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
OPENROUTER_MODEL=anthropic/claude-opus-4.5

# OpenAI (Latest)
OPENROUTER_MODEL=openai/gpt-5.2
OPENROUTER_MODEL=openai/gpt-5.1
OPENROUTER_MODEL=openai/gpt-4o

# Google (Latest)
OPENROUTER_MODEL=google/gemini-3-pro
OPENROUTER_MODEL=google/gemini-2.5-pro

# Meta Llama
OPENROUTER_MODEL=meta-llama/llama-3.1-405b-instruct
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
```

See all available models at: [https://openrouter.ai/models](https://openrouter.ai/models)

## Advanced Usage

### Use Different Models in Code

```python
from deepagents import create_deep_agent, get_openrouter_model

# Override the environment variable (latest models as of 2026)
model = get_openrouter_model("openai/gpt-5.2")

agent = create_deep_agent(model=model)
```

### Use Different Models for Subagents

```python
from deepagents import create_deep_agent, get_openrouter_model

research_subagent = {
    "name": "researcher",
    "description": "Expert researcher",
    "system_prompt": "You are a research expert.",
    "model": get_openrouter_model("anthropic/claude-3-opus"),
}

agent = create_deep_agent(
    model=get_openrouter_model("anthropic/claude-3.5-sonnet"),
    subagents=[research_subagent]
)
```

## Testing Your Setup

Run the test script:

```bash
cd libs/deepagents
uv run python ../../test_openrouter.py
```

## Need Help?

- **Full Documentation**: See [OPENROUTER_INTEGRATION.md](OPENROUTER_INTEGRATION.md)
- **Examples**: Check [examples/openrouter_example.py](examples/openrouter_example.py)
- **OpenRouter Docs**: [https://openrouter.ai/docs](https://openrouter.ai/docs)
- **DeepAgents Docs**: [https://docs.langchain.com/oss/python/deepagents/overview](https://docs.langchain.com/oss/python/deepagents/overview)

## Why OpenRouter?

✅ Access 300+ AI models through one API
✅ Latest models (Claude 4.5, GPT-5.2, Gemini 3)
✅ Easy model switching without code changes
✅ Competitive pricing
✅ Automatic fallbacks if a model is unavailable
✅ Unified billing across all models

Happy building! 🚀

