# OpenRouter Integration for DeepAgents

This guide explains how to use OpenRouter with DeepAgents, giving you access to a wide variety of AI models through a single API.

## What is OpenRouter?

[OpenRouter](https://openrouter.ai/) is a unified API that provides access to multiple AI models from different providers (OpenAI, Anthropic, Google, Meta, Mistral, and more) through a single interface. This allows you to:

- Switch between models easily
- Compare different models for your use case
- Access models that might not be directly available in your region
- Potentially reduce costs by choosing the most cost-effective model

## Setup

### 1. Get an OpenRouter API Key

1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up or log in
3. Create a new API key
4. Copy your API key

### 2. Configure Environment Variables

Create a `.env` file in your project root (or copy from `.env.example`):

```bash
# Required: Your OpenRouter API key
OPENROUTER_API_KEY=your_api_key_here

# Optional: Specify which model to use (defaults to anthropic/claude-sonnet-4.5)
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
```

### 3. Install Dependencies

The OpenRouter integration is included in the base deepagents installation:

```bash
cd libs/deepagents
uv sync
```

## Usage

### Automatic Configuration (Recommended)

When you set `OPENROUTER_API_KEY` in your `.env` file, DeepAgents will automatically use OpenRouter:

```python
from deepagents import create_deep_agent

# Automatically uses OpenRouter if OPENROUTER_API_KEY is set
agent = create_deep_agent(
    system_prompt="You are a helpful AI assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

### Explicit Model Selection

You can explicitly specify which OpenRouter model to use:

```python
from deepagents import create_deep_agent, get_openrouter_model

# Use a specific model (latest as of 2026)
model = get_openrouter_model("openai/gpt-5.2")

agent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful AI assistant."
)
```

### Available Models (Latest as of January 2026)

OpenRouter supports 300+ models. Here are some popular options:

**Anthropic (Latest):**
- `anthropic/claude-sonnet-4.5` (recommended)
- `anthropic/claude-opus-4.5`
- `anthropic/claude-3.5-sonnet` (previous generation)

**OpenAI (Latest):**
- `openai/gpt-5.2`
- `openai/gpt-5.1`
- `openai/gpt-4o`
- `openai/gpt-4-turbo` (previous generation)

**Google (Latest):**
- `google/gemini-3-pro`
- `google/gemini-2.5-pro`
- `google/gemini-pro-1.5` (previous generation)

**Meta:**
- `meta-llama/llama-3.1-405b-instruct`
- `meta-llama/llama-3.1-70b-instruct`
- `meta-llama/llama-3.1-8b-instruct`

**Mistral:**
- `mistralai/mistral-large`
- `mistralai/mistral-medium`
- `mistralai/mixtral-8x7b-instruct`

For a complete list, visit: [https://openrouter.ai/models](https://openrouter.ai/models)

### Programmatic Model Selection

```python
from deepagents import get_openrouter_model, create_deep_agent

# Create different models for different purposes
fast_model = get_openrouter_model("anthropic/claude-3-haiku")
powerful_model = get_openrouter_model("anthropic/claude-3-opus")
cost_effective = get_openrouter_model("meta-llama/llama-3.1-70b-instruct")

# Use in agents
agent = create_deep_agent(
    model=fast_model,
    system_prompt="You are a helpful assistant."
)
```

### Using with Subagents

You can use different OpenRouter models for different subagents:

```python
from deepagents import create_deep_agent, get_openrouter_model

research_subagent = {
    "name": "research-agent",
    "description": "Used to research in-depth questions",
    "system_prompt": "You are an expert researcher",
    "model": get_openrouter_model("anthropic/claude-3-opus"),
}

agent = create_deep_agent(
    model=get_openrouter_model("anthropic/claude-sonnet-4.5"),
    subagents=[research_subagent]
)
```

## Testing

Run the test script to verify your OpenRouter integration:

```bash
cd libs/deepagents
uv run python ../../test_openrouter.py
```

This will test:
- Configuration detection
- Model creation
- Deep agent creation
- Simple invocation (if API key is valid)

## Checking Configuration

You can programmatically check if OpenRouter is configured:

```python
from deepagents import is_openrouter_configured

if is_openrouter_configured():
    print("OpenRouter is ready to use!")
else:
    print("Please set OPENROUTER_API_KEY in your .env file")
```

## Fallback Behavior

If `OPENROUTER_API_KEY` is not set, DeepAgents will fall back to the default Anthropic Claude model (requires `ANTHROPIC_API_KEY`).

## Troubleshooting

### "OPENROUTER_API_KEY environment variable is not set"

Make sure you have:
1. Created a `.env` file in your project root
2. Added `OPENROUTER_API_KEY=your_key_here` to the file
3. The `.env` file is in the same directory where you run your script

### Model not found errors

- Check that the model name is correct at [https://openrouter.ai/models](https://openrouter.ai/models)
- Some models may require additional credits or permissions

### Rate limiting

OpenRouter has rate limits. If you hit them:
- Add delays between requests
- Consider upgrading your OpenRouter plan
- Use a different model with higher limits

## Benefits of OpenRouter

1. **Model Flexibility**: Easily switch between models without changing code
2. **Cost Optimization**: Choose the most cost-effective model for each task
3. **Availability**: Access models that might not be available in your region
4. **Unified Billing**: One bill for all your AI model usage
5. **Fallback Options**: Automatically fall back to alternative models if one is unavailable

## Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)

