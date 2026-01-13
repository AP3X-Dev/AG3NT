# OpenRouter Integration Update - January 2026

## Summary

Updated the DeepAgents OpenRouter integration to use the official OpenRouter Python SDK and reflect the latest available models as of January 2026.

## Key Changes

### 1. Official SDK Support

**Added dependency:**
- `openrouter>=0.1.1` - Official OpenRouter Python SDK

**Benefits:**
- Type-safe API
- Better error handling
- Official support from OpenRouter
- Access to 300+ models

### 2. Updated Default Model

**Changed from:** `anthropic/claude-3.5-sonnet`  
**Changed to:** `anthropic/claude-sonnet-4.5`

This reflects the latest Claude model available as of January 2026.

### 3. Updated Model References

All documentation and examples now reference the latest models:

**Anthropic (Latest):**
- `anthropic/claude-sonnet-4.5` (new default)
- `anthropic/claude-opus-4.5`

**OpenAI (Latest):**
- `openai/gpt-5.2`
- `openai/gpt-5.1`
- `openai/gpt-4o`

**Google (Latest):**
- `google/gemini-3-pro`
- `google/gemini-2.5-pro`

**Meta:**
- `meta-llama/llama-3.1-405b-instruct`

### 4. Updated Documentation

**Files Updated:**
- `libs/deepagents/pyproject.toml` - Added openrouter dependency
- `libs/deepagents/deepagents/openrouter.py` - Updated docstrings and defaults
- `libs/deepagents/README.md` - Added OpenRouter section with latest models
- `.env.example` - Updated with latest model examples
- `test_openrouter.py` - Updated test cases with latest models
- `OPENROUTER_QUICKSTART.md` - Updated all model references
- `OPENROUTER_INTEGRATION.md` - Updated comprehensive documentation
- `OPENROUTER_IMPLEMENTATION_SUMMARY.md` - Updated model lists
- `examples/openrouter_example.py` - Updated all example code

**Key Documentation Updates:**
- Changed "100+ models" to "300+ models"
- Updated all code examples with latest model names
- Added notes about model availability as of January 2026
- Updated default model in all examples

## Migration Guide

### For Existing Users

If you're already using OpenRouter with DeepAgents, you have two options:

**Option 1: Keep using your current model**
- No changes needed if you've set `OPENROUTER_MODEL` in your `.env` file
- Your existing configuration will continue to work

**Option 2: Upgrade to latest models**
1. Update your `.env` file:
   ```bash
   OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
   ```
2. Install the official SDK:
   ```bash
   uv pip install openrouter
   ```

### For New Users

1. Get your API key from [OpenRouter](https://openrouter.ai/keys)
2. Create a `.env` file:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   OPENROUTER_MODEL=anthropic/claude-sonnet-4.5  # Optional
   ```
3. Use DeepAgents as normal - it will automatically use OpenRouter

## Backward Compatibility

✅ **Fully backward compatible**
- Existing code continues to work without changes
- Old model names still supported (e.g., `anthropic/claude-3.5-sonnet`)
- No breaking changes to the API

## Testing

Run the test suite to verify the integration:

```bash
cd libs/deepagents
uv run python ../../test_openrouter.py
```

## Benefits of This Update

1. **Latest Models**: Access to the newest AI models (Claude 4.5, GPT-5.2, Gemini 3)
2. **Official SDK**: Better support and type safety
3. **More Models**: 300+ models instead of 100+
4. **Better Documentation**: Clear examples with latest models
5. **Future-Proof**: Easy to update as new models are released

## Notes

- Model names and availability may change over time
- Check [OpenRouter Models](https://openrouter.ai/models) for the most up-to-date list
- Pricing varies by model - see OpenRouter for current pricing
- Some models may have usage limits or require additional permissions

## Support

- **Quick Start**: See `OPENROUTER_QUICKSTART.md`
- **Full Documentation**: See `OPENROUTER_INTEGRATION.md`
- **Examples**: See `examples/openrouter_example.py`
- **OpenRouter Docs**: https://openrouter.ai/docs

