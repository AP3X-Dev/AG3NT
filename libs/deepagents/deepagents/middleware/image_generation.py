"""Middleware for AI-powered image generation and editing using Gemini 3 Pro.

This middleware provides tools for generating and editing images using Google's
Gemini 3 Pro Image Preview model via OpenRouter API.

Features:
- High-fidelity image generation from text prompts
- Image editing with text instructions
- Text rendering in images (including long passages, multilingual)
- Multi-image blending and compositing
- Identity preservation across subjects
- Flexible aspect ratios and resolutions (1K, 2K, 4K)
- Localized edits, lighting/focus adjustments
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

# --- Configuration ---

DEFAULT_IMAGE_MODEL = "google/gemini-3-pro-image-preview"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_workspace_path() -> Path:
    """Get the workspace directory for generated files.

    Returns the path from DEEPAGENTS_WORKSPACE env var, or defaults to
    ./workspace relative to the project root.
    """
    workspace = os.getenv("DEEPAGENTS_WORKSPACE")
    if workspace:
        path = Path(workspace)
    else:
        # Default to ./workspace relative to the deepagents package
        # Try to find project root by looking for common markers
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".env").exists() or (parent / "workspace").exists():
                path = parent / "workspace"
                break
        else:
            # Fallback to cwd/workspace
            path = Path.cwd() / "workspace"

    # Ensure workspace exists
    path.mkdir(parents=True, exist_ok=True)
    return path


# Supported aspect ratios and their dimensions
ASPECT_RATIOS = {
    "1:1": "1024×1024",
    "2:3": "832×1248",
    "3:2": "1248×832",
    "3:4": "864×1184",
    "4:3": "1184×864",
    "4:5": "896×1152",
    "5:4": "1152×896",
    "9:16": "768×1344",
    "16:9": "1344×768",
    "21:9": "1536×672",
}

IMAGE_SIZES = ["1K", "2K", "4K"]

# --- Tool Descriptions ---

GENERATE_IMAGE_DESCRIPTION = """Generate images from text prompts using AI.

Uses Gemini 3 Pro for high-quality image generation with advanced features:
- Accurate text rendering in images
- Context-rich graphics (infographics, diagrams, composites)
- Professional-grade output quality

Parameters:
- prompt: Detailed description of the image to generate (required)
- aspect_ratio: Output aspect ratio (optional, default "1:1")
  Options: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- image_size: Output resolution (optional, default "1K")
  Options: 1K (standard), 2K (higher), 4K (highest)
- output_path: Path to save the generated image (optional)
  If not provided, returns base64 data

Examples:
- "A futuristic cityscape at sunset with flying cars"
- "Professional product photo of a sleek smartphone on marble"
- "Infographic showing the water cycle with labeled arrows"
- "Portrait of a Renaissance-style painting of a golden retriever"

Returns the generated image as base64 data or saves to the specified path."""

EDIT_IMAGE_DESCRIPTION = """Edit an existing image using AI with text instructions.

Uses Gemini 3 Pro for advanced image editing capabilities:
- Localized edits (modify specific regions)
- Lighting and focus adjustments
- Style transformations
- Object addition/removal
- Multi-image blending (up to 5 source images)
- Identity preservation

Parameters:
- image_path: Path to the source image to edit (required)
- instruction: Text describing the desired edit (required)
- additional_images: Paths to additional reference images (optional)
  Use for style transfer, identity preservation, or blending
- aspect_ratio: Output aspect ratio (optional, preserves original if not set)
- image_size: Output resolution (optional, default "1K")
- output_path: Path to save the edited image (optional)

Examples:
- "Remove the person in the background"
- "Change the sky to a dramatic sunset"
- "Add soft studio lighting to the portrait"
- "Apply the artistic style from the reference image"
- "Replace the car's color with metallic blue"

Returns the edited image as base64 data or saves to the specified path."""

IMAGE_GEN_SYSTEM_PROMPT = """You have access to AI image generation and editing tools.

For image GENERATION (generate_image):
- Be specific and descriptive in prompts
- Include style, lighting, composition details
- Specify text content if text should appear in the image
- Choose appropriate aspect ratios for the content type

For image EDITING (edit_image):
- Describe the specific changes you want
- Reference specific parts of the image when needed
- Use additional reference images for style transfer
- Be clear about what should change vs. stay the same

The image model excels at:
- Accurate text rendering (signs, labels, documents)
- Professional product photography
- Artistic compositions and style matching
- Identity-consistent character generation
"""


# --- Core Image Generation Functions ---


def _extract_image_from_response(data: dict[str, Any]) -> dict[str, Any]:
    """Extract image data from an OpenRouter/Gemini API response.

    Handles various response formats from different models.

    Args:
        data: The JSON response from the API.

    Returns:
        Dict with 'success', 'image_data' (base64 or URL), 'error' keys.
    """
    choices = data.get("choices", [])
    if not choices:
        logger.warning("No choices in response: %s", list(data.keys()))
        return {"success": False, "image_data": None, "error": "No response from model"}

    message = choices[0].get("message", {})
    logger.debug("Message keys: %s", list(message.keys()))

    # Check for images array in the response (OpenRouter format)
    images = message.get("images", [])
    if images:
        img = images[0]
        # Handle various image URL formats
        image_url = (
            img.get("imageUrl", {}).get("url") or img.get("image_url", {}).get("url") or img.get("url") or (img if isinstance(img, str) else None)
        )
        if image_url:
            if image_url.startswith("data:"):
                _, b64_data = image_url.split(",", 1)
                return {"success": True, "image_data": b64_data, "error": None}
            return {"success": True, "image_data": image_url, "error": None}

    # Check content array for image parts
    content = message.get("content", [])
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type", "")
                # Handle image_url type
                if part_type == "image_url":
                    url = part.get("image_url", {}).get("url", "") or part.get("url", "")
                    if url:
                        if url.startswith("data:"):
                            _, b64_data = url.split(",", 1)
                            return {"success": True, "image_data": b64_data, "error": None}
                        return {"success": True, "image_data": url, "error": None}
                # Handle inline_data type (Gemini native format)
                elif part_type == "inline_data" or "inline_data" in part:
                    inline = part.get("inline_data", part)
                    if isinstance(inline, dict) and "data" in inline:
                        return {"success": True, "image_data": inline["data"], "error": None}
                # Handle image type
                elif part_type == "image":
                    img_data = part.get("source", {}).get("data") or part.get("data")
                    if img_data:
                        return {"success": True, "image_data": img_data, "error": None}

    # If content is a string, the model might have declined to generate
    if isinstance(content, str):
        if any(word in content.lower() for word in ["cannot", "can't", "unable", "sorry", "inappropriate"]):
            return {"success": False, "image_data": None, "error": f"Model declined: {content[:500]}"}
        return {"success": False, "image_data": None, "error": f"No image generated. Model said: {content[:300]}"}

    # Log the full message structure for debugging
    logger.warning("Could not extract image from message. Keys: %s, Content type: %s", list(message.keys()), type(content).__name__)
    if content:
        logger.debug("Content structure: %s", content[:2] if isinstance(content, list) else str(content)[:200])

    return {"success": False, "image_data": None, "error": f"No image in response. Message keys: {list(message.keys())}"}


async def _generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate an image using OpenRouter's Gemini 3 Pro model.

    Args:
        prompt: Text description of the image to generate.
        aspect_ratio: Aspect ratio (e.g., "16:9", "1:1").
        image_size: Resolution ("1K", "2K", "4K").
        api_key: OpenRouter API key (uses env var if not provided).
        model: Model to use (uses env var or default if not provided).

    Returns:
        Dict with 'success', 'image_data' (base64), 'error' keys.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    model = model or os.getenv("OPENROUTER_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)

    if not api_key:
        return {"success": False, "image_data": None, "error": "OPENROUTER_API_KEY not set"}

    # Validate inputs
    if aspect_ratio not in ASPECT_RATIOS:
        return {
            "success": False,
            "image_data": None,
            "error": f"Invalid aspect_ratio. Options: {', '.join(ASPECT_RATIOS.keys())}",
        }
    if image_size not in IMAGE_SIZES:
        return {
            "success": False,
            "image_data": None,
            "error": f"Invalid image_size. Options: {', '.join(IMAGE_SIZES)}",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return _extract_image_from_response(data)

    except httpx.HTTPStatusError as e:
        error_text = e.response.text[:500] if e.response.text else str(e)
        return {"success": False, "image_data": None, "error": f"HTTP {e.response.status_code}: {error_text}"}
    except Exception as e:
        logger.exception("Error generating image")
        return {"success": False, "image_data": None, "error": f"Error: {e!s}"}


async def _edit_image(
    image_bytes: bytes,
    instruction: str,
    additional_images: list[bytes] | None = None,
    aspect_ratio: str | None = None,
    image_size: str = "1K",
    api_key: str | None = None,
    model: str | None = None,
    file_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Edit an image using OpenRouter's Gemini 3 Pro model.

    Args:
        image_bytes: Raw bytes of the source image.
        instruction: Text describing the desired edit.
        additional_images: Optional list of additional image bytes for reference.
        aspect_ratio: Output aspect ratio (optional).
        image_size: Resolution ("1K", "2K", "4K").
        api_key: OpenRouter API key.
        model: Model to use.
        file_type: MIME type of the image.

    Returns:
        Dict with 'success', 'image_data' (base64), 'error' keys.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    model = model or os.getenv("OPENROUTER_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)

    if not api_key:
        return {"success": False, "image_data": None, "error": "OPENROUTER_API_KEY not set"}

    if image_size not in IMAGE_SIZES:
        return {
            "success": False,
            "image_data": None,
            "error": f"Invalid image_size. Options: {', '.join(IMAGE_SIZES)}",
        }

    # Encode source image
    b64_source = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{file_type};base64,{b64_source}"

    # Build content array with text and images
    content: list[dict[str, Any]] = [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    # Add reference images if provided
    if additional_images:
        for i, img_bytes in enumerate(additional_images[:4]):  # Max 5 total images
            b64_ref = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{file_type};base64,{b64_ref}"},
                }
            )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": {"image_size": image_size},
    }

    if aspect_ratio and aspect_ratio in ASPECT_RATIOS:
        payload["image_config"]["aspect_ratio"] = aspect_ratio

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return _extract_image_from_response(data)

    except httpx.HTTPStatusError as e:
        error_text = e.response.text[:500] if e.response.text else str(e)
        return {"success": False, "image_data": None, "error": f"HTTP {e.response.status_code}: {error_text}"}
    except Exception as e:
        logger.exception("Error editing image")
        return {"success": False, "image_data": None, "error": f"Error: {e!s}"}


# --- Tool Generators ---


def _make_generate_image_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
) -> BaseTool:
    """Create the generate_image tool."""

    async def async_generate_image(
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        output_path: str = "",
        runtime: ToolRuntime = None,
    ) -> str:
        """Generate an image from a text prompt."""
        import uuid
        from datetime import datetime

        result = await _generate_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )

        if not result["success"]:
            return f"Image generation failed: {result['error']}"

        image_data = result["image_data"]

        # Decode image data
        try:
            if not image_data.startswith("http"):
                img_bytes = base64.b64decode(image_data)
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image_data)
                    img_bytes = resp.content
        except Exception as e:
            return f"Image generated but failed to decode: {e}"

        # Determine save path - use workspace if no output_path specified
        workspace = get_workspace_path()
        if output_path:
            # Expand ~ to user home and handle relative paths
            save_file = Path(output_path).expanduser()

            # Handle Unix-style root paths on Windows (e.g., /ww3_assessment.html)
            # These should be relative to workspace, not C:\
            if os.name == "nt" and str(output_path).startswith("/"):
                # Strip leading slash and make relative to workspace
                relative_path = output_path.lstrip("/")
                save_file = workspace / relative_path
            elif not save_file.is_absolute():
                # For relative paths, resolve relative to workspace
                save_file = workspace / output_path
            # For absolute paths, try to use them directly but fall back to workspace
            elif save_file.is_absolute():
                try:
                    # Test if we can write to the parent directory
                    save_file.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    # Fall back to workspace with just the filename
                    save_file = workspace / save_file.name
        else:
            # Generate unique filename in workspace
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            save_file = workspace / f"image_{timestamp}_{short_id}.png"

        # Save the image directly to filesystem
        try:
            save_file.parent.mkdir(parents=True, exist_ok=True)
            save_file.write_bytes(img_bytes)
            return f"Image generated and saved to: {save_file}"
        except Exception as e:
            return f"Image generated but failed to save: {e}"

    def sync_generate_image(
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        output_path: str = "",
        runtime: ToolRuntime = None,
    ) -> str:
        """Sync version of generate_image."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(async_generate_image(prompt, aspect_ratio, image_size, output_path, runtime))

    return StructuredTool.from_function(
        name="generate_image",
        description=GENERATE_IMAGE_DESCRIPTION,
        func=sync_generate_image,
        coroutine=async_generate_image,
    )


def _make_edit_image_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
) -> BaseTool:
    """Create the edit_image tool."""

    async def async_edit_image(
        image_path: str,
        instruction: str,
        aspect_ratio: str = "",
        image_size: str = "1K",
        output_path: str = "",
        runtime: ToolRuntime = None,
    ) -> str:
        """Edit an existing image with text instructions."""
        import uuid
        from datetime import datetime

        # Try to read source image from filesystem or backend
        source_path = Path(image_path)
        if source_path.is_absolute() and source_path.exists():
            image_bytes = source_path.read_bytes()
        elif runtime:
            be = backend(runtime) if callable(backend) else backend
            try:
                responses = await be.adownload_files([image_path])
                if not responses or responses[0].error:
                    return f"Failed to read source image: {responses[0].error if responses else 'No response'}"
                image_bytes = responses[0].content
            except Exception as e:
                return f"Failed to read source image: {e}"
        else:
            return f"Failed to read source image: file not found at {image_path}"

        # Determine file type from path
        suffix = Path(image_path).suffix.lower()
        file_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/jpeg")

        result = await _edit_image(
            image_bytes=image_bytes,
            instruction=instruction,
            additional_images=None,
            aspect_ratio=aspect_ratio if aspect_ratio else None,
            image_size=image_size,
            file_type=file_type,
        )

        if not result["success"]:
            return f"Image editing failed: {result['error']}"

        image_data = result["image_data"]

        # Decode image data
        try:
            if not image_data.startswith("http"):
                img_bytes = base64.b64decode(image_data)
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image_data)
                    img_bytes = resp.content
        except Exception as e:
            return f"Image edited but failed to decode: {e}"

        # Determine save path - use workspace if no output_path specified
        workspace = get_workspace_path()
        if output_path:
            # Expand ~ to user home and handle relative paths
            save_file = Path(output_path).expanduser()

            # Handle Unix-style root paths on Windows (e.g., /ww3_assessment.html)
            # These should be relative to workspace, not C:\
            if os.name == "nt" and str(output_path).startswith("/"):
                # Strip leading slash and make relative to workspace
                relative_path = output_path.lstrip("/")
                save_file = workspace / relative_path
            elif not save_file.is_absolute():
                # For relative paths, resolve relative to workspace
                save_file = workspace / output_path
            # For absolute paths, try to use them directly but fall back to workspace
            elif save_file.is_absolute():
                try:
                    # Test if we can write to the parent directory
                    save_file.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    # Fall back to workspace with just the filename
                    save_file = workspace / save_file.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            save_file = workspace / f"edited_{timestamp}_{short_id}.png"

        # Save directly to filesystem
        try:
            save_file.parent.mkdir(parents=True, exist_ok=True)
            save_file.write_bytes(img_bytes)
            return f"Image edited and saved to: {save_file}"
        except Exception as e:
            return f"Image edited but failed to save: {e}"

    def sync_edit_image(
        image_path: str,
        instruction: str,
        aspect_ratio: str = "",
        image_size: str = "1K",
        output_path: str = "",
        runtime: ToolRuntime = None,
    ) -> str:
        """Sync version of edit_image."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(async_edit_image(image_path, instruction, aspect_ratio, image_size, output_path, runtime))

    return StructuredTool.from_function(
        name="edit_image",
        description=EDIT_IMAGE_DESCRIPTION,
        func=sync_edit_image,
        coroutine=async_edit_image,
    )


# --- Tool Collection ---


def _get_image_tools(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    enabled_tools: list[str] | None = None,
) -> list[BaseTool]:
    """Get image generation tools.

    Args:
        backend: Backend for file operations.
        enabled_tools: Optional list of tool names to enable.

    Returns:
        List of configured image tools.
    """
    tools_to_generate = enabled_tools or ["generate_image", "edit_image"]
    tools: list[BaseTool] = []

    if "generate_image" in tools_to_generate:
        tools.append(_make_generate_image_tool(backend))

    if "edit_image" in tools_to_generate:
        tools.append(_make_edit_image_tool(backend))

    return tools


# --- Middleware Class ---


class ImageGenerationMiddleware(AgentMiddleware):
    """Middleware for AI-powered image generation and editing.

    This middleware adds tools for generating and editing images using
    Google's Gemini 3 Pro Image Preview model via OpenRouter.

    Features:
    - generate_image: Create images from text prompts
    - edit_image: Modify existing images with text instructions

    Args:
        backend: Backend for file operations.
        system_prompt: Optional custom system prompt override.
        enabled_tools: Optional list of tool names to enable.

    Example:
        ```python
        from deepagents.middleware.image_generation import ImageGenerationMiddleware
        from deepagents.backends import FilesystemBackend

        backend = FilesystemBackend(root_dir="/path/to/workspace")
        middleware = ImageGenerationMiddleware(backend=backend)

        # Use with create_agent
        agent = create_agent(
            model=model,
            middleware=[middleware],
        )
        ```
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol] | None = None,
        system_prompt: str | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        """Initialize the image generation middleware.

        Args:
            backend: Backend for file operations. Defaults to StateBackend.
            system_prompt: Optional custom system prompt override.
            enabled_tools: Optional list of tool names to enable.
        """
        from deepagents.backends import StateBackend

        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))
        self._custom_system_prompt = system_prompt
        enabled = enabled_tools or ["generate_image", "edit_image"]

        # Build tools list and store in self.tools (used by create_agent)
        self.tools = _get_image_tools(self.backend, enabled)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add image generation system prompt."""
        system_prompt = self._custom_system_prompt or IMAGE_GEN_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Add image generation system prompt."""
        system_prompt = self._custom_system_prompt or IMAGE_GEN_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return await handler(request)
