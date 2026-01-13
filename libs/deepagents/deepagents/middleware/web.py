"""Middleware for web tools: web_search, read_web_page."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

# --- Tool Descriptions ---

WEB_SEARCH_DESCRIPTION = """Search the web for information.

Parameters:
- objective: What you want to find (required) - be specific about the information needed
- search_queries: Optional list of specific search keywords to use

When to use:
- Finding up-to-date documentation or API references
- Researching libraries, frameworks, or tools
- Getting current information that may not be in training data
- Verifying facts or checking for updates

Best practices:
- Use specific, targeted search queries
- Combine with read_web_page to get full content from relevant results
- For multiple topics, make separate search calls

Returns search results with:
- Title and URL of each result
- Snippet/description of the content
- Relevance ranking

Note: For internal/private documentation, use other tools like read_file or grep."""

READ_WEB_PAGE_DESCRIPTION = """Read and extract content from a web page.

Parameters:
- url: The URL to fetch (required)
- objective: What specific information to extract (optional) - filters to relevant excerpts
- forceRefetch: Set to true to bypass cache and get fresh content (default: false)

Returns the page content converted to markdown format.

When to use:
- Following up on web_search results to get full content
- Reading documentation pages
- Extracting specific information from known URLs

Limitations:
- Do NOT use for localhost URLs - use curl via execute tool instead
- Some pages may block automated access
- JavaScript-rendered content may not be available

Best practices:
- Provide an 'objective' to filter content and reduce response size
- Use web_search first to find relevant URLs if you don't have a specific URL"""


# --- Tool Implementations ---


async def _search_web(
    objective: str,
    search_queries: list[str] | None = None,
) -> str:
    """Search the web for information."""
    try:
        # Try to use httpx for async HTTP requests
        import httpx
    except ImportError:
        return "Web search requires httpx. Install with: pip install httpx"

    # Use DuckDuckGo HTML search (no API key required)
    queries = search_queries if search_queries else [objective]
    all_results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for query in queries[:3]:  # Limit to 3 queries
            try:
                # Use DuckDuckGo HTML endpoint
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
                )
                response.raise_for_status()

                # Parse results (basic HTML parsing)
                from html.parser import HTMLParser

                class DDGParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.results = []
                        self.current_result = {}
                        self.in_result = False
                        self.in_title = False
                        self.in_snippet = False

                    def handle_starttag(self, tag, attrs):
                        attrs_dict = dict(attrs)
                        if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                            self.in_result = True
                            self.in_title = True
                            self.current_result = {"url": attrs_dict.get("href", "")}
                        elif tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                            self.in_snippet = True

                    def handle_endtag(self, tag):
                        if tag == "a" and self.in_title:
                            self.in_title = False
                        elif tag == "a" and self.in_snippet:
                            self.in_snippet = False
                            if self.current_result:
                                self.results.append(self.current_result)
                                self.current_result = {}
                            self.in_result = False

                    def handle_data(self, data):
                        if self.in_title:
                            self.current_result["title"] = data.strip()
                        elif self.in_snippet:
                            self.current_result["snippet"] = data.strip()

                parser = DDGParser()
                parser.feed(response.text)

                for result in parser.results[:5]:  # Top 5 per query
                    all_results.append(result)

            except Exception as e:
                logger.warning(f"Search query failed: {e}")
                continue

    if not all_results:
        return f"No results found for: {objective}"

    # Format results
    output = [f"Search results for: {objective}\n"]
    for i, result in enumerate(all_results[:10], 1):  # Max 10 results
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        output.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}\n")

    return "\n".join(output)


async def _read_web_page(
    url: str,
    objective: str | None = None,
    forceRefetch: bool = False,
) -> str:
    """Read and extract content from a web page."""
    # Check for localhost - should use execute/curl instead
    if "localhost" in url or "127.0.0.1" in url or url.startswith("file://"):
        return "Cannot read localhost URLs. Use the execute tool with curl instead."

    try:
        import httpx
    except ImportError:
        return "Web page reading requires httpx. Install with: pip install httpx"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            if forceRefetch:
                headers["Cache-Control"] = "no-cache"

            response = await client.get(url, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                # Return JSON as-is formatted
                import json

                try:
                    data = response.json()
                    return f"```json\n{json.dumps(data, indent=2)}\n```"
                except Exception:
                    return response.text

            elif "text/html" in content_type or "application/xhtml" in content_type:
                # Convert HTML to markdown
                html_content = response.text
                markdown = _html_to_markdown(html_content)

                # If objective provided, try to extract relevant sections
                if objective:
                    markdown = _filter_by_objective(markdown, objective)

                # Truncate if too long
                if len(markdown) > 50000:
                    markdown = markdown[:50000] + "\n\n[Content truncated due to length]"

                return markdown

            else:
                # Return text content as-is
                return response.text[:50000]

    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code}: {e.response.reason_phrase}"
    except httpx.TimeoutException:
        return f"Request timed out for URL: {url}"
    except Exception as e:
        return f"Failed to read web page: {e}"


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown (basic implementation)."""
    import re
    from html.parser import HTMLParser

    class HTMLToMarkdown(HTMLParser):
        def __init__(self):
            super().__init__()
            self.output = []
            self.current_text = []
            self.in_pre = False
            self.in_code = False
            self.in_script = False
            self.in_style = False
            self.list_stack = []
            self.heading_level = 0

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag in ("script", "style", "nav", "footer", "header"):
                self.in_script = True
            elif tag == "pre":
                self.in_pre = True
                self.output.append("\n```\n")
            elif tag == "code" and not self.in_pre:
                self.in_code = True
                self.output.append("`")
            elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self.heading_level = int(tag[1])
                self.output.append("\n" + "#" * self.heading_level + " ")
            elif tag == "p":
                self.output.append("\n\n")
            elif tag == "br":
                self.output.append("\n")
            elif tag == "a":
                attrs_dict = dict(attrs)
                href = attrs_dict.get("href", "")
                self.output.append("[")
                # Store href for later
                self._current_href = href
            elif tag in ("ul", "ol"):
                self.list_stack.append(tag)
            elif tag == "li":
                prefix = "- " if self.list_stack and self.list_stack[-1] == "ul" else "1. "
                self.output.append("\n" + "  " * (len(self.list_stack) - 1) + prefix)
            elif tag in ("strong", "b"):
                self.output.append("**")
            elif tag in ("em", "i"):
                self.output.append("*")

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in ("script", "style", "nav", "footer", "header"):
                self.in_script = False
            elif tag == "pre":
                self.in_pre = False
                self.output.append("\n```\n")
            elif tag == "code" and not self.in_pre:
                self.in_code = False
                self.output.append("`")
            elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self.heading_level = 0
                self.output.append("\n")
            elif tag == "a":
                href = getattr(self, "_current_href", "")
                self.output.append(f"]({href})")
            elif tag in ("ul", "ol"):
                if self.list_stack:
                    self.list_stack.pop()
            elif tag in ("strong", "b"):
                self.output.append("**")
            elif tag in ("em", "i"):
                self.output.append("*")

        def handle_data(self, data):
            if self.in_script or self.in_style:
                return
            # Clean up whitespace
            if not self.in_pre:
                data = re.sub(r"\s+", " ", data)
            self.output.append(data)

    parser = HTMLToMarkdown()
    parser.feed(html)
    result = "".join(parser.output)

    # Clean up multiple newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _filter_by_objective(content: str, objective: str) -> str:
    """Filter content to sections relevant to the objective."""
    # Simple keyword-based filtering
    keywords = objective.lower().split()
    lines = content.split("\n")
    relevant_sections = []
    current_section = []
    section_relevant = False

    for line in lines:
        # Check if this is a heading
        if line.startswith("#"):
            # Save previous section if relevant
            if section_relevant and current_section:
                relevant_sections.extend(current_section)
            current_section = [line]
            section_relevant = any(kw in line.lower() for kw in keywords)
        else:
            current_section.append(line)
            if any(kw in line.lower() for kw in keywords):
                section_relevant = True

    # Don't forget last section
    if section_relevant and current_section:
        relevant_sections.extend(current_section)

    if relevant_sections:
        return "\n".join(relevant_sections)
    # Fall back to returning start of content
    return content[:10000] + "\n\n[Filtered content - no sections matched objective]"


# --- Tool Generators ---


def _web_search_tool_generator(custom_description: str | None = None) -> BaseTool:
    """Generate the web_search tool."""
    tool_description = custom_description or WEB_SEARCH_DESCRIPTION

    async def async_web_search(
        objective: str,
        search_queries: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        return await _search_web(objective, search_queries)

    def sync_web_search(
        objective: str,
        search_queries: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(_search_web(objective, search_queries))

    return StructuredTool.from_function(
        name="web_search",
        description=tool_description,
        func=sync_web_search,
        coroutine=async_web_search,
    )


def _read_web_page_tool_generator(custom_description: str | None = None) -> BaseTool:
    """Generate the read_web_page tool."""
    tool_description = custom_description or READ_WEB_PAGE_DESCRIPTION

    async def async_read_web_page(
        url: str,
        objective: str | None = None,
        forceRefetch: bool = False,
        runtime: ToolRuntime = None,
    ) -> str:
        return await _read_web_page(url, objective, forceRefetch)

    def sync_read_web_page(
        url: str,
        objective: str | None = None,
        forceRefetch: bool = False,
        runtime: ToolRuntime = None,
    ) -> str:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(_read_web_page(url, objective, forceRefetch))

    return StructuredTool.from_function(
        name="read_web_page",
        description=tool_description,
        func=sync_read_web_page,
        coroutine=async_read_web_page,
    )


# --- Tool Registry ---

WEB_TOOL_GENERATORS = {
    "web_search": lambda desc: _web_search_tool_generator(desc),
    "read_web_page": lambda desc: _read_web_page_tool_generator(desc),
}


def _get_web_tools(
    custom_tool_descriptions: dict[str, str] | None = None,
    enabled_tools: list[str] | None = None,
) -> list[BaseTool]:
    """Get web tools."""
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}

    tools_to_generate = enabled_tools or list(WEB_TOOL_GENERATORS.keys())
    tools = []

    for tool_name in tools_to_generate:
        if tool_name in WEB_TOOL_GENERATORS:
            tool_generator = WEB_TOOL_GENERATORS[tool_name]
            tool = tool_generator(custom_tool_descriptions.get(tool_name))
            tools.append(tool)

    return tools


WEB_SYSTEM_PROMPT = """## Web Tools

You have access to web tools for searching and reading online content:

- web_search: Search the web for information (documentation, tutorials, API references)
- read_web_page: Read and extract content from a specific URL

**Autonomous Research Behavior:**

Research anything you don't fully understand BEFORE responding. When you encounter unfamiliar terms, technologies, platforms, or concepts - use web_search immediately. Never ask the user to explain something you can research.

**Research triggers (search automatically):**
- Unfamiliar platform/service names
- Technologies or frameworks you're uncertain about
- Domain-specific terminology you don't recognize
- Current market data, trending items, or real-time information
- Best practices or standards you're not confident about

**Research pattern:**
1. Identify knowledge gaps
2. Generate targeted queries, search broadly
3. Rank results, read only the best sources via read_web_page
4. Extract concrete facts with source references
5. Execute the task with full understanding

**Best practices:**
- Use web_search to find relevant URLs, then read_web_page for full content
- For localhost URLs, use the execute tool with curl instead
- Provide specific objectives when reading pages to filter content
- Combine multiple searches to triangulate information from different sources
- When time-sensitive, prefer current sources and capture dates when available
"""


class WebMiddleware(AgentMiddleware):
    """Middleware for web tools: web_search, read_web_page.

    This middleware adds tools for searching and reading web content.

    Args:
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        enabled_tools: Optional list of tool names to enable. Defaults to all tools.

    Example:
        ```python
        from deepagents.middleware.web import WebMiddleware
        from langchain.agents import create_agent

        # Enable all web tools
        agent = create_agent(middleware=[WebMiddleware()])

        # Enable only web_search
        agent = create_agent(middleware=[WebMiddleware(enabled_tools=["web_search"])])
        ```
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        """Initialize the web middleware."""
        self._custom_system_prompt = system_prompt
        self.tools = _get_web_tools(custom_tool_descriptions, enabled_tools)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add web tools system prompt."""
        system_prompt = self._custom_system_prompt or WEB_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Add web tools system prompt."""
        system_prompt = self._custom_system_prompt or WEB_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return await handler(request)
