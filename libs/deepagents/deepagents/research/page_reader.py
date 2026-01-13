"""Page Reader for fetching and extracting content from web pages.

The PageReader handles HTTP-based content extraction:
- Fetches pages via HTTP with proper headers and timeouts
- Extracts main content using readability algorithms
- Normalizes content to markdown format
- Extracts metadata (title, author, publish date)
- Detects when browser mode is needed (JS-heavy sites)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from deepagents.research.config import ResearchConfig

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Extracted content from a web page."""

    url: str
    title: str | None
    content: str  # Markdown content
    raw_html: str | None = None

    # Metadata
    author: str | None = None
    publish_date: datetime | None = None
    description: str | None = None

    # Extraction info
    word_count: int = 0
    extraction_method: str = "unknown"

    # Flags
    needs_browser: bool = False
    error: str | None = None


class PageReader:
    """Reads and extracts content from web pages.

    The PageReader uses HTTP requests to fetch pages and extracts
    the main content using readability-style algorithms. It detects
    when JavaScript rendering is required and signals for browser mode.

    Args:
        config: Research configuration.
        use_mock: If True, use mock session for testing.
    """

    def __init__(self, config: ResearchConfig, use_mock: bool = False) -> None:
        self.config = config
        self._session = None
        self._use_mock = use_mock

    async def _get_session(self) -> Any:
        """Get or create an aiohttp session."""
        if self._session is None:
            if self._use_mock:
                self._session = MockSession()
            else:
                try:
                    import aiohttp

                    timeout = aiohttp.ClientTimeout(total=self.config.page_fetch_timeout_seconds)
                    self._session = aiohttp.ClientSession(timeout=timeout)
                except ImportError:
                    logger.warning("aiohttp not available, using mock session")
                    self._session = MockSession()
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    def _extract_title(self, html: str) -> str | None:
        """Extract title from HTML."""
        # Try <title> tag
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try og:title
        match = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _extract_description(self, html: str) -> str | None:
        """Extract description from HTML."""
        # Try meta description
        match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try og:description
        match = re.search(
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _extract_publish_date(self, html: str) -> datetime | None:
        """Extract publish date from HTML."""
        if not self.config.extract_publish_dates:
            return None

        # Try common date meta tags
        patterns = [
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']+)["\']',
            r'<time[^>]+datetime=["\']([^"\']+)["\']',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    # Try ISO format
                    if "T" in date_str:
                        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except Exception:
                    pass

        return None

    def _extract_author(self, html: str) -> str | None:
        """Extract author from HTML."""
        if not self.config.extract_author_info:
            return None

        # Try meta author
        match = re.search(
            r'<meta[^>]+name=["\']author["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown.

        This is a simplified conversion - in production, use a proper
        library like html2text or markdownify.
        """
        content = html

        # Remove script and style tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<nav[^>]*>.*?</nav>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<footer[^>]*>.*?</footer>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<header[^>]*>.*?</header>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Convert headers
        for i in range(6, 0, -1):
            content = re.sub(
                rf"<h{i}[^>]*>(.*?)</h{i}>",
                rf"\n{'#' * i} \1\n",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Convert paragraphs
        content = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", content, flags=re.DOTALL | re.IGNORECASE)

        # Convert lists
        content = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", content, flags=re.DOTALL | re.IGNORECASE)

        # Convert links
        content = re.sub(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            r"[\2](\1)",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert bold/strong
        content = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", content, flags=re.DOTALL | re.IGNORECASE)

        # Convert italic/em
        content = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", content, flags=re.DOTALL | re.IGNORECASE)

        # Convert code
        content = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove remaining HTML tags
        content = re.sub(r"<[^>]+>", "", content)

        # Clean up whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        # Decode common HTML entities
        content = content.replace("&nbsp;", " ")
        content = content.replace("&amp;", "&")
        content = content.replace("&lt;", "<")
        content = content.replace("&gt;", ">")
        content = content.replace("&quot;", '"')
        content = content.replace("&#39;", "'")

        return content.strip()

    def _detect_needs_browser(self, html: str, content: str) -> bool:
        """Detect if the page needs browser rendering."""
        # Check for common SPA indicators
        spa_indicators = [
            "window.__INITIAL_STATE__",
            "window.__NUXT__",
            "window.__NEXT_DATA__",
            "__GATSBY",
            "ng-app",
            "data-reactroot",
        ]

        for indicator in spa_indicators:
            if indicator in html:
                # Check if content is too short (likely not rendered)
                if len(content) < 500:
                    return True

        # Check for noscript warning
        if "<noscript>" in html.lower() and len(content) < 500:
            return True

        # Check for very short content with lots of JS
        js_count = html.lower().count("<script")
        if js_count > 10 and len(content) < 300:
            return True

        return False

    async def read(self, url: str) -> PageContent:
        """Read and extract content from a URL.

        Args:
            url: The URL to read.

        Returns:
            PageContent with extracted content and metadata.
        """
        session = await self._get_session()

        try:
            # Fetch the page
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return PageContent(
                        url=url,
                        title=None,
                        content="",
                        error=f"HTTP {response.status}",
                    )

                html = await response.text()
        except Exception as e:
            return PageContent(
                url=url,
                title=None,
                content="",
                error=str(e),
            )

        # Extract metadata
        title = self._extract_title(html)
        description = self._extract_description(html)
        publish_date = self._extract_publish_date(html)
        author = self._extract_author(html)

        # Convert to markdown
        content = self._html_to_markdown(html)

        # Truncate if needed
        if len(content) > self.config.max_content_chars_per_source:
            content = content[: self.config.max_content_chars_per_source] + "\n\n[Content truncated]"

        # Check if browser is needed
        needs_browser = self._detect_needs_browser(html, content)

        # Count words
        word_count = len(content.split())

        return PageContent(
            url=url,
            title=title,
            content=content,
            raw_html=html if len(html) < 100000 else None,
            author=author,
            publish_date=publish_date,
            description=description,
            word_count=word_count,
            extraction_method="http_reader",
            needs_browser=needs_browser,
        )


class MockSession:
    """Mock HTTP session for testing."""

    def get(self, url: str, **kwargs: Any) -> MockResponse:
        """Return a MockResponse that can be used as an async context manager."""
        return MockResponse(url)

    async def close(self) -> None:
        pass


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, url: str):
        self.url = url
        self.status = 200

    async def text(self) -> str:
        return f"""
        <html>
        <head>
            <title>Mock Page for {self.url}</title>
            <meta name="description" content="This is a mock page for testing.">
        </head>
        <body>
            <h1>Mock Content</h1>
            <p>This is mock content for the URL: {self.url}</p>
            <p>It contains some sample text for testing the page reader.</p>
        </body>
        </html>
        """

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass
