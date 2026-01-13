"""Tests for PageReader."""

import pytest

from deepagents.research import PageContent, PageReader, ResearchConfig


class TestPageReader:
    """Tests for PageReader."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResearchConfig(
            page_fetch_timeout_seconds=10,
            max_content_chars_per_source=10000,
        )

    @pytest.fixture
    def reader(self, config):
        """Create a page reader with mock session."""
        return PageReader(config, use_mock=True)

    @pytest.mark.asyncio
    async def test_read_mock_page(self, reader):
        """Test reading a mock page."""
        content = await reader.read("https://example.com/test")

        assert content.url == "https://example.com/test"
        assert content.title is not None
        assert len(content.content) > 0
        assert content.error is None

    @pytest.mark.asyncio
    async def test_content_extraction(self, reader):
        """Test content extraction from HTML."""
        content = await reader.read("https://example.com/article")

        # Should have extracted markdown content
        assert "Mock Content" in content.content or "mock" in content.content.lower()
        assert content.word_count > 0

    def test_html_to_markdown_headers(self, reader):
        """Test HTML header conversion."""
        html = "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>"
        md = reader._html_to_markdown(html)

        assert "# Title" in md
        assert "## Subtitle" in md
        assert "Content" in md

    def test_html_to_markdown_links(self, reader):
        """Test HTML link conversion."""
        html = '<a href="https://example.com">Link Text</a>'
        md = reader._html_to_markdown(html)

        assert "[Link Text](https://example.com)" in md

    def test_html_to_markdown_lists(self, reader):
        """Test HTML list conversion."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        md = reader._html_to_markdown(html)

        assert "- Item 1" in md
        assert "- Item 2" in md

    def test_html_to_markdown_formatting(self, reader):
        """Test HTML formatting conversion."""
        html = "<strong>Bold</strong> and <em>italic</em>"
        md = reader._html_to_markdown(html)

        assert "**Bold**" in md
        assert "*italic*" in md

    def test_html_to_markdown_removes_scripts(self, reader):
        """Test that scripts are removed."""
        html = "<p>Content</p><script>alert('bad')</script><p>More</p>"
        md = reader._html_to_markdown(html)

        assert "alert" not in md
        assert "Content" in md
        assert "More" in md

    def test_extract_title(self, reader):
        """Test title extraction."""
        html = "<html><head><title>Page Title</title></head></html>"
        title = reader._extract_title(html)

        assert title == "Page Title"

    def test_extract_title_og(self, reader):
        """Test og:title extraction."""
        html = '<meta property="og:title" content="OG Title">'
        title = reader._extract_title(html)

        assert title == "OG Title"

    def test_extract_description(self, reader):
        """Test description extraction."""
        html = '<meta name="description" content="Page description">'
        desc = reader._extract_description(html)

        assert desc == "Page description"

    def test_detect_needs_browser_spa(self, reader):
        """Test SPA detection."""
        html = '<div id="app"></div><script>window.__NEXT_DATA__={}</script>'
        content = "Short"

        needs_browser = reader._detect_needs_browser(html, content)

        assert needs_browser is True

    def test_detect_needs_browser_normal(self, reader):
        """Test normal page detection."""
        html = "<html><body><p>Normal content here with enough text to pass the threshold.</p></body></html>"
        content = "Normal content here with enough text to pass the threshold. " * 20

        needs_browser = reader._detect_needs_browser(html, content)

        assert needs_browser is False


class TestPageContent:
    """Tests for PageContent dataclass."""

    def test_page_content_creation(self):
        """Test creating PageContent."""
        content = PageContent(
            url="https://example.com",
            title="Test Page",
            content="# Test\n\nContent here",
            word_count=3,
            extraction_method="http_reader",
        )

        assert content.url == "https://example.com"
        assert content.title == "Test Page"
        assert content.word_count == 3
        assert content.needs_browser is False
        assert content.error is None

    def test_page_content_with_error(self):
        """Test PageContent with error."""
        content = PageContent(
            url="https://example.com",
            title=None,
            content="",
            error="Connection timeout",
        )

        assert content.error == "Connection timeout"
        assert content.content == ""
