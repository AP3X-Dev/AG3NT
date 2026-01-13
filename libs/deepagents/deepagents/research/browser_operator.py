"""Browser Operator for JS-heavy page extraction.

The BrowserOperator handles pages that require JavaScript rendering:
- Abstracts browser automation via a BrowserDriver interface
- Supports Playwright MCP backend adapter
- Implements observe/act/extract patterns for complex pages
- Manages step budgets and timeouts
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from deepagents.research.config import ResearchConfig
from deepagents.research.page_reader import PageContent

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BrowserActionType(str, Enum):
    """Types of browser actions."""
    
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"


@dataclass
class BrowserAction:
    """A single browser action."""
    
    action_type: BrowserActionType
    target: str | None = None  # CSS selector or URL
    value: str | None = None  # Text to type or scroll amount
    timeout_ms: int = 5000


@dataclass
class BrowserState:
    """Current state of the browser."""
    
    url: str
    title: str | None = None
    page_content: str | None = None
    screenshot_path: str | None = None
    
    # Observed elements
    clickable_elements: list[dict[str, Any]] = field(default_factory=list)
    input_elements: list[dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    step_count: int = 0
    last_action: BrowserAction | None = None
    error: str | None = None


class BrowserDriver(ABC):
    """Abstract interface for browser automation.
    
    Implementations can use Playwright, Selenium, or other backends.
    """
    
    @abstractmethod
    async def navigate(self, url: str) -> BrowserState:
        """Navigate to a URL."""
        ...
    
    @abstractmethod
    async def click(self, selector: str) -> BrowserState:
        """Click an element."""
        ...
    
    @abstractmethod
    async def type_text(self, selector: str, text: str) -> BrowserState:
        """Type text into an element."""
        ...
    
    @abstractmethod
    async def scroll(self, direction: str = "down", amount: int = 500) -> BrowserState:
        """Scroll the page."""
        ...
    
    @abstractmethod
    async def wait(self, selector: str | None = None, timeout_ms: int = 5000) -> BrowserState:
        """Wait for an element or timeout."""
        ...
    
    @abstractmethod
    async def get_state(self) -> BrowserState:
        """Get current browser state."""
        ...
    
    @abstractmethod
    async def extract_content(self) -> str:
        """Extract page content as markdown."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close the browser."""
        ...


class MockBrowserDriver(BrowserDriver):
    """Mock browser driver for testing."""
    
    def __init__(self) -> None:
        self._current_url = ""
        self._step_count = 0
    
    async def navigate(self, url: str) -> BrowserState:
        self._current_url = url
        self._step_count += 1
        return BrowserState(
            url=url,
            title=f"Mock Page: {url}",
            step_count=self._step_count,
        )
    
    async def click(self, selector: str) -> BrowserState:
        self._step_count += 1
        return BrowserState(
            url=self._current_url,
            title="Mock Page",
            step_count=self._step_count,
            last_action=BrowserAction(BrowserActionType.CLICK, selector),
        )
    
    async def type_text(self, selector: str, text: str) -> BrowserState:
        self._step_count += 1
        return BrowserState(
            url=self._current_url,
            title="Mock Page",
            step_count=self._step_count,
            last_action=BrowserAction(BrowserActionType.TYPE, selector, text),
        )
    
    async def scroll(self, direction: str = "down", amount: int = 500) -> BrowserState:
        self._step_count += 1
        return BrowserState(
            url=self._current_url,
            title="Mock Page",
            step_count=self._step_count,
            last_action=BrowserAction(BrowserActionType.SCROLL, value=f"{direction}:{amount}"),
        )
    
    async def wait(self, selector: str | None = None, timeout_ms: int = 5000) -> BrowserState:
        self._step_count += 1
        return BrowserState(
            url=self._current_url,
            title="Mock Page",
            step_count=self._step_count,
            last_action=BrowserAction(BrowserActionType.WAIT, selector),
        )
    
    async def get_state(self) -> BrowserState:
        return BrowserState(
            url=self._current_url,
            title="Mock Page",
            step_count=self._step_count,
        )
    
    async def extract_content(self) -> str:
        return f"# Mock Content\n\nThis is mock content from {self._current_url}"
    
    async def close(self) -> None:
        pass


@dataclass
class BrowserTask:
    """A task to be executed in browser mode.

    A BrowserTask describes what needs to be accomplished,
    and the BrowserOperator figures out how to do it.
    """

    goal: str  # What to accomplish
    url: str  # Starting URL

    # Optional hints
    target_selectors: list[str] = field(default_factory=list)
    expected_content: list[str] = field(default_factory=list)

    # Limits
    max_steps: int = 15
    timeout_seconds: int = 60


@dataclass
class BrowserTaskResult:
    """Result of a browser task execution."""

    success: bool
    content: PageContent

    # Execution details
    steps_taken: int = 0
    actions_performed: list[BrowserAction] = field(default_factory=list)

    # Errors
    error: str | None = None


class BrowserOperator:
    """Operates a browser to extract content from JS-heavy pages.

    The BrowserOperator uses an observe/act/extract pattern:
    1. Observe: Get current page state and identify interactive elements
    2. Act: Perform actions to navigate or reveal content
    3. Extract: Pull out the relevant content once found

    Args:
        config: Research configuration.
        driver: Browser driver implementation.
    """

    def __init__(
        self,
        config: ResearchConfig,
        driver: BrowserDriver | None = None,
    ) -> None:
        self.config = config
        self.driver = driver or MockBrowserDriver()
        self._step_count = 0

    async def execute_task(self, task: BrowserTask) -> BrowserTaskResult:
        """Execute a browser task.

        Args:
            task: The task to execute.

        Returns:
            BrowserTaskResult with extracted content.
        """
        actions_performed: list[BrowserAction] = []
        self._step_count = 0

        try:
            # Navigate to the URL
            state = await self.driver.navigate(task.url)
            self._step_count += 1
            actions_performed.append(BrowserAction(BrowserActionType.NAVIGATE, task.url))

            # Wait for initial load
            await self.driver.wait(timeout_ms=2000)
            self._step_count += 1

            # Observe/act loop
            while self._step_count < task.max_steps:
                state = await self.driver.get_state()

                # Check if we have enough content
                content = await self.driver.extract_content()
                if self._has_sufficient_content(content, task):
                    break

                # Try to reveal more content
                action = self._decide_next_action(state, task)
                if action is None:
                    break

                await self._execute_action(action)
                actions_performed.append(action)
                self._step_count += 1

            # Final extraction
            content = await self.driver.extract_content()
            state = await self.driver.get_state()

            page_content = PageContent(
                url=task.url,
                title=state.title,
                content=content,
                word_count=len(content.split()),
                extraction_method="browser",
            )

            return BrowserTaskResult(
                success=True,
                content=page_content,
                steps_taken=self._step_count,
                actions_performed=actions_performed,
            )

        except Exception as e:
            logger.error(f"Browser task failed: {e}")
            return BrowserTaskResult(
                success=False,
                content=PageContent(
                    url=task.url,
                    title=None,
                    content="",
                    error=str(e),
                ),
                steps_taken=self._step_count,
                actions_performed=actions_performed,
                error=str(e),
            )

    def _has_sufficient_content(self, content: str, task: BrowserTask) -> bool:
        """Check if we have enough content."""
        # Check minimum length
        if len(content) < 500:
            return False

        # Check for expected content
        if task.expected_content:
            found = sum(1 for exp in task.expected_content if exp.lower() in content.lower())
            if found < len(task.expected_content) / 2:
                return False

        return True

    def _decide_next_action(
        self,
        state: BrowserState,
        task: BrowserTask,
    ) -> BrowserAction | None:
        """Decide the next action to take."""
        # If we have target selectors, try clicking them
        if task.target_selectors:
            for selector in task.target_selectors:
                # Check if element exists in clickable elements
                for elem in state.clickable_elements:
                    if selector in str(elem):
                        return BrowserAction(BrowserActionType.CLICK, selector)

        # Default: scroll down to load more content
        if self._step_count < 5:
            return BrowserAction(BrowserActionType.SCROLL, value="down:500")

        # Give up after a few scrolls
        return None

    async def _execute_action(self, action: BrowserAction) -> BrowserState:
        """Execute a browser action."""
        if action.action_type == BrowserActionType.CLICK:
            return await self.driver.click(action.target or "")
        elif action.action_type == BrowserActionType.TYPE:
            return await self.driver.type_text(action.target or "", action.value or "")
        elif action.action_type == BrowserActionType.SCROLL:
            parts = (action.value or "down:500").split(":")
            direction = parts[0]
            amount = int(parts[1]) if len(parts) > 1 else 500
            return await self.driver.scroll(direction, amount)
        elif action.action_type == BrowserActionType.WAIT:
            return await self.driver.wait(action.target, action.timeout_ms)
        else:
            return await self.driver.get_state()

    async def close(self) -> None:
        """Close the browser."""
        await self.driver.close()
