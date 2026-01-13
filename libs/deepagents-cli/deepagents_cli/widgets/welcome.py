"""Welcome banner widget for AG3NT CLI."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

from deepagents_cli._version import __version__
from deepagents_cli.config import AP3X_ASCII_AP3_LINES, AP3X_ASCII_X_LINES


def _build_colored_banner() -> str:
    """Build the AP3X banner with AP3 in white and X in red.

    Combines the two ASCII art parts line by line for proper alignment.
    """
    # Combine each line with proper coloring
    combined_lines = []
    for ap3_part, x_part in zip(AP3X_ASCII_AP3_LINES, AP3X_ASCII_X_LINES):
        combined_lines.append(f"[bold white]{ap3_part}[/bold white][bold #ff3333]{x_part}[/bold #ff3333]")

    return '\n'.join(combined_lines)


class WelcomeBanner(Static):
    """Welcome banner displayed at startup."""

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the welcome banner."""
        # AP3X banner with AP3 in white, X in red
        banner_text = _build_colored_banner()
        banner_text += f"\n[dim]v{__version__}[/dim]\n"
        banner_text += "[#10b981]Ready to code! What would you like to build?[/#10b981]\n"
        banner_text += "[dim]Enter send • Ctrl+J newline • @ files • / commands[/dim]"
        super().__init__(banner_text, **kwargs)
