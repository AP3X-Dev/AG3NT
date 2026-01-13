#!/usr/bin/env python3
"""Example MCP server providing math tools.

This is a simple MCP server that provides basic math operations.
Run with: python math_server.py

Requires: pip install mcp
"""

import asyncio
import math

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    raise

# Create the MCP server
server = Server("math-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available math tools."""
    return [
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="multiply",
            description="Multiply two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="sqrt",
            description="Calculate the square root of a number",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Number to take square root of"},
                },
                "required": ["x"],
            },
        ),
        Tool(
            name="factorial",
            description="Calculate the factorial of a non-negative integer",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Non-negative integer"},
                },
                "required": ["n"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a math tool."""
    if name == "add":
        result = arguments["a"] + arguments["b"]
    elif name == "multiply":
        result = arguments["a"] * arguments["b"]
    elif name == "sqrt":
        x = arguments["x"]
        if x < 0:
            return [TextContent(type="text", text=f"Error: Cannot take square root of negative number {x}")]
        result = math.sqrt(x)
    elif name == "factorial":
        n = arguments["n"]
        if n < 0:
            return [TextContent(type="text", text=f"Error: Factorial not defined for negative numbers")]
        result = math.factorial(n)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return [TextContent(type="text", text=str(result))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())

