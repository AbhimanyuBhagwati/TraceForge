"""Mock tool registry and execution."""

from traceforge.models import ToolDefinition


class MockToolRegistry:
    """Registry that cycles through mock responses for each tool."""

    def __init__(self, tools: list[ToolDefinition]):
        self._tools = {t.name: t for t in tools}
        self._indices: dict[str, int] = {t.name: 0 for t in tools}

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Call a mock tool and return the next mock response."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}
        if not tool.mock_responses:
            return {"error": f"No mock responses for: {tool_name}"}
        idx = self._indices[tool_name] % len(tool.mock_responses)
        response = tool.mock_responses[idx]
        self._indices[tool_name] += 1
        return response

    def reset(self):
        """Reset all response indices to 0."""
        self._indices = {name: 0 for name in self._tools}
