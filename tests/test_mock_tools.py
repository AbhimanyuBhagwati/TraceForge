"""Tests for mock tool registry."""

from traceforge.mock_tools import MockToolRegistry
from traceforge.models import ToolDefinition


def make_tool(name, responses):
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object"},
        mock_responses=responses,
    )


class TestMockToolRegistry:
    def test_basic_call(self):
        reg = MockToolRegistry([make_tool("calc", [{"result": 42}])])
        assert reg.call("calc", {"expr": "1+1"}) == {"result": 42}

    def test_cycling(self):
        reg = MockToolRegistry([make_tool("calc", [{"r": 1}, {"r": 2}, {"r": 3}])])
        assert reg.call("calc", {}) == {"r": 1}
        assert reg.call("calc", {}) == {"r": 2}
        assert reg.call("calc", {}) == {"r": 3}
        assert reg.call("calc", {}) == {"r": 1}  # wraps around

    def test_unknown_tool(self):
        reg = MockToolRegistry([make_tool("calc", [{"r": 1}])])
        result = reg.call("unknown", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_no_mock_responses(self):
        tool = ToolDefinition(name="bare", description="x", parameters={})
        reg = MockToolRegistry([tool])
        result = reg.call("bare", {})
        assert "error" in result
        assert "No mock responses" in result["error"]

    def test_reset(self):
        reg = MockToolRegistry([make_tool("calc", [{"r": 1}, {"r": 2}])])
        reg.call("calc", {})
        reg.call("calc", {})
        reg.reset()
        assert reg.call("calc", {}) == {"r": 1}

    def test_multiple_tools(self):
        reg = MockToolRegistry([
            make_tool("a", [{"v": "a1"}, {"v": "a2"}]),
            make_tool("b", [{"v": "b1"}]),
        ])
        assert reg.call("a", {}) == {"v": "a1"}
        assert reg.call("b", {}) == {"v": "b1"}
        assert reg.call("a", {}) == {"v": "a2"}
        assert reg.call("b", {}) == {"v": "b1"}  # cycles back
