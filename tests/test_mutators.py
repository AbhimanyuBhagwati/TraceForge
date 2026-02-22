"""Tests for mutation operators."""

from traceforge.models import ToolDefinition
from traceforge.mutators import (
    ALL_MUTATORS,
    BoundaryValue,
    EmptyString,
    MissingRequired,
    NullInjection,
    NumericExtreme,
    TypeSwap,
    generate_mutations,
)


def make_tool():
    return ToolDefinition(
        name="calc",
        description="Calculator",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "integer", "maximum": 10},
            },
            "required": ["expression"],
        },
    )


class TestNumericExtreme:
    def test_can_mutate_number(self):
        m = NumericExtreme()
        assert m.can_mutate({"type": "number"}, "x", 5)
        assert m.can_mutate({"type": "integer"}, "x", 5)
        assert m.can_mutate({}, "x", 3.14)

    def test_cannot_mutate_string(self):
        m = NumericExtreme()
        assert not m.can_mutate({"type": "string"}, "x", "hello")

    def test_mutate(self):
        m = NumericExtreme()
        val, desc = m.mutate({"type": "number"}, "price", 10.0)
        assert isinstance(val, (int, float))
        assert "price" in desc


class TestMissingRequired:
    def test_always_can_mutate(self):
        m = MissingRequired()
        assert m.can_mutate({}, "x", "anything")

    def test_mutate_returns_remove(self):
        m = MissingRequired()
        val, desc = m.mutate({}, "name", "test")
        assert val == "__REMOVE__"
        assert "name" in desc


class TestTypeSwap:
    def test_string_to_number(self):
        m = TypeSwap()
        val, _ = m.mutate({}, "x", "hello")
        assert isinstance(val, int)

    def test_number_to_string(self):
        m = TypeSwap()
        val, _ = m.mutate({}, "x", 42)
        assert isinstance(val, str)


class TestEmptyString:
    def test_can_mutate_string(self):
        m = EmptyString()
        assert m.can_mutate({}, "x", "hello")
        assert not m.can_mutate({}, "x", 42)

    def test_mutate(self):
        m = EmptyString()
        val, _ = m.mutate({}, "x", "hello")
        assert val == ""


class TestNullInjection:
    def test_always_can_mutate(self):
        m = NullInjection()
        assert m.can_mutate({}, "x", "anything")

    def test_mutate(self):
        m = NullInjection()
        val, _ = m.mutate({}, "x", "hello")
        assert val is None


class TestBoundaryValue:
    def test_string_boundary(self):
        m = BoundaryValue()
        val, _ = m.mutate({}, "x", "hello")
        assert len(val) == 10000

    def test_number_boundary_uses_schema_max(self):
        m = BoundaryValue()
        val, _ = m.mutate({"maximum": 100}, "x", 5)
        assert val == 100


class TestGenerateMutations:
    def test_generates_mutations(self):
        tool = make_tool()
        mutations = generate_mutations(
            tool,
            {"expression": "6 * 7", "precision": 2},
            ["numeric_extreme", "empty_string", "null_injection"],
            max_mutations=10,
        )
        assert len(mutations) > 0
        for m in mutations:
            assert m.mutation_type in ["numeric_extreme", "empty_string", "null_injection"]

    def test_max_mutations_limit(self):
        tool = make_tool()
        mutations = generate_mutations(
            tool,
            {"expression": "6 * 7", "precision": 2},
            ["numeric_extreme", "empty_string", "null_injection", "type_swap", "missing_required"],
            max_mutations=3,
        )
        assert len(mutations) <= 3

    def test_remove_mutation(self):
        tool = make_tool()
        mutations = generate_mutations(
            tool,
            {"expression": "6 * 7"},
            ["missing_required"],
            max_mutations=5,
        )
        assert any("expression" not in m.mutated_args for m in mutations)

    def test_all_mutators_registered(self):
        assert len(ALL_MUTATORS) == 6
        expected = {"numeric_extreme", "missing_required", "type_swap",
                    "empty_string", "null_injection", "boundary"}
        assert set(ALL_MUTATORS.keys()) == expected
