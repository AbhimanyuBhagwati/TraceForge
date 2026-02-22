"""Fuzzing mutation operators (schema-aware)."""

from typing import Any

from traceforge.models import MutationRecord, ToolDefinition


class MutationOperator:
    """Base class for tool argument mutations."""
    name: str = ""

    def can_mutate(self, param_schema: dict, param_name: str, current_value: Any) -> bool:
        raise NotImplementedError

    def mutate(self, param_schema: dict, param_name: str, current_value: Any) -> tuple[Any, str]:
        raise NotImplementedError


class NumericExtreme(MutationOperator):
    """Replace numbers with extremes: 0, -1, MAX_INT, etc."""
    name = "numeric_extreme"

    def can_mutate(self, schema, name, value):
        return schema.get("type") in ("number", "integer") or isinstance(value, (int, float))

    def mutate(self, schema, name, value):
        extremes = [0, -1, -999999, 999999, 0.0001, float("inf")]
        mutated = extremes[hash(name) % len(extremes)]
        return mutated, f"Set '{name}' to extreme value {mutated}"


class MissingRequired(MutationOperator):
    """Remove a required field entirely."""
    name = "missing_required"

    def can_mutate(self, schema, name, value):
        return True

    def mutate(self, schema, name, value):
        return "__REMOVE__", f"Removed required field '{name}'"


class TypeSwap(MutationOperator):
    """Swap type: string->number, number->string, etc."""
    name = "type_swap"

    def can_mutate(self, schema, name, value):
        return True

    def mutate(self, schema, name, value):
        if isinstance(value, str):
            return 42, f"Swapped '{name}' from string to number"
        elif isinstance(value, (int, float)):
            return str(value), f"Swapped '{name}' from number to string"
        elif isinstance(value, bool):
            return "true", f"Swapped '{name}' from bool to string"
        return None, f"Swapped '{name}' to null"


class EmptyString(MutationOperator):
    """Replace strings with empty string."""
    name = "empty_string"

    def can_mutate(self, schema, name, value):
        return isinstance(value, str)

    def mutate(self, schema, name, value):
        return "", f"Set '{name}' to empty string"


class NullInjection(MutationOperator):
    """Replace any value with null/None."""
    name = "null_injection"

    def can_mutate(self, schema, name, value):
        return True

    def mutate(self, schema, name, value):
        return None, f"Set '{name}' to null"


class BoundaryValue(MutationOperator):
    """Boundary values: very long strings, schema min/max for numbers."""
    name = "boundary"

    def can_mutate(self, schema, name, value):
        return True

    def mutate(self, schema, name, value):
        if isinstance(value, str):
            return "A" * 10000, f"Set '{name}' to 10000-char string"
        elif isinstance(value, (int, float)):
            maximum = schema.get("maximum", 999999)
            return maximum, f"Set '{name}' to schema maximum {maximum}"
        return value, "No boundary mutation applied"


ALL_MUTATORS: dict[str, MutationOperator] = {
    "numeric_extreme": NumericExtreme(),
    "missing_required": MissingRequired(),
    "type_swap": TypeSwap(),
    "empty_string": EmptyString(),
    "null_injection": NullInjection(),
    "boundary": BoundaryValue(),
}


def generate_mutations(
    tool_def: ToolDefinition,
    original_args: dict,
    mutation_types: list[str],
    max_mutations: int = 5,
) -> list[MutationRecord]:
    """Generate mutated argument sets for a given tool call.

    Each mutation changes exactly one parameter.
    """
    mutations = []
    properties = tool_def.parameters.get("properties", {})

    for param_name, param_value in original_args.items():
        param_schema = properties.get(param_name, {})
        for mut_type in mutation_types:
            if len(mutations) >= max_mutations:
                return mutations
            mutator = ALL_MUTATORS.get(mut_type)
            if mutator and mutator.can_mutate(param_schema, param_name, param_value):
                new_val, desc = mutator.mutate(param_schema, param_name, param_value)
                mutated_args = {**original_args}
                if new_val == "__REMOVE__":
                    del mutated_args[param_name]
                else:
                    mutated_args[param_name] = new_val
                mutations.append(MutationRecord(
                    original_args=original_args,
                    mutated_args=mutated_args,
                    mutation_type=mut_type,
                    mutation_description=desc,
                ))
    return mutations
