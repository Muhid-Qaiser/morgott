"""Trusted lineage propagation immediately before side effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .policy import TRUSTED_PROVENANCE, UNTRUSTED_PROVENANCE, authorize

_KNOWN_PROVENANCE = TRUSTED_PROVENANCE | UNTRUSTED_PROVENANCE


@dataclass(frozen=True, slots=True)
class SourcedValue:
    """A runtime value with trusted, monotone data-flow metadata."""

    value: str
    sources: frozenset[str]
    provenance: frozenset[str]
    sensitive: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or not isinstance(self.sources, frozenset)
            or not self.sources
            or not all(
                isinstance(source, str) and source.strip() for source in self.sources
            )
            or not isinstance(self.provenance, frozenset)
            or not self.provenance
            or not self.provenance <= _KNOWN_PROVENANCE
            or type(self.sensitive) is not bool
        ):
            raise ValueError("invalid sourced value")

    @classmethod
    def source(
        cls,
        value: str,
        *,
        source: str,
        provenance: str,
        sensitive: bool = False,
    ) -> SourcedValue:
        return cls(value, frozenset({source}), frozenset({provenance}), sensitive)

    @classmethod
    def derived(
        cls,
        value: str,
        *inputs: SourcedValue,
        producer: str = "planner_output",
    ) -> SourcedValue:
        if not inputs or not all(isinstance(item, cls) for item in inputs):
            raise ValueError("derived values require sourced inputs")
        provenance = frozenset().union(*(item.provenance for item in inputs))
        if producer not in _KNOWN_PROVENANCE:
            raise ValueError("unknown producer provenance")
        provenance |= {producer}
        return cls(
            value=value,
            sources=frozenset().union(*(item.sources for item in inputs)),
            provenance=provenance,
            sensitive=any(item.sensitive for item in inputs),
        )


def enforce(
    policy: dict,
    tool: str,
    arguments: dict[str, SourcedValue],
    *,
    influenced_by: tuple[SourcedValue, ...],
    effect: Callable[[str, dict[str, str]], object],
) -> tuple[bool, str]:
    """Invoke ``effect`` once only when the sourced action is authorized."""
    # ponytail: this seam is synchronous; add an async twin only for a real
    # async tool adapter rather than maintaining two unused execution paths.
    if (
        not isinstance(tool, str)
        or not tool.strip()
        or not isinstance(arguments, dict)
        or not all(
            isinstance(name, str) and name and isinstance(value, SourcedValue)
            for name, value in arguments.items()
        )
        or not isinstance(influenced_by, tuple)
        or not influenced_by
        or not all(isinstance(value, SourcedValue) for value in influenced_by)
    ):
        return False, "invalid_runtime_metadata"
    if not callable(effect):
        raise TypeError("effect must be callable")

    values = (*arguments.values(), *influenced_by)
    context: dict[str, object] = {
        "contains_sensitive_data": any(value.sensitive for value in values),
        "provenance": sorted(
            frozenset().union(*(value.provenance for value in values))
        ),
    }
    capability = (
        policy.get("capabilities", {}).get(tool, {})
        if isinstance(policy, dict) and isinstance(policy.get("capabilities"), dict)
        else {}
    )
    source_bound = (
        capability.get("allowed_argument_sources")
        if isinstance(capability, dict)
        else None
    )
    if isinstance(source_bound, dict):
        context["argument_sources"] = {
            name: sorted(arguments[name].sources)
            for name in source_bound
            if name in arguments
        }

    action = {
        "tool": tool,
        "arguments": {name: value.value for name, value in arguments.items()},
    }
    allowed, reason = authorize(policy, action, context)
    if allowed:
        effect(action["tool"], dict(action["arguments"]))
    return allowed, reason
