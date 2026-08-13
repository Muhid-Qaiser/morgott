"""Validate and construct experimental mmBERT classification heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core import new_head

PRIMARY_TARGET = "instruction_subversion"
AUXILIARY_TARGET = "harmful_intent"
LEGACY_ARCHITECTURE = "legacy_sequential_binary_v1"
MULTITASK_ARCHITECTURE = "shared_trunk_separate_binary_projections_v1"


@dataclass(frozen=True)
class HeadContract:
    """The output layout recorded in a training run's ``result.json``."""

    architecture: str
    outputs: int
    columns: tuple[str, ...]
    primary_column: int


_LEGACY_CONTRACT = HeadContract(
    architecture=LEGACY_ARCHITECTURE,
    outputs=1,
    columns=(PRIMARY_TARGET,),
    primary_column=0,
)


def resolve_head_contract(result: Mapping[str, object]) -> HeadContract:
    """Return a strict output contract, with an absent field meaning legacy width 1."""

    value = result.get("head_contract")
    if value is None:
        return _LEGACY_CONTRACT
    if not isinstance(value, dict) or set(value) != {
        "architecture",
        "outputs",
        "columns",
        "primary_column",
    }:
        raise ValueError("invalid head contract fields")

    architecture = value["architecture"]
    outputs = value["outputs"]
    primary_column = value["primary_column"]
    columns = value["columns"]
    if type(outputs) is not int or outputs not in {1, 2}:  # bool is not a width
        raise ValueError("head contract outputs must be one or two")
    if type(primary_column) is not int or primary_column != 0:
        raise ValueError("head contract primary column must be zero")
    expected = {"0": PRIMARY_TARGET}
    expected_architecture = LEGACY_ARCHITECTURE
    if outputs == 2:
        expected["1"] = AUXILIARY_TARGET
        expected_architecture = MULTITASK_ARCHITECTURE
    if architecture != expected_architecture:
        raise ValueError("head contract architecture does not match the output width")
    if not isinstance(columns, dict) or columns != expected:
        raise ValueError("head contract columns do not match the output width")
    return HeadContract(
        architecture=architecture,
        outputs=outputs,
        columns=tuple(expected[str(index)] for index in range(outputs)),
        primary_column=primary_column,
    )


def new_multitask_head(hidden_size: int, seed: int):
    """Build the shared-trunk head with legacy-identical primary parameters."""

    import torch
    from torch import nn

    class MultitaskHead(nn.Module):
        def __init__(self, legacy, auxiliary) -> None:
            super().__init__()
            self.trunk = nn.Sequential(*list(legacy.children())[:-1])
            self.primary = legacy[-1]
            self.auxiliary = auxiliary

        def forward(self, features):
            shared = self.trunk(features)
            return torch.cat(
                (self.primary(shared), self.auxiliary(shared)),
                dim=-1,
            )

    legacy = new_head(hidden_size, seed)
    try:
        primary = legacy[-1]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("legacy head has no final projection") from error
    if not isinstance(primary, nn.Linear) or primary.out_features != 1:
        raise ValueError("legacy head final projection must be nn.Linear(..., 1)")

    # The auxiliary projection must be deterministic without perturbing the
    # global RNG state seen by the rest of training. Preserve exactly the state
    # left by constructing the historical head.
    post_legacy_rng = torch.random.get_rng_state()
    try:
        auxiliary = nn.Linear(primary.in_features, 1)
    finally:
        torch.random.set_rng_state(post_legacy_rng)
    return MultitaskHead(legacy, auxiliary)


def new_head_for_result(
    hidden_size: int,
    seed: int,
    result: Mapping[str, object],
):
    """Build the exact one- or two-output architecture declared by a run."""

    contract = resolve_head_contract(result)
    if contract.outputs == 1:
        return new_head(hidden_size, seed)
    return new_multitask_head(hidden_size, seed)
