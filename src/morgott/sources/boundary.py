from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator

from ..data import _sample, _set_source_role
from ._shared import _download_files

BOUNDARY_INSTRUCTION_FAMILIES = {
    "direct_instruction_override",
    "indirect_content_injection",
    "memory_context_poisoning",
    "multi_agent_trust_spoofing",
    "obfuscation_encoding",
    "rag_context_poisoning",
    "roleplay_jailbreak",
    "system_prompt_extraction",
}
BOUNDARY_AUTHORIZATION_FAMILIES = {
    "approval_workflow_bypass",
    "authority_claim_bypass",
    "sensitive_data_exfiltration",
    "tool_action_abuse",
}
BOUNDARY_FAMILIES = BOUNDARY_INSTRUCTION_FAMILIES | BOUNDARY_AUTHORIZATION_FAMILIES
BOUNDARY_DIRECT_CONTEXT = "direct_user"
BOUNDARY_FIELDS = {
    "attack_family",
    "category",
    "expected_action",
    "id",
    "label",
    "language",
    "pair_family",
    "pair_id",
    "risk_domain",
    "scenario_id",
    "source_context",
    "source_type",
    "split",
    "target_boundary",
    "text",
}


def _boundary_pair_sample(source_row: dict, split: str) -> dict:
    pair_family = source_row["pair_family"]
    source_context = source_row["source_context"]
    boundary_label = source_row["label"]
    instruction_family = pair_family in BOUNDARY_INSTRUCTION_FAMILIES
    input_channel = (
        "direct_user"
        if source_context == BOUNDARY_DIRECT_CONTEXT
        else "untrusted_content"
    )
    if instruction_family:
        label = boundary_label
        security_label = None
        if boundary_label == 0:
            attack_type = None
        elif pair_family == "roleplay_jailbreak":
            attack_type = "direct_jailbreak"
        elif input_channel == "untrusted_content":
            attack_type = "indirect_prompt_injection"
        else:
            attack_type = "direct_prompt_injection"
        label_basis = "synthetic_curated_matched_instruction_subversion_pair"
    else:
        label = None
        security_label = "uncertain"
        attack_type = None
        label_basis = "synthetic_curated_authorization_boundary_diagnostic"
    row = _sample(
        text=source_row["text"],
        label=label,
        attack_type=attack_type,
        source="agentic_boundary_pairs",
        source_split=split,
        source_id=source_row["id"],
        group_id=f"agentic_boundary_pairs:pair:{source_row['pair_id']}",
        split_group_id=(f"agentic_boundary_pairs:scenario:{source_row['scenario_id']}"),
        category=source_row["category"],
        input_channel=input_channel,
        label_basis=label_basis,
        security_label=security_label,
    )
    row.update(
        {
            "source_attack_family": source_row["attack_family"],
            "source_boundary_label": boundary_label,
            "source_expected_action": source_row["expected_action"],
            "source_language": source_row["language"],
            "source_pair_family": pair_family,
            "source_pair_id": source_row["pair_id"],
            "source_risk_domain": source_row["risk_domain"],
            "source_scenario_id": source_row["scenario_id"],
            "source_source_context": source_context,
            "source_source_type": source_row["source_type"],
            "source_target_boundary": source_row["target_boundary"],
        }
    )
    return _set_source_role(row, "auxiliary")


def _validate_boundary_rows(rows_by_split: dict[str, list[dict]]) -> dict:
    row_ids = set()
    pairs = defaultdict(list)
    scenario_splits = defaultdict(set)
    family_counts = Counter()
    for split, rows in rows_by_split.items():
        for source_row in rows:
            if set(source_row) != BOUNDARY_FIELDS:
                raise ValueError(
                    f"agentic_boundary_pairs:{split} has an unexpected schema"
                )
            if source_row["split"] != split:
                raise ValueError(
                    f"agentic_boundary_pairs:{source_row['id']} has a split mismatch"
                )
            if type(source_row["label"]) is not int or source_row["label"] not in {
                0,
                1,
            }:
                raise ValueError(
                    f"agentic_boundary_pairs:{source_row['id']} has an invalid label"
                )
            for field in BOUNDARY_FIELDS - {"label"}:
                if (
                    not isinstance(source_row[field], str)
                    or not source_row[field].strip()
                ):
                    raise ValueError(
                        f"agentic_boundary_pairs:{source_row['id']} has invalid {field}"
                    )
            row_id = source_row["id"]
            if row_id in row_ids:
                raise ValueError(f"agentic_boundary_pairs has duplicate id {row_id}")
            row_ids.add(row_id)
            pair_family = source_row["pair_family"]
            if pair_family not in BOUNDARY_FAMILIES:
                raise ValueError(
                    f"agentic_boundary_pairs:{row_id} has an unknown pair family"
                )
            expected_attack_family = pair_family if source_row["label"] else "none"
            if source_row["attack_family"] != expected_attack_family:
                raise ValueError(
                    f"agentic_boundary_pairs:{row_id} has inconsistent family labels"
                )
            expected_category = (
                "prompt_injection" if source_row["label"] else "benign_boundary"
            )
            expected_action = "block_or_review" if source_row["label"] else "allow"
            if (
                source_row["category"] != expected_category
                or source_row["expected_action"] != expected_action
                or source_row["language"] != "en"
                or source_row["source_type"] != "synthetic_curated"
            ):
                raise ValueError(
                    f"agentic_boundary_pairs:{row_id} has inconsistent metadata"
                )
            pair_key = source_row["pair_id"]
            pairs[pair_key].append(source_row)
            scenario_splits[source_row["scenario_id"]].add(split)
            family_counts[f"{split}:{pair_family}:{source_row['label']}"] += 1
    for pair_id, rows in pairs.items():
        if len(rows) != 2 or {row["label"] for row in rows} != {0, 1}:
            raise ValueError(
                f"agentic_boundary_pairs:{pair_id} is not one aligned binary pair"
            )
        if any(
            len({row[field] for row in rows}) != 1
            for field in (
                "pair_family",
                "scenario_id",
                "source_context",
                "risk_domain",
                "split",
                "target_boundary",
            )
        ):
            raise ValueError(
                f"agentic_boundary_pairs:{pair_id} has inconsistent pair lineage"
            )
    crossing = sorted(
        scenario_id
        for scenario_id, splits in scenario_splits.items()
        if len(splits) != 1
    )
    if crossing:
        raise ValueError(
            "agentic_boundary_pairs scenarios cross official splits: "
            + ", ".join(crossing[:5])
        )
    return {
        "official_rows": {
            split: len(rows) for split, rows in sorted(rows_by_split.items())
        },
        "pairs": len(pairs),
        "scenarios": len(scenario_splits),
        "family_label_rows": dict(sorted(family_counts.items())),
        "instruction_subversion_families": sorted(BOUNDARY_INSTRUCTION_FAMILIES),
        "authorization_diagnostic_families": sorted(BOUNDARY_AUTHORIZATION_FAMILIES),
        "projection": "raw paired detector text with complete boundary metadata",
    }


def _agentic_boundary_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    paths, downloads = _download_files("agentic_boundary_pairs")
    rows_by_split = {}
    for split, path in paths.items():
        rows_by_split[split] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    profile = _validate_boundary_rows(rows_by_split)

    def rows() -> Iterator[dict]:
        for split in ("train", "validation", "test"):
            for source_row in rows_by_split[split]:
                yield _boundary_pair_sample(source_row, split)

    return rows(), downloads, profile
