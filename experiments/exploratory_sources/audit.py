"""Offline projection contracts for deferred exploratory corpora."""

from __future__ import annotations

from collections.abc import Iterable

from vulsight_guard.data import normalize_text, text_hash


HACKAPROMPT_REVISION = "25b87fbedfb86840abaf8cd09af7a029208a971a"
HACKAPROMPT_CARD_FIELDS = {
    "level",
    "user_input",
    "prompt",
    "completion",
    "model",
    "expected_completion",
    "token_count",
    "correct",
    "error",
    "score",
    "dataset",
    "timestamp",
}
HACKAPROMPT_FORBIDDEN_MODEL_FIELDS = {
    "prompt",
    "completion",
    "expected_completion",
}
_PARTITIONS = {"playground_data", "submission_data"}
_MODELS = {"FlanT5-XXL", "gpt-3.5-turbo", "text-davinci-003"}


def project_hackaprompt_rows(rows: Iterable[dict]) -> list[dict]:
    """Project card-shaped rows to user input and target-specific success metadata.

    This makes no network request. HackAPrompt remains gated and is not active in
    the dataset build; the function only locks the safe field/label contract for a
    future explicitly authorized local audit.
    """
    grouped: dict[str, dict] = {}
    for index, row in enumerate(rows):
        missing = HACKAPROMPT_CARD_FIELDS - set(row)
        if missing:
            raise ValueError(f"row {index} is missing card fields: {sorted(missing)}")
        level = row["level"]
        text = row["user_input"]
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or not 0 <= level <= 10
        ):
            raise ValueError(f"row {index} has an invalid level")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {index} has empty user_input")
        if not isinstance(row["correct"], bool) or not isinstance(row["error"], bool):
            raise ValueError(f"row {index} has non-boolean outcome fields")
        if row["error"]:
            raise ValueError(f"row {index} is an errored submission")
        if row["dataset"] not in _PARTITIONS or row["model"] not in _MODELS:
            raise ValueError(f"row {index} has an unknown partition or model")

        text = text.strip()
        digest = text_hash(text)
        projected = grouped.setdefault(
            digest,
            {
                "schema_version": 1,
                "id": f"hackaprompt:{digest}",
                "text": text,
                "label": 1,
                "attack_type": "direct_prompt_injection",
                "attack_success_any": False,
                "successful_target_trials": 0,
                "target_trials": 0,
                "input_channel": "direct_user",
                "label_basis": "competition_submission_attack_attempt",
                "success_basis": "target_model_expected_completion_match",
                "source": "hackaprompt",
                "source_revision": HACKAPROMPT_REVISION,
                "license": "MIT",
                "group_id": f"hackaprompt:text:{digest}",
                "task_group_ids": set(),
                "levels": set(),
                "models": set(),
                "source_partitions": set(),
            },
        )
        if normalize_text(text) != normalize_text(projected["text"]):
            raise AssertionError("text hash collision")
        projected["text"] = min(projected["text"], text)
        projected["attack_success_any"] |= row["correct"]
        projected["successful_target_trials"] += int(row["correct"])
        projected["target_trials"] += 1
        projected["levels"].add(level)
        projected["models"].add(row["model"])
        projected["source_partitions"].add(row["dataset"])
        projected["task_group_ids"].add(f"hackaprompt:level:{level}")

    output = []
    for projected in grouped.values():
        for field in ("task_group_ids", "levels", "models", "source_partitions"):
            projected[field] = sorted(projected[field])
        output.append(projected)
    return sorted(output, key=lambda row: row["id"])
