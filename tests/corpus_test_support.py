import hashlib
import json
from pathlib import Path

from morgott.data import _sample, _set_source_role


def _row(
    source_id: str,
    text: str,
    group: str,
    role: str = "candidate",
    *,
    label: int = 1,
    source: str = "gandalf",
) -> dict:
    row = _sample(
        text=text,
        label=label,
        attack_type="direct_prompt_injection" if label else None,
        source=source,
        source_split="train",
        source_id=source_id,
        group_id=group,
    )
    return _set_source_role(row, role)


def _source_output(root: Path, rows: list[dict]) -> dict:
    source = rows[0]["source"]
    path = root / "sources" / f"{source}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    path.write_bytes(data)
    return {
        "path": str(path.relative_to(root)),
        "rows": len(rows),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _boundary_row(
    row_id: str,
    pair_id: str,
    label: int,
    *,
    family: str = "direct_instruction_override",
    scenario_id: str = "scenario-1",
    split: str = "train",
    source_context: str = "direct_user",
) -> dict:
    return {
        "attack_family": family if label else "none",
        "category": "prompt_injection" if label else "benign_boundary",
        "expected_action": "block_or_review" if label else "allow",
        "id": row_id,
        "label": label,
        "language": "en",
        "pair_family": family,
        "pair_id": pair_id,
        "risk_domain": "finance",
        "scenario_id": scenario_id,
        "source_context": source_context,
        "source_type": "synthetic_curated",
        "split": split,
        "target_boundary": "instruction_integrity",
        "text": f"boundary example {row_id}",
    }
