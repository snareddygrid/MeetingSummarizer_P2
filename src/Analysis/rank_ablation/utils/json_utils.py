"""JSON and path helpers for LoRA rank ablation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def ensure_dir(path_like: Path | str) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path_like: Path | str, payload: Dict[str, Any]) -> None:
    path = Path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path_like: Path | str) -> Dict[str, Any]:
    path = Path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_json_loads(text: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except Exception:  # noqa: BLE001
        return False, None


def build_structured_prompt(dialogue: str) -> str:
    return (
        "Summarize the conversation in JSON format:\n"
        "{\n"
        '  "topics": [],\n'
        '  "action_items": [],\n'
        '  "decisions": []\n'
        "}\n\n"
        f"Conversation: {dialogue}"
    )


def rank_key(rank: int | str) -> str:
    rank_value = int(str(rank).replace("r", ""))
    return f"r{rank_value}"


def rank_value(rank: int | str) -> int:
    return int(str(rank).replace("r", ""))


def as_records(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return list(payload)
    return []


def iter_rank_keys(payload: Dict[str, Any]) -> Iterable[str]:
    return sorted(payload.keys(), key=rank_value)
