"""Repair structured output files into strict JSON-schema-valid predictions.

This is a no-retraining post-processing step for fast JSON-validity recovery.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from utils.json_utils import as_records, load_json, rank_key, rank_value, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RANKS = [2, 4, 8, 16, 32]

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ACTION_RE = re.compile(r"\b(will|should|need to|must|plan to|going to|todo|follow up|send|call|buy|meet)\b", re.I)
_DECISION_RE = re.compile(r"\b(decide|decided|agreed|approved|finalized|confirmed|chose|selected)\b", re.I)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "it", "this", "that", "we", "they", "he", "she", "you", "i", "as", "at", "be", "by", "from", "about",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Repair structured outputs to valid JSON schema.")
    parser.add_argument(
        "--input-dir",
        default="outputs/analysis/rank_ablation/structured_outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/rank_ablation/structured_outputs_repaired",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/analysis/rank_ablation/validity/repair_report.json",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    return parser.parse_args()


def _strip_code_fence(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_brace_span(text: str) -> str:
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return text
    return text[left : right + 1]


def _parse_candidate_json(text: str) -> Any:
    candidate = _strip_code_fence(text)
    candidates = [
        candidate,
        _extract_brace_span(candidate),
        _extract_brace_span(candidate).replace("'", '"'),
    ]
    for item in candidates:
        try:
            return json.loads(item)
        except Exception:  # noqa: BLE001
            pass
    try:
        parsed = ast.literal_eval(_extract_brace_span(candidate))
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass
    return None


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "\n" in text:
            items = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
            return [item for item in items if item]
        if ";" in text:
            items = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
            return items
        return [text]
    return [str(value).strip()]


def _summarize_topic_from_text(text: str) -> List[str]:
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text)]
    keywords = []
    seen = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= 3:
            break
    if keywords:
        return keywords
    compact = " ".join(text.split())
    return [compact[:80]] if compact else []


def _heuristic_from_text(text: str) -> Dict[str, List[str]]:
    cleaned = " ".join(str(text or "").strip().split())
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(cleaned) if s.strip()]
    if not sentences and cleaned:
        sentences = [cleaned]

    topics = _summarize_topic_from_text(cleaned)
    action_items = [sent for sent in sentences if _ACTION_RE.search(sent)]
    decisions = [sent for sent in sentences if _DECISION_RE.search(sent)]

    # If no explicit signals found, place main summary sentence under topics.
    if not action_items and sentences:
        action_items = []
    if not decisions and any(" will " in f" {sent.lower()} " for sent in sentences):
        decisions = [sent for sent in sentences if " will " in f" {sent.lower()} "][:1]

    return {
        "topics": topics,
        "action_items": action_items,
        "decisions": decisions,
    }


def _normalize_schema(obj: Any, raw_text: str) -> Dict[str, List[str]]:
    if isinstance(obj, dict):
        topics = _to_list(obj.get("topics"))
        action_items = _to_list(obj.get("action_items"))
        decisions = _to_list(obj.get("decisions"))
        if topics or action_items or decisions:
            return {
                "topics": topics,
                "action_items": action_items,
                "decisions": decisions,
            }
    return _heuristic_from_text(raw_text)


def _strict_json_valid(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for rank in tqdm(args.ranks, desc="Repairing structured outputs"):
        key = rank_key(rank)
        src = input_dir / f"{key}.json"
        if not src.exists():
            report[key] = {"status": "missing_input_file"}
            continue

        payload = load_json(src)
        records = as_records(payload)

        repaired_records = []
        before_valid = 0
        after_valid = 0
        repaired_count = 0

        for row in records:
            raw_prediction = str(row.get("prediction", ""))
            was_valid_before = _strict_json_valid(raw_prediction)
            if was_valid_before:
                before_valid += 1

            parsed = _parse_candidate_json(raw_prediction)
            normalized = _normalize_schema(parsed, raw_prediction)
            repaired_prediction = json.dumps(normalized, ensure_ascii=False)
            if not was_valid_before:
                repaired_count += 1
            if _strict_json_valid(repaired_prediction):
                after_valid += 1

            repaired_row = dict(row)
            repaired_row["raw_prediction"] = raw_prediction
            repaired_row["prediction"] = repaired_prediction
            repaired_row["was_valid_before"] = bool(was_valid_before)
            repaired_records.append(repaired_row)

        output_payload = dict(payload)
        output_payload["mode"] = "structured_repaired"
        output_payload["records"] = repaired_records
        output_payload["repair_stats"] = {
            "total": int(len(records)),
            "valid_before": int(before_valid),
            "valid_after": int(after_valid),
            "repaired_count": int(repaired_count),
            "validity_before": float(before_valid / len(records)) if records else 0.0,
            "validity_after": float(after_valid / len(records)) if records else 0.0,
        }
        save_json(output_dir / f"{key}.json", output_payload)

        report[key] = {
            "total": int(len(records)),
            "valid_before": int(before_valid),
            "valid_after": int(after_valid),
            "repaired_count": int(repaired_count),
            "validity_before": float(before_valid / len(records)) if records else 0.0,
            "validity_after": float(after_valid / len(records)) if records else 0.0,
            "status": "ok",
        }

    ordered = dict(sorted(report.items(), key=lambda item: rank_value(item[0])))
    report_path = (PROJECT_ROOT / args.report_path).resolve()
    save_json(report_path, ordered)
    print(f"Saved repair report: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
