"""
Create a manual rating template for steering outputs.

Run:
    python src/analysis/steering/build_manual_rating_sheet.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from utils.activation_utils import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build manual action-clarity rating template.")
    parser.add_argument("--output-root", default="outputs/analysis/steering")
    parser.add_argument("--scales", nargs="+", default=["0.0", "0.5", "1.0", "1.5"])
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument(
        "--output-path",
        default="outputs/analysis/steering/evaluations/manual_ratings_template.json",
    )
    return parser.parse_args()


def _normalize_scale(scale_value: str) -> str:
    try:
        return str(float(scale_value))
    except Exception:  # noqa: BLE001
        return str(scale_value)


def _build_rows(records: List[Dict], sample_limit: int) -> List[Dict]:
    rows = records[: max(0, int(sample_limit))]
    template_rows = []
    for row in rows:
        template_rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "source_index": row.get("source_index"),
                "prediction": str(row.get("prediction", "")),
                "manual_action_score": None,
            }
        )
    return template_rows


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    generated_dir = output_root / "generated"

    payload: Dict[str, List[Dict]] = {}
    for raw_scale in args.scales:
        scale = _normalize_scale(raw_scale)
        source_path = generated_dir / f"{scale}.json"
        if not source_path.exists():
            continue
        source_payload = load_json(source_path)
        records = list(source_payload.get("records", []))
        payload[scale] = _build_rows(records=records, sample_limit=args.sample_limit)

    result = {
        "instructions": (
            "Fill manual_action_score on a 1-5 scale for action-item clarity, "
            "then save as manual_ratings.json and pass --manual-ratings-path to evaluate_steering.py."
        ),
        "scales": payload,
    }
    save_json(Path(args.output_path), result)
    print(f"Saved template: {Path(args.output_path).as_posix()}")


if __name__ == "__main__":
    main()
