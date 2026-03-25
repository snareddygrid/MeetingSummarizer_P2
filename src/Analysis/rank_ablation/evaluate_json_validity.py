"""Evaluate strict JSON validity rate for structured generation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from utils.json_utils import as_records, load_json, rank_key, rank_value, safe_json_loads, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute JSON validity rate for each rank output.")
    parser.add_argument(
        "--structured-dir",
        default="outputs/analysis/rank_ablation/structured_outputs",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument(
        "--output-path",
        default="outputs/analysis/rank_ablation/validity/validity.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    structured_dir = (PROJECT_ROOT / args.structured_dir).resolve()
    payload = {}

    for rank in tqdm(args.ranks, desc="JSON validity by rank"):
        key = rank_key(rank)
        path = structured_dir / f"{key}.json"
        if not path.exists():
            payload[key] = {"status": "missing_structured_file"}
            continue

        records = as_records(load_json(path))
        total = len(records)
        valid = 0
        for row in records:
            ok, _ = safe_json_loads(str(row.get("prediction", "")))
            if ok:
                valid += 1

        payload[key] = {
            "valid": int(valid),
            "total": int(total),
            "validity_rate": float(valid / total) if total else 0.0,
            "status": "ok",
        }

    ordered = dict(sorted(payload.items(), key=lambda item: rank_value(item[0])))
    output_path = (PROJECT_ROOT / args.output_path).resolve()
    save_json(output_path, ordered)
    print(f"Saved JSON validity metrics: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
