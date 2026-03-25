"""Measure disk footprint of each LoRA rank checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from utils.json_utils import rank_key, rank_value, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Measure model directory size by rank.")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument(
        "--output-path",
        default="outputs/analysis/rank_ablation/model_size/size.json",
    )
    return parser.parse_args()


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return int(total)


def main():
    args = parse_args()
    payload = {}

    for rank in tqdm(args.ranks, desc="Model size by rank"):
        key = rank_key(rank)
        model_dir = (PROJECT_ROOT / args.experiments_root / f"t5_small_lora_r{rank}").resolve()
        if not model_dir.exists():
            payload[key] = {"status": "missing_model_dir"}
            continue

        size_bytes = directory_size_bytes(model_dir)
        payload[key] = {
            "size_bytes": int(size_bytes),
            "size_mb": float(size_bytes / (1024 ** 2)),
            "status": "ok",
        }

    ordered = dict(sorted(payload.items(), key=lambda item: rank_value(item[0])))
    output_path = (PROJECT_ROOT / args.output_path).resolve()
    save_json(output_path, ordered)
    print(f"Saved model size metrics: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
