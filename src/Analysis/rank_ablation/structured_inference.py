"""Generate structured JSON-like summaries for each LoRA rank checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference import generate_summary  # noqa: E402

from evaluate_ranks import load_rank_model  # noqa: E402
from utils.json_utils import (  # noqa: E402
    build_structured_prompt,
    rank_key,
    rank_value,
    save_json,
)
from utils.timing_utils import set_seed  # noqa: E402


DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Run structured-output inference by LoRA rank.")
    parser.add_argument("--raw-data-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/rank_ablation/structured_outputs",
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _num_samples(dataset, max_samples: int) -> int:
    if max_samples is None or max_samples < 0:
        return len(dataset)
    return min(int(max_samples), len(dataset))


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = load_from_disk((PROJECT_ROOT / args.raw_data_path).as_posix())[args.split]
    sample_count = _num_samples(dataset, args.max_samples)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank in tqdm(args.ranks, desc="Structured generation by rank"):
        key = rank_key(rank)
        model_dir = (PROJECT_ROOT / args.experiments_root / f"t5_small_lora_r{rank}").resolve()
        if not model_dir.exists():
            continue

        model, tokenizer, device = load_rank_model(
            model_dir=model_dir,
            default_base_model=args.default_base_model,
        )

        records = []
        for idx in tqdm(range(sample_count), desc=f"Structured {key}", leave=False):
            row = dataset[idx]
            dialogue = str(row["dialogue"])
            prompt = build_structured_prompt(dialogue)
            prediction = generate_summary(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=prompt,
                architecture=None,
            )
            records.append(
                {
                    "id": row.get("id", idx),
                    "reference": str(row.get("summary", "")),
                    "prediction": prediction,
                }
            )

        save_json(
            output_dir / f"{key}.json",
            {
                "rank": key,
                "mode": "structured",
                "num_samples": int(sample_count),
                "records": records,
            },
        )

    print(f"Saved structured outputs to: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
