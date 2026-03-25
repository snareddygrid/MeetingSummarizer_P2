"""Create adversarial meeting transcript dataset for robustness testing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utils.evaluation_utils import ensure_dir, resolve_path, save_json, set_seed
from utils.noise_utils import create_adversarial_dialogue


def _load_rows_from_raw(raw_path: Path) -> List[Dict]:
    from datasets import load_from_disk

    dataset = load_from_disk(raw_path.as_posix())
    split = dataset["test"]
    rows: List[Dict] = []
    for idx, row in enumerate(split):
        rows.append(
            {
                "id": str(row.get("id", f"test_{idx:04d}")),
                "dialogue": str(row.get("dialogue", "")).strip(),
                "summary": str(row.get("summary", "")).strip(),
            }
        )
    return rows


def _load_rows_from_processed(processed_path: Path, model_name: str = "t5-small") -> List[Dict]:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dataset = load_from_disk(processed_path.as_posix())
    split = dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    rows: List[Dict] = []
    for idx, row in enumerate(split):
        input_ids = [int(x) for x in row["input_ids"]]
        labels = [int(x) for x in row["labels"] if int(x) >= 0]
        dialogue = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
        dialogue = dialogue.replace("Summarize the following conversation:\n", "", 1)
        summary = tokenizer.decode(labels, skip_special_tokens=True).strip()
        rows.append(
            {
                "id": f"processed_{idx:04d}",
                "dialogue": dialogue,
                "summary": summary,
            }
        )
    return rows


def load_source_rows(raw_path: Path, processed_path: Path) -> List[Dict]:
    if raw_path.exists():
        return _load_rows_from_raw(raw_path=raw_path)
    if processed_path.exists():
        return _load_rows_from_processed(processed_path=processed_path)
    raise FileNotFoundError(
        f"Neither raw nor processed dataset found. raw={raw_path.as_posix()} processed={processed_path.as_posix()}"
    )


def build_adversarial_dataset(source_rows: List[Dict], sample_size: int, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    sample_size = min(int(sample_size), len(source_rows))
    chosen_indices = sorted(rng.sample(range(len(source_rows)), sample_size))

    original_rows: List[Dict] = []
    adversarial_rows: List[Dict] = []

    for idx in tqdm(chosen_indices, desc="Create Adversarial"):
        row = source_rows[idx]
        sample_id = str(row.get("id") or f"sample_{idx:04d}")
        dialogue = str(row.get("dialogue", "")).strip()
        summary = str(row.get("summary", "")).strip()

        adversarial_dialogue, perturbations = create_adversarial_dialogue(dialogue=dialogue, rng=rng)

        original_rows.append(
            {
                "id": sample_id,
                "dialogue": dialogue,
                "summary": summary,
            }
        )
        adversarial_rows.append(
            {
                "id": sample_id,
                "dialogue": adversarial_dialogue,
                "summary": summary,
                "perturbations": perturbations,
            }
        )

    return {"original": original_rows, "adversarial": adversarial_rows}


def parse_args():
    parser = argparse.ArgumentParser(description="Create adversarial robustness dataset.")
    parser.add_argument("--raw-path", default="data/raw")
    parser.add_argument("--processed-path", default="data/processed")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--original-out", default="data/robustness/original/data.json")
    parser.add_argument("--adversarial-out", default="data/robustness/adversarial/data.json")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    raw_path = resolve_path(args.raw_path)
    processed_path = resolve_path(args.processed_path)

    source_rows = load_source_rows(raw_path=raw_path, processed_path=processed_path)
    payload = build_adversarial_dataset(source_rows=source_rows, sample_size=args.sample_size, seed=args.seed)

    original_out = resolve_path(args.original_out)
    adversarial_out = resolve_path(args.adversarial_out)
    ensure_dir(original_out.parent)
    ensure_dir(adversarial_out.parent)

    save_json(
        original_out,
        {
            "metadata": {
                "seed": int(args.seed),
                "sample_size": int(len(payload["original"])),
                "source": "raw" if raw_path.exists() else "processed",
            },
            "samples": payload["original"],
        },
    )
    save_json(
        adversarial_out,
        {
            "metadata": {
                "seed": int(args.seed),
                "sample_size": int(len(payload["adversarial"])),
                "source": "raw" if raw_path.exists() else "processed",
                "perturbations": ["overlap", "noise", "off_topic", "length"],
            },
            "samples": payload["adversarial"],
        },
    )

    print(f"Saved original dataset: {original_out.as_posix()}")
    print(f"Saved adversarial dataset: {adversarial_out.as_posix()}")
    print(f"Samples: {len(payload['original'])}")


if __name__ == "__main__":
    main()
