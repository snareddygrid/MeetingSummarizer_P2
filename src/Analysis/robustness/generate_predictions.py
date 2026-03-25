"""Generate model predictions for original and adversarial robustness datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utils.evaluation_utils import ensure_dir, load_data_rows, load_summarization_model, resolve_path, save_json, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from inference import generate_summary  # noqa: E402


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip()
    if not prefix:
        return ""
    return prefix if prefix.endswith("_") else f"{prefix}_"


def load_model_from_existing_pipeline(model_key: str):
    from model_loader import load_selected_model

    return load_selected_model(model_key)


def resolve_model(args):
    if args.model_key:
        return load_model_from_existing_pipeline(args.model_key)
    return load_summarization_model(
        model_dir=resolve_path(args.model_dir),
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
    )


def generate_for_split(
    model,
    tokenizer,
    device,
    rows: List[Dict],
    architecture: str,
) -> List[Dict]:
    records: List[Dict] = []
    for row in tqdm(rows, desc="Predict"):
        dialogue = str(row.get("dialogue", ""))
        reference = str(row.get("summary", ""))
        prediction = generate_summary(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=dialogue,
            architecture=architecture,
        )
        records.append(
            {
                "id": row.get("id"),
                "input": dialogue,
                "reference": reference,
                "prediction": prediction,
                "perturbations": row.get("perturbations", []),
            }
        )
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for robustness datasets.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--model-key", default="")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    parser.add_argument("--original-data", default="data/robustness/original/data.json")
    parser.add_argument("--adversarial-data", default="data/robustness/adversarial/data.json")
    parser.add_argument("--output-dir", default="outputs/analysis/robustness/predictions")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--architecture", default="t5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    prefix = _normalize_prefix(args.output_prefix)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    model, tokenizer, device = resolve_model(args)
    model.eval()

    original_rows = load_data_rows(resolve_path(args.original_data))
    adversarial_rows = load_data_rows(resolve_path(args.adversarial_data))

    architecture = args.architecture
    original_predictions = generate_for_split(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rows=original_rows,
        architecture=architecture,
    )
    adversarial_predictions = generate_for_split(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rows=adversarial_rows,
        architecture=architecture,
    )

    original_out = output_dir / f"{prefix}original.json"
    adversarial_out = output_dir / f"{prefix}adversarial.json"

    save_json(
        original_out,
        {
            "metadata": {
                "model_dir": args.model_dir,
                "model_key": args.model_key,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_samples": len(original_predictions),
            },
            "records": original_predictions,
        },
    )
    save_json(
        adversarial_out,
        {
            "metadata": {
                "model_dir": args.model_dir,
                "model_key": args.model_key,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_samples": len(adversarial_predictions),
            },
            "records": adversarial_predictions,
        },
    )

    print(f"Saved predictions: {original_out.as_posix()}")
    print(f"Saved predictions: {adversarial_out.as_posix()}")


if __name__ == "__main__":
    main()
