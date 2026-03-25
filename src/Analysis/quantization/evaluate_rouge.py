"""
ROUGE evaluation for quantized/baseline batch outputs.

Run:
    python src/analysis/quantization/evaluate_rouge.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

from batch_inference import resolve_model_specs, run_batch_inference
from common import QUANTIZATION_LEVELS, attach_subset_metadata, compute_rouge_scores, ensure_output_structure, load_json, save_json


def _normalize_text(text: str) -> str:
    normalized = " ".join(str(text).strip().split())
    return normalized


def evaluate_from_batch_outputs(
    batch_dir: Path,
    rouge_output_dir: Path,
    model_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Dict]:
    batch_dir = Path(batch_dir)
    rouge_output_dir = Path(rouge_output_dir)
    rouge_output_dir.mkdir(parents=True, exist_ok=True)

    labels_filter = set(model_labels) if model_labels else None
    results: Dict[str, Dict] = {}

    for predictions_file in sorted(batch_dir.glob("*_predictions.json")):
        label = predictions_file.stem.replace("_predictions", "")
        if labels_filter is not None and label not in labels_filter:
            continue

        payload = load_json(predictions_file)
        records = payload.get("records", [])
        if not records:
            continue
        subset_metadata = {
            "dataset_size": int(payload.get("dataset_size", 100)),
            "subset_indices": payload.get("subset_indices", []),
            "split": payload.get("subset_split", "test"),
            "selection_mode": payload.get("subset_selection_mode", "first_n"),
        }

        predictions = []
        references = []
        empty_prediction_count = 0
        empty_reference_count = 0
        for row in records:
            prediction = _normalize_text(row.get("prediction", ""))
            reference = _normalize_text(row.get("reference", ""))
            if not prediction:
                empty_prediction_count += 1
                prediction = "<empty_prediction>"
            if not reference:
                empty_reference_count += 1
                reference = "<empty_reference>"
            predictions.append(prediction)
            references.append(reference)

        if empty_prediction_count > 0:
            print(
                f"[ROUGE warning] {label}: empty_predictions={empty_prediction_count}/{len(records)}"
            )
        if empty_reference_count > 0:
            print(
                f"[ROUGE warning] {label}: empty_references={empty_reference_count}/{len(records)}"
            )

        rouge_scores = compute_rouge_scores(predictions=predictions, references=references)

        result_payload = {
            "model_label": label,
            "num_samples": len(records),
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "empty_prediction_count": empty_prediction_count,
            "empty_reference_count": empty_reference_count,
        }
        result_payload = attach_subset_metadata(result_payload, subset_metadata, num_samples=len(records))
        save_json(rouge_output_dir / f"{label}.json", result_payload)
        results[label] = result_payload

    return results


def evaluate_rouge(
    output_root: Path,
    model_dir: Path,
    dataset_path: str,
    processed_dataset_path: str,
    raw_dataset_path: str,
    split: str,
    num_samples: int,
    subset_size: int,
    quant_levels: Sequence[str],
    include_base: bool,
    force_regenerate_batch: bool,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
) -> Dict[str, Dict]:
    output_dirs = ensure_output_structure(output_root)
    model_specs = resolve_model_specs(
        model_dir=model_dir,
        models_output_dir=output_dirs["models"],
        quant_levels=quant_levels,
        include_base=include_base,
    )
    if not model_specs:
        raise RuntimeError("No model specs available for ROUGE evaluation.")

    existing_predictions = list(output_dirs["batch"].glob("*_predictions.json"))
    if force_regenerate_batch or not existing_predictions:
        run_batch_inference(
            model_specs=model_specs,
            dataset_path=dataset_path,
            split=split,
            output_root=output_root,
            num_samples=num_samples,
            default_base_model=default_base_model,
            fallback_local_base_model=fallback_local_base_model,
            processed_dataset_path=processed_dataset_path,
            raw_dataset_path=raw_dataset_path,
            subset_size=subset_size,
            subset_indices_path=output_dirs["reports"] / "fixed_test_subset.json",
        )

    label_list = list(model_specs.keys())
    return evaluate_from_batch_outputs(
        batch_dir=output_dirs["batch"],
        rouge_output_dir=output_dirs["rouge"],
        model_labels=label_list,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ROUGE metrics for quantized/baseline summaries.")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/raw")
    parser.add_argument("--processed-dataset-path", default="data/processed")
    parser.add_argument("--raw-dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--force-regenerate-batch", action="store_true")
    parser.add_argument("--quant-levels", nargs="+", default=list(QUANTIZATION_LEVELS.keys()))
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def main():
    args = parse_args()
    results = evaluate_rouge(
        output_root=Path(args.output_root),
        model_dir=Path(args.model_dir),
        dataset_path=args.dataset_path,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        split=args.split,
        num_samples=args.num_samples,
        subset_size=args.subset_size,
        quant_levels=args.quant_levels,
        include_base=args.include_base,
        force_regenerate_batch=args.force_regenerate_batch,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
    )
    rouge_dir = Path(args.output_root) / "rouge"
    first_payload = next(iter(results.values()), {})
    summary_payload = {
        "num_samples": int(first_payload.get("num_samples", 0)),
        "dataset_size": int(first_payload.get("dataset_size", args.subset_size)),
        "subset_indices": first_payload.get("subset_indices", []),
        "models": results,
    }
    save_json(rouge_dir / "rouge_summary.json", summary_payload)
    save_json(rouge_dir / "summary.json", summary_payload)
    print("ROUGE evaluation complete.")
    print(f"Models evaluated: {list(results.keys())}")


if __name__ == "__main__":
    main()
