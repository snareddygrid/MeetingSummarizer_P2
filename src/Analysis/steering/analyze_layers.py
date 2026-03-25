"""
Layer contribution analysis for steering.

Run:
    python src/analysis/steering/analyze_layers.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utils.activation_utils import (
    build_subset_samples,
    decode_labels,
    ensure_dir,
    get_device,
    load_json,
    load_lora_model_and_tokenizer,
    load_processed_split,
    load_subset_indices,
    resolve_decoder_layers,
    save_json,
)
from utils.steering_utils import evaluate_records, generate_summary_with_steering, rouge_drop_percent


def _load_direction(direction_path: Path) -> Dict[int, torch.Tensor]:
    import torch

    payload = torch.load(direction_path, map_location="cpu")
    direction_by_layer = payload.get("direction_by_layer", {})
    return {int(layer): tensor.float() for layer, tensor in direction_by_layer.items()}


def _build_records_for_layer(
    model,
    tokenizer,
    device,
    samples: List[Dict],
    direction_by_layer: Dict[int, torch.Tensor],
    scale: float,
    target_layers: List[int] | None,
    prompt_prefix: str,
    max_source_tokens: int,
) -> List[Dict]:
    records = []
    for sample in tqdm(samples, desc=f"Layer Analysis scale={scale} layers={target_layers}"):
        summary = generate_summary_with_steering(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            direction_by_layer=direction_by_layer,
            scale=scale,
            device=device,
            target_layers=target_layers,
            input_prefix=prompt_prefix,
            max_source_tokens=max_source_tokens,
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "source_index": sample["source_index"],
                "prediction": summary,
                "reference": decode_labels(tokenizer=tokenizer, labels=sample["labels"]),
            }
        )
    return records


def run_layer_analysis(args) -> Dict:
    import torch

    output_root = Path(args.output_root)
    reports_dir = ensure_dir(output_root / "reports")
    generated_dir = ensure_dir(output_root / "generated")

    rows = load_processed_split(dataset_path=args.dataset_path, split=args.split)
    subset_indices = load_subset_indices(
        total_size=len(rows),
        subset_size=args.subset_size,
        subset_indices_path=Path(args.subset_indices_path) if args.subset_indices_path else None,
    )
    samples_all = build_subset_samples(rows=rows, indices=subset_indices)
    samples = samples_all[: int(args.eval_samples)]

    device = get_device(prefer_mps=True) if args.device == "auto" else torch.device(args.device)
    model, tokenizer, device = load_lora_model_and_tokenizer(
        model_dir=Path(args.model_dir),
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        device=device,
        merge_lora=False,
    )

    direction_by_layer = _load_direction(Path(args.direction_path))
    if not direction_by_layer:
        raise FileNotFoundError(f"Direction vectors missing at {args.direction_path}")

    resolved_layers, layer_info = resolve_decoder_layers(model=model, requested_layers=sorted(direction_by_layer.keys()))
    direction_by_layer = {layer: direction_by_layer[layer] for layer in resolved_layers if layer in direction_by_layer}

    baseline_path = generated_dir / "0.json"
    if not baseline_path.exists():
        baseline_path = generated_dir / "0.0.json"
    if baseline_path.exists():
        baseline_records = load_json(baseline_path).get("records", [])[: len(samples)]
    else:
        baseline_records = _build_records_for_layer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            samples=samples,
            direction_by_layer=direction_by_layer,
            scale=0.0,
            target_layers=None,
            prompt_prefix=args.baseline_prompt_prefix,
            max_source_tokens=args.max_source_tokens,
        )
    baseline_metrics = evaluate_records(records=baseline_records, sample_limit=len(samples))

    per_layer = []
    for layer in resolved_layers:
        layer_records = _build_records_for_layer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            samples=samples,
            direction_by_layer=direction_by_layer,
            scale=float(args.layer_scale),
            target_layers=[layer],
            prompt_prefix=args.action_prompt_prefix,
            max_source_tokens=args.max_source_tokens,
        )
        metrics = evaluate_records(records=layer_records, sample_limit=len(samples))
        rouge_drop = rouge_drop_percent(
            candidate_rouge_l=float(metrics["rougeL"]),
            baseline_rouge_l=float(baseline_metrics["rougeL"]),
        )
        action_gain = float(metrics["action_score"] - baseline_metrics["action_score"])
        per_layer.append(
            {
                "layer": int(layer),
                "scale": float(args.layer_scale),
                "rougeL": float(metrics["rougeL"]),
                "action_score": float(metrics["action_score"]),
                "rouge_drop_pct": float(rouge_drop),
                "action_gain": action_gain,
                "eligible": bool(rouge_drop < float(args.rouge_drop_limit_pct)),
            }
        )

    eligible = [row for row in per_layer if row["eligible"]]
    ranked = sorted(eligible if eligible else per_layer, key=lambda row: row["action_gain"], reverse=True)
    best_layers = [row["layer"] for row in ranked[: max(1, int(args.top_k_layers))]]

    payload = {
        "eval_samples": int(len(samples)),
        "requested_layers": sorted(direction_by_layer.keys()),
        "resolved_layers": resolved_layers,
        "layer_resolution": layer_info,
        "layer_scale": float(args.layer_scale),
        "rouge_drop_limit_pct": float(args.rouge_drop_limit_pct),
        "baseline": baseline_metrics,
        "per_layer": per_layer,
        "best_layers": best_layers,
    }
    save_json(reports_dir / "layer_analysis.json", payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze layer-wise steering contribution.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/steering")
    parser.add_argument("--direction-path", default="outputs/analysis/steering/directions/direction.pt")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument(
        "--subset-indices-path",
        default="outputs/analysis/quantization/reports/fixed_test_subset.json",
    )
    parser.add_argument("--layer-scale", type=float, default=1.0)
    parser.add_argument("--rouge-drop-limit-pct", type=float, default=2.0)
    parser.add_argument("--top-k-layers", type=int, default=3)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--baseline-prompt-prefix", default="Summarize:\n")
    parser.add_argument("--action-prompt-prefix", default="Summarize with focus on action items:\n")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run_layer_analysis(args)
    print("Layer analysis complete.")
    print(f"Best layers: {payload['best_layers']}")


if __name__ == "__main__":
    main()
