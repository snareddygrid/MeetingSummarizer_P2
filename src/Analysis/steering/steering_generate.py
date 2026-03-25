"""
End-to-end steering pipeline:
1) Ensure activations exist
2) Ensure direction exists
3) Generate steered summaries for configured scales
4) Evaluate scales
5) Analyze layer contributions
6) Save final report

Run:
    python src/analysis/steering/steering_generate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from analyze_layers import run_layer_analysis
from compute_direction import run_direction_computation
from evaluate_steering import run_evaluation
from extract_activations import run_extraction
from utils.activation_utils import (
    build_subset_samples,
    decode_labels,
    ensure_dir,
    get_device,
    load_lora_model_and_tokenizer,
    load_processed_split,
    load_subset_indices,
    load_json,
    save_json,
)
from utils.steering_utils import action_verb_count, generate_summary_with_steering


def _load_direction(direction_path: Path) -> Dict[int, torch.Tensor]:
    payload = torch.load(direction_path, map_location="cpu")
    direction_by_layer = payload.get("direction_by_layer", {})
    return {int(layer): tensor.float() for layer, tensor in direction_by_layer.items()}


def _ensure_prerequisites(args) -> None:
    activation_report = Path(args.output_root) / "reports" / "activation_extraction.json"
    direction_path = Path(args.output_root) / "directions" / "direction.pt"

    if not activation_report.exists() or args.force_reextract:
        extract_args = argparse.Namespace(
            model_dir=args.model_dir,
            dataset_path=args.dataset_path,
            split=args.split,
            output_root=args.output_root,
            subset_size=args.subset_size,
            subset_indices_path=args.subset_indices_path,
            decoder_layers=args.decoder_layers,
            device=args.device,
            default_base_model=args.default_base_model,
            base_model_fallback=args.base_model_fallback,
        )
        run_extraction(extract_args)

    if not direction_path.exists() or args.force_recompute_direction:
        direction_args = argparse.Namespace(
            output_root=args.output_root,
            label_path=args.label_path,
            normalize_direction=args.normalize_direction,
            min_summary_tokens=args.min_summary_tokens,
            min_class_samples=args.min_class_samples,
        )
        run_direction_computation(direction_args)


def _run_generation(args) -> Dict:
    output_root = Path(args.output_root)
    generated_dir = ensure_dir(output_root / "generated")
    reports_dir = ensure_dir(output_root / "reports")

    rows = load_processed_split(dataset_path=args.dataset_path, split=args.split)
    subset_indices = load_subset_indices(
        total_size=len(rows),
        subset_size=args.subset_size,
        subset_indices_path=Path(args.subset_indices_path) if args.subset_indices_path else None,
    )
    samples = build_subset_samples(rows=rows, indices=subset_indices)

    device = get_device(prefer_mps=True) if args.device == "auto" else torch.device(args.device)
    model, tokenizer, device = load_lora_model_and_tokenizer(
        model_dir=Path(args.model_dir),
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        device=device,
        merge_lora=False,
    )

    direction_by_layer = _load_direction(Path(args.output_root) / "directions" / "direction.pt")
    if not direction_by_layer:
        raise FileNotFoundError("Direction vectors missing. Run compute_direction.py first.")

    requested_layers = [int(layer) for layer in args.steering_layers]
    active_layers = [layer for layer in requested_layers if layer in direction_by_layer]
    requested_max = max(requested_layers) if requested_layers else 0
    direction_max = max(direction_by_layer.keys()) if direction_by_layer else 0
    if (
        not active_layers
        or (
            requested_max > direction_max
            and len(active_layers) < min(2, len(direction_by_layer))
        )
    ):
        active_layers = sorted(direction_by_layer.keys())

    generated_files = {}
    generated_records: Dict[str, List[Dict]] = {}
    for scale in args.scales:
        scale_value = float(scale)
        scale_key = str(scale_value)
        prompt_prefix = args.baseline_prompt_prefix if scale_value == 0.0 else args.action_prompt_prefix
        records: List[Dict] = []
        for sample in tqdm(samples, desc=f"Steering Generate scale={scale_key}"):
            prediction = generate_summary_with_steering(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                direction_by_layer=direction_by_layer,
                scale=scale_value,
                device=device,
                target_layers=active_layers,
                input_prefix=prompt_prefix,
                max_source_tokens=args.max_source_tokens,
            )
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "source_index": sample["source_index"],
                    "scale": scale_value,
                    "prediction": prediction,
                    "reference": decode_labels(tokenizer=tokenizer, labels=sample["labels"]),
                }
            )

        payload = {
            "scale": scale_value,
            "num_samples": len(records),
            "subset_indices": subset_indices,
            "records": records,
        }
        output_path = generated_dir / f"{scale_key}.json"
        save_json(output_path, payload)
        generated_files[scale_key] = output_path.as_posix()
        generated_records[scale_key] = records

    debug_rows = []
    baseline_key = "0.0" if "0.0" in generated_records else "0"
    compare_scales = [key for key in ["1.5", "3.0"] if key in generated_records]
    if baseline_key in generated_records and compare_scales:
        for idx in range(min(int(args.debug_samples), len(generated_records[baseline_key]))):
            baseline = generated_records[baseline_key][idx]
            row = {
                "sample_id": baseline["sample_id"],
                "source_index": baseline["source_index"],
                "baseline_summary": baseline["prediction"],
                "baseline_action_verbs": action_verb_count(baseline["prediction"]),
            }
            for scale_key in compare_scales:
                pred = generated_records[scale_key][idx]["prediction"]
                row[f"scale_{scale_key}_summary"] = pred
                row[f"scale_{scale_key}_action_verbs"] = action_verb_count(pred)
            debug_rows.append(row)

    for row in debug_rows:
        print(
            f"[debug] {row['sample_id']} "
            f"baseline_verbs={row['baseline_action_verbs']} "
            f"scale_1.5_verbs={row.get('scale_1.5_action_verbs', 'n/a')} "
            f"scale_3.0_verbs={row.get('scale_3.0_action_verbs', 'n/a')}"
        )
        print(f"[debug] baseline: {row['baseline_summary']}")
        if "scale_1.5_summary" in row:
            print(f"[debug] scale 1.5: {row['scale_1.5_summary']}")
        if "scale_3.0_summary" in row:
            print(f"[debug] scale 3.0: {row['scale_3.0_summary']}")

    generation_summary = {
        "generated_files": generated_files,
        "subset_size": len(samples),
        "subset_indices": subset_indices,
        "scales": [float(scale) for scale in args.scales],
        "active_steering_layers": active_layers,
        "baseline_prompt_prefix": args.baseline_prompt_prefix,
        "action_prompt_prefix": args.action_prompt_prefix,
        "debug_samples": debug_rows,
    }
    save_json(reports_dir / "generation_summary.json", generation_summary)
    save_json(reports_dir / "debug_comparisons.json", {"debug_samples": debug_rows})
    return generation_summary


def run_pipeline(args) -> Dict:
    _ensure_prerequisites(args)
    generation_summary = _run_generation(args)

    evaluation = run_evaluation(
        argparse.Namespace(
            output_root=args.output_root,
            eval_samples=args.eval_samples,
            rouge_drop_limit_pct=args.rouge_drop_limit_pct,
            manual_ratings_path=args.manual_ratings_path,
        )
    )
    layer_analysis = run_layer_analysis(
        argparse.Namespace(
            model_dir=args.model_dir,
            dataset_path=args.dataset_path,
            split=args.split,
            output_root=args.output_root,
            direction_path=(Path(args.output_root) / "directions" / "direction.pt").as_posix(),
            subset_size=args.subset_size,
            eval_samples=args.eval_samples,
            subset_indices_path=args.subset_indices_path,
            layer_scale=args.layer_scale,
            rouge_drop_limit_pct=args.rouge_drop_limit_pct,
            top_k_layers=args.top_k_layers,
            max_source_tokens=args.max_source_tokens,
            baseline_prompt_prefix=args.baseline_prompt_prefix,
            action_prompt_prefix=args.action_prompt_prefix,
            device=args.device,
            default_base_model=args.default_base_model,
            base_model_fallback=args.base_model_fallback,
        )
    )

    final_report = {
        "best_scale": evaluation["best_scale"],
        "rouge_drop": float(evaluation["best_rouge_drop_pct"]),
        "action_score": float(evaluation["best_action_score"]),
        "best_layers": layer_analysis["best_layers"],
        "evaluation": evaluation,
        "layer_analysis": {
            "best_layers": layer_analysis["best_layers"],
            "layer_scale": layer_analysis["layer_scale"],
            "per_layer": layer_analysis["per_layer"],
        },
        "generation_summary": generation_summary,
    }
    save_json(Path(args.output_root) / "reports" / "final_report.json", final_report)
    return final_report


def parse_args():
    parser = argparse.ArgumentParser(description="Generate steered summaries and final Task-03 report.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/steering")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument(
        "--subset-indices-path",
        default="outputs/analysis/quantization/reports/fixed_test_subset.json",
    )
    parser.add_argument("--decoder-layers", nargs="+", type=int, default=[6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--steering-layers", nargs="+", type=int, default=[6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--rouge-drop-limit-pct", type=float, default=2.0)
    parser.add_argument("--layer-scale", type=float, default=1.0)
    parser.add_argument("--top-k-layers", type=int, default=3)
    parser.set_defaults(normalize_direction=True)
    parser.add_argument("--normalize-direction", dest="normalize_direction", action="store_true")
    parser.add_argument("--no-normalize-direction", dest="normalize_direction", action="store_false")
    parser.add_argument("--min-summary-tokens", type=int, default=6)
    parser.add_argument("--min-class-samples", type=int, default=12)
    parser.add_argument("--label-path", default=None)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--baseline-prompt-prefix", default="Summarize:\n")
    parser.add_argument("--action-prompt-prefix", default="Summarize with focus on action items:\n")
    parser.add_argument("--debug-samples", type=int, default=3)
    parser.add_argument(
        "--manual-ratings-path",
        default="outputs/analysis/steering/evaluations/manual_ratings.json",
    )
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--force-recompute-direction", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_pipeline(args)
    print("Steering pipeline complete.")
    print(f"Best scale: {report['best_scale']}")
    print(f"Best layers: {report['best_layers']}")
    print(f"Final report: {(Path(args.output_root) / 'reports' / 'final_report.json').as_posix()}")


if __name__ == "__main__":
    main()
