"""
Batch inference benchmarking for quantized T5-small LoRA models.

Run:
    python src/analysis/quantization/batch_inference.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from tqdm import tqdm

from common import (
    QUANTIZATION_LEVELS,
    attach_subset_metadata,
    collect_subset_samples,
    compute_latency_stats,
    ensure_output_structure,
    generate_summary,
    generate_summary_llama_cpp,
    get_fixed_test_subset,
    get_device,
    load_lora_model_and_tokenizer,
    save_json,
)
from quantize_model import load_quantized_model_and_tokenizer


def resolve_model_specs(
    model_dir: Path,
    models_output_dir: Path,
    quant_levels: Sequence[str],
    include_base: bool = True,
) -> Dict[str, Dict]:
    specs: Dict[str, Dict] = {}
    if include_base:
        specs["BASE_LORA"] = {
            "type": "lora",
            "model_dir": Path(model_dir).as_posix(),
        }

    for level in quant_levels:
        if level not in QUANTIZATION_LEVELS:
            raise ValueError(f"Unknown quantization level: {level}")
        artifact_gguf = models_output_dir / f"{level}.gguf"
        artifact_pt = models_output_dir / f"{level}.pt"
        if artifact_gguf.exists():
            specs[level] = {
                "type": "llama_cpp_gguf",
                "artifact_path": artifact_gguf.as_posix(),
            }
        elif artifact_pt.exists():
            specs[level] = {
                "type": "quantized_legacy",
                "artifact_path": artifact_pt.as_posix(),
            }
    return specs


def load_model_for_spec(
    label: str,
    spec: Dict,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
    device_preference: str = "auto",
):
    if spec["type"] == "llama_cpp_gguf":
        return {"model_path": spec["artifact_path"]}, None, "cpu"

    if device_preference == "cpu":
        device = get_device(prefer_mps=False)
    elif device_preference == "mps":
        device = get_device(prefer_mps=True)
    else:
        device = get_device(prefer_mps=True)

    if spec["type"] == "quantized_legacy":
        model, tokenizer, device, _ = load_quantized_model_and_tokenizer(
            artifact_path=Path(spec["artifact_path"]),
            device=device,
        )
        return model, tokenizer, device

    model, tokenizer, device, _ = load_lora_model_and_tokenizer(
        model_dir=Path(spec["model_dir"]),
        default_base_model=default_base_model,
        fallback_local_base_model=fallback_local_base_model,
        device=device,
        merge_lora=True,
    )
    return model, tokenizer, device


def run_batch_inference(
    model_specs: Dict[str, Dict],
    output_root: Path,
    dataset_path: str = "data/raw",
    split: str = "test",
    num_samples: int = 100,
    target_utterances: Optional[int] = None,
    generation_config: Optional[Dict] = None,
    max_input_length: int = 1024,
    device_preference: str = "auto",
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
    processed_dataset_path: str = "data/processed",
    raw_dataset_path: str = "data/raw",
    subset_size: int = 100,
    subset_rows: Optional[Sequence[Dict]] = None,
    subset_metadata: Optional[Dict] = None,
    subset_indices_path: Optional[Path] = None,
    llama_completion_binary: Optional[str] = None,
    llama_device: str = "BLAS",
    llama_n_gpu_layers: int = 0,
) -> Dict[str, Dict]:
    output_dirs = ensure_output_structure(output_root)
    if subset_rows is None or subset_metadata is None:
        subset_rows, subset_metadata = get_fixed_test_subset(
            n=subset_size,
            processed_dataset_path=processed_dataset_path,
            raw_dataset_path=raw_dataset_path if raw_dataset_path else dataset_path,
            split=split,
            subset_indices_path=subset_indices_path,
        )
    samples = collect_subset_samples(
        subset_rows=subset_rows,
        num_samples=min(num_samples, len(subset_rows)),
        target_utterances=target_utterances,
    )

    results = {}
    for label, spec in model_specs.items():
        model, tokenizer, device = load_model_for_spec(
            label=label,
            spec=spec,
            default_base_model=default_base_model,
            fallback_local_base_model=fallback_local_base_model,
            device_preference=device_preference,
        )

        latencies = []
        records = []
        for sample in tqdm(samples, desc=f"Batch Inference [{label}]"):
            if spec["type"] == "llama_cpp_gguf":
                summary, latency, _, _ = generate_summary_llama_cpp(
                    model_path=Path(spec["artifact_path"]),
                    dialogue_text=sample["dialogue"],
                    generation_config=generation_config,
                    max_input_length=max_input_length,
                    llama_completion_binary=llama_completion_binary,
                    llama_device=llama_device,
                    llama_n_gpu_layers=llama_n_gpu_layers,
                    measure_peak_memory=False,
                )
            else:
                summary, latency = generate_summary(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dialogue_text=sample["dialogue"],
                    generation_config=generation_config,
                    max_input_length=max_input_length,
                )
            latencies.append(latency)
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "reference": sample["summary"],
                    "prediction": summary,
                    "latency_sec": float(latency),
                    "target_utterances": target_utterances,
                }
            )

        latency_stats = compute_latency_stats(latencies)
        throughput = 0.0
        if latency_stats["total_sec"] > 0:
            throughput = float(len(records) / latency_stats["total_sec"])

        metrics_payload = {
            "model_label": label,
            "num_samples": len(records),
            "target_utterances": target_utterances,
            "latency": latency_stats,
            "throughput_samples_per_sec": throughput,
            "device": str(device),
        }
        metrics_payload = attach_subset_metadata(metrics_payload, subset_metadata, num_samples=len(records))

        predictions_path = output_dirs["batch"] / f"{label}_predictions.json"
        metrics_path = output_dirs["batch"] / f"{label}_metrics.json"
        prediction_payload = attach_subset_metadata(
            {
                "model_label": label,
                "records": records,
            },
            subset_metadata,
            num_samples=len(records),
        )
        save_json(predictions_path, prediction_payload)
        save_json(metrics_path, metrics_payload)

        results[label] = {
            "predictions_path": predictions_path.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "metrics": metrics_payload,
        }

        if spec["type"] != "llama_cpp_gguf":
            del model

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference for LoRA + quantized models.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/raw")
    parser.add_argument("--processed-dataset-path", default="data/processed")
    parser.add_argument("--raw-dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--target-utterances", type=int, default=None)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--quant-levels", nargs="+", default=list(QUANTIZATION_LEVELS.keys()))
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    parser.add_argument("--llama-completion-binary", default=None)
    parser.add_argument("--llama-device", default="BLAS")
    parser.add_argument("--llama-n-gpu-layers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dirs = ensure_output_structure(Path(args.output_root))
    model_specs = resolve_model_specs(
        model_dir=Path(args.model_dir),
        models_output_dir=output_dirs["models"],
        quant_levels=args.quant_levels,
        include_base=args.include_base,
    )
    if not model_specs:
        raise RuntimeError(
            "No models found for batch inference. "
            "Run quantization first or pass --include-base to evaluate BASE_LORA."
        )

    results = run_batch_inference(
        model_specs=model_specs,
        dataset_path=args.dataset_path,
        split=args.split,
        output_root=Path(args.output_root),
        num_samples=args.num_samples,
        target_utterances=args.target_utterances,
        max_input_length=args.max_input_length,
        device_preference=args.device,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        subset_size=args.subset_size,
        subset_indices_path=Path(args.output_root) / "reports" / "fixed_test_subset.json",
        llama_completion_binary=args.llama_completion_binary,
        llama_device=args.llama_device,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
    )
    print("Batch inference complete.")
    print(f"Saved metrics for models: {list(results.keys())}")


if __name__ == "__main__":
    main()
