"""
Streaming vs batch inference comparison for quantized T5-small LoRA models.

Run:
    python src/analysis/quantization/streaming_inference.py
"""

from __future__ import annotations

import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, Optional, Sequence

from tqdm import tqdm

from batch_inference import load_model_for_spec, resolve_model_specs
from common import (
    QUANTIZATION_LEVELS,
    attach_subset_metadata,
    collect_subset_samples,
    compute_latency_stats,
    compute_rouge_scores,
    ensure_output_structure,
    generate_summary,
    generate_summary_llama_cpp,
    get_fixed_test_subset,
    save_json,
    split_dialogue_turns,
)


def _build_stream_steps(num_turns: int, step_size: int) -> Sequence[int]:
    if num_turns <= 1:
        return [1]
    steps = list(range(1, num_turns + 1, max(step_size, 1)))
    if steps[-1] != num_turns:
        steps.append(num_turns)
    return sorted(set(steps))


def _generate_with_memory(
    *,
    spec: Dict,
    model,
    tokenizer,
    device,
    dialogue_text: str,
    generation_config,
    max_input_length: int,
    llama_completion_binary: Optional[str],
    llama_device: str,
    llama_n_gpu_layers: int,
):
    if spec["type"] == "llama_cpp_gguf":
        summary, latency, peak_mb, _ = generate_summary_llama_cpp(
            model_path=Path(spec["artifact_path"]),
            dialogue_text=dialogue_text,
            generation_config=generation_config,
            max_input_length=max_input_length,
            llama_completion_binary=llama_completion_binary,
            llama_device=llama_device,
            llama_n_gpu_layers=llama_n_gpu_layers,
            measure_peak_memory=True,
        )
        return summary, latency, peak_mb

    tracemalloc.start()
    summary, latency = generate_summary(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dialogue_text=dialogue_text,
        generation_config=generation_config,
        max_input_length=max_input_length,
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return summary, latency, float(peak_bytes / (1024 ** 2))


def run_streaming_benchmark(
    model_specs: Dict[str, Dict],
    dataset_path: str,
    split: str,
    output_root: Path,
    num_samples: int = 100,
    step_size: int = 3,
    generation_config: Optional[Dict] = None,
    max_input_length: int = 1024,
    device_preference: str = "auto",
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
    subset_rows: Optional[Sequence[Dict]] = None,
    subset_metadata: Optional[Dict] = None,
    subset_size: int = 100,
    processed_dataset_path: str = "data/processed",
    raw_dataset_path: str = "data/raw",
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

        streaming_latencies = []
        batch_latencies = []
        streaming_peak_memories = []
        batch_peak_memories = []
        streaming_predictions = []
        batch_predictions = []
        references = []
        per_sample = []

        for row in tqdm(samples, desc=f"Streaming Benchmark [{label}]"):
            sample_id = row["sample_id"]
            dialogue = str(row["dialogue"])
            reference = str(row["summary"])
            turns = split_dialogue_turns(dialogue)
            if not turns:
                continue

            stream_steps = _build_stream_steps(num_turns=len(turns), step_size=step_size)
            final_stream_summary = ""
            sample_stream_latencies = []
            sample_stream_peaks = []

            for step in stream_steps:
                partial_dialogue = "\n".join(turns[:step])
                stream_summary, stream_latency, stream_peak_mb = _generate_with_memory(
                    spec=spec,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dialogue_text=partial_dialogue,
                    generation_config=generation_config,
                    max_input_length=max_input_length,
                    llama_completion_binary=llama_completion_binary,
                    llama_device=llama_device,
                    llama_n_gpu_layers=llama_n_gpu_layers,
                )
                final_stream_summary = stream_summary
                sample_stream_latencies.append(stream_latency)
                sample_stream_peaks.append(stream_peak_mb)
                streaming_latencies.append(stream_latency)
                streaming_peak_memories.append(stream_peak_mb)

            batch_summary, batch_latency, batch_peak_mb = _generate_with_memory(
                spec=spec,
                model=model,
                tokenizer=tokenizer,
                device=device,
                dialogue_text=dialogue,
                generation_config=generation_config,
                max_input_length=max_input_length,
                llama_completion_binary=llama_completion_binary,
                llama_device=llama_device,
                llama_n_gpu_layers=llama_n_gpu_layers,
            )
            batch_latencies.append(batch_latency)
            batch_peak_memories.append(batch_peak_mb)

            streaming_predictions.append(final_stream_summary)
            batch_predictions.append(batch_summary)
            references.append(reference)

            per_sample.append(
                {
                    "sample_id": sample_id,
                    "turn_count": len(turns),
                    "stream_steps": list(stream_steps),
                    "stream_total_latency_sec": float(sum(sample_stream_latencies)),
                    "stream_mean_update_latency_sec": float(sum(sample_stream_latencies) / max(len(sample_stream_latencies), 1)),
                    "batch_latency_sec": float(batch_latency),
                    "stream_peak_memory_mb": float(max(sample_stream_peaks) if sample_stream_peaks else 0.0),
                    "batch_peak_memory_mb": float(batch_peak_mb),
                }
            )

        streaming_rouge = compute_rouge_scores(streaming_predictions, references) if references else {"rougeL": 0.0, "rouge1": 0.0, "rouge2": 0.0}
        batch_rouge = compute_rouge_scores(batch_predictions, references) if references else {"rougeL": 0.0, "rouge1": 0.0, "rouge2": 0.0}

        stream_latency_stats = compute_latency_stats(streaming_latencies)
        batch_latency_stats = compute_latency_stats(batch_latencies)
        total_streaming_latency = float(sum(item["stream_total_latency_sec"] for item in per_sample))
        total_batch_latency = float(sum(item["batch_latency_sec"] for item in per_sample))

        payload = attach_subset_metadata(
            {
            "model_label": label,
            "num_samples": len(references),
            "streaming": {
                "latency": stream_latency_stats,
                "peak_memory_mb_mean": float(sum(streaming_peak_memories) / max(len(streaming_peak_memories), 1)),
                "peak_memory_mb_max": float(max(streaming_peak_memories) if streaming_peak_memories else 0.0),
                "rouge1": streaming_rouge["rouge1"],
                "rouge2": streaming_rouge["rouge2"],
                "rougeL": streaming_rouge["rougeL"],
            },
            "batch": {
                "latency": batch_latency_stats,
                "peak_memory_mb_mean": float(sum(batch_peak_memories) / max(len(batch_peak_memories), 1)),
                "peak_memory_mb_max": float(max(batch_peak_memories) if batch_peak_memories else 0.0),
                "rouge1": batch_rouge["rouge1"],
                "rouge2": batch_rouge["rouge2"],
                "rougeL": batch_rouge["rougeL"],
            },
            "quality_delta_rougeL": float(streaming_rouge["rougeL"] - batch_rouge["rougeL"]),
            "total_latency_delta_sec": float(total_streaming_latency - total_batch_latency),
            "total_streaming_latency_sec": total_streaming_latency,
            "total_batch_latency_sec": total_batch_latency,
            "per_sample": per_sample,
            },
            subset_metadata,
            num_samples=len(references),
        )

        path = output_dirs["streaming"] / f"{label}_streaming_vs_batch.json"
        save_json(path, payload)
        results[label] = payload
        if spec["type"] != "llama_cpp_gguf":
            del model

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Compare streaming vs batch inference.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/raw")
    parser.add_argument("--processed-dataset-path", default="data/processed")
    parser.add_argument("--raw-dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=3)
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
        raise RuntimeError("No model specs found for streaming benchmark.")

    results = run_streaming_benchmark(
        model_specs=model_specs,
        dataset_path=args.dataset_path,
        split=args.split,
        output_root=Path(args.output_root),
        num_samples=args.num_samples,
        step_size=args.step_size,
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
    print("Streaming benchmark complete.")
    print(f"Models benchmarked: {list(results.keys())}")


if __name__ == "__main__":
    main()
