"""
End-to-end quantization and benchmarking pipeline for Task-2.

Run:
    python src/analysis/quantization/benchmark_inference.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

from tqdm import tqdm

from batch_inference import load_model_for_spec, resolve_model_specs, run_batch_inference
from common import (
    DEFAULT_LENGTH_BUCKETS,
    QUANTIZATION_LEVELS,
    attach_subset_metadata,
    collect_length_bucket_samples_from_subset,
    compute_latency_stats,
    ensure_output_structure,
    generate_summary,
    generate_summary_llama_cpp,
    get_fixed_test_subset,
    save_json,
)
from compare_results import generate_comparison_report
from convert_to_gguf import convert_or_fallback
from evaluate_rouge import evaluate_from_batch_outputs
from parallel_benchmark import run_parallel_benchmark
from quantize_model import quantize_model_levels, test_quantized_models
from streaming_inference import run_streaming_benchmark


def _benchmark_variable_lengths(
    model_specs: Dict[str, Dict],
    subset_rows,
    subset_metadata: Dict,
    lengths: Sequence[int],
    samples_per_length: int,
    warmup_runs: int,
    measurement_iterations: int,
    generation_config: Dict,
    max_input_length: int,
    default_base_model: str,
    fallback_local_base_model: str,
    llama_completion_binary: Optional[str],
    llama_device: str,
    llama_n_gpu_layers: int,
    llama_threads: Optional[int] = None,
) -> Dict:
    length_buckets = collect_length_bucket_samples_from_subset(
        subset_rows=subset_rows,
        lengths=lengths,
        samples_per_length=samples_per_length,
    )

    metrics_by_model = {}
    for label, spec in model_specs.items():
        model, tokenizer, device = load_model_for_spec(
            label=label,
            spec=spec,
            default_base_model=default_base_model,
            fallback_local_base_model=fallback_local_base_model,
            device_preference="auto",
        )

        per_length = {}
        for length in lengths:
            samples = length_buckets[int(length)]
            latencies = []
            if not samples:
                continue

            warmup_count = min(warmup_runs, len(samples))
            for warmup_idx in range(warmup_count):
                warmup_sample = samples[warmup_idx % len(samples)]
                if spec["type"] == "llama_cpp_gguf":
                    generate_summary_llama_cpp(
                        model_path=Path(spec["artifact_path"]),
                        dialogue_text=warmup_sample["dialogue"],
                        generation_config=generation_config,
                        max_input_length=max_input_length,
                        llama_completion_binary=llama_completion_binary,
                        llama_device=llama_device,
                        llama_n_gpu_layers=llama_n_gpu_layers,
                        threads=llama_threads,
                        measure_peak_memory=False,
                    )
                else:
                    generate_summary(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        dialogue_text=warmup_sample["dialogue"],
                        generation_config=generation_config,
                        max_input_length=max_input_length,
                    )

            measured_iters = max(measurement_iterations, 1)
            progress = tqdm(total=len(samples) * measured_iters, desc=f"Length Benchmark [{label}] {length}")
            for _ in range(measured_iters):
                for sample in samples:
                    if spec["type"] == "llama_cpp_gguf":
                        _, latency, _, _ = generate_summary_llama_cpp(
                            model_path=Path(spec["artifact_path"]),
                            dialogue_text=sample["dialogue"],
                            generation_config=generation_config,
                            max_input_length=max_input_length,
                            llama_completion_binary=llama_completion_binary,
                            llama_device=llama_device,
                            llama_n_gpu_layers=llama_n_gpu_layers,
                            threads=llama_threads,
                            measure_peak_memory=False,
                        )
                    else:
                        _, latency = generate_summary(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            dialogue_text=sample["dialogue"],
                            generation_config=generation_config,
                            max_input_length=max_input_length,
                        )
                    latencies.append(latency)
                    progress.update(1)
            progress.close()

            stats = compute_latency_stats(latencies)
            throughput = float(len(latencies) / stats["total_sec"]) if stats["total_sec"] > 0 else 0.0
            per_length[str(length)] = {
                "num_samples": len(samples),
                "warmup_runs": int(warmup_count),
                "measurement_iterations": int(measured_iters),
                "measured_samples_total": len(latencies),
                "mean_latency_sec": stats["mean_sec"],
                "median_latency_sec": stats["median_sec"],
                "p95_latency_sec": stats["p95_sec"],
                "throughput_samples_per_sec": throughput,
            }

        metrics_by_model[label] = per_length
        if spec["type"] != "llama_cpp_gguf":
            del model

    payload = attach_subset_metadata(
        {
            "lengths": [int(length) for length in lengths],
            "samples_per_length": int(samples_per_length),
            "warmup_runs": int(warmup_runs),
            "measurement_iterations": int(max(measurement_iterations, 1)),
            "models": metrics_by_model,
        },
        subset_metadata,
        num_samples=len(subset_rows),
    )
    return payload


def run_full_pipeline(args) -> Dict:
    output_root = Path(args.output_root)
    output_dirs = ensure_output_structure(output_root)
    subset_rows, subset_metadata = get_fixed_test_subset(
        n=args.subset_size,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        split=args.split,
        subset_indices_path=output_dirs["reports"] / "fixed_test_subset.json",
        selection_mode=args.subset_selection_mode,
        random_seed=args.subset_seed,
    )

    conversion_metadata = convert_or_fallback(
        model_dir=Path(args.model_dir),
        output_root=output_root,
        llama_cpp_dir=Path(args.llama_cpp_dir) if args.llama_cpp_dir else None,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
    )

    quantize_model_levels(
        model_dir=Path(args.model_dir),
        output_root=output_root,
        quant_levels=args.quant_levels,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        force=args.force_requantize,
        llama_quantize_binary=args.llama_quantize_binary,
        threads=args.llama_threads,
    )
    if args.run_quant_sanity_test:
        print("Running quick quantization sanity test before benchmarks...")
        test_quantized_models(
            model_dir=Path(args.model_dir),
            output_root=output_root,
            dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
            split=args.split,
            num_samples=args.sanity_samples,
            default_base_model=args.default_base_model,
            fallback_local_base_model=args.base_model_fallback,
            llama_completion_binary=args.llama_completion_binary,
            llama_device=args.llama_device,
            llama_n_gpu_layers=args.llama_n_gpu_layers,
        )

    model_specs = resolve_model_specs(
        model_dir=Path(args.model_dir),
        models_output_dir=output_dirs["models"],
        quant_levels=args.quant_levels,
        include_base=args.include_base,
    )
    if not model_specs:
        raise RuntimeError("No model specs were resolved. Quantized artifacts may be missing.")

    length_metrics = _benchmark_variable_lengths(
        model_specs=model_specs,
        subset_rows=subset_rows,
        subset_metadata=subset_metadata,
        lengths=args.lengths,
        samples_per_length=min(args.samples_per_length, len(subset_rows)),
        warmup_runs=args.warmup_runs,
        measurement_iterations=args.measurement_iterations,
        generation_config=None,
        max_input_length=args.max_input_length,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        llama_completion_binary=args.llama_completion_binary,
        llama_device=args.llama_device,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
        llama_threads=args.llama_threads,
    )
    save_json(output_dirs["latency"] / "length_benchmark.json", length_metrics)
    save_json(output_dirs["throughput"] / "length_benchmark.json", length_metrics)

    batch_results = run_batch_inference(
        model_specs=model_specs,
        output_root=output_root,
        dataset_path=args.dataset_path,
        split=args.split,
        num_samples=min(args.num_samples_rouge, len(subset_rows)),
        max_input_length=args.max_input_length,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        subset_size=args.subset_size,
        subset_rows=subset_rows,
        subset_metadata=subset_metadata,
        subset_indices_path=output_dirs["reports"] / "fixed_test_subset.json",
        llama_completion_binary=args.llama_completion_binary,
        llama_device=args.llama_device,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
    )

    rouge_results_raw = evaluate_from_batch_outputs(
        batch_dir=output_dirs["batch"],
        rouge_output_dir=output_dirs["rouge"],
        model_labels=list(model_specs.keys()),
    )
    rouge_results = attach_subset_metadata(
        {"models": rouge_results_raw},
        subset_metadata,
        num_samples=min(args.num_samples_rouge, len(subset_rows)),
    )
    save_json(output_dirs["rouge"] / "rouge_summary.json", rouge_results)
    save_json(output_dirs["rouge"] / "summary.json", rouge_results)

    streaming_results = run_streaming_benchmark(
        model_specs=model_specs,
        dataset_path=args.dataset_path,
        split=args.split,
        output_root=output_root,
        num_samples=min(args.streaming_samples, len(subset_rows)),
        step_size=args.streaming_step_size,
        max_input_length=args.max_input_length,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        subset_rows=subset_rows,
        subset_metadata=subset_metadata,
        subset_size=args.subset_size,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        subset_indices_path=output_dirs["reports"] / "fixed_test_subset.json",
        llama_completion_binary=args.llama_completion_binary,
        llama_device=args.llama_device,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
    )

    parallel_results = run_parallel_benchmark(
        model_specs=model_specs,
        dataset_path=args.dataset_path,
        split=args.split,
        output_root=output_root,
        process_counts=args.parallel_processes,
        num_samples=min(args.parallel_samples, len(subset_rows)),
        max_input_length=args.max_input_length,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        subset_rows=subset_rows,
        subset_metadata=subset_metadata,
        subset_size=args.subset_size,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        subset_indices_path=output_dirs["reports"] / "fixed_test_subset.json",
        num_trials=args.parallel_trials,
        llama_completion_binary=args.llama_completion_binary,
        llama_device=args.llama_device,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
    )

    report = generate_comparison_report(output_root=output_root)

    summary = {
        "conversion": conversion_metadata,
        "models": list(model_specs.keys()),
        "subset_metadata": subset_metadata,
        "batch_models": list(batch_results.keys()),
        "length_metrics_file": (output_dirs["latency"] / "length_benchmark.json").as_posix(),
        "rouge_summary_file": (output_dirs["rouge"] / "rouge_summary.json").as_posix(),
        "streaming_models": list(streaming_results.keys()),
        "parallel_models": list(parallel_results.keys()),
        "final_report_file": (output_dirs["reports"] / "final_report.json").as_posix(),
        "deployment_guide_file": (output_dirs["reports"] / "deployment_guide.json").as_posix(),
        "recommended_quantization": report.get("best_quantization"),
        "recommended_parallel_config": report.get("recommended_parallel_config"),
    }
    save_json(output_dirs["reports"] / "pipeline_run_summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run full quantization + benchmarking pipeline.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/raw")
    parser.add_argument("--processed-dataset-path", default="data/processed")
    parser.add_argument("--raw-dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--llama-cpp-dir", default=None)
    parser.add_argument("--llama-quantize-binary", default=None)
    parser.add_argument("--llama-completion-binary", default=None)
    parser.add_argument("--llama-device", default="BLAS")
    parser.add_argument("--llama-n-gpu-layers", type=int, default=0)
    parser.add_argument("--llama-threads", type=int, default=1)
    parser.add_argument("--quant-levels", nargs="+", default=list(QUANTIZATION_LEVELS.keys()))
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTH_BUCKETS)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--subset-selection-mode", choices=["first_n", "random_seed"], default="first_n")
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--samples-per-length", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measurement-iterations", type=int, default=2)
    parser.add_argument("--num-samples-rouge", type=int, default=100)
    parser.add_argument("--streaming-samples", type=int, default=100)
    parser.add_argument("--streaming-step-size", type=int, default=3)
    parser.add_argument("--parallel-samples", type=int, default=50)
    parser.add_argument("--parallel-processes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--parallel-trials", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--force-requantize", action="store_true")
    parser.add_argument("--run-quant-sanity-test", action="store_true")
    parser.add_argument("--sanity-samples", type=int, default=3)
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_full_pipeline(args)
    print("Quantization benchmarking pipeline complete.")
    print(f"Recommended quantization: {summary.get('recommended_quantization')}")
    print(f"Report: {summary.get('final_report_file')}")


if __name__ == "__main__":
    main()
