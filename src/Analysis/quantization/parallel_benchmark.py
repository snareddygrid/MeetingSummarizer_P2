"""
Parallel inference benchmark (1, 2, 4 processes) for quantized T5-small LoRA models.

Run:
    python src/analysis/quantization/parallel_benchmark.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from batch_inference import resolve_model_specs
from common import (
    QUANTIZATION_LEVELS,
    attach_subset_metadata,
    collect_subset_samples,
    compute_rouge_scores,
    ensure_output_structure,
    generate_summary,
    generate_summary_llama_cpp,
    get_fixed_test_subset,
    load_lora_model_and_tokenizer,
    save_json,
)


_WORKER_SPEC = None
_WORKER_MODEL = None
_WORKER_TOKENIZER = None
_WORKER_DEVICE = None
_WORKER_GENERATION_CONFIG = None
_WORKER_MAX_INPUT_LENGTH = 1024
_WORKER_LLAMA_COMPLETION_BINARY = None
_WORKER_LLAMA_DEVICE = "BLAS"
_WORKER_LLAMA_N_GPU_LAYERS = 0
_WORKER_LLAMA_THREADS = None


def _init_worker(
    spec: Dict,
    default_base_model: str,
    fallback_local_base_model: str,
    generation_config: Optional[Dict],
    max_input_length: int,
    llama_completion_binary: Optional[str],
    llama_device: str,
    llama_n_gpu_layers: int,
    llama_threads: Optional[int],
):
    global _WORKER_SPEC
    global _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_DEVICE
    global _WORKER_GENERATION_CONFIG, _WORKER_MAX_INPUT_LENGTH
    global _WORKER_LLAMA_COMPLETION_BINARY, _WORKER_LLAMA_DEVICE, _WORKER_LLAMA_N_GPU_LAYERS
    global _WORKER_LLAMA_THREADS

    _WORKER_SPEC = spec
    _WORKER_GENERATION_CONFIG = generation_config
    _WORKER_MAX_INPUT_LENGTH = max_input_length
    _WORKER_LLAMA_COMPLETION_BINARY = llama_completion_binary
    _WORKER_LLAMA_DEVICE = llama_device
    _WORKER_LLAMA_N_GPU_LAYERS = int(llama_n_gpu_layers)
    _WORKER_LLAMA_THREADS = int(llama_threads) if llama_threads is not None else None

    if spec["type"] == "llama_cpp_gguf":
        _WORKER_MODEL = None
        _WORKER_TOKENIZER = None
        _WORKER_DEVICE = None
        return

    model, tokenizer, device, _ = load_lora_model_and_tokenizer(
        model_dir=Path(spec["model_dir"]),
        default_base_model=default_base_model,
        fallback_local_base_model=fallback_local_base_model,
        device=None,
        merge_lora=True,
    )

    # Parallel CPU inference is safer than sharing MPS across processes.
    if str(device) != "cpu":
        import torch

        device = torch.device("cpu")
        model.to(device)

    _WORKER_MODEL = model
    _WORKER_TOKENIZER = tokenizer
    _WORKER_DEVICE = device


def _worker_generate(dialogue_text: str) -> Tuple[str, float]:
    if _WORKER_SPEC["type"] == "llama_cpp_gguf":
        summary, latency, _, _ = generate_summary_llama_cpp(
            model_path=Path(_WORKER_SPEC["artifact_path"]),
            dialogue_text=dialogue_text,
            generation_config=_WORKER_GENERATION_CONFIG,
            max_input_length=_WORKER_MAX_INPUT_LENGTH,
            llama_completion_binary=_WORKER_LLAMA_COMPLETION_BINARY,
            llama_device=_WORKER_LLAMA_DEVICE,
            llama_n_gpu_layers=_WORKER_LLAMA_N_GPU_LAYERS,
            threads=_WORKER_LLAMA_THREADS,
            measure_peak_memory=False,
        )
        return summary, latency

    summary, latency = generate_summary(
        model=_WORKER_MODEL,
        tokenizer=_WORKER_TOKENIZER,
        device=_WORKER_DEVICE,
        dialogue_text=dialogue_text,
        generation_config=_WORKER_GENERATION_CONFIG,
        max_input_length=_WORKER_MAX_INPUT_LENGTH,
    )
    return summary, latency


def run_parallel_benchmark(
    model_specs: Dict[str, Dict],
    dataset_path: str,
    split: str,
    output_root: Path,
    process_counts: Sequence[int] = (1, 2, 4),
    num_samples: int = 100,
    generation_config: Optional[Dict] = None,
    max_input_length: int = 1024,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
    subset_rows: Optional[Sequence[Dict]] = None,
    subset_metadata: Optional[Dict] = None,
    subset_size: int = 100,
    processed_dataset_path: str = "data/processed",
    raw_dataset_path: str = "data/raw",
    subset_indices_path: Optional[Path] = None,
    num_trials: int = 2,
    llama_completion_binary: Optional[str] = None,
    llama_device: str = "BLAS",
    llama_n_gpu_layers: int = 0,
    llama_threads: Optional[int] = None,
) -> Dict[str, List[Dict]]:
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
    dialogues = [str(row["dialogue"]) for row in samples]
    references = [str(row["summary"]) for row in samples]

    all_results = {}
    for label, spec in model_specs.items():
        label_records = []
        output_path = output_dirs["parallel"] / f"{label}_parallel.json"
        for process_count in process_counts:
            trial_records = []
            for trial in range(max(num_trials, 1)):
                ctx = mp.get_context("spawn")
                start = time.perf_counter()
                with ctx.Pool(
                    processes=process_count,
                    initializer=_init_worker,
                    initargs=(
                        spec,
                        default_base_model,
                        fallback_local_base_model,
                        generation_config,
                        max_input_length,
                        llama_completion_binary,
                        llama_device,
                        llama_n_gpu_layers,
                        llama_threads,
                    ),
                ) as pool:
                    outputs = list(
                        tqdm(
                            pool.imap(_worker_generate, dialogues),
                            total=len(dialogues),
                            desc=f"Parallel [{label}] x{process_count} trial={trial + 1}",
                        )
                    )
                elapsed = time.perf_counter() - start

                predictions = [item[0] for item in outputs]
                per_sample_latencies = [float(item[1]) for item in outputs]
                throughput = float(len(dialogues) / elapsed) if elapsed > 0 else 0.0
                rouge = compute_rouge_scores(predictions=predictions, references=references)
                trial_records.append(
                    {
                        "trial": int(trial + 1),
                        "wall_time_sec": float(elapsed),
                        "throughput_samples_per_sec": throughput,
                        "mean_worker_latency_sec": float(sum(per_sample_latencies) / max(len(per_sample_latencies), 1)),
                        "rougeL": rouge["rougeL"],
                    }
                )

            throughputs = [record["throughput_samples_per_sec"] for record in trial_records]
            wall_times = [record["wall_time_sec"] for record in trial_records]
            rouge_vals = [record["rougeL"] for record in trial_records]
            record = {
                "model_label": label,
                "process_count": int(process_count),
                "num_samples": len(dialogues),
                "num_trials": int(len(trial_records)),
                "throughput_samples_per_sec_mean": float(statistics.mean(throughputs) if throughputs else 0.0),
                "throughput_samples_per_sec_std": float(statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0),
                "wall_time_sec_mean": float(statistics.mean(wall_times) if wall_times else 0.0),
                "wall_time_sec_std": float(statistics.pstdev(wall_times) if len(wall_times) > 1 else 0.0),
                "rougeL_mean": float(statistics.mean(rouge_vals) if rouge_vals else 0.0),
                "trial_records": trial_records,
            }
            label_records.append(record)

            # Persist progress after each process-count block so interrupted runs
            # still keep completed trial results.
            payload = attach_subset_metadata(
                {
                    "model_label": label,
                    "records": label_records,
                },
                subset_metadata,
                num_samples=len(dialogues),
            )
            save_json(output_path, payload)
        all_results[label] = label_records

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run parallel summarization benchmark.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/raw")
    parser.add_argument("--processed-dataset-path", default="data/processed")
    parser.add_argument("--raw-dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--quant-levels", nargs="+", default=list(QUANTIZATION_LEVELS.keys()))
    parser.add_argument("--process-counts", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    parser.add_argument("--llama-completion-binary", default=None)
    parser.add_argument("--llama-device", default="BLAS")
    parser.add_argument("--llama-n-gpu-layers", type=int, default=0)
    parser.add_argument("--llama-threads", type=int, default=1)
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
        raise RuntimeError("No model specs found for parallel benchmark.")

    results = run_parallel_benchmark(
        model_specs=model_specs,
        dataset_path=args.dataset_path,
        split=args.split,
        output_root=Path(args.output_root),
        process_counts=args.process_counts,
        num_samples=args.num_samples,
        max_input_length=args.max_input_length,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        processed_dataset_path=args.processed_dataset_path,
        raw_dataset_path=args.raw_dataset_path if args.raw_dataset_path else args.dataset_path,
        subset_size=args.subset_size,
        subset_indices_path=Path(args.output_root) / "reports" / "fixed_test_subset.json",
        num_trials=args.num_trials,
        llama_completion_binary=args.llama_completion_binary,
        llama_device=args.llama_device,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
        llama_threads=args.llama_threads,
    )
    print("Parallel benchmark complete.")
    print(f"Models benchmarked: {list(results.keys())}")


if __name__ == "__main__":
    main()
