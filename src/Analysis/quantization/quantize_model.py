"""
Create true llama.cpp quantized variants for T5-small LoRA model.

Run:
    python src/analysis/quantization/quantize_model.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common import (
    QUANTIZATION_LEVELS,
    collect_test_samples,
    compute_rouge_scores,
    ensure_output_structure,
    generate_summary,
    generate_summary_llama_cpp,
    load_lora_model_and_tokenizer,
    load_raw_split,
    resolve_executable_path,
    save_json,
)


_TORCH = None


def _import_torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def _resolve_f16_gguf_path(output_root: Path) -> Path:
    output_dirs = ensure_output_structure(output_root)
    candidate = output_dirs["models"] / "intermediate" / "model.gguf"
    if not candidate.exists():
        raise FileNotFoundError(
            "Missing base GGUF model for quantization. "
            f"Expected at {candidate.as_posix()}. Run convert_to_gguf.py first."
        )
    return candidate


def load_quantized_model_and_tokenizer(
    artifact_path: Path,
    device=None,
):
    # Kept for compatibility with existing imports.
    torch = _import_torch()
    resolved_device = device or torch.device("cpu")
    spec = {
        "type": "llama_cpp_gguf",
        "artifact_path": Path(artifact_path).as_posix(),
    }
    metadata = {
        "backend": "llama_cpp_gguf",
        "artifact_path": Path(artifact_path).as_posix(),
    }
    return spec, None, resolved_device, metadata


def quantize_model_levels(
    model_dir: Path,
    output_root: Path,
    quant_levels: Optional[Sequence[str]] = None,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
    force: bool = False,
    verbose: bool = True,
    llama_quantize_binary: Optional[str] = None,
    f16_gguf_path: Optional[Path] = None,
    threads: Optional[int] = None,
) -> Dict[str, Path]:
    # model_dir/default_base_model/fallback_local_base_model kept for signature compatibility.
    del model_dir, default_base_model, fallback_local_base_model

    output_dirs = ensure_output_structure(output_root)
    models_dir = output_dirs["models"]
    quant_levels = list(quant_levels or QUANTIZATION_LEVELS.keys())

    f16_source = Path(f16_gguf_path) if f16_gguf_path else _resolve_f16_gguf_path(output_root=output_root)
    quantize_bin = resolve_executable_path(
        preferred=llama_quantize_binary,
        candidates=("llama-quantize",),
    )
    n_threads = int(threads if threads is not None and int(threads) > 0 else max(os.cpu_count() or 1, 1))

    artifact_map: Dict[str, Path] = {}
    quantization_index: Dict[str, Dict] = {}

    for quant_name in quant_levels:
        if quant_name not in QUANTIZATION_LEVELS:
            raise ValueError(f"Unsupported quantization level '{quant_name}'.")

        output_path = models_dir / f"{quant_name}.gguf"
        if output_path.exists() and not force:
            size_mb = float(output_path.stat().st_size / (1024 ** 2))
            artifact_map[quant_name] = output_path
            quantization_index[quant_name] = {
                "artifact_path": output_path.as_posix(),
                "bits": int(QUANTIZATION_LEVELS[quant_name]),
                "quantization_type": quant_name,
                "backend": "llama_cpp_quantize",
                "status": "reused",
                "serialized_size_mb": size_mb,
            }
            continue

        command = [
            quantize_bin,
            f16_source.as_posix(),
            output_path.as_posix(),
            quant_name,
            str(n_threads),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        if completed.returncode != 0:
            raise RuntimeError(
                "llama-quantize failed for "
                f"{quant_name} (code={completed.returncode}). stderr_tail={completed.stderr[-800:]}"
            )

        size_mb = float(output_path.stat().st_size / (1024 ** 2))
        metadata = {
            "backend": "llama_cpp_quantize",
            "quantization_type": quant_name,
            "bits": int(QUANTIZATION_LEVELS[quant_name]),
            "source_f16_gguf": f16_source.as_posix(),
            "artifact_path": output_path.as_posix(),
            "serialized_size_mb": size_mb,
            "threads": n_threads,
            "elapsed_sec": float(elapsed),
            "stdout_tail": completed.stdout[-1200:],
            "stderr_tail": completed.stderr[-1200:],
        }
        save_json(models_dir / f"{quant_name}_metadata.json", metadata)

        artifact_map[quant_name] = output_path
        quantization_index[quant_name] = {
            "artifact_path": output_path.as_posix(),
            "bits": int(QUANTIZATION_LEVELS[quant_name]),
            "quantization_type": quant_name,
            "backend": "llama_cpp_quantize",
            "status": "created",
            "serialized_size_mb": size_mb,
            "elapsed_sec": float(elapsed),
        }
        if verbose:
            print(f"[quantize] {quant_name}: size_mb={size_mb:.2f}, elapsed_sec={elapsed:.2f}")

    save_json(models_dir / "quantization_index.json", quantization_index)
    return artifact_map


def test_quantized_models(
    model_dir: Path,
    output_root: Path,
    dataset_path: str = "data/raw",
    split: str = "test",
    num_samples: int = 3,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
    llama_completion_binary: Optional[str] = None,
    llama_device: str = "BLAS",
    llama_n_gpu_layers: int = 0,
):
    """
    Sanity test for BASE/Q8/Q5/Q4 generation quality with printed outputs and ROUGE.
    """
    output_dirs = ensure_output_structure(output_root)
    raw_split = load_raw_split(dataset_path=dataset_path, split=split)
    samples = collect_test_samples(raw_split=raw_split, num_samples=max(1, num_samples))

    torch = _import_torch()
    device = torch.device("cpu")
    base_model, base_tokenizer, _, _ = load_lora_model_and_tokenizer(
        model_dir=model_dir,
        default_base_model=default_base_model,
        fallback_local_base_model=fallback_local_base_model,
        device=device,
        merge_lora=True,
    )

    model_entries: List[Tuple[str, Dict]] = [("BASE_LORA", {"type": "hf_base"})]
    for quant_name in ("Q8_0", "Q5_K_M", "Q4_K_M"):
        artifact_path = output_dirs["models"] / f"{quant_name}.gguf"
        if artifact_path.exists():
            model_entries.append(
                (
                    quant_name,
                    {
                        "type": "llama_cpp_gguf",
                        "artifact_path": artifact_path.as_posix(),
                    },
                )
            )

    all_predictions: Dict[str, List[str]] = {label: [] for label, _ in model_entries}
    references: List[str] = []
    sample_outputs = []

    for idx, sample in enumerate(samples):
        dialogue = sample["dialogue"]
        reference = sample["summary"]
        references.append(reference)
        print(f"\n[quant-test] Sample {idx + 1} / {len(samples)} | id={sample['sample_id']}")
        print(f"Reference: {reference}")

        per_model = {}
        for label, spec in model_entries:
            if spec["type"] == "hf_base":
                summary, _ = generate_summary(
                    model=base_model,
                    tokenizer=base_tokenizer,
                    device=device,
                    dialogue_text=dialogue,
                )
            else:
                summary, _, _, _ = generate_summary_llama_cpp(
                    model_path=Path(spec["artifact_path"]),
                    dialogue_text=dialogue,
                    max_input_length=1024,
                    llama_completion_binary=llama_completion_binary,
                    llama_device=llama_device,
                    llama_n_gpu_layers=llama_n_gpu_layers,
                    measure_peak_memory=False,
                )
            all_predictions[label].append(summary)
            per_model[label] = summary
            print(f"{label} output: {summary}")

        sample_outputs.append(
            {
                "sample_id": sample["sample_id"],
                "reference": reference,
                "predictions": per_model,
            }
        )

    rouge_by_model = {}
    for label, predictions in all_predictions.items():
        scores = compute_rouge_scores(predictions=predictions, references=references)
        rouge_by_model[label] = scores

    report = {
        "num_samples": len(samples),
        "rouge_by_model": rouge_by_model,
        "samples": sample_outputs,
    }
    save_json(output_dirs["reports"] / "quant_sanity_test.json", report)

    print("\n[quant-test] ROUGE summary:")
    for label, scores in rouge_by_model.items():
        print(
            f"{label}: rouge1={scores['rouge1']:.4f}, rouge2={scores['rouge2']:.4f}, rougeL={scores['rougeL']:.4f}"
        )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Create llama.cpp quantized variants for T5-small LoRA model.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--quant-levels", nargs="+", default=list(QUANTIZATION_LEVELS.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--llama-quantize-binary", default=None)
    parser.add_argument("--f16-gguf-path", default=None)
    parser.add_argument("--run-sanity-test", action="store_true")
    parser.add_argument("--sanity-samples", type=int, default=3)
    parser.add_argument("--dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    parser.add_argument("--llama-completion-binary", default=None)
    parser.add_argument("--llama-device", default="BLAS")
    parser.add_argument("--llama-n-gpu-layers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_map = quantize_model_levels(
        model_dir=Path(args.model_dir),
        output_root=Path(args.output_root),
        quant_levels=args.quant_levels,
        force=args.force,
        verbose=True,
        llama_quantize_binary=args.llama_quantize_binary,
        f16_gguf_path=Path(args.f16_gguf_path) if args.f16_gguf_path else None,
        threads=args.threads,
    )

    print("Quantization complete.")
    for name, path in artifact_map.items():
        print(f"  {name}: {path}")

    if args.run_sanity_test:
        print("\nRunning quantization sanity test...")
        test_quantized_models(
            model_dir=Path(args.model_dir),
            output_root=Path(args.output_root),
            dataset_path=args.dataset_path,
            split=args.split,
            num_samples=args.sanity_samples,
            default_base_model=args.default_base_model,
            fallback_local_base_model=args.base_model_fallback,
            llama_completion_binary=args.llama_completion_binary,
            llama_device=args.llama_device,
            llama_n_gpu_layers=args.llama_n_gpu_layers,
        )


if __name__ == "__main__":
    main()
