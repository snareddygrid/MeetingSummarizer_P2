"""
Validate quantization benchmarking outputs for consistency and reproducibility.

Run:
    python src/analysis/quantization/validate_quantization_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from common import load_json, save_json


def _check_subset_metadata(
    payload: Dict,
    expected_size: int,
    errors: List[str],
    context: str,
    allow_partial_num_samples: bool = False,
) -> None:
    dataset_size = int(payload.get("dataset_size", -1))
    subset_indices = payload.get("subset_indices", [])
    num_samples = int(payload.get("num_samples", -1))
    if dataset_size != expected_size:
        errors.append(f"{context}: dataset_size={dataset_size}, expected={expected_size}")
    if len(subset_indices) != expected_size:
        errors.append(f"{context}: subset_indices length={len(subset_indices)}, expected={expected_size}")
    if num_samples != -1 and not allow_partial_num_samples and num_samples != expected_size:
        errors.append(f"{context}: num_samples={num_samples}, expected={expected_size}")


def _validate_latency(payload: Dict, expected_lengths: Sequence[int], errors: List[str], warnings: List[str]) -> None:
    models = payload.get("models", {})
    if not models:
        errors.append("latency: no models found.")
        return

    reference_counts = {}
    for model_label, model_payload in models.items():
        for length in expected_lengths:
            length_key = str(length)
            if length_key not in model_payload:
                errors.append(f"latency: missing length={length_key} for model={model_label}")
                continue
            row = model_payload[length_key]
            num_samples = int(row.get("num_samples", 0))
            reference_counts.setdefault(length_key, num_samples)
            if reference_counts[length_key] != num_samples:
                errors.append(
                    f"latency: inconsistent num_samples for length={length_key}. "
                    f"expected={reference_counts[length_key]}, model={model_label}, got={num_samples}"
                )
            if float(row.get("mean_latency_sec", 0.0)) <= 0:
                errors.append(f"latency: non-positive mean latency for model={model_label}, length={length_key}")
            if float(row.get("throughput_samples_per_sec", 0.0)) <= 0:
                errors.append(f"latency: non-positive throughput for model={model_label}, length={length_key}")
            if int(row.get("warmup_runs", 0)) < 2:
                warnings.append(f"latency: warmup_runs < 2 for model={model_label}, length={length_key}")


def _validate_rouge(payload: Dict, min_rouge_l: float, errors: List[str]) -> None:
    models = payload.get("models", {})
    if not models:
        errors.append("rouge: no models found.")
        return
    for label, row in models.items():
        rouge_l = float(row.get("rougeL", 0.0))
        if rouge_l <= min_rouge_l:
            errors.append(f"rouge: model={label} has suspiciously low rougeL={rouge_l}")


def _validate_streaming(
    streaming_dir: Path,
    expected_size: int,
    errors: List[str],
    allow_partial_num_samples: bool = False,
) -> None:
    files = sorted(streaming_dir.glob("*_streaming_vs_batch.json"))
    if not files:
        errors.append("streaming: no streaming result files found.")
        return
    for path in files:
        payload = load_json(path)
        _check_subset_metadata(
            payload,
            expected_size,
            errors,
            f"streaming:{path.name}",
            allow_partial_num_samples=allow_partial_num_samples,
        )
        if float(payload.get("total_batch_latency_sec", 0.0)) <= 0:
            errors.append(f"streaming:{path.name}: total_batch_latency_sec should be > 0")


def _validate_parallel(
    parallel_dir: Path,
    expected_size: int,
    expected_processes: Sequence[int],
    errors: List[str],
    warnings: List[str],
    allow_partial_num_samples: bool = False,
) -> None:
    files = sorted(parallel_dir.glob("*_parallel.json"))
    if not files:
        errors.append("parallel: no parallel result files found.")
        return
    expected_set = set(int(x) for x in expected_processes)
    for path in files:
        payload = load_json(path)
        _check_subset_metadata(
            payload,
            expected_size,
            errors,
            f"parallel:{path.name}",
            allow_partial_num_samples=allow_partial_num_samples,
        )
        records = payload.get("records", [])
        process_counts = {int(row.get("process_count", -1)) for row in records}
        missing = sorted(expected_set - process_counts)
        if missing:
            errors.append(f"parallel:{path.name}: missing process counts {missing}")
        for row in records:
            if float(row.get("throughput_samples_per_sec_mean", 0.0)) <= 0:
                errors.append(
                    f"parallel:{path.name}: non-positive throughput for process={row.get('process_count')}"
                )
            if int(row.get("num_trials", 0)) < 1:
                warnings.append(f"parallel:{path.name}: num_trials < 1 for process={row.get('process_count')}")


def _validate_deployment_guide(path: Path, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"deployment: missing file {path.as_posix()}")
        return
    payload = load_json(path)
    default_profile = payload.get("default_profile", {})
    throughput_profile = payload.get("throughput_profile", {})
    if not default_profile.get("quantization"):
        errors.append("deployment: default_profile.quantization missing")
    if default_profile.get("parallel_processes") in (None, 0):
        errors.append("deployment: default_profile.parallel_processes missing")
    if not isinstance(default_profile.get("batch_config", {}), dict):
        errors.append("deployment: default_profile.batch_config missing/invalid")
    if not throughput_profile.get("quantization"):
        errors.append("deployment: throughput_profile.quantization missing")


def _validate_conversion_backend(
    models_dir: Path,
    require_llama_cpp: bool,
    errors: List[str],
    warnings: List[str],
) -> None:
    conversion_path = models_dir / "conversion_metadata.json"
    if not conversion_path.exists():
        warnings.append(f"conversion: missing {conversion_path.as_posix()}")
        return
    payload = load_json(conversion_path)
    gguf_result = payload.get("gguf_result", {})
    gguf_status = str(gguf_result.get("status", "unknown"))
    if gguf_status not in {"success", "reused"}:
        message = (
            "conversion: llama.cpp GGUF conversion not successful "
            f"(status={gguf_status}, reason={gguf_result.get('reason')})"
        )
        if require_llama_cpp:
            errors.append(message)
        else:
            warnings.append(message)


def _validate_quantized_artifacts(
    models_dir: Path,
    expected_quant_levels: Sequence[str],
    errors: List[str],
    warnings: List[str],
) -> None:
    index_path = models_dir / "quantization_index.json"
    if not index_path.exists():
        errors.append(f"quantization: missing {index_path.as_posix()}")
        return

    payload = load_json(index_path)
    for level in expected_quant_levels:
        info = payload.get(level)
        if not isinstance(info, dict):
            errors.append(f"quantization: missing index entry for {level}")
            continue
        artifact_path = Path(str(info.get("artifact_path", "")))
        if artifact_path.suffix.lower() != ".gguf":
            errors.append(
                f"quantization:{level}: expected GGUF artifact, got {artifact_path.as_posix()}"
            )
            continue
        if not artifact_path.exists():
            errors.append(f"quantization:{level}: artifact missing at {artifact_path.as_posix()}")
        if str(info.get("backend", "")) not in {"llama_cpp_quantize", "llama_cpp_gguf"}:
            warnings.append(
                f"quantization:{level}: unexpected backend={info.get('backend')}, expected llama_cpp_quantize"
            )


def validate_outputs(
    output_root: Path,
    expected_subset_size: int = 100,
    expected_lengths: Sequence[int] = (10, 50, 100, 200),
    expected_processes: Sequence[int] = (1, 2, 4),
    expected_quant_levels: Sequence[str] = ("Q4_K_M", "Q5_K_M", "Q8_0"),
    min_rouge_l: float = 1e-6,
    allow_partial_num_samples: bool = False,
    require_llama_cpp: bool = False,
) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []

    reports_dir = output_root / "reports"
    subset_path = reports_dir / "fixed_test_subset.json"
    latency_path = output_root / "latency" / "length_benchmark.json"
    throughput_path = output_root / "throughput" / "length_benchmark.json"
    rouge_path = output_root / "rouge" / "summary.json"
    deployment_path = reports_dir / "deployment_guide.json"

    required_files = [subset_path, latency_path, throughput_path, rouge_path, deployment_path]
    missing_files = [path.as_posix() for path in required_files if not path.exists()]
    if missing_files:
        errors.append(f"missing required files: {missing_files}")

    subset_payload = load_json(subset_path) if subset_path.exists() else {}
    latency_payload = load_json(latency_path) if latency_path.exists() else {}
    throughput_payload = load_json(throughput_path) if throughput_path.exists() else {}
    rouge_payload = load_json(rouge_path) if rouge_path.exists() else {}

    if subset_payload:
        _check_subset_metadata(
            subset_payload,
            expected_subset_size,
            errors,
            "subset",
            allow_partial_num_samples=allow_partial_num_samples,
        )
    if latency_payload:
        _check_subset_metadata(
            latency_payload,
            expected_subset_size,
            errors,
            "latency",
            allow_partial_num_samples=allow_partial_num_samples,
        )
        _validate_latency(latency_payload, expected_lengths, errors, warnings)
    if throughput_payload:
        _check_subset_metadata(
            throughput_payload,
            expected_subset_size,
            errors,
            "throughput",
            allow_partial_num_samples=allow_partial_num_samples,
        )
    if rouge_payload:
        _check_subset_metadata(
            rouge_payload,
            expected_subset_size,
            errors,
            "rouge",
            allow_partial_num_samples=allow_partial_num_samples,
        )
        _validate_rouge(rouge_payload, min_rouge_l, errors)

    _validate_streaming(
        output_root / "streaming",
        expected_subset_size,
        errors,
        allow_partial_num_samples=allow_partial_num_samples,
    )
    _validate_parallel(
        output_root / "parallel",
        expected_subset_size,
        expected_processes,
        errors,
        warnings,
        allow_partial_num_samples=allow_partial_num_samples,
    )
    _validate_deployment_guide(deployment_path, errors)
    _validate_quantized_artifacts(
        output_root / "models",
        expected_quant_levels=expected_quant_levels,
        errors=errors,
        warnings=warnings,
    )
    _validate_conversion_backend(output_root / "models", require_llama_cpp=require_llama_cpp, errors=errors, warnings=warnings)

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "expected_subset_size": int(expected_subset_size),
        "expected_lengths": [int(x) for x in expected_lengths],
        "expected_processes": [int(x) for x in expected_processes],
        "expected_quant_levels": list(expected_quant_levels),
    }
    save_json(reports_dir / "validation_report.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Validate quantization outputs for consistency.")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--expected-subset-size", type=int, default=100)
    parser.add_argument("--expected-lengths", nargs="+", type=int, default=[10, 50, 100, 200])
    parser.add_argument("--expected-processes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--expected-quant-levels", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"])
    parser.add_argument("--min-rouge-l", type=float, default=1e-6)
    parser.add_argument("--allow-partial-num-samples", action="store_true")
    parser.add_argument("--require-llama-cpp", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = validate_outputs(
        output_root=Path(args.output_root),
        expected_subset_size=args.expected_subset_size,
        expected_lengths=args.expected_lengths,
        expected_processes=args.expected_processes,
        expected_quant_levels=args.expected_quant_levels,
        min_rouge_l=args.min_rouge_l,
        allow_partial_num_samples=args.allow_partial_num_samples,
        require_llama_cpp=args.require_llama_cpp,
    )
    print(f"Validation status: {'PASS' if result['valid'] else 'FAIL'}")
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error}")
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    if args.fail_on_error and not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
