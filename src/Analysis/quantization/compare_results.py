"""
Aggregate quantization benchmark metrics and generate final report.

Run:
    python src/analysis/quantization/compare_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from common import ensure_output_structure, load_json, save_json


def _safe_get_latency_50(latency_payload: Dict, label: str) -> float:
    model_root = latency_payload.get("models", latency_payload)
    model_info = model_root.get(label, {})
    length_key = "50"
    if 50 in model_info:
        length_key = 50
    entry = model_info.get(length_key, {})
    return float(entry.get("mean_latency_sec", 0.0))


def _safe_get_throughput_50(throughput_payload: Dict, label: str) -> float:
    model_root = throughput_payload.get("models", throughput_payload)
    model_info = model_root.get(label, {})
    length_key = "50"
    if 50 in model_info:
        length_key = 50
    entry = model_info.get(length_key, {})
    return float(entry.get("throughput_samples_per_sec", 0.0))


def _load_rouge_results(rouge_dir: Path) -> Dict[str, Dict]:
    results = {}
    for path in rouge_dir.glob("*.json"):
        if path.stem in {"rouge_summary", "summary"}:
            continue
        payload = load_json(path)
        label = payload.get("model_label", path.stem)
        results[label] = payload
    summary_path = rouge_dir / "summary.json"
    if summary_path.exists():
        payload = load_json(summary_path)
        models = payload.get("models", {})
        for label, model_payload in models.items():
            if label not in results:
                results[label] = model_payload
    return results


def _load_streaming_results(streaming_dir: Path) -> Dict[str, Dict]:
    results = {}
    for path in streaming_dir.glob("*_streaming_vs_batch.json"):
        payload = load_json(path)
        label = payload.get("model_label", path.stem.replace("_streaming_vs_batch", ""))
        results[label] = payload
    return results


def _load_parallel_results(parallel_dir: Path) -> Dict[str, List[Dict]]:
    results = {}
    for path in parallel_dir.glob("*_parallel.json"):
        payload = load_json(path)
        label = payload.get("model_label", path.stem.replace("_parallel", ""))
        results[label] = payload.get("records", [])
    return results


def _recommend_parallel(parallel_results: Dict[str, List[Dict]], preferred_model: Optional[str] = None) -> Dict:
    if preferred_model:
        rows = parallel_results.get(preferred_model, [])
        if not rows:
            return {
                "score": 0.0,
                "model_label": preferred_model,
                "process_count": None,
                "throughput_samples_per_sec": 0.0,
                "wall_time_sec": 0.0,
            }
        best_row = max(
            rows,
            key=lambda row: float(row.get("throughput_samples_per_sec_mean", row.get("throughput_samples_per_sec", 0.0))),
        )
        score = float(best_row.get("throughput_samples_per_sec_mean", best_row.get("throughput_samples_per_sec", 0.0)))
        return {
            "score": score,
            "model_label": preferred_model,
            "process_count": int(best_row.get("process_count", 1)),
            "throughput_samples_per_sec": score,
            "wall_time_sec": float(best_row.get("wall_time_sec_mean", best_row.get("wall_time_sec", 0.0))),
        }

    best = None
    for label, records in parallel_results.items():
        for row in records:
            score = float(row.get("throughput_samples_per_sec_mean", row.get("throughput_samples_per_sec", 0.0)))
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "model_label": label,
                    "process_count": int(row.get("process_count", 1)),
                    "throughput_samples_per_sec": score,
                    "wall_time_sec": float(row.get("wall_time_sec_mean", row.get("wall_time_sec", 0.0))),
                }
    return best or {
        "model_label": None,
        "process_count": None,
        "throughput_samples_per_sec": 0.0,
        "wall_time_sec": 0.0,
    }


def _load_conversion_metadata(models_dir: Path) -> Dict:
    conversion_path = models_dir / "conversion_metadata.json"
    if not conversion_path.exists():
        return {}
    return load_json(conversion_path)


def _load_quantization_index(models_dir: Path) -> Dict:
    index_path = models_dir / "quantization_index.json"
    if not index_path.exists():
        return {}
    return load_json(index_path)


def _build_backend_summary(conversion_metadata: Dict, quantization_index: Dict) -> Dict:
    gguf = conversion_metadata.get("gguf_result", {})
    fallback = conversion_metadata.get("fallback_result")
    gguf_status = gguf.get("status")
    quant_entries = [value for value in quantization_index.values() if isinstance(value, dict)]
    has_gguf_quant = bool(
        quant_entries
        and all(str(entry.get("artifact_path", "")).lower().endswith(".gguf") for entry in quant_entries)
    )

    if gguf_status in {"success", "reused"} and has_gguf_quant:
        backend = "llama_cpp_gguf"
        backend_note = "llama.cpp GGUF conversion and quantization succeeded."
    elif gguf_status in {"success", "reused"}:
        backend = "partial_llama_cpp"
        backend_note = "GGUF conversion succeeded, but quantized GGUF artifacts were not fully detected."
    elif fallback:
        backend = "simulated_fallback"
        backend_note = (
            "llama.cpp GGUF was unavailable/unsupported for this run; "
            "used simulated selective quantization fallback."
        )
    else:
        backend = "unknown"
        backend_note = "Quantization backend could not be inferred from conversion metadata."

    return {
        "backend": backend,
        "gguf_status": gguf_status,
        "gguf_reason": gguf.get("reason"),
        "fallback_format": fallback.get("format") if isinstance(fallback, dict) else None,
        "note": backend_note,
    }


def _build_deployment_recommendation(
    rows: List[Dict],
    parallel_results: Dict[str, List[Dict]],
    backend_summary: Dict,
) -> Dict:
    quantized_rows = [row for row in rows if row.get("model_label") != "BASE_LORA"]
    if not quantized_rows:
        return {
            "backend": backend_summary,
            "default_profile": {},
            "throughput_profile": {},
            "notes": ["No quantized model rows were available to build deployment recommendations."],
        }

    best_quality = max(quantized_rows, key=lambda row: float(row.get("rougeL", 0.0)))
    best_throughput = max(quantized_rows, key=lambda row: float(row.get("throughput_50_samples_per_sec", 0.0)))

    quality_parallel = _recommend_parallel(parallel_results, preferred_model=best_quality.get("model_label"))
    throughput_parallel = _recommend_parallel(parallel_results, preferred_model=best_throughput.get("model_label"))

    default_profile = {
        "objective": "quality_first_realtime",
        "quantization": best_quality.get("model_label"),
        "parallel_processes": quality_parallel.get("process_count"),
        "batch_config": {
            "batch_size": 1,
            "streaming_step_size": 3,
            "max_input_length": 1024,
        },
        "expected_metrics": {
            "rougeL": float(best_quality.get("rougeL", 0.0)),
            "latency_50_sec": float(best_quality.get("latency_50_sec", 0.0)),
            "throughput_50_samples_per_sec": float(best_quality.get("throughput_50_samples_per_sec", 0.0)),
            "parallel_throughput_samples_per_sec": float(quality_parallel.get("throughput_samples_per_sec", 0.0)),
        },
    }

    throughput_profile = {
        "objective": "throughput_first_realtime",
        "quantization": best_throughput.get("model_label"),
        "parallel_processes": throughput_parallel.get("process_count"),
        "batch_config": {
            "batch_size": 1,
            "streaming_step_size": 3,
            "max_input_length": 1024,
        },
        "expected_metrics": {
            "rougeL": float(best_throughput.get("rougeL", 0.0)),
            "latency_50_sec": float(best_throughput.get("latency_50_sec", 0.0)),
            "throughput_50_samples_per_sec": float(best_throughput.get("throughput_50_samples_per_sec", 0.0)),
            "parallel_throughput_samples_per_sec": float(throughput_parallel.get("throughput_samples_per_sec", 0.0)),
        },
    }

    return {
        "backend": backend_summary,
        "default_profile": default_profile,
        "throughput_profile": throughput_profile,
        "notes": [
            "Default profile prioritizes ROUGE-L stability while remaining real-time capable.",
            "Throughput profile maximizes summaries/sec and may trade off summary quality.",
        ],
    }


def generate_comparison_report(output_root: Path) -> Dict:
    output_dirs = ensure_output_structure(output_root)

    latency_path = output_dirs["latency"] / "length_benchmark.json"
    throughput_path = output_dirs["throughput"] / "length_benchmark.json"

    latency_payload = load_json(latency_path) if latency_path.exists() else {}
    throughput_payload = load_json(throughput_path) if throughput_path.exists() else {}
    rouge_results = _load_rouge_results(output_dirs["rouge"])
    streaming_results = _load_streaming_results(output_dirs["streaming"])
    parallel_results = _load_parallel_results(output_dirs["parallel"])
    conversion_metadata = _load_conversion_metadata(output_dirs["models"])
    quantization_index = _load_quantization_index(output_dirs["models"])
    backend_summary = _build_backend_summary(conversion_metadata, quantization_index)

    latency_models = latency_payload.get("models", latency_payload)
    throughput_models = throughput_payload.get("models", throughput_payload)

    latency_keys = set(latency_models.keys())
    throughput_keys = set(throughput_models.keys())
    if latency_keys or throughput_keys:
        # Anchor comparison to the current benchmark run and ignore stale artifacts.
        model_labels = sorted(latency_keys | throughput_keys)
    else:
        model_labels = sorted(
            set(rouge_results.keys())
            | set(streaming_results.keys())
        )

    rows = []
    for label in model_labels:
        rouge_l = float(rouge_results.get(label, {}).get("rougeL", 0.0))
        latency_50 = _safe_get_latency_50(latency_payload, label)
        throughput_50 = _safe_get_throughput_50(throughput_payload, label)
        streaming_delta = float(streaming_results.get(label, {}).get("quality_delta_rougeL", 0.0))

        # Simple trade-off score: prioritize ROUGE-L, then throughput, then low latency.
        score = rouge_l * 100.0 + throughput_50 - (latency_50 * 10.0) + (streaming_delta * 20.0)
        rows.append(
            {
                "model_label": label,
                "rougeL": rouge_l,
                "latency_50_sec": latency_50,
                "throughput_50_samples_per_sec": throughput_50,
                "streaming_delta_rougeL": streaming_delta,
                "score": float(score),
            }
        )

    rows = sorted(rows, key=lambda item: item["score"], reverse=True)
    best_model = rows[0] if rows else {}
    best_model_parallel = _recommend_parallel(parallel_results, preferred_model=best_model.get("model_label"))
    fastest_parallel = _recommend_parallel(parallel_results)
    deployment = _build_deployment_recommendation(rows, parallel_results, backend_summary)

    summary = {
        "best_quantization": best_model.get("model_label"),
        "best_quantization_score": best_model.get("score", 0.0),
        "recommended_parallel_config": best_model_parallel,
        "fastest_parallel_config": fastest_parallel,
        "quantization_backend": backend_summary,
        "deployment_recommendation": deployment,
        "num_samples": int(latency_payload.get("num_samples", 0)),
        "dataset_size": int(latency_payload.get("dataset_size", 0)),
        "subset_indices": latency_payload.get("subset_indices", []),
        "model_comparison": rows,
        "notes": [
            backend_summary.get("note"),
            "Recommendation balances ROUGE-L, latency, throughput, and streaming quality delta.",
        ],
    }

    save_json(output_dirs["reports"] / "final_report.json", summary)

    markdown_lines = [
        "# Quantization Benchmark Report",
        "",
        f"Recommended quantization: **{summary['best_quantization']}**",
        (
            "Recommended parallel setup: "
            f"**{best_model_parallel.get('process_count')} processes** on {best_model_parallel.get('model_label')} "
            f"(throughput={best_model_parallel.get('throughput_samples_per_sec', 0.0):.4f} samples/sec)"
        ),
        (
            "Fastest overall parallel setup: "
            f"**{fastest_parallel.get('process_count')} processes** on {fastest_parallel.get('model_label')} "
            f"(throughput={fastest_parallel.get('throughput_samples_per_sec', 0.0):.4f} samples/sec)"
        ),
        f"Quantization backend: **{backend_summary.get('backend')}**",
        "",
        "## Model Comparison",
        "",
        "| Model | ROUGE-L | Latency@50 (s) | Throughput@50 (samples/s) | Streaming ΔROUGE-L | Score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            f"{row['model_label']} | "
            f"{row['rougeL']:.4f} | "
            f"{row['latency_50_sec']:.4f} | "
            f"{row['throughput_50_samples_per_sec']:.4f} | "
            f"{row['streaming_delta_rougeL']:.4f} | "
            f"{row['score']:.4f} |"
        )

    report_md = "\n".join(markdown_lines)
    (output_dirs["reports"] / "final_report.md").write_text(report_md, encoding="utf-8")

    save_json(output_dirs["reports"] / "deployment_guide.json", deployment)
    deployment_md = [
        "# Deployment Guide",
        "",
        f"Backend: **{backend_summary.get('backend')}**",
        "",
        "## Default Profile (Recommended)",
        (
            f"- Quantization: `{deployment['default_profile'].get('quantization')}`\n"
            f"- Parallel workers: `{deployment['default_profile'].get('parallel_processes')}`\n"
            f"- Batch config: `{deployment['default_profile'].get('batch_config')}`\n"
            f"- Expected metrics: `{deployment['default_profile'].get('expected_metrics')}`"
        ),
        "",
        "## Throughput Profile",
        (
            f"- Quantization: `{deployment['throughput_profile'].get('quantization')}`\n"
            f"- Parallel workers: `{deployment['throughput_profile'].get('parallel_processes')}`\n"
            f"- Batch config: `{deployment['throughput_profile'].get('batch_config')}`\n"
            f"- Expected metrics: `{deployment['throughput_profile'].get('expected_metrics')}`"
        ),
        "",
        "## Notes",
        "- Default profile prioritizes quality for real-time production.",
        "- Throughput profile is useful for high-volume traffic where slight quality loss is acceptable.",
    ]
    (output_dirs["reports"] / "deployment_guide.md").write_text("\n".join(deployment_md), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Combine benchmark metrics into final quantization report.")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    return parser.parse_args()


def main():
    args = parse_args()
    report = generate_comparison_report(output_root=Path(args.output_root))
    print("Comparison report generated.")
    print(f"Recommended quantization: {report.get('best_quantization')}")


if __name__ == "__main__":
    main()
