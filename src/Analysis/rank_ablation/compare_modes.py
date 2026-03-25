"""Compare structured vs free-form generation and build final rank recommendation."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk
from evaluate import load as load_metric
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from utils.json_utils import (  # noqa: E402
    as_records,
    load_json,
    rank_key,
    rank_value,
    safe_json_loads,
    save_json,
)
from utils.timing_utils import set_seed  # noqa: E402


DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare free-form and structured generation by rank.")
    parser.add_argument("--raw-data-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument(
        "--structured-dir",
        default="outputs/analysis/rank_ablation/structured_outputs",
    )
    parser.add_argument(
        "--freeform-dir",
        default="outputs/analysis/rank_ablation/comparisons/free_form",
    )
    parser.add_argument(
        "--comparison-output",
        default="outputs/analysis/rank_ablation/comparisons/comparison.json",
    )
    parser.add_argument(
        "--metrics-path",
        default="outputs/analysis/rank_ablation/metrics/rouge_scores.json",
    )
    parser.add_argument(
        "--validity-path",
        default="outputs/analysis/rank_ablation/validity/validity.json",
    )
    parser.add_argument(
        "--latency-path",
        default="outputs/analysis/rank_ablation/latency/latency.json",
    )
    parser.add_argument(
        "--size-path",
        default="outputs/analysis/rank_ablation/model_size/size.json",
    )
    parser.add_argument(
        "--final-report-path",
        default="outputs/analysis/rank_ablation/final_report.json",
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--force-regenerate-freeform", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _num_samples(dataset, max_samples: int) -> int:
    if max_samples is None or max_samples < 0:
        return len(dataset)
    return min(int(max_samples), len(dataset))


def _length_variance(predictions: List[str]) -> float:
    if not predictions:
        return 0.0
    lengths = [len(str(pred).split()) for pred in predictions]
    if len(lengths) <= 1:
        return 0.0
    return float(statistics.pvariance(lengths))


def _json_validity_rate(predictions: List[str]) -> float:
    if not predictions:
        return 0.0
    valid = 0
    for pred in predictions:
        ok, _ = safe_json_loads(str(pred))
        if ok:
            valid += 1
    return float(valid / len(predictions))


def _rouge_l(predictions: List[str], references: List[str]) -> float:
    if not predictions or not references:
        return 0.0
    n = min(len(predictions), len(references))
    rouge = load_metric("rouge")
    result = rouge.compute(
        predictions=[str(x) for x in predictions[:n]],
        references=[str(x) for x in references[:n]],
        use_stemmer=True,
    )
    return float(result["rougeL"])


def _extract_predictions_and_refs(records: List[Dict]) -> tuple[list[str], list[str]]:
    preds = [str(row.get("prediction", "")) for row in records]
    refs = [str(row.get("reference", "")) for row in records]
    return preds, refs


def _generate_freeform_outputs(
    rank: int,
    dataset,
    sample_count: int,
    args,
    output_path: Path,
) -> List[Dict]:
    # Lazy imports keep this script usable even in environments that cannot import torch,
    # as long as free-form outputs are already present and regeneration is not requested.
    import sys

    SRC_ROOT = PROJECT_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from inference import generate_summary
    from evaluate_ranks import load_rank_model

    model_dir = (PROJECT_ROOT / args.experiments_root / f"t5_small_lora_r{rank}").resolve()
    if not model_dir.exists():
        return []

    model, tokenizer, device = load_rank_model(
        model_dir=model_dir,
        default_base_model=args.default_base_model,
    )

    records = []
    for idx in tqdm(range(sample_count), desc=f"Free-form r{rank}", leave=False):
        row = dataset[idx]
        prediction = generate_summary(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=str(row["dialogue"]),
            architecture="t5-small",
        )
        records.append(
            {
                "id": row.get("id", idx),
                "reference": str(row.get("summary", "")),
                "prediction": prediction,
            }
        )

    save_json(
        output_path,
        {
            "rank": rank_key(rank),
            "mode": "free_form",
            "num_samples": int(sample_count),
            "records": records,
        },
    )
    return records


def _read_mode_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return as_records(load_json(path))


def _safe_metric(entry: Dict, key: str, default: float = 0.0) -> float:
    value = entry.get(key, default)
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _build_final_report(
    ranks: List[int],
    metrics_payload: Dict,
    validity_payload: Dict,
    latency_payload: Dict,
    size_payload: Dict,
    comparison_payload: Dict,
) -> Dict:
    rank_keys = [rank_key(rank) for rank in sorted(ranks)]
    rouge_by_rank = {
        key: _safe_metric(metrics_payload.get(key, {}), "rougeL")
        for key in rank_keys
    }
    validity_by_rank = {
        key: _safe_metric(validity_payload.get(key, {}), "validity_rate")
        for key in rank_keys
    }

    best_rouge_rank = max(rank_keys, key=lambda key: rouge_by_rank.get(key, 0.0))
    best_rouge = rouge_by_rank.get(best_rouge_rank, 0.0)
    rouge_threshold = 0.99 * best_rouge if best_rouge > 0 else 0.0

    eligible = []
    for key in rank_keys:
        if rouge_by_rank.get(key, 0.0) >= rouge_threshold and validity_by_rank.get(key, 0.0) >= 0.95:
            eligible.append(key)

    selected_key = eligible[0] if eligible else best_rouge_rank
    selected_rank = rank_value(selected_key)
    selected_latency = latency_payload.get(selected_key, {})
    selected_size = size_payload.get(selected_key, {})

    min_rank_key = rank_key(min(ranks))
    max_rank_key = rank_key(max(ranks))
    low_rouge = rouge_by_rank.get(min_rank_key, 0.0)
    high_rouge = rouge_by_rank.get(max_rank_key, 0.0)
    low_latency = _safe_metric(latency_payload.get(min_rank_key, {}), "avg_latency_sec")
    high_latency = _safe_metric(latency_payload.get(max_rank_key, {}), "avg_latency_sec")
    low_size = _safe_metric(size_payload.get(min_rank_key, {}), "size_mb")
    high_size = _safe_metric(size_payload.get(max_rank_key, {}), "size_mb")

    structured_vs_free = comparison_payload.get(selected_key, {}).get("delta", {})
    delta_rouge = _safe_metric(structured_vs_free, "rougeL")

    if eligible:
        conclusion = (
            f"Rank {selected_rank} provides optimal balance between performance and efficiency "
            f"while meeting JSON validity >= 95%."
        )
    else:
        conclusion = (
            f"No rank met both constraints; selected rank {selected_rank} because it delivered the highest ROUGE-L."
        )

    tradeoff_analysis = (
        f"Across ranks {min(ranks)}->{max(ranks)}, ROUGE-L changed from {low_rouge:.4f} to {high_rouge:.4f}, "
        f"average latency changed from {low_latency:.4f}s to {high_latency:.4f}s, and model size changed from "
        f"{low_size:.2f}MB to {high_size:.2f}MB. Structured decoding delta at selected rank is {delta_rouge:.4f} ROUGE-L."
    )

    return {
        "best_rank": int(selected_rank),
        "best_rougeL": float(rouge_by_rank.get(selected_key, 0.0)),
        "json_validity": float(validity_by_rank.get(selected_key, 0.0)),
        "latency": {
            "avg_latency_sec": _safe_metric(selected_latency, "avg_latency_sec"),
            "p95_latency_sec": _safe_metric(selected_latency, "p95_latency_sec"),
        },
        "model_size_mb": _safe_metric(selected_size, "size_mb"),
        "tradeoff_analysis": tradeoff_analysis,
        "conclusion": conclusion,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    structured_dir = (PROJECT_ROOT / args.structured_dir).resolve()
    freeform_dir = (PROJECT_ROOT / args.freeform_dir).resolve()
    freeform_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk((PROJECT_ROOT / args.raw_data_path).as_posix())[args.split]
    sample_count = _num_samples(dataset, args.max_samples)
    ranks = sorted(args.ranks)
    comparison_payload = {}

    for rank in tqdm(ranks, desc="Comparing modes by rank"):
        key = rank_key(rank)
        structured_path = structured_dir / f"{key}.json"
        freeform_path = freeform_dir / f"{key}.json"

        structured_records = _read_mode_records(structured_path)
        if args.force_regenerate_freeform or not freeform_path.exists():
            freeform_records = _generate_freeform_outputs(
                rank=rank,
                dataset=dataset,
                sample_count=sample_count,
                args=args,
                output_path=freeform_path,
            )
        else:
            freeform_records = _read_mode_records(freeform_path)

        if not structured_records or not freeform_records:
            comparison_payload[key] = {"status": "missing_mode_outputs"}
            continue

        structured_preds, structured_refs = _extract_predictions_and_refs(structured_records)
        free_preds, free_refs = _extract_predictions_and_refs(freeform_records)
        n = min(len(structured_preds), len(free_preds), len(structured_refs), len(free_refs))
        structured_preds = structured_preds[:n]
        structured_refs = structured_refs[:n]
        free_preds = free_preds[:n]
        free_refs = free_refs[:n]

        structured_rouge_l = _rouge_l(structured_preds, structured_refs)
        free_rouge_l = _rouge_l(free_preds, free_refs)
        structured_validity = _json_validity_rate(structured_preds)
        free_validity = _json_validity_rate(free_preds)
        structured_var = _length_variance(structured_preds)
        free_var = _length_variance(free_preds)

        comparison_payload[key] = {
            "num_samples": int(n),
            "free_form": {
                "rougeL": float(free_rouge_l),
                "json_validity": float(free_validity),
                "length_variance": float(free_var),
            },
            "structured": {
                "rougeL": float(structured_rouge_l),
                "json_validity": float(structured_validity),
                "length_variance": float(structured_var),
            },
            "delta": {
                "rougeL": float(structured_rouge_l - free_rouge_l),
                "json_validity": float(structured_validity - free_validity),
                "length_variance": float(structured_var - free_var),
            },
            "status": "ok",
        }

    comparison_output = (PROJECT_ROOT / args.comparison_output).resolve()
    save_json(comparison_output, comparison_payload)
    print(f"Saved comparison metrics: {comparison_output.as_posix()}")

    metrics_payload = load_json((PROJECT_ROOT / args.metrics_path).resolve())
    validity_payload = load_json((PROJECT_ROOT / args.validity_path).resolve())
    latency_payload = load_json((PROJECT_ROOT / args.latency_path).resolve())
    size_payload = load_json((PROJECT_ROOT / args.size_path).resolve())

    report = _build_final_report(
        ranks=ranks,
        metrics_payload=metrics_payload,
        validity_payload=validity_payload,
        latency_payload=latency_payload,
        size_payload=size_payload,
        comparison_payload=comparison_payload,
    )
    final_report_path = (PROJECT_ROOT / args.final_report_path).resolve()
    save_json(final_report_path, report)
    print(f"Saved final report: {final_report_path.as_posix()}")


if __name__ == "__main__":
    main()
