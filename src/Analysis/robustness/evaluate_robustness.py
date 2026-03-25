"""Evaluate robustness on original vs adversarial predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from utils.evaluation_utils import aggregate_prediction_metrics, ensure_dir, load_json, resolve_path, save_json


def _load_prediction_records(path: Path):
    payload = load_json(path)
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported prediction JSON format: {path.as_posix()}")


def evaluate_pair(predictions_dir: Path, prefix: str, output_path: Path) -> Optional[Dict]:
    original_path = predictions_dir / f"{prefix}original.json"
    adversarial_path = predictions_dir / f"{prefix}adversarial.json"
    if not original_path.exists() or not adversarial_path.exists():
        return None

    original_records = _load_prediction_records(original_path)
    adversarial_records = _load_prediction_records(adversarial_path)

    original_metrics = aggregate_prediction_metrics(original_records)
    adversarial_metrics = aggregate_prediction_metrics(adversarial_records)

    summary = {
        "prefix": prefix,
        "original": original_metrics,
        "adversarial": adversarial_metrics,
        "degradation": {
            "rougeL_drop": float(original_metrics["rougeL"] - adversarial_metrics["rougeL"]),
            "coherence_drop": float(original_metrics["coherence"] - adversarial_metrics["coherence"]),
            "action_drop": float(original_metrics["action_completeness"] - adversarial_metrics["action_completeness"]),
            "action_verb_drop": float(original_metrics["action_verb_mean"] - adversarial_metrics["action_verb_mean"]),
        },
    }

    save_json(output_path, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness metrics for prediction files.")
    parser.add_argument("--predictions-dir", default="outputs/analysis/robustness/predictions")
    parser.add_argument("--evaluations-dir", default="outputs/analysis/robustness/evaluations")
    parser.add_argument("--mode", choices=["all", "pre", "post"], default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_dir = resolve_path(args.predictions_dir)
    evaluations_dir = ensure_dir(resolve_path(args.evaluations_dir))

    wrote_any = False

    if args.mode in {"all", "pre"}:
        summary = evaluate_pair(
            predictions_dir=predictions_dir,
            prefix="",
            output_path=evaluations_dir / "pre_training.json",
        )
        if summary is not None:
            wrote_any = True
            print(f"Saved: {(evaluations_dir / 'pre_training.json').as_posix()}")

    if args.mode in {"all", "post"}:
        summary = evaluate_pair(
            predictions_dir=predictions_dir,
            prefix="post_",
            output_path=evaluations_dir / "post_training.json",
        )
        if summary is not None:
            wrote_any = True
            print(f"Saved: {(evaluations_dir / 'post_training.json').as_posix()}")

    if not wrote_any:
        raise FileNotFoundError(
            "No matching prediction files found. Expected one of: "
            "original/adversarial or post_original/post_adversarial."
        )


if __name__ == "__main__":
    main()
