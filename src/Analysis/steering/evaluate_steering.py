"""
Evaluate steered summaries across scales.

Run:
    python src/analysis/steering/evaluate_steering.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from utils.activation_utils import ensure_dir, load_json, save_json
from utils.steering_utils import evaluate_records, pick_best_scale, rouge_drop_percent


def _load_manual_ratings(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}

    payload = load_json(path)
    by_scale: Dict[str, Dict[str, float]] = {}

    if isinstance(payload, dict):
        root = payload.get("scales", payload)
        for scale_key, value in root.items():
            if not isinstance(value, (dict, list)):
                continue
            scale_ratings: Dict[str, float] = {}
            if isinstance(value, dict):
                for sample_id, score in value.items():
                    try:
                        scale_ratings[str(sample_id)] = float(score)
                    except Exception:  # noqa: BLE001
                        continue
            else:
                for row in value:
                    if not isinstance(row, dict):
                        continue
                    sample_id = row.get("sample_id", row.get("id"))
                    score = row.get("manual_action_score", row.get("score"))
                    if sample_id is None or score is None:
                        continue
                    try:
                        scale_ratings[str(sample_id)] = float(score)
                    except Exception:  # noqa: BLE001
                        continue
            if scale_ratings:
                by_scale[str(scale_key)] = scale_ratings

    return by_scale


def _manual_action_score(records, scale_ratings: Dict[str, float], sample_limit: int) -> tuple[Optional[float], int]:
    if sample_limit > 0:
        rows = list(records[:sample_limit])
    else:
        rows = list(records)

    scores = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in scale_ratings:
            scores.append(float(scale_ratings[sample_id]))
    if not scores:
        return None, 0
    return float(sum(scores) / len(scores)), int(len(scores))


def run_evaluation(args) -> Dict:
    output_root = Path(args.output_root)
    generated_dir = output_root / "generated"
    evaluation_dir = ensure_dir(output_root / "evaluations")
    manual_ratings = _load_manual_ratings(
        Path(args.manual_ratings_path) if args.manual_ratings_path else None
    )

    results_by_scale: Dict[str, Dict] = {}
    has_manual_scores = False
    for path in sorted(generated_dir.glob("*.json")):
        scale_key = path.stem
        payload = load_json(path)
        records = payload.get("records", [])
        metrics = evaluate_records(records=records, sample_limit=args.eval_samples)
        manual_score, rated_count = _manual_action_score(
            records=records,
            scale_ratings=manual_ratings.get(scale_key, {}),
            sample_limit=args.eval_samples,
        )
        metrics["manual_action_score"] = manual_score
        metrics["manual_rated_samples"] = int(rated_count)
        if manual_score is not None:
            has_manual_scores = True
        metrics["scale"] = scale_key
        results_by_scale[scale_key] = metrics

    if not results_by_scale:
        raise FileNotFoundError(
            f"No generated scale files found in {generated_dir.as_posix()}. Run steering_generate.py first."
        )

    baseline = results_by_scale.get("0.0", results_by_scale.get("0"))
    baseline_rouge = float(baseline.get("rougeL", 0.0)) if baseline else 0.0
    baseline_action_verbs = float(baseline.get("action_verb_count_mean", 0.0)) if baseline else 0.0
    for metrics in results_by_scale.values():
        metrics["rouge_drop_pct_vs_baseline"] = rouge_drop_percent(
            candidate_rouge_l=float(metrics.get("rougeL", 0.0)),
            baseline_rouge_l=baseline_rouge,
        )
        metrics["action_verb_gain_vs_baseline"] = float(metrics.get("action_verb_count_mean", 0.0)) - baseline_action_verbs

    action_metric_key = "manual_action_score" if has_manual_scores else "action_score"
    selection = pick_best_scale(
        results_by_scale=results_by_scale,
        rouge_drop_limit_pct=args.rouge_drop_limit_pct,
        action_metric_key=action_metric_key,
    )
    final = {
        "eval_samples": int(args.eval_samples),
        "rouge_drop_limit_pct": float(args.rouge_drop_limit_pct),
        "manual_ratings_path": Path(args.manual_ratings_path).as_posix() if args.manual_ratings_path else None,
        "manual_ratings_used": bool(has_manual_scores),
        "action_metric_used_for_selection": action_metric_key,
        "results_by_scale": results_by_scale,
        "selection": selection,
        "best_scale": selection["best"]["scale"],
        "best_action_score": float(selection["best"]["action_score"]),
        "best_rougeL": float(selection["best"]["rougeL"]),
        "best_rouge_drop_pct": float(selection["best"]["rouge_drop_pct"]),
    }
    save_json(evaluation_dir / "results.json", final)
    return final


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate steering output quality and action clarity.")
    parser.add_argument("--output-root", default="outputs/analysis/steering")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--rouge-drop-limit-pct", type=float, default=2.0)
    parser.add_argument(
        "--manual-ratings-path",
        default="outputs/analysis/steering/evaluations/manual_ratings.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_evaluation(args)
    print("Steering evaluation complete.")
    print(f"Best scale: {result['best_scale']}")
    print(f"Best action score: {result['best_action_score']:.4f}")


if __name__ == "__main__":
    main()
