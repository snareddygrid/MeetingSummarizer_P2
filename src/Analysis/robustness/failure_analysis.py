"""Analyze robustness failure modes between clean and adversarial predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from utils.evaluation_utils import (
    coherence_score,
    ensure_dir,
    entity_recall,
    hallucination_ratio,
    load_json,
    resolve_path,
    save_json,
    token_overlap_ratio,
)


PERTURBATION_KEYS = ["noise", "overlap", "length", "off_topic"]


def _load_prediction_records(path: Path) -> List[Dict]:
    payload = load_json(path)
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported prediction file format: {path.as_posix()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run failure mode analysis for robustness outputs.")
    parser.add_argument("--predictions-dir", default="outputs/analysis/robustness/predictions")
    parser.add_argument("--output-path", default="outputs/analysis/robustness/failure_analysis/failures.json")
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_dir = resolve_path(args.predictions_dir)

    original_records = _load_prediction_records(predictions_dir / "original.json")
    adversarial_records = _load_prediction_records(predictions_dir / "adversarial.json")

    original_by_id = {str(row["id"]): row for row in original_records}
    adversarial_by_id = {str(row["id"]): row for row in adversarial_records}

    failures: List[Dict] = []
    failure_type_counts = {
        "missing_key_info": 0,
        "wrong_entities": 0,
        "broken_grammar": 0,
        "hallucinations": 0,
    }
    failure_by_perturbation = {f"{key}_failure": 0 for key in PERTURBATION_KEYS}

    for sample_id, adv_row in adversarial_by_id.items():
        if sample_id not in original_by_id:
            continue
        orig_row = original_by_id[sample_id]

        reference = str(orig_row.get("reference", ""))
        original_prediction = str(orig_row.get("prediction", ""))
        adversarial_prediction = str(adv_row.get("prediction", ""))
        adversarial_input = str(adv_row.get("input", ""))
        perturbations = list(adv_row.get("perturbations", []))

        orig_overlap = token_overlap_ratio(original_prediction, reference)
        adv_overlap = token_overlap_ratio(adversarial_prediction, reference)

        orig_entity = entity_recall(reference, original_prediction)
        adv_entity = entity_recall(reference, adversarial_prediction)

        orig_coherence = coherence_score(original_prediction)
        adv_coherence = coherence_score(adversarial_prediction)

        orig_halluc = hallucination_ratio(original_prediction, str(orig_row.get("input", "")))
        adv_halluc = hallucination_ratio(adversarial_prediction, adversarial_input)

        sample_failure_types: List[str] = []
        if adv_overlap + 0.08 < orig_overlap:
            sample_failure_types.append("missing_key_info")
        if adv_entity + 0.20 < orig_entity:
            sample_failure_types.append("wrong_entities")
        if adv_coherence + 0.6 < orig_coherence and adv_coherence <= 3.0:
            sample_failure_types.append("broken_grammar")
        if adv_halluc > 0.60 and adv_halluc > orig_halluc + 0.15:
            sample_failure_types.append("hallucinations")

        if not sample_failure_types:
            continue

        for failure_type in sample_failure_types:
            failure_type_counts[failure_type] += 1
        for tag in perturbations:
            key = f"{tag}_failure"
            if key in failure_by_perturbation:
                failure_by_perturbation[key] += 1

        failures.append(
            {
                "id": sample_id,
                "perturbations": perturbations,
                "failure_types": sample_failure_types,
                "reference": reference,
                "original_prediction": original_prediction,
                "adversarial_prediction": adversarial_prediction,
                "scores": {
                    "overlap_original": orig_overlap,
                    "overlap_adversarial": adv_overlap,
                    "entity_original": orig_entity,
                    "entity_adversarial": adv_entity,
                    "coherence_original": orig_coherence,
                    "coherence_adversarial": adv_coherence,
                    "hallucination_original": orig_halluc,
                    "hallucination_adversarial": adv_halluc,
                },
            }
        )

    output_path = resolve_path(args.output_path)
    ensure_dir(output_path.parent)
    classified = {f"{key}_failure": int(failure_by_perturbation[f"{key}_failure"]) for key in PERTURBATION_KEYS}
    payload = {
        "num_samples": int(len(adversarial_by_id)),
        "num_failures": int(len(failures)),
        "failure_type_counts": failure_type_counts,
        "failure_by_perturbation": failure_by_perturbation,
        "classification": classified,
        **classified,
        "failures": failures,
    }
    save_json(output_path, payload)

    print(f"Saved failure analysis: {output_path.as_posix()}")
    print(f"Failure count: {len(failures)}")


if __name__ == "__main__":
    main()
