"""Compare pre- vs post-training robustness evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from utils.evaluation_utils import ensure_dir, load_json, resolve_path, save_json


def _load_eval(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path.as_posix()}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported evaluation format: {path.as_posix()}")
    return payload


def _metric_block(payload: Dict, split: str) -> Dict:
    block = payload.get(split)
    if not isinstance(block, dict):
        raise ValueError(f"Expected '{split}' metrics in evaluation payload.")
    return block


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare robustness metrics before and after adversarial training.")
    parser.add_argument("--pre-eval", default="outputs/analysis/robustness/evaluations/pre_training.json")
    parser.add_argument("--post-eval", default="outputs/analysis/robustness/evaluations/post_training.json")
    parser.add_argument("--output-path", default="outputs/analysis/robustness/reports/final_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    pre = _load_eval(resolve_path(args.pre_eval))
    post = _load_eval(resolve_path(args.post_eval))

    pre_adv = _metric_block(pre, "adversarial")
    post_adv = _metric_block(post, "adversarial")
    pre_orig = _metric_block(pre, "original")
    post_orig = _metric_block(post, "original")

    rouge_gain = float(post_adv.get("rougeL", 0.0) - pre_adv.get("rougeL", 0.0))
    coherence_gain = float(post_adv.get("coherence", 0.0) - pre_adv.get("coherence", 0.0))
    action_gain = float(post_adv.get("action_completeness", 0.0) - pre_adv.get("action_completeness", 0.0))

    clean_rouge_shift = float(post_orig.get("rougeL", 0.0) - pre_orig.get("rougeL", 0.0))
    clean_coherence_shift = float(post_orig.get("coherence", 0.0) - pre_orig.get("coherence", 0.0))
    clean_action_shift = float(post_orig.get("action_completeness", 0.0) - pre_orig.get("action_completeness", 0.0))

    if rouge_gain > 0 and coherence_gain >= 0 and action_gain >= 0:
        summary = "Adversarial training improved robustness"
    elif rouge_gain > 0 or coherence_gain > 0 or action_gain > 0:
        summary = "Adversarial training gave mixed robustness gains."
    else:
        summary = "Adversarial training did not improve robustness."

    report = {
        "rouge_gain": _round(rouge_gain),
        "coherence_gain": _round(coherence_gain),
        "action_gain": _round(action_gain),
        "summary": summary,
        "pre_training": {
            "original": pre_orig,
            "adversarial": pre_adv,
        },
        "post_training": {
            "original": post_orig,
            "adversarial": post_adv,
        },
        "clean_set_shift": {
            "rouge_shift": _round(clean_rouge_shift),
            "coherence_shift": _round(clean_coherence_shift),
            "action_shift": _round(clean_action_shift),
        },
    }

    output_path = resolve_path(args.output_path)
    ensure_dir(output_path.parent)
    save_json(output_path, report)
    print(f"Saved final robustness report: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
