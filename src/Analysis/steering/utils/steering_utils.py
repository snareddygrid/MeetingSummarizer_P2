"""
Steering helpers for generation-time activation control.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence


ACTION_PATTERN = re.compile(
    r"\b("
    r"will|should|plan|planned|planning|next step|next steps|action item|"
    r"follow up|follow-up|todo|to-do|assign|assigned|owner|deadline|by\s+\w+|"
    r"need to|must|deliver|send|schedule|prepare|confirm|review"
    r")\b",
    flags=re.IGNORECASE,
)

ACTION_VERB_PATTERN = re.compile(
    r"\b(will|should|plan|assign|need to|going to)\b",
    flags=re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"\s+", _normalize_text(text).lower()) if token]


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap <= 0 or pred_total <= 0 or ref_total <= 0:
        return 0.0
    precision = float(overlap / pred_total)
    recall = float(overlap / ref_total)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _rouge_n_single(prediction: str, reference: str, n: int) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0
    pred_counts = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_counts = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum(min(count, ref_counts.get(key, 0)) for key, count in pred_counts.items())
    return _f1(overlap=overlap, pred_total=sum(pred_counts.values()), ref_total=sum(ref_counts.values()))


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for idx, other in enumerate(b, start=1):
            if token == other:
                current.append(prev[idx - 1] + 1)
            else:
                current.append(max(prev[idx], current[-1]))
        prev = current
    return int(prev[-1])


def _rouge_l_single(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    lcs = _lcs_len(pred_tokens, ref_tokens)
    return _f1(overlap=lcs, pred_total=len(pred_tokens), ref_total=len(ref_tokens))


def compute_rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    total = min(len(predictions), len(references))
    if total == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    rouge1 = 0.0
    rouge2 = 0.0
    rouge_l = 0.0
    for idx in range(total):
        pred = predictions[idx]
        ref = references[idx]
        rouge1 += _rouge_n_single(pred, ref, n=1)
        rouge2 += _rouge_n_single(pred, ref, n=2)
        rouge_l += _rouge_l_single(pred, ref)
    return {
        "rouge1": float(rouge1 / total),
        "rouge2": float(rouge2 / total),
        "rougeL": float(rouge_l / total),
    }


def action_clarity_score(summary_text: str) -> float:
    text = _normalize_text(summary_text)
    if not text:
        return 0.0
    matches = ACTION_VERB_PATTERN.findall(text)
    sentence_count = max(1, len(re.split(r"[.!?]+", text)) - 1)
    cue_density = float(len(matches) / sentence_count)
    # Soft-cap to keep score in [0, 1].
    return float(min(1.0, cue_density / 2.0))


def average_action_score(predictions: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    scores = [action_clarity_score(text) for text in predictions]
    return float(sum(scores) / len(scores))


def action_verb_count(summary_text: str) -> int:
    text = _normalize_text(summary_text)
    if not text:
        return 0
    return int(len(ACTION_VERB_PATTERN.findall(text)))


def average_action_verb_count(predictions: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    counts = [action_verb_count(text) for text in predictions]
    return float(sum(counts) / len(counts))


def validate_generated_records(records: Sequence[Dict]) -> None:
    for idx, row in enumerate(records):
        if "sample_id" not in row:
            raise ValueError(f"Missing sample_id at row {idx}")
        if "prediction" not in row:
            raise ValueError(f"Missing prediction at row {idx}")
        if "reference" not in row:
            raise ValueError(f"Missing reference at row {idx}")


def evaluate_records(records: Sequence[Dict], sample_limit: int = 50) -> Dict:
    if sample_limit > 0:
        rows = list(records[:sample_limit])
    else:
        rows = list(records)
    validate_generated_records(rows)
    predictions = [_normalize_text(row["prediction"]) for row in rows]
    references = [_normalize_text(row["reference"]) for row in rows]
    rouge = compute_rouge_scores(predictions=predictions, references=references)
    action_score = average_action_score(predictions=predictions)
    action_verbs_mean = average_action_verb_count(predictions=predictions)
    return {
        "num_samples": len(rows),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "action_score": action_score,
        "action_verb_count_mean": action_verbs_mean,
        "manual_action_score": None,
    }


def _coerce_direction_tensor(direction_tensor, reference_tensor):
    import torch

    return direction_tensor.to(device=reference_tensor.device, dtype=reference_tensor.dtype).view(1, 1, -1)


@contextmanager
def apply_steering_hooks(
    model,
    direction_by_layer: Dict[int, "torch.Tensor"],
    scale: float,
    target_layers: Optional[Sequence[int]] = None,
):
    """
    Temporarily inject additive steering vectors into decoder block outputs.
    """

    active_layers = [int(layer) for layer in (target_layers or direction_by_layer.keys()) if int(layer) in direction_by_layer]
    handles = []
    decoder_depth = int(len(model.decoder.block))

    def make_hook(layer_id: int):
        direction = direction_by_layer[int(layer_id)]

        def hook(_module, _inputs, output):
            if scale == 0.0:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                steer = _coerce_direction_tensor(direction, hidden) * float(scale)
                steered = hidden + steer
                return (steered, *output[1:])
            steer = _coerce_direction_tensor(direction, output) * float(scale)
            return output + steer

        return hook

    for layer in active_layers:
        layer_id = int(layer)
        if layer_id == decoder_depth:
            # For T5 decoder_hidden_states indexing, the final index corresponds to
            # decoder.final_layer_norm output, not the raw last block output.
            handle = model.decoder.final_layer_norm.register_forward_hook(make_hook(layer_id))
            handles.append(handle)
            continue

        block_idx = layer_id - 1
        if block_idx < 0 or block_idx >= len(model.decoder.block):
            continue
        handle = model.decoder.block[block_idx].register_forward_hook(make_hook(layer_id))
        handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_summary_with_steering(
    model,
    tokenizer,
    sample: Dict,
    direction_by_layer: Dict[int, "torch.Tensor"],
    scale: float,
    device,
    generation_kwargs: Optional[Dict] = None,
    target_layers: Optional[Sequence[int]] = None,
    input_prefix: Optional[str] = None,
    max_source_tokens: int = 512,
) -> str:
    import torch

    config = {
        "num_beams": 4,
        "max_new_tokens": 140,
        "min_new_tokens": 16,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    if generation_kwargs:
        config.update(generation_kwargs)

    if input_prefix:
        source_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True).strip()
        conditioned = tokenizer(
            f"{input_prefix}{source_text}",
            return_tensors="pt",
            truncation=True,
            max_length=int(max_source_tokens),
        )
        input_ids = conditioned["input_ids"].to(device)
        attention_mask = conditioned["attention_mask"].to(device)
    else:
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long, device=device)

    with torch.no_grad():
        with apply_steering_hooks(
            model=model,
            direction_by_layer=direction_by_layer,
            scale=float(scale),
            target_layers=target_layers,
        ):
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **config,
            )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return summary


def rouge_drop_percent(candidate_rouge_l: float, baseline_rouge_l: float) -> float:
    if baseline_rouge_l <= 0:
        return 0.0
    return float(max(0.0, (baseline_rouge_l - candidate_rouge_l) / baseline_rouge_l * 100.0))


def pick_best_scale(
    results_by_scale: Dict[str, Dict],
    rouge_drop_limit_pct: float = 2.0,
    action_metric_key: str = "action_score",
) -> Dict:
    baseline = results_by_scale.get("0", results_by_scale.get("0.0"))
    baseline_rouge = float(baseline["rougeL"]) if baseline else 0.0

    candidates = []
    for scale_key, metrics in results_by_scale.items():
        rouge_l = float(metrics.get("rougeL", 0.0))
        action_value = metrics.get(action_metric_key, None)
        if action_value is None:
            action_value = metrics.get("action_score", 0.0)
        action = float(action_value)
        drop = rouge_drop_percent(rouge_l, baseline_rouge)
        candidates.append(
            {
                "scale": scale_key,
                "rougeL": rouge_l,
                "action_score": action,
                "action_metric_key": action_metric_key,
                "rouge_drop_pct": drop,
                "eligible": drop < float(rouge_drop_limit_pct),
            }
        )

    eligible = [row for row in candidates if row["eligible"]]
    if eligible:
        best = max(eligible, key=lambda row: row["action_score"])
    else:
        best = max(candidates, key=lambda row: row["action_score"]) if candidates else {
            "scale": "0",
            "rougeL": 0.0,
            "action_score": 0.0,
            "rouge_drop_pct": 0.0,
            "eligible": False,
        }

    return {
        "best": best,
        "baseline_rougeL": baseline_rouge,
        "candidates": candidates,
        "rouge_drop_limit_pct": float(rouge_drop_limit_pct),
    }
