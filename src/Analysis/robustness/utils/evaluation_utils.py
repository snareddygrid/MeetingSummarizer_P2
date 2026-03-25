"""Evaluation and model-loading helpers for robustness analysis."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np


ACTION_TERMS = ["will", "should", "plan", "need", "going to"]
_TOKEN_RE = re.compile(r"[A-Za-z']+")
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass


def load_data_rows(path: Path) -> List[Dict]:
    payload = load_json(path)
    if isinstance(payload, dict) and "samples" in payload:
        return list(payload["samples"])
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON format in {path.as_posix()}")


def _read_adapter_base_model(model_dir: Path, default_base_model: str) -> str:
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.exists():
        return default_base_model
    payload = load_json(adapter_config)
    return str(payload.get("base_model_name_or_path", default_base_model))


def _resolve_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_summarization_model(
    model_dir: Path,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
):
    """Load either a full seq2seq checkpoint or a LoRA adapter checkpoint."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir = Path(model_dir)
    device = _resolve_device()

    def _load_tokenizer(candidates: Sequence[str]):
        tokenizer = None
        for candidate in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True, local_files_only=True)
                break
            except Exception:  # noqa: BLE001
                continue
        if tokenizer is None:
            for candidate in candidates:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if tokenizer is None:
            raise RuntimeError(f"Unable to load tokenizer from candidates: {candidates}")
        return tokenizer

    model_dir = Path(model_dir)
    resolved_model_dir = resolve_path(model_dir)
    fallback_base_path = resolve_path(fallback_local_base_model)
    adapter_dir = resolved_model_dir if (resolved_model_dir / "adapter_config.json").exists() else model_dir
    non_adapter_source = resolved_model_dir.as_posix() if resolved_model_dir.exists() else str(model_dir)

    if (adapter_dir / "adapter_config.json").exists():
        from peft import PeftModel

        base_from_adapter = _read_adapter_base_model(model_dir=adapter_dir, default_base_model=default_base_model)
        model_candidates = [base_from_adapter, default_base_model, fallback_base_path.as_posix()]
        base_model = None
        for candidate in model_candidates:
            try:
                base_model = AutoModelForSeq2SeqLM.from_pretrained(candidate, local_files_only=True)
                break
            except Exception:  # noqa: BLE001
                continue
        if base_model is None:
            for candidate in model_candidates:
                try:
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(candidate)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if base_model is None:
            raise RuntimeError(f"Unable to load base model for adapter: {model_candidates}")
        model = PeftModel.from_pretrained(base_model, adapter_dir.as_posix())
        tokenizer = _load_tokenizer([adapter_dir.as_posix(), *model_candidates])
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(non_adapter_source, local_files_only=True)
        except Exception:  # noqa: BLE001
            model = AutoModelForSeq2SeqLM.from_pretrained(non_adapter_source)
        tokenizer = _load_tokenizer([non_adapter_source, default_base_model, fallback_base_path.as_posix()])

    model.to(device)
    model.eval()
    return model, tokenizer, device


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def tokenize_words(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(_normalize_text(text))]


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap <= 0 or pred_total <= 0 or ref_total <= 0:
        return 0.0
    p = overlap / pred_total
    r = overlap / ref_total
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok in a:
        cur = [0]
        for j, other in enumerate(b, start=1):
            if tok == other:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return int(prev[-1])


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """ROUGE-L F1 average with evaluate fallback."""
    predictions = list(predictions)
    references = list(references)
    total = min(len(predictions), len(references))
    if total == 0:
        return 0.0

    try:
        import evaluate

        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=predictions[:total], references=references[:total], use_stemmer=True)
        return float(result["rougeL"])
    except Exception:  # noqa: BLE001
        scores = []
        for pred, ref in zip(predictions[:total], references[:total]):
            p_tokens = tokenize_words(pred)
            r_tokens = tokenize_words(ref)
            lcs = _lcs_len(p_tokens, r_tokens)
            scores.append(_f1(lcs, len(p_tokens), len(r_tokens)))
        return float(sum(scores) / len(scores)) if scores else 0.0


def action_verb_count(text: str) -> int:
    lowered = _normalize_text(text).lower()
    return sum(1 for term in ACTION_TERMS if term in lowered)


def action_completeness_score(summary: str) -> int:
    hits = action_verb_count(summary)
    if hits == 0:
        return 1
    if hits <= 2:
        return 3
    return 5


def coherence_score(summary: str) -> float:
    text = _normalize_text(summary)
    if not text:
        return 1.0

    tokens = tokenize_words(text)
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    repeated_ratio = 1.0 - unique_ratio

    score = 5.0
    if repeated_ratio > 0.45:
        score -= 2.0
    elif repeated_ratio > 0.30:
        score -= 1.0

    if len(tokens) < 8:
        score -= 1.0

    if not re.search(r"[.!?]$", text):
        score -= 0.5

    grammar_like_noise = len(re.findall(r"\b\w{1,2}\b", text)) / max(1, len(tokens))
    if grammar_like_noise > 0.35:
        score -= 0.5

    return float(max(1.0, min(5.0, score)))


def aggregate_prediction_metrics(records: Sequence[Dict]) -> Dict[str, float]:
    rows = list(records)
    predictions = [_normalize_text(row.get("prediction", "")) for row in rows]
    references = [_normalize_text(row.get("reference", "")) for row in rows]

    rouge_l = compute_rouge_l(predictions=predictions, references=references)
    coherence = [coherence_score(pred) for pred in predictions]
    action = [action_completeness_score(pred) for pred in predictions]
    verbs = [action_verb_count(pred) for pred in predictions]

    return {
        "num_samples": int(len(rows)),
        "rougeL": float(rouge_l),
        "coherence": float(sum(coherence) / max(1, len(coherence))),
        "action_completeness": float(sum(action) / max(1, len(action))),
        "action_verb_mean": float(sum(verbs) / max(1, len(verbs))),
    }


def token_overlap_ratio(a: str, b: str) -> float:
    a_set = set(tokenize_words(a))
    b_set = set(tokenize_words(b))
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return float(len(a_set & b_set) / max(1, len(a_set | b_set)))


def extract_entities(text: str) -> List[str]:
    return re.findall(r"\b[A-Z][a-z]+\b", str(text or ""))


def entity_recall(reference: str, prediction: str) -> float:
    ref_entities = {ent.lower() for ent in extract_entities(reference)}
    pred_entities = {ent.lower() for ent in extract_entities(prediction)}
    if not ref_entities:
        return 1.0
    return float(len(ref_entities & pred_entities) / len(ref_entities))


def hallucination_ratio(prediction: str, source_text: str) -> float:
    pred = set(tokenize_words(prediction))
    src = set(tokenize_words(source_text))
    if not pred:
        return 0.0
    unseen = pred - src
    return float(len(unseen) / len(pred))
