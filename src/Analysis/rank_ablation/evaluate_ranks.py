"""Evaluate ROUGE-L for LoRA rank ablation checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from datasets import load_from_disk
from evaluate import load as load_metric
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_loader import load_model_registry  # noqa: E402

from utils.json_utils import rank_key, rank_value, save_json  # noqa: E402
from utils.timing_utils import resolve_device, set_seed  # noqa: E402


DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA rank checkpoints with ROUGE.")
    parser.add_argument("--processed-data-path", default="data/processed")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument(
        "--output-path",
        default="outputs/analysis/rank_ablation/metrics/rouge_scores.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_fallback_base(default_base_model: str) -> str:
    try:
        registry = load_model_registry()
    except Exception:  # noqa: BLE001
        return default_base_model

    for key, value in registry.items():
        architecture = str(value.get("architecture", "")).lower()
        if key.upper() == "T5-SMALL":
            return value.get("path", default_base_model)
        if "t5" in architecture:
            return value.get("path", default_base_model)
    return default_base_model


def _load_first_model(candidates):
    last_error = None
    for candidate in candidates:
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(candidate, local_files_only=True)
        except Exception as error:  # noqa: BLE001
            last_error = error
    for candidate in candidates:
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(candidate)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(f"Unable to load model from candidates={candidates}; last_error={last_error}")


def _load_first_tokenizer(candidates):
    last_error = None
    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, use_fast=True, local_files_only=True)
        except Exception as error:  # noqa: BLE001
            last_error = error
    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, use_fast=True)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(f"Unable to load tokenizer from candidates={candidates}; last_error={last_error}")


def load_rank_model(
    model_dir: Path,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str | None = None,
) -> Tuple:
    model_dir = Path(model_dir)
    adapter_config = model_dir / "adapter_config.json"
    device = resolve_device()

    if adapter_config.exists():
        with adapter_config.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        base_model_name = payload.get("base_model_name_or_path", default_base_model)
        candidates = [base_model_name]
        if fallback_local_base_model:
            candidates.append(fallback_local_base_model)
        if default_base_model not in candidates:
            candidates.append(default_base_model)

        base_model = _load_first_model(candidates)
        model = PeftModel.from_pretrained(base_model, model_dir.as_posix())
        tokenizer = _load_first_tokenizer([model_dir.as_posix(), *candidates])
    else:
        model = _load_first_model([model_dir.as_posix(), default_base_model])
        tokenizer = _load_first_tokenizer([model_dir.as_posix(), default_base_model])

    model.to(device)
    model.eval()
    return model, tokenizer, device


def compute_metrics_builder(tokenizer):
    rouge = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.array(predictions)
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]

        labels_np = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {
            "rouge1": float(result["rouge1"]),
            "rouge2": float(result["rouge2"]),
            "rougeL": float(result["rougeL"]),
        }

    return compute_metrics


def evaluate_rank(
    rank: int,
    test_dataset,
    args,
    fallback_local_base_model: str,
) -> Dict:
    model_dir = (PROJECT_ROOT / args.experiments_root / f"t5_small_lora_r{rank}").resolve()
    if not model_dir.exists():
        return {"status": "missing_model_dir"}

    model, tokenizer, _ = load_rank_model(
        model_dir=model_dir,
        default_base_model=args.default_base_model,
        fallback_local_base_model=fallback_local_base_model,
    )

    eval_args = Seq2SeqTrainingArguments(
        output_dir=(PROJECT_ROOT / "temp_eval_rank_ablation").as_posix(),
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    metrics = trainer.evaluate(test_dataset)
    return {
        "rouge1": float(metrics.get("eval_rouge1", 0.0)),
        "rouge2": float(metrics.get("eval_rouge2", 0.0)),
        "rougeL": float(metrics.get("eval_rougeL", 0.0)),
        "eval_loss": float(metrics.get("eval_loss", 0.0)),
        "num_samples": int(len(test_dataset)),
        "status": "ok",
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    test_dataset = load_from_disk((PROJECT_ROOT / args.processed_data_path).as_posix())["test"]
    fallback_base = _resolve_fallback_base(args.default_base_model)

    payload = {}
    for rank in tqdm(args.ranks, desc="Evaluating ROUGE by rank"):
        key = rank_key(rank)
        payload[key] = evaluate_rank(
            rank=rank,
            test_dataset=test_dataset,
            args=args,
            fallback_local_base_model=fallback_base,
        )

    ordered = dict(sorted(payload.items(), key=lambda item: rank_value(item[0])))
    output_path = (PROJECT_ROOT / args.output_path).resolve()
    save_json(output_path, ordered)
    print(f"Saved ROUGE metrics: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
