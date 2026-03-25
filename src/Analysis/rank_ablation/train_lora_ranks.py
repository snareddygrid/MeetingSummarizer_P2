"""Train T5-small LoRA models across multiple ranks for ablation study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_from_disk
from evaluate import load as load_metric
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model import build_model  # noqa: E402

try:
    from preprocess.preprocess import get_tokenizer  # noqa: E402
except Exception:  # noqa: BLE001
    from transformers import AutoTokenizer  # noqa: E402

    def get_tokenizer(model_name: str):
        return AutoTokenizer.from_pretrained(model_name)

from utils.json_utils import rank_key, save_json  # noqa: E402
from utils.timing_utils import set_seed  # noqa: E402


DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA T5-small models at multiple ranks.")
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--processed-data-path", default="data/processed")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_lora_config(rank: int) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=max(16, rank * 2),
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
    )


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


def train_one_rank(
    rank: int,
    dataset,
    tokenizer,
    args,
) -> Dict:
    output_dir = (PROJECT_ROOT / args.experiments_root / f"t5_small_lora_r{rank}").resolve()
    if output_dir.exists() and not args.overwrite:
        return {
            "rank": rank_key(rank),
            "output_dir": output_dir.as_posix(),
            "status": "skipped_existing",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        model_name=args.model_name,
        mode="lora",
        lora_config=make_lora_config(rank),
    )

    model.config.num_beams = 8
    model.config.max_new_tokens = 140
    model.config.min_new_tokens = 16
    model.config.length_penalty = 1.05
    model.config.no_repeat_ngram_size = 3
    model.config.repetition_penalty = 1.1
    model.config.early_stopping = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir.as_posix(),
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        label_smoothing_factor=0.05,
        max_grad_norm=1.0,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        generation_num_beams=8,
        generation_max_length=140,
        predict_with_generate=True,
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=1e-4,
            )
        ],
    )

    trainer.train()
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())

    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    test_metrics = {key: float(value) if isinstance(value, (float, int)) else value for key, value in test_metrics.items()}
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    return {
        "rank": rank_key(rank),
        "output_dir": output_dir.as_posix(),
        "status": "trained",
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = load_from_disk((PROJECT_ROOT / args.processed_data_path).as_posix())
    tokenizer = get_tokenizer(args.model_name)

    results: List[Dict] = []
    for rank in tqdm(args.ranks, desc="Training LoRA ranks"):
        result = train_one_rank(rank=rank, dataset=dataset, tokenizer=tokenizer, args=args)
        results.append(result)

    summary_path = PROJECT_ROOT / "outputs/analysis/rank_ablation/training_summary.json"
    save_json(
        summary_path,
        {
            "model_name": args.model_name,
            "processed_data_path": (PROJECT_ROOT / args.processed_data_path).as_posix(),
            "ranks": [int(rank) for rank in args.ranks],
            "results": results,
        },
    )
    print(f"Saved training summary to: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
