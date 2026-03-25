"""Adversarial fine-tuning for T5-small LoRA robustness."""

from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader, Sampler

from utils.evaluation_utils import ensure_dir, load_data_rows, resolve_path, set_seed


def _normalize_rows(rows: List[Dict], is_adversarial: int) -> List[Dict]:
    return [
        {
            "dialogue": str(row.get("dialogue", "")),
            "summary": str(row.get("summary", "")),
            "is_adversarial": int(is_adversarial),
        }
        for row in rows
    ]


def split_rows(rows: List[Dict], test_size: float, seed: int) -> Dict[str, List[Dict]]:
    rows = list(rows)
    if len(rows) <= 1:
        return {"train": rows, "eval": []}

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    eval_count = max(1, int(round(len(rows) * float(test_size))))
    eval_indices = set(indices[:eval_count])

    train_rows = [rows[idx] for idx in range(len(rows)) if idx not in eval_indices]
    eval_rows = [rows[idx] for idx in range(len(rows)) if idx in eval_indices]
    return {"train": train_rows, "eval": eval_rows}


def build_weighted_train_rows(
    clean_rows: List[Dict],
    adversarial_rows: List[Dict],
    clean_ratio: float,
    seed: int,
) -> List[Dict]:
    clean_rows = _normalize_rows(clean_rows, is_adversarial=0)
    adversarial_rows = _normalize_rows(adversarial_rows, is_adversarial=1)

    if not clean_rows or not adversarial_rows:
        return clean_rows + adversarial_rows

    ratio = min(max(float(clean_ratio), 0.01), 0.99)
    max_total_by_clean = int(len(clean_rows) / ratio)
    max_total_by_adv = int(len(adversarial_rows) / (1.0 - ratio))
    total = max(2, min(max_total_by_clean, max_total_by_adv))

    n_clean = int(round(total * ratio))
    n_clean = max(1, min(n_clean, len(clean_rows)))
    n_adv = max(1, total - n_clean)
    n_adv = min(n_adv, len(adversarial_rows))

    rng = random.Random(seed)
    sampled_clean = rng.sample(clean_rows, n_clean)
    sampled_adv = rng.sample(adversarial_rows, n_adv)
    mixed = sampled_clean + sampled_adv
    rng.shuffle(mixed)
    return mixed


def build_eval_rows(clean_eval_rows: List[Dict], adversarial_eval_rows: List[Dict], seed: int) -> List[Dict]:
    rows = _normalize_rows(clean_eval_rows, is_adversarial=0) + _normalize_rows(adversarial_eval_rows, is_adversarial=1)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def tokenize_dataset(dataset: Dataset, tokenizer, max_input_length: int, max_target_length: int):
    def _preprocess(batch):
        prompts = [f"Summarize the following conversation:\n{d}" for d in batch["dialogue"]]
        model_inputs = tokenizer(
            prompts,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["is_adversarial"] = [int(x) for x in batch["is_adversarial"]]
        return model_inputs

    return dataset.map(_preprocess, batched=True, remove_columns=dataset.column_names)


class RobustDataCollator:
    def __init__(self, base_collator: DataCollatorForSeq2Seq):
        self.base_collator = base_collator

    def __call__(self, features):
        adv_flags = torch.tensor([int(row.get("is_adversarial", 0)) for row in features], dtype=torch.long)
        stripped = [{k: v for k, v in row.items() if k != "is_adversarial"} for row in features]
        batch = self.base_collator(stripped)
        batch["is_adversarial"] = adv_flags
        return batch


class MixedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        clean_indices: List[int],
        adversarial_indices: List[int],
        batch_size: int,
        clean_ratio: float,
        seed: int = 42,
    ):
        if batch_size < 2:
            raise ValueError("per_device_train_batch_size must be >=2 to guarantee both clean and adversarial samples.")
        if not clean_indices or not adversarial_indices:
            raise ValueError("Both clean and adversarial samples are required for mixed-batch sampling.")

        self.clean_indices = list(clean_indices)
        self.adversarial_indices = list(adversarial_indices)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._iteration = 0

        clean_per_batch = int(round(self.batch_size * float(clean_ratio)))
        clean_per_batch = min(max(clean_per_batch, 1), self.batch_size - 1)
        adv_per_batch = self.batch_size - clean_per_batch
        if adv_per_batch < 1:
            adv_per_batch = 1
            clean_per_batch = self.batch_size - 1
        self.clean_per_batch = clean_per_batch
        self.adv_per_batch = adv_per_batch

    def __len__(self) -> int:
        return min(
            len(self.clean_indices) // self.clean_per_batch,
            len(self.adversarial_indices) // self.adv_per_batch,
        )

    def __iter__(self):
        self._iteration += 1
        rng = random.Random(self.seed + self._iteration)

        clean = list(self.clean_indices)
        adversarial = list(self.adversarial_indices)
        rng.shuffle(clean)
        rng.shuffle(adversarial)

        num_batches = len(self)
        for batch_idx in range(num_batches):
            clean_chunk = clean[
                batch_idx * self.clean_per_batch : (batch_idx + 1) * self.clean_per_batch
            ]
            adv_chunk = adversarial[
                batch_idx * self.adv_per_batch : (batch_idx + 1) * self.adv_per_batch
            ]
            batch = clean_chunk + adv_chunk
            rng.shuffle(batch)
            yield batch


class RobustSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        clean_mix_ratio: float = 0.8,
        adversarial_loss_weight: float = 0.5,
        sampler_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clean_mix_ratio = float(clean_mix_ratio)
        self.adversarial_loss_weight = float(adversarial_loss_weight)
        self.sampler_seed = int(sampler_seed)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        flags = [int(x) for x in self.train_dataset["is_adversarial"]]
        clean_indices = [idx for idx, flag in enumerate(flags) if flag == 0]
        adv_indices = [idx for idx, flag in enumerate(flags) if flag == 1]

        batch_sampler = MixedBatchSampler(
            clean_indices=clean_indices,
            adversarial_indices=adv_indices,
            batch_size=int(self.args.per_device_train_batch_size),
            clean_ratio=self.clean_mix_ratio,
            seed=self.sampler_seed,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        adv_flags = inputs.pop("is_adversarial", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if adv_flags is None or labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        vocab_size = logits.shape[-1]
        token_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.shape)
        valid_mask = (labels != -100).float()
        per_sample_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        sample_weights = torch.where(
            adv_flags.view(-1).bool(),
            torch.full_like(per_sample_loss, float(self.adversarial_loss_weight)),
            torch.ones_like(per_sample_loss),
        )
        loss = (per_sample_loss * sample_weights).mean()
        return (loss, outputs) if return_outputs else loss


def build_training_args(args, output_dir: Path) -> Seq2SeqTrainingArguments:
    kwargs = {
        "output_dir": output_dir.as_posix(),
        "overwrite_output_dir": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "predict_with_generate": False,
        "logging_steps": 25,
        "fp16": False,
        "report_to": "none",
    }
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return Seq2SeqTrainingArguments(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA model on original + adversarial data.")
    parser.add_argument("--original-data", default="data/robustness/original/data.json")
    parser.add_argument("--adversarial-data", default="data/robustness/adversarial/data.json")
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--output-dir", default="experiments/t5_small_lora_robust")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=140)
    parser.add_argument("--per-device-train-batch-size", type=int, default=5)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--clean-mix-ratio", type=float, default=0.8, help="Clean sample ratio in training mix (e.g., 0.7 or 0.8).")
    parser.add_argument("--adversarial-loss-weight", type=float, default=0.5, help="Relative loss weight for adversarial samples.")
    parser.add_argument("--eval-split-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(resolve_path(args.output_dir))

    original_rows = load_data_rows(resolve_path(args.original_data))
    adversarial_rows = load_data_rows(resolve_path(args.adversarial_data))
    clean_split = split_rows(rows=original_rows, test_size=args.eval_split_ratio, seed=args.seed)
    adversarial_split = split_rows(rows=adversarial_rows, test_size=args.eval_split_ratio, seed=args.seed + 1)
    train_rows = build_weighted_train_rows(
        clean_rows=clean_split["train"],
        adversarial_rows=adversarial_split["train"],
        clean_ratio=args.clean_mix_ratio,
        seed=args.seed,
    )
    eval_rows = build_eval_rows(
        clean_eval_rows=clean_split["eval"],
        adversarial_eval_rows=adversarial_split["eval"],
        seed=args.seed + 2,
    )

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    train_clean = sum(1 for row in train_rows if int(row["is_adversarial"]) == 0)
    train_adv = sum(1 for row in train_rows if int(row["is_adversarial"]) == 1)
    print(
        "Train mix after split:",
        f"clean={train_clean}",
        f"adversarial={train_adv}",
        f"clean_ratio={train_clean / max(1, len(train_rows)):.3f}",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
    )
    model = get_peft_model(model, lora_config)

    tokenized_train = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    tokenized_eval = tokenize_dataset(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    data_collator = RobustDataCollator(DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    ))

    training_args = build_training_args(args=args, output_dir=output_dir)

    trainer = RobustSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        clean_mix_ratio=args.clean_mix_ratio,
        adversarial_loss_weight=args.adversarial_loss_weight,
        sampler_seed=args.seed,
    )

    print("Starting adversarial LoRA training...")
    trainer.train()
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())
    print(f"Saved robust model to: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
