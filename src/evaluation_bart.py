"""
evaluation_bart.py

Test evaluation script for facebook/bart-base experiment.
Uses the same decoding configuration as training.
"""

import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from evaluate import load as load_metric


# ===============================
# CONFIGURATION
# ===============================

MODEL_DIR = "experiments/bart_base_full"
PROCESSED_DATA_PATH = "data/processed_bart"
OUTPUT_METRICS_FILE = os.path.join(MODEL_DIR, "test_metrics.json")


# ===============================
# Load Dataset
# ===============================

def load_test_dataset():
    dataset = load_from_disk(PROCESSED_DATA_PATH)
    return dataset["test"]


# ===============================
# ROUGE Metric
# ===============================

rouge = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }


# ===============================
# Evaluation
# ===============================

def evaluate():

    global tokenizer

    print("Loading test dataset...")
    test_dataset = load_test_dataset()

    print("Loading trained BART model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # ✅ Match EXACT decoding config used in training
    model.config.num_beams = 5
    model.config.max_new_tokens = 140
    model.config.length_penalty = 1.0
    model.config.early_stopping = True

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    eval_args = Seq2SeqTrainingArguments(
        output_dir="temp_eval_bart",
        per_device_eval_batch_size=2,   # Safe for MPS
        predict_with_generate=True,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation on test set...")
    metrics = trainer.evaluate(test_dataset)

    print("\nTest Results:")
    print(metrics)

    final_metrics = {
        "test_loss": metrics.get("eval_loss"),
        "rouge1": metrics.get("eval_rouge1"),
        "rouge2": metrics.get("eval_rouge2"),
        "rougeL": metrics.get("eval_rougeL"),
    }

    with open(OUTPUT_METRICS_FILE, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\nTest metrics saved at: {OUTPUT_METRICS_FILE}")


if __name__ == "__main__":
    evaluate()