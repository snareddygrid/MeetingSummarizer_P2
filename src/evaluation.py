"""
evaluation.py

Responsible for:
- Loading trained model
- Loading processed test dataset
- Running evaluation on test set
- Saving clean test metrics JSON
"""

import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from evaluate import load as load_metric


# ===============================
# CONFIG
# ===============================

MODEL_DIR = "experiments/t5_small_optimized"  # Change this to your trained model directory['flan_t5_base']
PROCESSED_DATA_PATH = "data/processed"
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

    vocab_size = tokenizer.vocab_size
    predictions = np.clip(predictions, 0, vocab_size - 1)

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

    print("Loading trained model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    # Ensure same generation settings as training
    model.config.num_beams = 4
    model.config.max_new_tokens = 140
    model.config.length_penalty = 1.1
    model.config.early_stopping = True

    print("Loading tokenizer from trained folder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Minimal evaluation arguments (clean + safe)
    eval_args = Seq2SeqTrainingArguments(
        output_dir="temp_eval",
        per_device_eval_batch_size=2,
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

    # Save only important metrics
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
