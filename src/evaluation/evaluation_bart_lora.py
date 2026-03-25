"""
evaluation_bart_lora.py

Test evaluation for BART-base LoRA experiment.
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
# CONFIG
# ===============================

MODEL_DIR = "experiments/bart_base_lora"
PROCESSED_DATA_PATH = "data/processed_bart"
OUTPUT_METRICS_FILE = os.path.join(MODEL_DIR, "test_metrics.json")


# ===============================
# Load Dataset
# ===============================

def load_test_dataset():
    dataset = load_from_disk(PROCESSED_DATA_PATH)
    return dataset["test"]


# ===============================
# ROUGE
# ===============================

rouge = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)

    # Prevent decoding overflow
    predictions = np.clip(
        predictions,
        0,
        tokenizer.vocab_size - 1
    )

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

    print("Loading trained LoRA-BART model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # Match training decoding config
    model.config.num_beams = 5
    model.config.max_new_tokens = 180
    model.config.length_penalty = 1.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    eval_args = Seq2SeqTrainingArguments(
        output_dir="temp_eval_bart_lora",
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

    print("Running test evaluation...")
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