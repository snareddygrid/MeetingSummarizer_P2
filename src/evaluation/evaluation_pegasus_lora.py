"""
evaluation_pegasus_lora.py

Evaluate PEGASUS-LoRA model on test set
"""

import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel
from evaluate import load as load_metric


# ===============================
# CONFIG
# ===============================

MODEL_DIR = "experiments/pegasus_lora"
PROCESSED_DATA_PATH = "data/processed_pegasus_speaker"
OUTPUT_FILE = os.path.join(MODEL_DIR, "test_metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ===============================
# Load Dataset
# ===============================

print("Loading test dataset...")
dataset = load_from_disk(PROCESSED_DATA_PATH)
test_dataset = dataset["test"]


# ===============================
# Load Base + LoRA Model
# ===============================

print("Loading base PEGASUS model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

model.eval()


# ===============================
# Generation Settings
# ===============================

model.config.num_beams = 4
model.config.max_new_tokens = 180
model.config.length_penalty = 1.2
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True


# ===============================
# Tokenizer
# ===============================

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ===============================
# ROUGE
# ===============================

rouge = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)

    # Prevent overflow errors
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
# Evaluation Arguments
# ===============================

eval_args = Seq2SeqTrainingArguments(
    output_dir="temp_eval_pegasus_lora",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    do_train=False,
    do_eval=True,
    report_to="none",
)


# ===============================
# Trainer
# ===============================

trainer = Seq2SeqTrainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("Running evaluation on test set...")
metrics = trainer.evaluate(test_dataset)

print("\nRaw Metrics:")
print(metrics)


# ===============================
# Save Clean Metrics
# ===============================

final_metrics = {
    "test_loss": metrics.get("eval_loss"),
    "rouge1": metrics.get("eval_rouge1"),
    "rouge2": metrics.get("eval_rouge2"),
    "rougeL": metrics.get("eval_rougeL"),
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(final_metrics, f, indent=4)

print(f"\nTest metrics saved at: {OUTPUT_FILE}")