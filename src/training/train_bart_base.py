"""
train_bart_base.py

Full Fine-Tuning for facebook/bart-base
Uses dataset from data/processed_bart
"""

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load as load_metric

from model import build_model
from preprocess import get_tokenizer


# ===============================
# CONFIGURATION
# ===============================

MODEL_NAME = "facebook/bart-base"
MODE = "full"
PROCESSED_DATA_PATH = "data/processed_bart"
OUTPUT_DIR = "experiments/bart_base_full"


# ===============================
# Load Dataset
# ===============================

print("Loading processed BART dataset...")
dataset = load_from_disk(PROCESSED_DATA_PATH)


# ===============================
# Load Model
# ===============================

print("Loading BART-base model...")
model = build_model(
    model_name=MODEL_NAME,
    mode=MODE
)

# Generation settings (important for ROUGE)
model.config.num_beams = 5
model.config.max_new_tokens = 140
model.config.length_penalty = 1.0
model.config.early_stopping = True


tokenizer = get_tokenizer(MODEL_NAME)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ===============================
# ROUGE Metric
# ===============================

rouge = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)

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
# Training Arguments
# ===============================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,

    learning_rate=5e-5,               # BART prefers lower LR
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=7,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.1,

    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,   # MPS safe
    report_to="none",
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("Starting BART-base full fine-tuning...")
trainer.train()

print("Saving model...")
trainer.save_model(OUTPUT_DIR)

print("Training complete.")