"""
train_bart_base_lora.py

LoRA fine-tuning for facebook/bart-base
Reusable via model.py
"""

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType
from evaluate import load as load_metric

from model import build_model


# ===============================
# CONFIG
# ===============================

MODEL_NAME = "facebook/bart-base"
MODE = "lora"
OUTPUT_DIR = "experiments/bart_base_lora"
PROCESSED_DATA_PATH = "data/processed_bart"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# Custom LoRA Config (BART specific)
# ===============================

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2"
    ]
)


# ===============================
# Load Dataset
# ===============================

dataset = load_from_disk(PROCESSED_DATA_PATH)


# ===============================
# Load Model (Reusable!)
# ===============================
2
model = build_model(
    model_name=MODEL_NAME,
    mode=MODE,
    lora_config=peft_config
)


# ===============================
# Generation Config
# ===============================

model.config.num_beams = 5
model.config.max_new_tokens = 180
model.config.length_penalty = 1.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3


# ===============================
# Tokenizer
# ===============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
# Training Arguments
# ===============================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,

    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,

    num_train_epochs=8,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,

    fp16=False,
    report_to="none",
)


# ===============================
# Trainer
# ===============================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("Starting LoRA training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print("Training complete.")