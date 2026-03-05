import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType
from evaluate import load as load_metric

from model import build_model
from preprocess import get_tokenizer


# ===============================
# CONFIG
# ===============================

MODEL_NAME = "t5-small"
MODE = "lora"
OUTPUT_DIR = "experiments/t5_small_lora"
PROCESSED_DATA_PATH = "data/processed"


# ===============================
# Aggressive LoRA Config
# ===============================

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "k", "v", "o", "wi", "wo"]
    #["q","k","v","o","wi_0","wi_1","wo"]
)


# ===============================
# Load Dataset
# ===============================

dataset = load_from_disk(PROCESSED_DATA_PATH)


# ===============================
# Load Model
# ===============================

model = build_model(
    model_name=MODEL_NAME,
    mode=MODE,
    lora_config=peft_config
)

model.config.num_beams = 5
model.config.max_new_tokens = 128
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
# Training Arguments
# ===============================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",   # Trainer automatically uses eval_rougeL
    greater_is_better=True,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,

    learning_rate=2.5e-4,
    num_train_epochs=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    label_smoothing_factor=0.1,

    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    save_total_limit=2,
    fp16=False,   # Keep False for MPS
    report_to="none",
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,   # ← THIS FIXES EVERYTHING
)


print("Starting LoRA training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print("LoRA training complete.")
