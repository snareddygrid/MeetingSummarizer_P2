import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    EarlyStoppingCallback,
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# High-capacity LoRA Config (ROUGE-L focused)
# ===============================

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q", "k", "v", "o", "wi", "wo"]
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

# Generation config used by Trainer during evaluation/prediction.
model.config.num_beams = 8
model.config.max_new_tokens = 140
model.config.min_new_tokens = 16
model.config.length_penalty = 1.05
model.config.no_repeat_ngram_size = 3
model.config.repetition_penalty = 1.1
model.config.early_stopping = True

tokenizer = get_tokenizer(MODEL_NAME)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
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
    decoded_preds = [pred.strip() for pred in decoded_preds]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )
    decoded_labels = [label.strip() for label in decoded_labels]

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
    metric_for_best_model="rougeL",
    greater_is_better=True,
    seed=42,
    data_seed=42,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,

    learning_rate=1.5e-4,
    num_train_epochs=5,
    weight_decay=0.01,
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
    save_total_limit=3,
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
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=1e-4
        )
    ],
)


print("Starting LoRA training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print("LoRA training complete.")
