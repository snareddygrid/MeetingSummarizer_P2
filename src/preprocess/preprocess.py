"""
preprocess.py

Responsible for:
- Loading raw dataset
- Adding instruction prefix
- Tokenizing dialogues and summaries
- Applying truncation and padding
- Saving processed dataset
"""

import os
from typing import Dict
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


# ===============================
# Configuration
# ===============================

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 140


# ===============================
# Load Dataset
# ===============================

def load_raw_dataset(path: str = "data/raw") -> DatasetDict:
    return load_from_disk(path)


# ===============================
# Initialize Tokenizer
# ===============================

def get_tokenizer(model_name: str):
    """
    Load tokenizer for any seq2seq model.
    """
    return AutoTokenizer.from_pretrained(model_name)


# ===============================
# Preprocessing Function
# ===============================

def preprocess_function(examples: Dict, tokenizer) -> Dict:

    # Better instruction prompt for FLAN
    inputs = [
        "Summarize the following conversation:\n" + dialogue
        for dialogue in examples["dialogue"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets (modern way, no deprecated API)
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ===============================
# Apply Preprocessing
# ===============================

def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:

    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset


# ===============================
# Save Processed Dataset
# ===============================

def save_processed_dataset(dataset: DatasetDict, path: str = "data/processed"):

    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    print(f"Processed dataset saved at: {path}")


# ===============================
# Main Execution
# ===============================

def main():

    MODEL_NAME = "google/flan-t5-base"   #Change here if needed(e.g. "t5-small")

    print("Loading raw dataset...")
    dataset = load_raw_dataset()

    print("Initializing tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    print("Saving processed dataset...")
    save_processed_dataset(tokenized_dataset)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
