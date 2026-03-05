"""
preprocess_bart.py

Preprocessing specifically for facebook/bart-base.

- No task prefix
- Uses BART tokenizer
- Saves dataset to data/processed_bart
"""

import os
from typing import Dict
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


# ===============================
# Configuration
# ===============================

MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 140
OUTPUT_PATH = "data/processed_bart"


# ===============================
# Load Dataset
# ===============================

def load_raw_dataset(path: str = "data/raw") -> DatasetDict:
    return load_from_disk(path)


# ===============================
# Preprocessing Function
# ===============================

def preprocess_function(examples: Dict, tokenizer) -> Dict:

    # IMPORTANT: No prefix for BART
    inputs = examples["dialogue"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

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

def save_processed_dataset(dataset: DatasetDict, path: str = OUTPUT_PATH):

    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    print(f"Processed BART dataset saved at: {path}")


# ===============================
# Main Execution
# ===============================

def main():

    print("Loading raw dataset...")
    dataset = load_raw_dataset()

    print("Loading BART tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset for BART...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    print("Saving processed dataset...")
    save_processed_dataset(tokenized_dataset)

    print("BART preprocessing complete.")


if __name__ == "__main__":
    main()