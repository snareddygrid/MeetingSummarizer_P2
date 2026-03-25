"""
preprocess_pegasus_speaker.py

Preprocessing for PEGASUS with Speaker Tags
Creates data/processed_pegasus_speaker
"""

import os
from typing import Dict
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


# ===============================
# CONFIG
# ===============================

MODEL_NAME = "google/pegasus-xsum"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 140
OUTPUT_PATH = "data/processed_pegasus_speaker"


# ===============================
# Load Raw Dataset
# ===============================

def load_raw_dataset(path: str = "data/raw") -> DatasetDict:
    return load_from_disk(path)


# ===============================
# Add Speaker Tags
# ===============================

def add_speaker_tags(dialogue: str) -> str:
    """
    Convert:
    John: Hi
    Mary: Hello

    To:
    [Speaker1] John: Hi
    [Speaker2] Mary: Hello
    """

    lines = dialogue.split("\n")
    speaker_map = {}
    speaker_id = 1

    new_lines = []

    for line in lines:
        if ":" in line:
            speaker_name = line.split(":")[0].strip()

            if speaker_name not in speaker_map:
                speaker_map[speaker_name] = f"[Speaker{speaker_id}]"
                speaker_id += 1

            tag = speaker_map[speaker_name]
            new_lines.append(f"{tag} {line}")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


# ===============================
# Preprocessing Function
# ===============================

def preprocess_function(examples: Dict, tokenizer):

    # Add speaker tags
    inputs = [
        add_speaker_tags(dialogue)
        for dialogue in examples["dialogue"]
    ]

    # PEGASUS does NOT require instruction prefix
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
# Tokenize Entire Dataset
# ===============================

def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:

    return dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )


# ===============================
# Save Processed Dataset
# ===============================

def save_processed_dataset(dataset: DatasetDict):

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved to: {OUTPUT_PATH}")


# ===============================
# Main
# ===============================

def main():

    print("Loading raw dataset...")
    dataset = load_raw_dataset()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Applying speaker tagging + tokenization...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    print("Saving processed dataset...")
    save_processed_dataset(tokenized_dataset)

    print("Done.")


if __name__ == "__main__":
    main()