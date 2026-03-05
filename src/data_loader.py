"""
data_loader.py

Responsible for:
- Loading SAMSum dataset from HuggingFace
- Saving raw splits locally for reproducibility
- Printing basic dataset statistics
"""

import os
from datasets import load_dataset, DatasetDict
from typing import Dict


def load_samsum_dataset() -> DatasetDict:
    """
    Load SAMSum dataset from HuggingFace.

    Returns:
        DatasetDict: containing train, validation, and test splits.
    """
    dataset = load_dataset("knkarthick/samsum")
    return dataset


def print_dataset_statistics(dataset: DatasetDict) -> None:
    """
    Print basic statistics about dataset splits.

    Args:
        dataset (DatasetDict): Loaded SAMSum dataset.
    """
    print("\nDataset Statistics:")
    for split in dataset.keys():
        print(f"Split: {split}")
        print(f"Number of samples: {len(dataset[split])}")
        print("-" * 40)


def save_dataset_locally(dataset: DatasetDict, save_dir: str = "data/raw") -> None:
    """
    Save dataset splits locally for reproducibility.

    Args:
        dataset (DatasetDict): Loaded dataset.
        save_dir (str): Directory to save dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)
    print(f"\nDataset saved locally at: {save_dir}")


def main():
    dataset = load_samsum_dataset()
    print_dataset_statistics(dataset)
    save_dataset_locally(dataset)


if __name__ == "__main__":
    main()
