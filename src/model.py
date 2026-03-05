"""
model.py

Responsible for:
- Loading any Seq2Seq model (T5, FLAN-T5, BART, etc.)
- Supporting both LoRA and Full Fine-Tuning
- Printing trainable parameter statistics
- Moving model to appropriate device
"""

import torch
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model


# ===============================
# Device Setup
# ===============================

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


# ===============================
# Print Trainable Parameters
# ===============================

def print_trainable_parameters(model):
    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params}")
    print(f"Total params: {total_params}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


# ===============================
# Build Model (LoRA or Full)
# ===============================

def build_model(model_name="t5-small", mode="lora",lora_config=None):
    """
    model_name:
        Any HuggingFace seq2seq model
        e.g.
            - t5-small
            - google/flan-t5-base
            - facebook/bart-base

    mode:
        - "lora"
        - "full"
    """

    device = get_device()

    print(f"Loading base model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if mode == "lora":
        print("Applying LoRA...")
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        model = get_peft_model(model, lora_config)

    elif mode == "full":
        print("Full fine-tuning selected (all parameters trainable).")

    else:
        raise ValueError("Mode must be either 'lora' or 'full'")

    print_trainable_parameters(model)

    model.to(device)
    return model
