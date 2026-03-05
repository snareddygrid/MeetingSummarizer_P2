"""
inference.py

Handles summary generation only.
"""

import torch


def generate_summary(model, tokenizer, device, text, architecture=None):

    # Add instruction prefix only for T5-based models
    if architecture and "t5" in architecture.lower():
        prompt = "Summarize the following conversation:\n" + text
    else:
        prompt = text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=6,
            max_new_tokens=180,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary