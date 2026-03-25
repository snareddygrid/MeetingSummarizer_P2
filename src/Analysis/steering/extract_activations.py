"""
Extract decoder activations for steering analysis.

Run:
    python src/analysis/steering/extract_activations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utils.activation_utils import (
    build_subset_samples,
    decode_labels,
    ensure_dir,
    extract_decoder_hidden_states,
    get_device,
    load_lora_model_and_tokenizer,
    load_processed_split,
    load_subset_indices,
    pool_hidden_states,
    save_json,
    tensorize_sample,
    resolve_decoder_layers,
)


DEFAULT_REQUESTED_LAYERS = [6, 7, 8, 9, 10, 11, 12]


def run_extraction(args) -> Dict:
    import torch

    output_root = Path(args.output_root)
    activations_dir = ensure_dir(output_root / "activations")
    reports_dir = ensure_dir(output_root / "reports")

    rows = load_processed_split(dataset_path=args.dataset_path, split=args.split)
    subset_indices = load_subset_indices(
        total_size=len(rows),
        subset_size=args.subset_size,
        subset_indices_path=Path(args.subset_indices_path) if args.subset_indices_path else None,
    )
    samples = build_subset_samples(rows=rows, indices=subset_indices)

    device = get_device(prefer_mps=True) if args.device == "auto" else torch.device(args.device)
    model, tokenizer, device = load_lora_model_and_tokenizer(
        model_dir=Path(args.model_dir),
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
        device=device,
        merge_lora=False,
    )
    resolved_layers, layer_metadata = resolve_decoder_layers(model=model, requested_layers=args.decoder_layers)

    activation_index: List[Dict] = []
    for sample in tqdm(samples, desc="Extract Activations"):
        input_ids, attention_mask, labels = tensorize_sample(sample=sample, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
        layer_hidden_states = extract_decoder_hidden_states(
            decoder_hidden_states=outputs.decoder_hidden_states,
            resolved_layers=resolved_layers,
        )
        pooled = pool_hidden_states(layer_hidden_states)

        sample_payload = {
            "sample_id": sample["sample_id"],
            "source_index": sample["source_index"],
            "resolved_decoder_layers": resolved_layers,
            "reference_summary": decode_labels(tokenizer=tokenizer, labels=sample["labels"]),
            "decoder_hidden_states": {
                str(layer): tensor.to(dtype=torch.float16) for layer, tensor in layer_hidden_states.items()
            },
            "pooled_activations": {
                str(layer): tensor.to(dtype=torch.float32) for layer, tensor in pooled.items()
            },
        }

        sample_path = activations_dir / f"{sample['sample_id']}.pt"
        torch.save(sample_payload, sample_path)
        activation_index.append(
            {
                "sample_id": sample["sample_id"],
                "source_index": sample["source_index"],
                "activation_path": sample_path.as_posix(),
            }
        )

    metadata = {
        "model_dir": Path(args.model_dir).as_posix(),
        "dataset_path": args.dataset_path,
        "split": args.split,
        "subset_size": len(samples),
        "subset_indices": subset_indices,
        "resolved_decoder_layers": resolved_layers,
        "layer_selection": layer_metadata,
        "device": str(device),
        "activation_index": activation_index,
    }
    save_json(reports_dir / "activation_extraction.json", metadata)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Extract decoder hidden activations for steering.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/analysis/steering")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument(
        "--subset-indices-path",
        default="outputs/analysis/quantization/reports/fixed_test_subset.json",
    )
    parser.add_argument("--decoder-layers", nargs="+", type=int, default=DEFAULT_REQUESTED_LAYERS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = run_extraction(args)
    print("Activation extraction complete.")
    print(f"Resolved decoder layers: {metadata['resolved_decoder_layers']}")
    print(f"Saved activations for samples: {metadata['subset_size']}")


if __name__ == "__main__":
    main()
