"""
Task-1 Attention Analysis Pipeline

Run:
    python src/analysis/attention/extract_attention.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

# Ensure local module imports work regardless of directory case on macOS.
CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR / "utils"
for local_path in (CURRENT_DIR, UTILS_DIR):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from attention_visualization import create_attention_heatmap, compute_rollout_flow_map
from compute_entropy import compute_entropy_for_sample
from generate_report import generate_final_report
from key_moment_detection import detect_key_moments
from speaker_analysis import analyze_speaker_distribution
from utils.attention_helpers import (
    build_prompt_text,
    convert_ids_to_clean_tokens,
    decode_processed_dialogue,
    ensure_output_structure,
    load_model_and_tokenizer,
    map_tokens_to_turns,
    normalize_cross_attention,
    parse_dialogue_turns,
    tokenize_text_with_offsets,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Task-1 attention extraction and analysis pipeline.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--dataset-path", default="data/processed")
    parser.add_argument("--raw-dataset-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-root", default="outputs/analysis/attention")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--num-beams", type=int, default=6)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def _safe_sample_id(index: int, split_name: str) -> str:
    return f"{split_name}_{index:04d}"


def _stack_attention_layers(attention_tuple):
    return torch.stack([layer[0].detach().cpu() for layer in attention_tuple], dim=0)


def run_pipeline(args):
    output_root = Path(args.output_root)
    output_dirs = ensure_output_structure(output_root)

    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(
        model_dir=Path(args.model_dir),
        default_base_model=args.base_model,
        fallback_local_base_model=args.base_model_fallback,
    )
    print(f"Using device: {device}")

    print("Loading processed dataset...")
    dataset = load_from_disk(args.dataset_path)
    split_dataset = dataset[args.split]
    raw_split_dataset = None
    raw_dataset_path = Path(args.raw_dataset_path)
    if raw_dataset_path.exists():
        raw_dataset = load_from_disk(raw_dataset_path.as_posix())
        if args.split in raw_dataset:
            raw_split_dataset = raw_dataset[args.split]
            if len(raw_split_dataset) != len(split_dataset):
                print("Warning: raw and processed split sizes differ; falling back to processed decode.")
                raw_split_dataset = None
    total = min(args.num_samples, len(split_dataset))
    print(f"Processing {total} samples from split='{args.split}'")

    failures = []

    progress = tqdm(range(total), desc="Task-1 Attention")
    for index in progress:
        sample_id = _safe_sample_id(index=index, split_name=args.split)
        sample = split_dataset[index]

        try:
            # Prefer raw dialogue text for clean speaker-turn parsing.
            if raw_split_dataset is not None:
                dialogue_text = str(raw_split_dataset[index]["dialogue"])
                full_text = build_prompt_text(dialogue_text)
                dialogue_start_char = full_text.find(dialogue_text)
                if dialogue_start_char < 0:
                    dialogue_start_char = 0
            else:
                # Fallback to processed decode when raw is unavailable.
                full_text, dialogue_text, dialogue_start_char = decode_processed_dialogue(
                    tokenizer=tokenizer,
                    input_ids=sample["input_ids"],
                )

            turns = parse_dialogue_turns(dialogue_text)
            if not turns:
                turns = [
                    {
                        "turn_id": 1,
                        "speaker": "UNKNOWN",
                        "utterance": dialogue_text,
                        "raw_line": dialogue_text,
                        "char_start": 0,
                        "char_end": len(dialogue_text),
                    }
                ]

            # Re-tokenize with offsets for reliable token->turn mapping.
            input_ids, attention_mask, offsets = tokenize_text_with_offsets(
                tokenizer=tokenizer,
                text=full_text,
                device=device,
                max_length=args.max_input_length,
            )
            token_to_turn = map_tokens_to_turns(
                offsets=offsets,
                turns=turns,
                dialogue_start_char=dialogue_start_char,
            )

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    early_stopping=True,
                    return_dict_in_generate=True,
                )

            generated_ids = generated.sequences
            summary_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Use generated sequence as decoder input to capture attentions.
            decoder_input_ids = generated_ids[:, :-1]
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            encoder_attentions = _stack_attention_layers(outputs.encoder_attentions)
            decoder_attentions = _stack_attention_layers(outputs.decoder_attentions)
            cross_attentions = _stack_attention_layers(outputs.cross_attentions)
            rollout_map = compute_rollout_flow_map(
                encoder_attentions=encoder_attentions,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
            )

            normalized_cross = normalize_cross_attention(cross_attentions)
            source_token_ids = input_ids[0].detach().cpu().tolist()
            summary_token_ids = decoder_input_ids[0].detach().cpu().tolist()
            input_tokens = convert_ids_to_clean_tokens(tokenizer, source_token_ids)
            summary_tokens = convert_ids_to_clean_tokens(tokenizer, summary_token_ids)

            # 1) Save raw extracted tensors.
            tensor_path = output_dirs["attention_tensors"] / f"{sample_id}.pt"
            tensor_payload = {
                "sample_id": sample_id,
                "input_ids": torch.tensor(source_token_ids, dtype=torch.long),
                "summary_ids": torch.tensor(summary_token_ids, dtype=torch.long),
                "encoder_attentions": encoder_attentions.to(torch.float16),
                "decoder_attentions": decoder_attentions.to(torch.float16),
                "cross_attentions": cross_attentions.to(torch.float16),
                "rollout_flow": rollout_map.to(torch.float16),
            }
            torch.save(tensor_payload, tensor_path)

            # 2) Speaker analysis.
            speaker_distribution, _ = analyze_speaker_distribution(
                sample_id=sample_id,
                cross_attention_map=normalized_cross,
                token_to_turn=token_to_turn,
                turns=turns,
                output_dir=output_dirs["speaker_distribution"],
            )

            # 3) Key moment detection.
            key_moment_payload, _ = detect_key_moments(
                sample_id=sample_id,
                cross_attention_map=normalized_cross,
                summary_tokens=summary_tokens,
                input_tokens=input_tokens,
                token_to_turn=token_to_turn,
                turns=turns,
                output_dir=output_dirs["key_moments"],
            )

            # 4) Entropy.
            entropy_payload, _ = compute_entropy_for_sample(
                sample_id=sample_id,
                speaker_distribution=speaker_distribution,
                output_dir=output_dirs["entropy"],
            )

            # 5) Heatmap visualization.
            create_attention_heatmap(
                sample_id=sample_id,
                cross_attention_map=normalized_cross,
                rollout_attention_map=rollout_map,
                input_tokens=input_tokens,
                summary_tokens=summary_tokens,
                output_dir=output_dirs["heatmaps"],
            )

            # 6) Final report.
            generate_final_report(
                sample_id=sample_id,
                summary_text=summary_text,
                speaker_distribution=speaker_distribution,
                entropy_payload=entropy_payload,
                key_moment_payload=key_moment_payload,
                output_dir=output_dirs["reports"],
            )

        except Exception as error:
            failures.append({"sample_id": sample_id, "error": str(error)})
            progress.set_postfix({"failed": len(failures)})

    summary = {
        "split": args.split,
        "requested_samples": total,
        "processed_samples": total - len(failures),
        "failed_samples": len(failures),
        "output_root": output_root.as_posix(),
    }

    summary_path = output_dirs["reports"] / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if failures:
        failures_path = output_dirs["reports"] / "failures.json"
        with failures_path.open("w", encoding="utf-8") as file:
            json.dump(failures, file, indent=2)

    print("\nTask-1 pipeline completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_pipeline(parse_args())
