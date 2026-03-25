"""
Reusable helpers for Task-1 attention analysis.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROMPT_PREFIX = "Summarize the following conversation:"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_output_structure(output_root: Path) -> Dict[str, Path]:
    paths = {
        "attention_tensors": output_root / "attention_tensors",
        "speaker_distribution": output_root / "speaker_distribution",
        "key_moments": output_root / "key_moments",
        "entropy": output_root / "entropy",
        "heatmaps": output_root / "heatmaps",
        "reports": output_root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _read_adapter_base_model(model_dir: Path, default_base_model: str) -> str:
    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return default_base_model
    with adapter_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("base_model_name_or_path", default_base_model)


def _resolve_tokenizer_source(model_dir: Path, base_model_name: str) -> str:
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "spiece.model", "vocab.json"]
    if any((model_dir / name).exists() for name in tokenizer_files):
        return model_dir.as_posix()
    return base_model_name


def load_model_and_tokenizer(
    model_dir: Path,
    default_base_model: str = "t5-small",
    fallback_local_base_model: Optional[str] = "experiments/t5_small_optimized",
    device: Optional[torch.device] = None,
):
    """
    Loads a T5 / T5-LoRA model with AutoModelForSeq2SeqLM as base loader.
    """
    device = device or get_device()
    model_dir = Path(model_dir)
    base_model_name = _read_adapter_base_model(model_dir, default_base_model)

    # Always load base using AutoModelForSeq2SeqLM (per task requirement).
    load_candidates = [base_model_name]
    if fallback_local_base_model:
        fallback_candidate = Path(fallback_local_base_model).as_posix()
        if fallback_candidate not in load_candidates:
            load_candidates.append(fallback_candidate)

    base_model = None
    selected_base_source = None
    last_error = None
    for candidate in load_candidates:
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(candidate, local_files_only=True)
            selected_base_source = candidate
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
            continue

    if base_model is None:
        raise RuntimeError(
            "Unable to load base model locally for LoRA adapter. "
            f"Tried: {load_candidates}. Last error: {last_error}"
        )

    if (model_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, model_dir.as_posix())
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir.as_posix(), local_files_only=True)

    tokenizer_source = _resolve_tokenizer_source(model_dir, selected_base_source or base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, local_files_only=True)

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.to(device)
    model.eval()

    return model, tokenizer, device


def decode_processed_dialogue(tokenizer, input_ids: List[int]) -> Tuple[str, str, int]:
    """
    Converts tokenized sample back to text and extracts dialogue body.
    Returns:
      full_text, dialogue_text, dialogue_start_char
    """
    token_tensor = torch.tensor(input_ids, dtype=torch.long)
    non_pad_ids = token_tensor[token_tensor != tokenizer.pad_token_id]
    full_text = tokenizer.decode(non_pad_ids, skip_special_tokens=True).strip()

    marker_index = full_text.lower().find(PROMPT_PREFIX.lower())
    if marker_index == -1:
        return full_text, full_text, 0

    start_index = marker_index + len(PROMPT_PREFIX)
    while start_index < len(full_text) and full_text[start_index] in {" ", "\n", "\t"}:
        start_index += 1
    dialogue_text = full_text[start_index:]
    return full_text, dialogue_text, start_index


def build_prompt_text(dialogue_text: str) -> str:
    return f"{PROMPT_PREFIX}\n{dialogue_text.strip()}"


def parse_dialogue_turns(dialogue_text: str) -> List[Dict]:
    """
    Parses speaker turns from both:
    1) newline-separated format
    2) flattened single-line format with repeated `Speaker: ...` segments
    """
    dialogue_text = dialogue_text.strip()
    if not dialogue_text:
        return []

    # Preferred path: line-separated dialogue (as in raw SAMSum).
    if "\n" in dialogue_text:
        turns: List[Dict] = []
        cursor = 0
        for turn_id, line in enumerate(dialogue_text.splitlines(), start=1):
            line_start = cursor
            line_end = line_start + len(line)
            cursor = line_end + 1

            if ":" in line:
                speaker, utterance = line.split(":", 1)
                speaker = speaker.strip() or f"Turn{turn_id}"
                utterance = utterance.strip()
            else:
                speaker = "UNKNOWN"
                utterance = line.strip()

            turns.append(
                {
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "utterance": utterance,
                    "raw_line": line,
                    "char_start": line_start,
                    "char_end": line_end,
                }
            )
        return turns

    # Robust pattern for SAMSum-like speaker tags in flattened text.
    speaker_pattern = re.compile(
        r"(?<!\S)([A-Z][A-Za-z0-9_.\-]*(?: [A-Z][A-Za-z0-9_.\-]*){0,1}):"
    )
    matches = list(speaker_pattern.finditer(dialogue_text))

    if len(matches) >= 2:
        turns: List[Dict] = []
        for turn_id, match in enumerate(matches, start=1):
            speaker = match.group(1).strip()
            utterance_start = match.end()
            utterance_end = matches[turn_id].start() if turn_id < len(matches) else len(dialogue_text)
            utterance = dialogue_text[utterance_start:utterance_end].strip()
            raw_line = dialogue_text[match.start():utterance_end].strip()

            turns.append(
                {
                    "turn_id": turn_id,
                    "speaker": speaker if speaker else f"Turn{turn_id}",
                    "utterance": utterance,
                    "raw_line": raw_line,
                    "char_start": match.start(),
                    "char_end": utterance_end,
                }
            )
        return turns

    # Last fallback.
    turns: List[Dict] = []
    cursor = 0
    for turn_id, line in enumerate(dialogue_text.splitlines(), start=1):
        line_start = cursor
        line_end = line_start + len(line)
        cursor = line_end + 1

        if ":" in line:
            speaker, utterance = line.split(":", 1)
            speaker = speaker.strip() or f"Turn{turn_id}"
            utterance = utterance.strip()
        else:
            speaker = "UNKNOWN"
            utterance = line.strip()

        turns.append(
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "utterance": utterance,
                "raw_line": line,
                "char_start": line_start,
                "char_end": line_end,
            }
        )
    return turns


def tokenize_text_with_offsets(
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask, offsets


def map_tokens_to_turns(
    offsets: List[Tuple[int, int]],
    turns: List[Dict],
    dialogue_start_char: int,
) -> List[int]:
    """
    Maps each source token index to a turn index (-1 means non-dialogue tokens).
    """
    token_to_turn = [-1 for _ in range(len(offsets))]

    for turn_idx, turn in enumerate(turns):
        start = turn["char_start"] + dialogue_start_char
        end = turn["char_end"] + dialogue_start_char
        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:
                continue
            if tok_end <= start or tok_start >= end:
                continue
            token_to_turn[token_idx] = turn_idx

    return token_to_turn


def normalize_cross_attention(cross_attentions: torch.Tensor) -> torch.Tensor:
    """
    Expects [layers, heads, tgt_len, src_len], returns normalized [tgt_len, src_len].
    """
    cross = cross_attentions.float().mean(dim=(0, 1))
    cross = torch.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)
    cross = torch.clamp(cross, min=0.0)
    denom = cross.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return cross / denom


def normalize_distribution(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0.0:
        return {}
    return {key: float(max(0.0, value) / total) for key, value in scores.items()}


def convert_ids_to_clean_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    cleaned = []
    for token in tokens:
        stripped = token.replace("▁", " ").strip()
        cleaned.append(stripped if stripped else token)
    return cleaned


def sanitize_token_for_display(token: str) -> str:
    token = token.replace("\n", " ").strip()
    if not token:
        return "<blank>"
    return token
