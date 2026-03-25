"""
Utilities for Task-03 steering analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_device(prefer_mps: bool = True):
    import torch

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_processed_split(dataset_path: str, split: str = "test") -> List[Dict]:
    """
    Load tokenized processed split from local HF Arrow files.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    split_dir = Path(dataset_path) / split
    state_path = split_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path.as_posix()}")

    state = load_json(state_path)
    files = []
    for row in state.get("_data_files", []):
        filename = row.get("filename")
        if filename:
            files.append(split_dir / filename)
    if not files:
        files = sorted(split_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow files found in {split_dir.as_posix()}")

    rows: List[Dict] = []
    for file_path in files:
        with pa.memory_map(file_path.as_posix(), "r") as source:
            table = ipc.open_stream(source).read_all()
        rows.extend(table.to_pylist())
    return rows


def load_subset_indices(
    total_size: int,
    subset_size: int = 100,
    subset_indices_path: Optional[Path] = None,
) -> List[int]:
    subset_size = min(int(subset_size), int(total_size))
    if subset_indices_path and subset_indices_path.exists():
        payload = load_json(subset_indices_path)
        indices = payload.get("subset_indices", [])
        if isinstance(indices, list) and len(indices) >= subset_size:
            return [int(x) for x in indices[:subset_size]]
    return list(range(subset_size))


def build_subset_samples(
    rows: Sequence[Dict],
    indices: Sequence[int],
) -> List[Dict]:
    samples = []
    for idx in indices:
        row = rows[int(idx)]
        samples.append(
            {
                "sample_id": f"test_{int(idx):04d}",
                "source_index": int(idx),
                "input_ids": [int(x) for x in row["input_ids"]],
                "attention_mask": [int(x) for x in row["attention_mask"]],
                "labels": [int(x) for x in row["labels"]],
            }
        )
    return samples


def _read_adapter_base_model(model_dir: Path, default_base_model: str) -> str:
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.exists():
        return default_base_model
    payload = load_json(adapter_config)
    return str(payload.get("base_model_name_or_path", default_base_model))


def load_lora_model_and_tokenizer(
    model_dir: Path,
    default_base_model: str = "t5-small",
    fallback_local_base_model: Optional[str] = "experiments/t5_small_optimized",
    device=None,
    merge_lora: bool = False,
):
    """
    Load local T5-small LoRA adapter and tokenizer.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir = Path(model_dir)
    device = device or get_device()

    base_model_name = _read_adapter_base_model(model_dir, default_base_model)
    candidates = [base_model_name]
    if fallback_local_base_model:
        candidate = Path(fallback_local_base_model).as_posix()
        if candidate not in candidates:
            candidates.append(candidate)

    if (model_dir / "adapter_config.json").exists():
        base_model = None
        for candidate in candidates:
            try:
                base_model = AutoModelForSeq2SeqLM.from_pretrained(candidate, local_files_only=True)
                break
            except Exception:  # noqa: BLE001
                continue
        if base_model is None:
            raise RuntimeError(f"Unable to load base model for LoRA adapter from {candidates}")
        model = PeftModel.from_pretrained(base_model, model_dir.as_posix())
        if merge_lora:
            model = model.merge_and_unload()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir.as_posix(), local_files_only=True)

    tokenizer_candidates = [model_dir.as_posix(), *candidates]
    tokenizer = None
    for source in tokenizer_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=True)
            break
        except Exception:  # noqa: BLE001
            continue
    if tokenizer is None:
        raise RuntimeError(f"Unable to load tokenizer from {tokenizer_candidates}")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def resolve_decoder_layers(
    model,
    requested_layers: Sequence[int],
) -> Tuple[List[int], Dict]:
    """
    Resolve 1-based decoder layers. If requested layers exceed model depth, fallback
    to available decoder layers while recording metadata.
    """
    decoder_depth = int(len(model.decoder.block))
    requested = [int(x) for x in requested_layers]
    valid = [layer for layer in requested if 1 <= layer <= decoder_depth]

    def _upper_half_layers(depth: int) -> List[int]:
        start = max(1, int(depth // 2))
        return list(range(start, depth + 1))

    fallback_used = False
    fallback_reason = None
    requested_max = max(requested) if requested else 0

    if not valid:
        # If all requested layers are out of range, use the upper-half decoder band.
        # For T5-small (depth=6), this yields layers 3..6.
        valid = _upper_half_layers(decoder_depth)
        fallback_used = True
        fallback_reason = "all_requested_layers_out_of_range"
    elif requested_max > decoder_depth and len(valid) < max(2, decoder_depth // 3):
        # If most requested layers are out of range and the surviving overlap is too small,
        # broaden to the upper-half band so steering analysis remains meaningful.
        valid = _upper_half_layers(decoder_depth)
        fallback_used = True
        fallback_reason = "sparse_overlap_with_requested_layers"

    valid = sorted(set(valid))

    metadata = {
        "requested_layers": requested,
        "resolved_layers": valid,
        "decoder_depth": decoder_depth,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }
    return valid, metadata


def decode_labels(tokenizer, labels: Sequence[int]) -> str:
    cleaned = [int(token) for token in labels if int(token) >= 0]
    if not cleaned:
        return ""
    return tokenizer.decode(cleaned, skip_special_tokens=True).strip()


def tensorize_sample(sample: Dict, device):
    import torch

    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
    return input_ids, attention_mask, labels


def extract_decoder_hidden_states(
    decoder_hidden_states: Sequence,
    resolved_layers: Sequence[int],
):
    """
    Extract decoder hidden states using 1-based layer indexing.
    """
    extracted = {}
    for layer in resolved_layers:
        # HF returns decoder_hidden_states with embedding output at index 0.
        hidden = decoder_hidden_states[int(layer)]
        extracted[int(layer)] = hidden.squeeze(0).detach().cpu()
    return extracted


def pool_hidden_states(layer_hidden_states: Dict[int, "torch.Tensor"]) -> Dict[int, "torch.Tensor"]:
    pooled = {}
    for layer, hidden in layer_hidden_states.items():
        pooled[int(layer)] = hidden.mean(dim=0)
    return pooled


def average_activations(vectors: Iterable["torch.Tensor"]) -> "torch.Tensor":
    import torch

    vectors = list(vectors)
    if not vectors:
        raise ValueError("Cannot average empty activation list.")
    stacked = torch.stack(vectors, dim=0)
    return stacked.mean(dim=0)


def normalize_vector(vector: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    import torch

    norm = torch.linalg.norm(vector)
    if float(norm) < eps:
        return vector
    return vector / norm
