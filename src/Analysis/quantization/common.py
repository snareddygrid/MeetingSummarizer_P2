"""
Shared utilities for Task-2 quantization analysis pipeline.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psutil


PROMPT_PREFIX = "Summarize the following conversation:"

QUANTIZATION_LEVELS: Dict[str, int] = {
    "Q4_K_M": 4,
    "Q5_K_M": 5,
    "Q8_0": 8,
}

DEFAULT_GENERATION_CONFIG = {
    "num_beams": 8,
    "max_new_tokens": 140,
    "min_new_tokens": 16,
    "length_penalty": 1.05,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

DEFAULT_LENGTH_BUCKETS = [10, 50, 100, 200]

_ROUGE_METRIC = None
_DEFAULT_EVAL_CACHE = Path(os.environ.get("HF_EVALUATE_CACHE", "outputs/analysis/quantization/.hf_cache"))

_TORCH = None
_AUTO_MODEL = None
_AUTO_TOKENIZER = None
_PEFT_MODEL = None
_DATASETS_LOAD_FROM_DISK = None
_EVALUATE_LOAD = None
_PYARROW = None
_PYARROW_IPC = None


def _import_torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def _import_transformers():
    global _AUTO_MODEL, _AUTO_TOKENIZER
    if _AUTO_MODEL is None or _AUTO_TOKENIZER is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _AUTO_MODEL = AutoModelForSeq2SeqLM
        _AUTO_TOKENIZER = AutoTokenizer
    return _AUTO_MODEL, _AUTO_TOKENIZER


def _import_peft():
    global _PEFT_MODEL
    if _PEFT_MODEL is None:
        from peft import PeftModel

        _PEFT_MODEL = PeftModel
    return _PEFT_MODEL


def _import_datasets_load_from_disk():
    global _DATASETS_LOAD_FROM_DISK
    if _DATASETS_LOAD_FROM_DISK is None:
        from datasets import load_from_disk

        _DATASETS_LOAD_FROM_DISK = load_from_disk
    return _DATASETS_LOAD_FROM_DISK


def _import_evaluate_load():
    global _EVALUATE_LOAD
    if _EVALUATE_LOAD is None:
        from evaluate import load as load_metric

        _EVALUATE_LOAD = load_metric
    return _EVALUATE_LOAD


def _import_pyarrow():
    global _PYARROW, _PYARROW_IPC
    if _PYARROW is None or _PYARROW_IPC is None:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        _PYARROW = pa
        _PYARROW_IPC = ipc
    return _PYARROW, _PYARROW_IPC


def ensure_output_structure(output_root: Path) -> Dict[str, Path]:
    paths = {
        "models": output_root / "models",
        "latency": output_root / "latency",
        "throughput": output_root / "throughput",
        "rouge": output_root / "rouge",
        "streaming": output_root / "streaming",
        "batch": output_root / "batch",
        "parallel": output_root / "parallel",
        "reports": output_root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_device(prefer_mps: bool = True) -> torch.device:
    torch = _import_torch()
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    torch = _import_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _read_adapter_base_model(model_dir: Path, default_base_model: str) -> str:
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.exists():
        return default_base_model
    with adapter_config.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("base_model_name_or_path", default_base_model)


def _resolve_tokenizer_source(model_dir: Path, default_source: str) -> str:
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "spiece.model",
        "vocab.json",
    ]
    if any((model_dir / name).exists() for name in tokenizer_files):
        return model_dir.as_posix()
    return default_source


def _load_base_model_from_candidates(candidates: Sequence[str]):
    AutoModelForSeq2SeqLM, _ = _import_transformers()
    selected_source = None
    base_model = None
    last_error = None
    for candidate in candidates:
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(candidate, local_files_only=True)
            selected_source = candidate
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
            continue

    if base_model is None:
        raise RuntimeError(
            "Unable to load base model locally for LoRA model. "
            f"Tried candidates={list(candidates)}; last_error={last_error}"
        )
    return base_model, selected_source


def load_lora_model_and_tokenizer(
    model_dir: Path,
    default_base_model: str = "t5-small",
    fallback_local_base_model: Optional[str] = "experiments/t5_small_optimized",
    device: Optional[torch.device] = None,
    merge_lora: bool = False,
):
    """
    Load T5-small LoRA model and tokenizer from local artifacts only.
    """
    model_dir = Path(model_dir)
    PeftModel = _import_peft()
    _, AutoTokenizer = _import_transformers()
    device = device or get_device()

    base_model_name = _read_adapter_base_model(model_dir, default_base_model)
    candidates: List[str] = [base_model_name]
    if fallback_local_base_model:
        fallback_candidate = Path(fallback_local_base_model).as_posix()
        if fallback_candidate not in candidates:
            candidates.append(fallback_candidate)

    model = None
    selected_base_source = None

    if (model_dir / "adapter_config.json").exists():
        base_model, selected_base_source = _load_base_model_from_candidates(candidates)
        model = PeftModel.from_pretrained(base_model, model_dir.as_posix())
        if merge_lora:
            model = model.merge_and_unload()
    else:
        load_candidates = [model_dir.as_posix(), *candidates]
        model, selected_base_source = _load_base_model_from_candidates(load_candidates)

    tokenizer_candidates = [_resolve_tokenizer_source(model_dir, selected_base_source or base_model_name)]
    tokenizer_candidates.extend([c for c in candidates if c not in tokenizer_candidates])

    tokenizer = None
    selected_tokenizer_source = None
    tokenizer_error = None
    for source in tokenizer_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=True)
            selected_tokenizer_source = source
            break
        except Exception as error:  # noqa: BLE001
            tokenizer_error = error
            continue

    if tokenizer is None:
        raise RuntimeError(
            "Unable to load tokenizer locally. "
            f"Tried candidates={tokenizer_candidates}; last_error={tokenizer_error}"
        )

    model.to(device)
    model.eval()

    load_info = {
        "base_model_candidates": candidates,
        "selected_base_source": selected_base_source,
        "tokenizer_source_candidates": tokenizer_candidates,
        "selected_tokenizer_source": selected_tokenizer_source,
    }
    return model, tokenizer, device, load_info


def _load_split_with_pyarrow(dataset_path: str, split: str) -> List[Dict[str, Any]]:
    pa, ipc = _import_pyarrow()
    split_dir = Path(dataset_path) / split
    state_path = split_dir / "state.json"
    data_files: List[Path] = []

    if state_path.exists():
        payload = load_json(state_path)
        for row in payload.get("_data_files", []):
            filename = row.get("filename")
            if filename:
                data_files.append(split_dir / filename)

    if not data_files:
        data_files = sorted(split_dir.glob("data-*.arrow"))

    if not data_files:
        raise FileNotFoundError(f"No Arrow data files found under {split_dir.as_posix()}")

    rows: List[Dict[str, Any]] = []
    for arrow_file in data_files:
        with pa.memory_map(arrow_file.as_posix(), "r") as source:
            reader = ipc.open_stream(source)
            table = reader.read_all()
        rows.extend(table.to_pylist())
    return rows


def _load_split(dataset_path: str, split: str):
    arrow_error = None
    try:
        return _load_split_with_pyarrow(dataset_path=dataset_path, split=split)
    except Exception as error:  # noqa: BLE001
        arrow_error = error

    allow_datasets_import = os.environ.get("QUANTIZATION_ALLOW_DATASETS_IMPORT", "0") == "1"
    if not allow_datasets_import:
        raise RuntimeError(
            "Unable to load dataset split. "
            f"dataset_path={dataset_path}, split={split}, arrow_error={arrow_error}. "
            "Set QUANTIZATION_ALLOW_DATASETS_IMPORT=1 to allow Hugging Face datasets fallback."
        ) from arrow_error

    try:
        load_from_disk = _import_datasets_load_from_disk()
        dataset = load_from_disk(dataset_path)
        if split in dataset:
            return dataset[split]
        raise KeyError(
            f"Split '{split}' not found in dataset at {dataset_path}. Available: {list(dataset.keys())}"
        )
    except Exception as hf_error:  # noqa: BLE001
        raise RuntimeError(
            "Unable to load dataset split from both pyarrow and datasets backends. "
            f"dataset_path={dataset_path}, split={split}, arrow_error={arrow_error}, hf_error={hf_error}"
        ) from hf_error


def load_raw_split(dataset_path: str, split: str):
    return _load_split(dataset_path=dataset_path, split=split)


def get_fixed_test_subset(
    n: int = 100,
    processed_dataset_path: str = "data/processed",
    raw_dataset_path: str = "data/raw",
    split: str = "test",
    subset_indices_path: Optional[Path] = None,
    selection_mode: str = "first_n",
    random_seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    """
    Build one reproducible subset of test samples from processed indices and map to raw dialogues.
    """
    processed_split = _load_split(dataset_path=processed_dataset_path, split=split)
    raw_split = _load_split(dataset_path=raw_dataset_path, split=split)
    if len(processed_split) != len(raw_split):
        raise RuntimeError(
            "Processed/raw split size mismatch. "
            f"processed={len(processed_split)}, raw={len(raw_split)}, split={split}"
        )

    total_size = len(processed_split)
    subset_size = min(n, total_size)

    indices: List[int]
    if subset_indices_path is not None and subset_indices_path.exists():
        saved = load_json(subset_indices_path)
        saved_indices = saved.get("subset_indices", [])
        if isinstance(saved_indices, list) and len(saved_indices) >= subset_size:
            indices = [int(idx) for idx in saved_indices[:subset_size]]
        else:
            indices = []
    else:
        indices = []

    if not indices:
        if selection_mode == "random_seed":
            rng = random.Random(random_seed)
            indices = sorted(rng.sample(range(total_size), subset_size))
        else:
            indices = list(range(subset_size))

    subset_rows: List[Dict] = []
    for idx in indices:
        row = raw_split[idx]
        sample_id = row["id"] if "id" in row else f"{split}_{idx:04d}"
        subset_rows.append(
            {
                "sample_id": sample_id,
                "source_index": int(idx),
                "dialogue": str(row["dialogue"]),
                "summary": str(row["summary"]),
            }
        )

    metadata = {
        "split": split,
        "selection_mode": selection_mode,
        "random_seed": int(random_seed),
        "num_samples": len(indices),
        "dataset_size": len(indices),  # standardized benchmarking dataset size
        "full_test_size": total_size,
        "subset_indices": indices,
        "processed_dataset_path": processed_dataset_path,
        "raw_dataset_path": raw_dataset_path,
    }
    if subset_indices_path is not None:
        save_json(subset_indices_path, metadata)

    return subset_rows, metadata


def attach_subset_metadata(payload: Dict, subset_metadata: Dict, num_samples: Optional[int] = None) -> Dict:
    updated = dict(payload)
    updated["num_samples"] = int(num_samples if num_samples is not None else updated.get("num_samples", 0))
    updated["dataset_size"] = int(subset_metadata.get("dataset_size", 100))
    updated["subset_indices"] = [int(idx) for idx in subset_metadata.get("subset_indices", [])]
    updated["subset_split"] = subset_metadata.get("split", "test")
    updated["subset_selection_mode"] = subset_metadata.get("selection_mode", "first_n")
    return updated


def split_dialogue_turns(dialogue_text: str) -> List[str]:
    lines = [line.strip() for line in str(dialogue_text).splitlines() if line.strip()]
    if lines:
        return lines
    flattened = str(dialogue_text).strip()
    if flattened:
        return [flattened]
    return []


def reshape_dialogue_for_utterances(dialogue_text: str, target_utterances: int) -> str:
    turns = split_dialogue_turns(dialogue_text)
    if not turns:
        return ""
    if target_utterances <= 0:
        return "\n".join(turns)

    if len(turns) >= target_utterances:
        return "\n".join(turns[:target_utterances])

    expanded = list(turns)
    cursor = 0
    repeat_round = 1
    while len(expanded) < target_utterances:
        original = turns[cursor % len(turns)]
        if ":" in original:
            speaker, utterance = original.split(":", 1)
            repeated = f"{speaker.strip()}: {utterance.strip()} (repeat {repeat_round})"
        else:
            repeated = f"{original} (repeat {repeat_round})"
        expanded.append(repeated)
        cursor += 1
        if cursor % len(turns) == 0:
            repeat_round += 1
    return "\n".join(expanded[:target_utterances])


def build_prompt(dialogue_text: str) -> str:
    return f"{PROMPT_PREFIX}\n{dialogue_text.strip()}"


def truncate_dialogue_for_context(dialogue_text: str, max_input_length: int) -> str:
    """
    Approximate token-safe truncation for llama.cpp CLI runs without tokenizer access.
    Keeps the most recent turns and preserves structure.
    """
    text = str(dialogue_text or "").strip()
    if not text:
        return text

    # Conservative heuristics to avoid llama.cpp prompt-limit assertions.
    max_chars = max(int(max_input_length * 2.8), 384)
    max_words = max(int(max_input_length * 0.55), 96)
    if len(text) <= max_chars and len(text.split()) <= max_words:
        return text

    turns = split_dialogue_turns(text)
    if not turns:
        return text[-max_chars:]

    selected_reversed: List[str] = []
    used_chars = 0
    used_words = 0
    # Reserve space for prompt prefix and "Summary:" suffix.
    char_budget = max(max_chars - len(PROMPT_PREFIX) - 64, 128)
    word_budget = max(max_words - 16, 48)
    for turn in reversed(turns):
        turn_words = len(turn.split())
        projected_chars = used_chars + len(turn) + (1 if selected_reversed else 0)
        projected_words = used_words + turn_words
        if (projected_chars > char_budget or projected_words > word_budget) and selected_reversed:
            break
        selected_reversed.append(turn)
        used_chars = projected_chars
        used_words = projected_words
        if projected_chars >= char_budget or projected_words >= word_budget:
            break

    selected = list(reversed(selected_reversed))
    return "\n".join(selected)


def merge_generation_config(user_config: Optional[Dict]) -> Dict:
    merged = dict(DEFAULT_GENERATION_CONFIG)
    if user_config:
        merged.update(user_config)
    return merged


def resolve_executable_path(preferred: Optional[str], candidates: Sequence[str]) -> str:
    if preferred:
        direct = Path(preferred)
        if direct.exists():
            return direct.as_posix()
        discovered = shutil.which(preferred)
        if discovered:
            return discovered
        raise FileNotFoundError(f"Executable not found: {preferred}")

    for candidate in candidates:
        discovered = shutil.which(candidate)
        if discovered:
            return discovered
    raise FileNotFoundError(f"No executable found from candidates={list(candidates)}")


def parse_llama_completion_output(stdout: str, stderr: str) -> str:
    # llama-completion prints generated text to stdout. Keep parsing strict and predictable.
    text = str(stdout or "").strip()
    if text:
        text = text.replace("[end of text]", "").strip()
        return " ".join(text.split())

    # Fallback path if output was routed unexpectedly.
    combined = f"{stdout or ''}\n{stderr or ''}"
    lines = []
    for line in combined.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("load_backend:"):
            continue
        if line.startswith("build:"):
            continue
        if line.endswith("[end of text]"):
            line = line.replace("[end of text]", "").strip()
        if not line:
            continue
        lines.append(line)
    return " ".join(lines).strip()


def generate_summary_llama_cpp(
    model_path: Path,
    dialogue_text: str,
    generation_config: Optional[Dict] = None,
    max_input_length: int = 1024,
    llama_completion_binary: Optional[str] = None,
    llama_device: str = "BLAS",
    llama_n_gpu_layers: int = 0,
    seed: int = 42,
    threads: Optional[int] = None,
    measure_peak_memory: bool = False,
) -> Tuple[str, float, float, Dict]:
    config = merge_generation_config(generation_config)
    max_new_tokens = int(config.get("max_new_tokens", DEFAULT_GENERATION_CONFIG["max_new_tokens"]))
    temperature = float(config.get("temperature", 0.0))
    top_k = int(config.get("top_k", 40))
    top_p = float(config.get("top_p", 0.95))

    binary = resolve_executable_path(
        preferred=llama_completion_binary,
        candidates=("llama-completion", "llama-cli"),
    )
    binary_name = Path(binary).name

    env = os.environ.copy()
    env.setdefault("LLAMA_OFFLINE", "1")

    start = time.perf_counter()
    peak_memory_bytes = 0
    last_stdout = ""
    last_stderr = ""
    last_command: List[str] = []
    effective_input_length = max(int(max_input_length), 128)

    for attempt in range(3):
        dialogue_budget_tokens = max(effective_input_length - 96, 128)
        safe_dialogue = truncate_dialogue_for_context(
            dialogue_text=dialogue_text,
            max_input_length=dialogue_budget_tokens,
        )
        prompt = f"{build_prompt(safe_dialogue)}\nSummary:"
        ctx_size = max(int(effective_input_length + max_new_tokens + 64), 512)

        command = [
            binary,
            "-m",
            Path(model_path).as_posix(),
            "--device",
            str(llama_device),
            "--split-mode",
            "none",
            "--n-gpu-layers",
            str(int(llama_n_gpu_layers)),
            "--main-gpu",
            "0",
            "--simple-io",
            "--no-display-prompt",
            "--no-perf",
            "--no-warmup",
            "--verbosity",
            "1",
            "-p",
            prompt,
            "-n",
            str(max_new_tokens),
            "--temp",
            str(temperature),
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--seed",
            str(int(seed)),
            "--ctx-size",
            str(ctx_size),
            "--batch-size",
            str(ctx_size),
            "--ubatch-size",
            str(ctx_size),
        ]
        if threads is not None and int(threads) > 0:
            command.extend(["--threads", str(int(threads)), "--threads-batch", str(int(threads))])
        if binary_name == "llama-cli":
            command.extend(["--single-turn"])

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        process_handle = None
        if measure_peak_memory:
            try:
                process_handle = psutil.Process(proc.pid)
            except Exception:  # noqa: BLE001
                process_handle = None

        while proc.poll() is None:
            if process_handle is not None:
                try:
                    peak_memory_bytes = max(peak_memory_bytes, process_handle.memory_info().rss)
                except Exception:  # noqa: BLE001
                    process_handle = None
            time.sleep(0.005)

        stdout, stderr = proc.communicate()
        last_stdout = stdout
        last_stderr = stderr
        last_command = command

        if process_handle is not None:
            try:
                peak_memory_bytes = max(peak_memory_bytes, process_handle.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass

        if proc.returncode == 0:
            latency = time.perf_counter() - start
            summary = parse_llama_completion_output(stdout=stdout, stderr=stderr)
            peak_memory_mb = float(peak_memory_bytes / (1024 ** 2))
            debug = {
                "binary": binary,
                "binary_name": binary_name,
                "attempt": int(attempt + 1),
                "command": command,
                "stdout_tail": stdout[-500:],
                "stderr_tail": stderr[-500:],
            }
            return summary, latency, peak_memory_mb, debug

        prompt_too_long = "prompt is too long" in str(stderr).lower()
        if prompt_too_long and attempt < 2:
            effective_input_length = max(int(effective_input_length * 0.75), 256)
            continue

        break

    latency = time.perf_counter() - start
    raise RuntimeError(
        "llama.cpp completion failed "
        f"(model={Path(model_path).name}, command={last_command}, stderr_tail={last_stderr[-800:]}, "
        f"stdout_tail={last_stdout[-200:]}, latency_sec={latency:.3f})"
    )


def generate_summary(
    model,
    tokenizer,
    device: torch.device,
    dialogue_text: str,
    generation_config: Optional[Dict] = None,
    max_input_length: int = 1024,
) -> Tuple[str, float]:
    torch = _import_torch()
    prompt = build_prompt(dialogue_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    config = merge_generation_config(generation_config)
    with torch.no_grad():
        start = time.perf_counter()
        output_ids = model.generate(**inputs, **config)
        synchronize_device(device)
        latency = time.perf_counter() - start

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return summary, latency


def _normalize_rouge_score(value) -> float:
    if hasattr(value, "mid"):
        return float(value.mid.fmeasure)
    return float(value)


def _tokenize_rouge(text: str) -> List[str]:
    return [token for token in re.split(r"\s+", str(text).strip().lower()) if token]


def _f1_from_counts(overlap: int, predicted_total: int, reference_total: int) -> float:
    if predicted_total <= 0 or reference_total <= 0 or overlap <= 0:
        return 0.0
    precision = float(overlap / predicted_total)
    recall = float(overlap / reference_total)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _rouge_n_single(prediction: str, reference: str, n: int) -> float:
    pred_tokens = _tokenize_rouge(prediction)
    ref_tokens = _tokenize_rouge(reference)
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    pred_counts = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_counts = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum(min(pred_counts[key], ref_counts.get(key, 0)) for key in pred_counts)
    return _f1_from_counts(
        overlap=overlap,
        predicted_total=sum(pred_counts.values()),
        reference_total=sum(ref_counts.values()),
    )


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0]
        for idx, other in enumerate(b, start=1):
            if token == other:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_single(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_rouge(prediction)
    ref_tokens = _tokenize_rouge(reference)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    return _f1_from_counts(
        overlap=lcs,
        predicted_total=len(pred_tokens),
        reference_total=len(ref_tokens),
    )


def _compute_rouge_scores_fallback(predictions: List[str], references: List[str]) -> Dict[str, float]:
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    total = min(len(predictions), len(references))
    rouge1 = 0.0
    rouge2 = 0.0
    rouge_l = 0.0
    for idx in range(total):
        pred = predictions[idx]
        ref = references[idx]
        rouge1 += _rouge_n_single(prediction=pred, reference=ref, n=1)
        rouge2 += _rouge_n_single(prediction=pred, reference=ref, n=2)
        rouge_l += _rouge_l_single(prediction=pred, reference=ref)
    return {
        "rouge1": float(rouge1 / total),
        "rouge2": float(rouge2 / total),
        "rougeL": float(rouge_l / total),
    }


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    global _ROUGE_METRIC
    use_evaluate = os.environ.get("QUANTIZATION_USE_EVALUATE_ROUGE", "0") == "1"
    if _ROUGE_METRIC is None and use_evaluate:
        try:
            _DEFAULT_EVAL_CACHE.mkdir(parents=True, exist_ok=True)
            load_metric = _import_evaluate_load()
            _ROUGE_METRIC = load_metric("rouge", cache_dir=_DEFAULT_EVAL_CACHE.as_posix())
        except Exception:  # noqa: BLE001
            _ROUGE_METRIC = False

    if _ROUGE_METRIC:
        result = _ROUGE_METRIC.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )

        return {
            "rouge1": _normalize_rouge_score(result["rouge1"]),
            "rouge2": _normalize_rouge_score(result["rouge2"]),
            "rougeL": _normalize_rouge_score(result["rougeL"]),
        }

    return _compute_rouge_scores_fallback(predictions=predictions, references=references)


def compute_latency_stats(latencies: Iterable[float]) -> Dict[str, float]:
    values = sorted(float(x) for x in latencies)
    if not values:
        return {
            "count": 0,
            "mean_sec": 0.0,
            "median_sec": 0.0,
            "p95_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
            "total_sec": 0.0,
        }

    count = len(values)
    p95_index = min(int(0.95 * (count - 1)), count - 1)

    return {
        "count": count,
        "mean_sec": float(sum(values) / count),
        "median_sec": float(values[count // 2]),
        "p95_sec": float(values[p95_index]),
        "min_sec": float(values[0]),
        "max_sec": float(values[-1]),
        "total_sec": float(sum(values)),
    }


def collect_test_samples(
    raw_split: Sequence[Dict],
    num_samples: int,
    target_utterances: Optional[int] = None,
) -> List[Dict]:
    samples: List[Dict] = []
    total = min(num_samples, len(raw_split))

    for idx in range(total):
        row = raw_split[idx]
        dialogue = str(row["dialogue"])
        if target_utterances is not None:
            dialogue = reshape_dialogue_for_utterances(dialogue, target_utterances)

        sample_id = row["id"] if "id" in row else f"sample_{idx:04d}"
        samples.append(
            {
                "sample_id": sample_id,
                "dialogue": dialogue,
                "summary": str(row["summary"]),
            }
        )
    return samples


def collect_subset_samples(
    subset_rows: Sequence[Dict],
    num_samples: int,
    target_utterances: Optional[int] = None,
) -> List[Dict]:
    samples: List[Dict] = []
    total = min(num_samples, len(subset_rows))
    for idx in range(total):
        row = subset_rows[idx]
        dialogue = str(row["dialogue"])
        if target_utterances is not None:
            dialogue = reshape_dialogue_for_utterances(dialogue, target_utterances)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "source_index": int(row.get("source_index", idx)),
                "dialogue": dialogue,
                "summary": str(row["summary"]),
            }
        )
    return samples


def collect_length_bucket_samples(
    raw_split: Sequence[Dict],
    lengths: Sequence[int],
    samples_per_length: int,
) -> Dict[int, List[Dict]]:
    total = min(samples_per_length, len(raw_split))
    rows = [raw_split[idx] for idx in range(total)]

    buckets: Dict[int, List[Dict]] = {}
    for length in lengths:
        bucket: List[Dict] = []
        for idx, row in enumerate(rows):
            sample_id = row["id"] if "id" in row else f"sample_{idx:04d}"
            dialogue = reshape_dialogue_for_utterances(row["dialogue"], length)
            bucket.append(
                {
                    "sample_id": sample_id,
                    "dialogue": dialogue,
                    "summary": str(row["summary"]),
                    "target_utterances": int(length),
                }
            )
        buckets[int(length)] = bucket
    return buckets


def collect_length_bucket_samples_from_subset(
    subset_rows: Sequence[Dict],
    lengths: Sequence[int],
    samples_per_length: int,
) -> Dict[int, List[Dict]]:
    total = min(samples_per_length, len(subset_rows))
    rows = list(subset_rows[:total])

    buckets: Dict[int, List[Dict]] = {}
    for length in lengths:
        bucket: List[Dict] = []
        for row in rows:
            dialogue = reshape_dialogue_for_utterances(str(row["dialogue"]), length)
            bucket.append(
                {
                    "sample_id": row["sample_id"],
                    "source_index": int(row.get("source_index", -1)),
                    "dialogue": dialogue,
                    "summary": str(row["summary"]),
                    "target_utterances": int(length),
                }
            )
        buckets[int(length)] = bucket
    return buckets
