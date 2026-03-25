"""
Convert T5-small LoRA model to GGUF if possible, otherwise export fallback artifacts.

Run:
    python src/analysis/quantization/convert_to_gguf.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from common import ensure_output_structure, load_lora_model_and_tokenizer, save_json


_TORCH = None


def _import_torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def _try_llama_cpp_conversion(
    hf_model_dir: Path,
    output_gguf_path: Path,
    llama_cpp_dir: Optional[Path],
) -> Dict:
    converter_script = None
    if llama_cpp_dir is not None:
        candidate = llama_cpp_dir / "convert_hf_to_gguf.py"
        if candidate.exists():
            converter_script = candidate
        else:
            return {
                "status": "skipped",
                "reason": f"Converter script not found at {candidate.as_posix()}",
            }

    if converter_script is None:
        discovered = shutil.which("convert_hf_to_gguf.py")
        if discovered:
            converter_script = Path(discovered)
        else:
            return {"status": "skipped", "reason": "convert_hf_to_gguf.py not found in PATH"}

    command = [
        sys.executable,
        Path(converter_script).as_posix(),
        hf_model_dir.as_posix(),
        "--outfile",
        output_gguf_path.as_posix(),
        "--outtype",
        "f16",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "status": "success",
            "gguf_path": output_gguf_path.as_posix(),
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    except Exception as error:  # noqa: BLE001
        return {"status": "failed", "reason": str(error)}


def _export_torchscript_fallback(model, tokenizer, output_dir: Path) -> Dict:
    torch = _import_torch()
    nn = torch.nn

    class _ForwardWrapper(nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.model = wrapped_model

        def forward(self, input_ids, attention_mask, decoder_input_ids):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True,
                use_cache=False,
            )
            return outputs.logits

    wrapper = _ForwardWrapper(model.cpu().eval())
    script_path = output_dir / "model_forward.ts.pt"

    dummy_inputs = tokenizer(
        "Summarize the following conversation:\nAlice: hello\nBob: hi",
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long)

    try:
        traced = torch.jit.trace(
            wrapper,
            (
                dummy_inputs["input_ids"],
                dummy_inputs["attention_mask"],
                decoder_input_ids,
            ),
            strict=False,
        )
        traced.save(script_path.as_posix())
        return {"status": "success", "artifact_path": script_path.as_posix(), "format": "torchscript"}
    except Exception as error:  # noqa: BLE001
        state_path = output_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), state_path.as_posix())
        return {
            "status": "fallback_state_dict",
            "artifact_path": state_path.as_posix(),
            "format": "state_dict",
            "reason": str(error),
        }


def convert_or_fallback(
    model_dir: Path,
    output_root: Path,
    llama_cpp_dir: Optional[Path] = None,
    default_base_model: str = "t5-small",
    fallback_local_base_model: str = "experiments/t5_small_optimized",
) -> Dict:
    output_dirs = ensure_output_structure(output_root)
    models_dir = output_dirs["models"]
    intermediate_dir = models_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    hf_export_dir = intermediate_dir / "hf_merged"
    gguf_path = intermediate_dir / "model.gguf"

    # Fast path for local/offline benchmark reruns where conversion artifacts already exist.
    if hf_export_dir.exists() and gguf_path.exists():
        metadata = {
            "source_model_dir": Path(model_dir).as_posix(),
            "hf_export_dir": hf_export_dir.as_posix(),
            "gguf_result": {
                "status": "reused",
                "gguf_path": gguf_path.as_posix(),
                "reason": "existing_artifact",
            },
            "fallback_result": None,
            "load_info": {
                "reuse_existing_artifacts": True,
            },
        }
        save_json(models_dir / "conversion_metadata.json", metadata)
        return metadata

    torch = _import_torch()
    model, tokenizer, _, load_info = load_lora_model_and_tokenizer(
        model_dir=Path(model_dir),
        default_base_model=default_base_model,
        fallback_local_base_model=fallback_local_base_model,
        device=torch.device("cpu"),
        merge_lora=True,
    )

    model.save_pretrained(hf_export_dir.as_posix())
    tokenizer.save_pretrained(hf_export_dir.as_posix())

    gguf_result = _try_llama_cpp_conversion(
        hf_model_dir=hf_export_dir,
        output_gguf_path=gguf_path,
        llama_cpp_dir=llama_cpp_dir,
    )

    fallback_result = None
    if gguf_result.get("status") != "success":
        fallback_result = _export_torchscript_fallback(
            model=model,
            tokenizer=tokenizer,
            output_dir=intermediate_dir,
        )

    metadata = {
        "source_model_dir": Path(model_dir).as_posix(),
        "hf_export_dir": hf_export_dir.as_posix(),
        "gguf_result": gguf_result,
        "fallback_result": fallback_result,
        "load_info": load_info,
    }
    save_json(models_dir / "conversion_metadata.json", metadata)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HF T5 LoRA model to GGUF or fallback artifact.")
    parser.add_argument("--model-dir", default="experiments/t5_small_lora")
    parser.add_argument("--output-root", default="outputs/analysis/quantization")
    parser.add_argument("--llama-cpp-dir", default=None)
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--base-model-fallback", default="experiments/t5_small_optimized")
    return parser.parse_args()


def main():
    args = parse_args()
    llama_cpp_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else None
    metadata = convert_or_fallback(
        model_dir=Path(args.model_dir),
        output_root=Path(args.output_root),
        llama_cpp_dir=llama_cpp_dir,
        default_base_model=args.default_base_model,
        fallback_local_base_model=args.base_model_fallback,
    )
    print("Conversion complete.")
    print(f"Metadata saved with gguf status={metadata['gguf_result'].get('status')}")


if __name__ == "__main__":
    main()
