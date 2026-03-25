"""Measure per-sample inference latency across LoRA rank checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference import generate_summary  # noqa: E402

from evaluate_ranks import load_rank_model  # noqa: E402
from utils.json_utils import rank_key, rank_value, save_json  # noqa: E402
from utils.timing_utils import set_seed, summarize_latencies, time_callable  # noqa: E402


DEFAULT_RANKS = [2, 4, 8, 16, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Measure latency for LoRA rank models.")
    parser.add_argument("--raw-data-path", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--default-base-model", default="t5-small")
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--output-path",
        default="outputs/analysis/rank_ablation/latency/latency.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = load_from_disk((PROJECT_ROOT / args.raw_data_path).as_posix())[args.split]
    num_samples = min(args.num_samples, len(dataset))

    payload = {}
    for rank in tqdm(args.ranks, desc="Latency by rank"):
        key = rank_key(rank)
        model_dir = (PROJECT_ROOT / args.experiments_root / f"t5_small_lora_r{rank}").resolve()
        if not model_dir.exists():
            payload[key] = {"status": "missing_model_dir"}
            continue

        model, tokenizer, device = load_rank_model(
            model_dir=model_dir,
            default_base_model=args.default_base_model,
        )

        latencies = []
        for idx in tqdm(range(num_samples), desc=f"Latency {key}", leave=False):
            dialogue = str(dataset[idx]["dialogue"])
            _, elapsed = time_callable(
                lambda d=dialogue: generate_summary(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    text=d,
                    architecture="t5-small",
                ),
                device=device,
            )
            latencies.append(float(elapsed))

        stats = summarize_latencies(latencies)
        stats["num_samples"] = int(num_samples)
        stats["status"] = "ok"
        payload[key] = stats

    ordered = dict(sorted(payload.items(), key=lambda item: rank_value(item[0])))
    output_path = (PROJECT_ROOT / args.output_path).resolve()
    save_json(output_path, ordered)
    print(f"Saved latency metrics: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
