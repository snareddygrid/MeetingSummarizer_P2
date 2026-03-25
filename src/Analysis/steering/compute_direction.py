"""
Compute steering direction from extracted activations.

Run:
    python src/analysis/steering/compute_direction.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from utils.activation_utils import ensure_dir, load_json, normalize_vector, save_json


ACTION_LABEL_PATTERN = re.compile(
    r"\b(will|should|plan|assign|need to|going to)\b",
    flags=re.IGNORECASE,
)
VALID_LABELS = {"action", "topic"}


def _classify_sample(reference_summary: str) -> str:
    text = str(reference_summary or "").strip()
    if ACTION_LABEL_PATTERN.search(text):
        return "action"
    return "topic"


def _is_high_quality_summary(reference_summary: str, min_tokens: int) -> bool:
    text = str(reference_summary or "").strip()
    if not text:
        return False
    tokens = [token for token in re.split(r"\s+", text) if token]
    if len(tokens) < int(min_tokens):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    return True


def _load_activation_paths(activations_dir: Path, activation_index_path: Path) -> List[Path]:
    if activation_index_path.exists():
        payload = load_json(activation_index_path)
        index_rows = payload.get("activation_index", [])
        if index_rows:
            return [Path(row["activation_path"]) for row in index_rows]
    return sorted(activations_dir.glob("*.pt"))


def _normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in VALID_LABELS:
        return label
    return None


def _load_label_map(label_path: Optional[Path]) -> Dict[str, str]:
    if label_path is None or not label_path.exists():
        return {}

    payload = load_json(label_path)
    label_map: Dict[str, str] = {}

    def _ingest_row(row: Dict):
        sample_id = row.get("sample_id", row.get("id"))
        source_index = row.get("source_index")
        label = _normalize_label(row.get("label"))
        if label is None:
            return
        if sample_id is not None:
            label_map[str(sample_id)] = label
        if source_index is not None:
            label_map[f"source_index:{int(source_index)}"] = label

    if isinstance(payload, dict):
        if "labels" in payload and isinstance(payload["labels"], dict):
            for key, value in payload["labels"].items():
                label = _normalize_label(value)
                if label is not None:
                    label_map[str(key)] = label
        elif "records" in payload and isinstance(payload["records"], list):
            for row in payload["records"]:
                if isinstance(row, dict):
                    _ingest_row(row)
        else:
            for key, value in payload.items():
                label = _normalize_label(value)
                if label is not None:
                    label_map[str(key)] = label
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                _ingest_row(row)

    return label_map


def _build_balanced_subset(rows: List[Dict], min_class_samples: int) -> List[Dict]:
    action_rows = [row for row in rows if row["label"] == "action"]
    topic_rows = [row for row in rows if row["label"] == "topic"]
    balanced_size = min(len(action_rows), len(topic_rows))
    if balanced_size < int(min_class_samples):
        raise RuntimeError(
            "Insufficient balanced samples after filtering. "
            f"action={len(action_rows)} topic={len(topic_rows)} min_class_samples={min_class_samples}"
        )
    action_rows = sorted(action_rows, key=lambda row: row["sample_id"])[:balanced_size]
    topic_rows = sorted(topic_rows, key=lambda row: row["sample_id"])[:balanced_size]
    return sorted([*action_rows, *topic_rows], key=lambda row: row["sample_id"])


def run_direction_computation(args) -> Dict:
    import torch
    import torch.nn.functional as F

    output_root = Path(args.output_root)
    activations_dir = output_root / "activations"
    reports_dir = ensure_dir(output_root / "reports")
    directions_dir = ensure_dir(output_root / "directions")

    activation_paths = _load_activation_paths(
        activations_dir=activations_dir,
        activation_index_path=reports_dir / "activation_extraction.json",
    )
    if not activation_paths:
        raise FileNotFoundError(f"No activation files found in {activations_dir.as_posix()}")

    label_map = _load_label_map(Path(args.label_path) if args.label_path else None)
    labeled_count = 0
    heuristic_count = 0

    sample_rows: List[Dict] = []
    dropped_low_quality = 0

    for path in tqdm(activation_paths, desc="Compute Direction"):
        payload = torch.load(path, map_location="cpu")
        reference = str(payload.get("reference_summary", ""))
        sample_id = payload.get("sample_id", path.stem)
        source_index = payload.get("source_index")

        if not _is_high_quality_summary(reference_summary=reference, min_tokens=args.min_summary_tokens):
            dropped_low_quality += 1
            continue

        label = label_map.get(str(sample_id))
        if label is None and source_index is not None:
            label = label_map.get(f"source_index:{int(source_index)}")
        if label is None:
            label = _classify_sample(reference_summary=reference)
            heuristic_count += 1
        else:
            labeled_count += 1

        sample_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "reference_summary": reference,
            }
        )

    if not sample_rows:
        raise RuntimeError("No usable samples available after filtering.")

    selected_rows = _build_balanced_subset(rows=sample_rows, min_class_samples=args.min_class_samples)
    selected_ids = {row["sample_id"] for row in selected_rows}
    label_by_id = {row["sample_id"]: row["label"] for row in selected_rows}
    action_count = sum(1 for row in selected_rows if row["label"] == "action")
    topic_count = sum(1 for row in selected_rows if row["label"] == "topic")

    token_sums = {"action": {}, "topic": {}}
    token_counts = {"action": {}, "topic": {}}
    selected_rows_preview = []

    for path in tqdm(activation_paths, desc="Aggregate Direction"):
        payload = torch.load(path, map_location="cpu")
        sample_id = payload.get("sample_id", path.stem)
        if sample_id not in selected_ids:
            continue
        label = label_by_id[sample_id]
        hidden_states = payload.get("decoder_hidden_states", {})
        for layer_key, hidden in hidden_states.items():
            layer = int(layer_key)
            hidden = hidden.float()
            if hidden.ndim != 2:
                hidden = hidden.view(-1, hidden.shape[-1])
            hidden = F.normalize(hidden, p=2, dim=-1, eps=1e-8)
            layer_sum = hidden.sum(dim=0)
            layer_count = int(hidden.shape[0])
            if layer not in token_sums[label]:
                token_sums[label][layer] = layer_sum
                token_counts[label][layer] = layer_count
            else:
                token_sums[label][layer] += layer_sum
                token_counts[label][layer] += layer_count
        if len(selected_rows_preview) < 10:
            selected_rows_preview.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "reference_summary": str(payload.get("reference_summary", "")),
                }
            )

    direction_by_layer = {}
    direction_norms = {}
    action_layers = set(token_sums["action"].keys())
    topic_layers = set(token_sums["topic"].keys())
    common_layers = sorted(action_layers & topic_layers)
    if not common_layers:
        raise RuntimeError("No common layers found for action/topic direction computation.")

    for layer in common_layers:
        action_mean = token_sums["action"][layer] / max(1, token_counts["action"][layer])
        topic_mean = token_sums["topic"][layer] / max(1, token_counts["topic"][layer])
        direction = action_mean - topic_mean
        if args.normalize_direction:
            direction = normalize_vector(direction)
        direction_by_layer[int(layer)] = direction
        direction_norms[str(layer)] = float(torch.linalg.norm(direction))

    direction_payload = {
        "direction_by_layer": direction_by_layer,
        "metadata": {
            "action_count": action_count,
            "topic_count": topic_count,
            "layers": sorted(direction_by_layer.keys()),
            "normalized": bool(args.normalize_direction),
            "direction_norms": direction_norms,
        },
    }
    torch.save(direction_payload, directions_dir / "direction.pt")

    summary = {
        "direction_path": (directions_dir / "direction.pt").as_posix(),
        "action_count": action_count,
        "topic_count": topic_count,
        "balanced_subset_size": int(len(selected_rows)),
        "dropped_low_quality": int(dropped_low_quality),
        "label_path": Path(args.label_path).as_posix() if args.label_path else None,
        "manual_label_count": int(labeled_count),
        "heuristic_label_count": int(heuristic_count),
        "layers": sorted(direction_by_layer.keys()),
        "normalized": bool(args.normalize_direction),
        "direction_norms": direction_norms,
        "label_preview": selected_rows_preview,
    }
    save_json(reports_dir / "direction_summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compute steering direction (action - topic).")
    parser.add_argument("--output-root", default="outputs/analysis/steering")
    parser.add_argument("--label-path", default=None)
    parser.add_argument("--min-summary-tokens", type=int, default=6)
    parser.add_argument("--min-class-samples", type=int, default=12)
    parser.set_defaults(normalize_direction=True)
    parser.add_argument("--normalize-direction", dest="normalize_direction", action="store_true")
    parser.add_argument("--no-normalize-direction", dest="normalize_direction", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_direction_computation(args)
    print("Direction computation complete.")
    print(f"Layers: {summary['layers']}")
    print(f"Action samples: {summary['action_count']} | Topic samples: {summary['topic_count']}")


if __name__ == "__main__":
    main()
