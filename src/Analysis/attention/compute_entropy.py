"""
Speaker imbalance metric based on entropy.
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple


def compute_entropy(distribution: Dict[str, float]) -> float:
    if not distribution:
        return 0.0
    entropy = 0.0
    for probability in distribution.values():
        p = max(0.0, float(probability))
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy)


def compute_entropy_for_sample(
    sample_id: str,
    speaker_distribution: Dict[str, float],
    output_dir: Path,
) -> Tuple[Dict, Path]:
    entropy_value = compute_entropy(speaker_distribution)
    payload = {
        "sample_id": sample_id,
        "entropy": entropy_value,
        "speaker_distribution": speaker_distribution,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return payload, output_path

