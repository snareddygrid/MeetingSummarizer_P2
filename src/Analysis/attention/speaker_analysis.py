"""
Speaker-aware attention aggregation using cross-attention.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from utils.attention_helpers import normalize_distribution


def analyze_speaker_distribution(
    sample_id: str,
    cross_attention_map: torch.Tensor,
    token_to_turn: List[int],
    turns: List[Dict],
    output_dir: Path,
) -> Tuple[Dict[str, float], Path]:
    """
    Aggregates cross-attention over source tokens and maps it to speakers.
    """
    # Average over summary tokens -> source-token importance.
    source_importance = cross_attention_map.mean(dim=0)
    speaker_scores: Dict[str, float] = {}

    for source_index, score in enumerate(source_importance.tolist()):
        if source_index >= len(token_to_turn):
            continue
        turn_index = token_to_turn[source_index]
        if turn_index < 0 or turn_index >= len(turns):
            continue
        speaker = turns[turn_index]["speaker"]
        speaker_scores[speaker] = speaker_scores.get(speaker, 0.0) + float(score)

    speaker_distribution = normalize_distribution(speaker_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(speaker_distribution, file, indent=2)

    return speaker_distribution, output_path
