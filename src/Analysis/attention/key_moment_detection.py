"""
Key moment detection from decoder-to-encoder cross-attention.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from utils.attention_helpers import sanitize_token_for_display


def detect_key_moments(
    sample_id: str,
    cross_attention_map: torch.Tensor,
    summary_tokens: List[str],
    input_tokens: List[str],
    token_to_turn: List[int],
    turns: List[Dict],
    output_dir: Path,
    top_k_source_tokens: int = 5,
    top_k_turns: int = 3,
) -> Tuple[Dict, Path]:
    """
    1) For each summary token, get top attended input tokens.
    2) Aggregate attention by dialogue turn and return top-3 turns.
    """
    summary_alignment = []
    turn_scores = {turn["turn_id"]: 0.0 for turn in turns}

    for target_index in range(cross_attention_map.shape[0]):
        row = cross_attention_map[target_index]
        k = min(top_k_source_tokens, row.numel())
        top_values, top_indices = torch.topk(row, k=k)

        top_sources = []
        for value, source_index in zip(top_values.tolist(), top_indices.tolist()):
            source_token = sanitize_token_for_display(
                input_tokens[source_index] if source_index < len(input_tokens) else ""
            )
            turn_idx = token_to_turn[source_index] if source_index < len(token_to_turn) else -1
            turn_info = turns[turn_idx] if 0 <= turn_idx < len(turns) else None

            if turn_info is not None:
                turn_scores[turn_info["turn_id"]] = turn_scores.get(turn_info["turn_id"], 0.0) + float(value)

            top_sources.append(
                {
                    "source_index": int(source_index),
                    "source_token": source_token,
                    "attention_score": float(value),
                    "turn_id": int(turn_info["turn_id"]) if turn_info else None,
                    "speaker": turn_info["speaker"] if turn_info else None,
                    "utterance": turn_info["utterance"] if turn_info else None,
                }
            )

        summary_alignment.append(
            {
                "summary_index": int(target_index),
                "summary_token": sanitize_token_for_display(
                    summary_tokens[target_index] if target_index < len(summary_tokens) else ""
                ),
                "top_attended_inputs": top_sources,
            }
        )

    ranked_turns = []
    turn_lookup = {turn["turn_id"]: turn for turn in turns}
    for turn_id, score in turn_scores.items():
        turn_info = turn_lookup.get(turn_id)
        if not turn_info:
            continue
        ranked_turns.append(
            {
                "turn_id": int(turn_info["turn_id"]),
                "speaker": turn_info["speaker"],
                "utterance": turn_info["utterance"],
                "attention_score": float(score),
            }
        )

    ranked_turns.sort(key=lambda item: item["attention_score"], reverse=True)
    top_turns = ranked_turns[:top_k_turns]

    payload = {
        "sample_id": sample_id,
        "top_turns": top_turns,
        "summary_token_alignment": summary_alignment,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return payload, output_path
