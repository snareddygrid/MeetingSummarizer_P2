"""
Cross-attention heatmap visualization.
"""

import os
from pathlib import Path
from typing import List

# Keep Matplotlib cache inside project workspace when home cache is not writable.
if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(__file__).resolve().parents[3] / ".pycache_local" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache_dir.as_posix()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.attention_helpers import sanitize_token_for_display


def _ensure_row_stochastic(matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = torch.clamp(matrix, min=0.0)
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return matrix / denom


def _compute_attention_rollout(self_attentions: torch.Tensor) -> torch.Tensor:
    """
    self_attentions: [layers, heads, tokens, tokens]
    rollout: [tokens, tokens]
    """
    if self_attentions.ndim != 4:
        raise ValueError("Expected attention shape [layers, heads, tokens, tokens].")

    layers = self_attentions.shape[0]
    tokens = self_attentions.shape[-1]
    identity = torch.eye(tokens, dtype=torch.float32)
    rollout = identity.clone()

    for layer_index in range(layers):
        layer_map = self_attentions[layer_index].mean(dim=0)
        layer_map = _ensure_row_stochastic(layer_map)
        layer_map = _ensure_row_stochastic((layer_map + identity) / 2.0)
        rollout = _ensure_row_stochastic(layer_map @ rollout)

    return rollout


def compute_rollout_flow_map(
    encoder_attentions: torch.Tensor,
    decoder_attentions: torch.Tensor,
    cross_attentions: torch.Tensor,
) -> torch.Tensor:
    """
    Computes dialogue->summary attention flow using multi-head rollout:
      decoder_rollout @ cross_mean @ encoder_rollout
    """
    encoder_rollout = _compute_attention_rollout(encoder_attentions)
    decoder_rollout = _compute_attention_rollout(decoder_attentions)
    cross_mean = _ensure_row_stochastic(cross_attentions.mean(dim=(0, 1)))

    flow = decoder_rollout @ cross_mean @ encoder_rollout
    return _ensure_row_stochastic(flow)


def create_attention_heatmap(
    sample_id: str,
    cross_attention_map: torch.Tensor,
    rollout_attention_map: torch.Tensor,
    input_tokens: List[str],
    summary_tokens: List[str],
    output_dir: Path,
    max_input_tokens: int = 120,
    max_summary_tokens: int = 40,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}.png"

    cross_heatmap = cross_attention_map.detach().cpu()
    rollout_heatmap = rollout_attention_map.detach().cpu()
    summary_len = min(cross_heatmap.shape[0], max_summary_tokens, len(summary_tokens))
    input_len = min(cross_heatmap.shape[1], max_input_tokens, len(input_tokens))
    cross_heatmap = cross_heatmap[:summary_len, :input_len]
    rollout_heatmap = rollout_heatmap[:summary_len, :input_len]

    x_labels = [sanitize_token_for_display(token) for token in input_tokens[:input_len]]
    y_labels = [sanitize_token_for_display(token) for token in summary_tokens[:summary_len]]

    fig_width = max(14, min(36, 5.0 + input_len * 0.35))
    fig_height = max(6, min(20, 2.0 + summary_len * 0.3))
    figure, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)
    left_axis, right_axis = axes

    cross_image = left_axis.imshow(cross_heatmap.numpy(), aspect="auto", cmap="viridis")
    left_axis.set_title("Cross-Attention (Mean Heads/Layers)")
    left_axis.set_xlabel("Input Tokens")
    left_axis.set_ylabel("Summary Tokens")

    rollout_image = right_axis.imshow(rollout_heatmap.numpy(), aspect="auto", cmap="magma")
    right_axis.set_title("Multi-Head Rollout Flow")
    right_axis.set_xlabel("Input Tokens")
    right_axis.set_ylabel("Summary Tokens")

    x_step = max(1, input_len // 30)
    y_step = max(1, summary_len // 30)
    x_ticks = list(range(0, input_len, x_step))
    y_ticks = list(range(0, summary_len, y_step))

    for axis in (left_axis, right_axis):
        axis.set_xticks(x_ticks)
        axis.set_xticklabels([x_labels[idx] for idx in x_ticks], rotation=90, fontsize=7)
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([y_labels[idx] for idx in y_ticks], fontsize=8)

    figure.suptitle(f"Attention Flow Mapping ({sample_id})", fontsize=12)
    figure.colorbar(cross_image, ax=left_axis, fraction=0.046, pad=0.02, label="Cross-Attn Weight")
    figure.colorbar(rollout_image, ax=right_axis, fraction=0.046, pad=0.02, label="Rollout Weight")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    return output_path
