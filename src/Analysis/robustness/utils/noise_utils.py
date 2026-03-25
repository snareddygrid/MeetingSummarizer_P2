"""Noise and perturbation helpers for robustness testing."""

from __future__ import annotations

import random
import re
from typing import List, Sequence, Tuple

import numpy as np


OFF_TOPIC_TANGENTS = [
    "By the way, did anyone watch the game last night?",
    "Unrelated, the office coffee machine is broken again.",
    "Side note: the weather tomorrow looks really bad.",
    "Random thought: we should plan a team lunch soon.",
    "This is off-topic, but parking was impossible today.",
]

FILLER_PHRASES = [
    "um",
    "you know",
    "just to reiterate",
    "as I said before",
    "to be honest",
]

WORD_LEVEL_TYPOS = {
    "will": "wil",
    "tomorrow": "tomorow",
    "meeting": "meting",
    "because": "becuase",
    "action": "aciton",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _split_turn(turn: str) -> Tuple[str, str]:
    turn = str(turn).strip()
    if not turn:
        return "Speaker", ""
    if ":" in turn:
        speaker, text = turn.split(":", 1)
        return speaker.strip() or "Speaker", text.strip()
    return "Speaker", turn


def _join_turn(speaker: str, text: str) -> str:
    speaker = (speaker or "Speaker").strip()
    text = (text or "").strip()
    if not text:
        return f"{speaker}:"
    return f"{speaker}: {text}"


def _split_text_midway(text: str) -> Tuple[str, str]:
    words = [w for w in text.split() if w]
    if len(words) < 4:
        return text, ""
    cut = max(1, len(words) // 2)
    return " ".join(words[:cut]), " ".join(words[cut:])


def _char_swap(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    idx = rng.randint(1, len(word) - 2)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def inject_noise(dialogue: str, rng: random.Random, typo_prob: float = 0.08) -> str:
    """Inject spelling noise via common typos and random char swaps."""
    tokens = re.split(r"(\s+)", dialogue)
    out: List[str] = []
    for token in tokens:
        if not token or token.isspace():
            out.append(token)
            continue

        core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
        if not core:
            out.append(token)
            continue

        replacement = core
        lower = core.lower()
        if lower in WORD_LEVEL_TYPOS and rng.random() < 0.5:
            replacement = WORD_LEVEL_TYPOS[lower]
        elif rng.random() < typo_prob:
            replacement = _char_swap(core, rng)

        out.append(token.replace(core, replacement, 1))
    return "".join(out)


def interleave_overlapping_speakers(dialogue: str, rng: random.Random, interleave_prob: float = 0.5) -> str:
    """Create overlap-like interruptions by splitting and interleaving turns."""
    lines = [line.strip() for line in str(dialogue).splitlines() if line.strip()]
    if len(lines) < 2:
        return dialogue

    turns = [_split_turn(line) for line in lines]
    transformed: List[str] = []
    idx = 0
    while idx < len(turns):
        if idx + 1 < len(turns) and rng.random() < interleave_prob:
            spk_a, txt_a = turns[idx]
            spk_b, txt_b = turns[idx + 1]
            first_a, second_a = _split_text_midway(txt_a)
            first_b, second_b = _split_text_midway(txt_b)

            transformed.append(_join_turn(spk_a, f"{first_a} ..." if second_a else first_a))
            transformed.append(_join_turn(spk_b, f"{first_b} ..." if second_b else first_b))
            if second_a:
                transformed.append(_join_turn(spk_a, second_a))
            if second_b:
                transformed.append(_join_turn(spk_b, second_b))
            idx += 2
        else:
            spk, txt = turns[idx]
            transformed.append(_join_turn(spk, txt))
            idx += 1

    return "\n".join(transformed)


def insert_off_topic_tangents(dialogue: str, rng: random.Random, max_insertions: int = 2) -> str:
    """Insert irrelevant tangent lines at random points."""
    lines = [line for line in str(dialogue).splitlines() if line.strip()]
    if not lines:
        return dialogue

    n_insert = rng.randint(1, max_insertions)
    for _ in range(n_insert):
        tangent = rng.choice(OFF_TOPIC_TANGENTS)
        speaker = rng.choice(["Random", "SideThread", "Note"])
        insert_idx = rng.randint(0, len(lines))
        lines.insert(insert_idx, f"{speaker}: {tangent}")
    return "\n".join(lines)


def expand_length(dialogue: str, rng: random.Random, repeat_prob: float = 0.25) -> str:
    """Expand transcript length via repeated turns and filler phrases."""
    lines = [line for line in str(dialogue).splitlines() if line.strip()]
    if not lines:
        return dialogue

    expanded: List[str] = []
    for line in lines:
        expanded.append(line)
        if rng.random() < repeat_prob:
            speaker, text = _split_turn(line)
            filler = rng.choice(FILLER_PHRASES)
            expanded.append(_join_turn(speaker, f"{text} {filler}."))
    return "\n".join(expanded)


def create_adversarial_dialogue(dialogue: str, rng: random.Random) -> Tuple[str, List[str]]:
    """Apply all perturbation families and return transformed text + tags."""
    perturbed = str(dialogue)
    tags: List[str] = []

    overlapped = interleave_overlapping_speakers(perturbed, rng=rng)
    if overlapped != perturbed:
        tags.append("overlap")
        perturbed = overlapped

    noised = inject_noise(perturbed, rng=rng)
    if noised != perturbed:
        tags.append("noise")
        perturbed = noised

    off_topic = insert_off_topic_tangents(perturbed, rng=rng)
    if off_topic != perturbed:
        tags.append("off_topic")
        perturbed = off_topic

    lengthened = expand_length(perturbed, rng=rng)
    if lengthened != perturbed:
        tags.append("length")
        perturbed = lengthened

    return perturbed, tags
