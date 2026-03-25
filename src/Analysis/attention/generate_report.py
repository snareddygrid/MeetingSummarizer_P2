"""
Final structured report generation for Task-1.
"""

import json
from pathlib import Path
from typing import Dict, Tuple


def generate_final_report(
    sample_id: str,
    summary_text: str,
    speaker_distribution: Dict[str, float],
    entropy_payload: Dict,
    key_moment_payload: Dict,
    output_dir: Path,
) -> Tuple[Dict, Path]:
    report = {
        "sample_id": sample_id,
        "generated_summary": summary_text,
        "speaker_distribution": speaker_distribution,
        "speaker_entropy": entropy_payload.get("entropy", 0.0),
        "top_3_turns": key_moment_payload.get("top_turns", [])[:3],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report, output_path

