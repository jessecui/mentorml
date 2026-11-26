#!/usr/bin/env python3
"""
Export the SigLIP 2 text tower on its own.

At query time the app only embeds text; image embeddings are precomputed by
scripts/curate_corpus.py. This script pulls the text tower, its tokenizer and
the learned score calibration out of the full checkpoint and saves them to
model/siglip2_text/, which model.scorer.SigLIPScorer(text_only=True) loads
without touching the network or the vision tower.

The Dockerfile runs this at build time and then deletes the full download.

Usage:
    python scripts/export_text_tower.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from model.siglip_config import MODEL_ID, TEXT_TOWER_DIR, read_calibration  # noqa: E402


def main() -> None:
    from transformers import AutoTokenizer, SiglipTextModel

    print(f"Exporting text tower of {MODEL_ID} to {TEXT_TOWER_DIR}...")
    TEXT_TOWER_DIR.mkdir(parents=True, exist_ok=True)
    SiglipTextModel.from_pretrained(MODEL_ID).save_pretrained(TEXT_TOWER_DIR)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(TEXT_TOWER_DIR)

    temperature, bias = read_calibration(MODEL_ID)
    (TEXT_TOWER_DIR / "calibration.json").write_text(
        json.dumps({"model_id": MODEL_ID, "temperature": temperature, "bias": bias}, indent=1)
    )

    size_mb = sum(f.stat().st_size for f in TEXT_TOWER_DIR.iterdir() if f.is_file()) / 1e6
    print(f"✅ Saved ({size_mb:.0f} MB), temperature={temperature:.3f}, bias={bias:.3f}")


if __name__ == "__main__":
    main()
