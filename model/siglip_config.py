"""
Which SigLIP 2 model the project uses, where its text-tower export lives, and how
to read the model's learned score calibration.

Kept apart from scorer.py so the Docker build can export the text tower from this
file alone: editing scorer.py then does not invalidate the layer that downloads the
multi-gigabyte checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# SigLIP 2 So400m/14 at 384px (fixed resolution variant)
MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip2-so400m-patch14-384")
# Local text-tower export written by scripts/export_text_tower.py
TEXT_TOWER_DIR = Path(os.getenv("SIGLIP_TEXT_TOWER_DIR", PROJECT_ROOT / "model" / "siglip2_text"))


def read_calibration(model_id: str) -> tuple[float, float]:
    """Read the learned temperature and bias without instantiating the vision tower."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors import safe_open

    try:
        single = hf_hub_download(model_id, "model.safetensors")
        files = {"logit_scale": single, "logit_bias": single}
    except EntryNotFoundError:
        # Larger checkpoints are split across files; the index says which holds each tensor
        index = json.loads(Path(hf_hub_download(model_id, "model.safetensors.index.json")).read_text())
        files = {name: hf_hub_download(model_id, index["weight_map"][name]) for name in ("logit_scale", "logit_bias")}

    def tensor(name: str):
        with safe_open(files[name], framework="pt") as f:
            return f.get_tensor(name)

    return float(tensor("logit_scale").exp().item()), float(tensor("logit_bias").item())
