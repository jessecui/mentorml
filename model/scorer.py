"""
SigLIP 2 scorer (PyTorch, via Hugging Face transformers).

Embeds text queries and images into the same space for query-image retrieval.
Scores follow SigLIP's sigmoid formulation:

    score = sigmoid(temperature * <image_emb, text_emb> + bias)

Two loading modes:

- Full model (`SigLIPScorer()`): both towers. Used offline by the curation
  and benchmark scripts, which need to embed images.
- Text tower only (`SigLIPScorer(text_only=True)`): used by the MCP server at
  query time, where image embeddings are precomputed. Loads from a local
  export (see scripts/export_text_tower.py) when one exists, which roughly
  halves memory and avoids downloading the vision tower in production.

Usage:
    from model.scorer import SigLIPScorer
    scorer = SigLIPScorer()
    img = scorer.get_image_embedding("diagram.png")     # [1, 1152], L2-normalised
    txt = scorer.get_text_embedding("attention heads")  # [1, 1152], L2-normalised
    score = scorer.score("diagram.png", "attention heads")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.siglip_config import MODEL_ID, PROJECT_ROOT, TEXT_TOWER_DIR  # noqa: F401  (re-exported)
from model.siglip_config import read_calibration as _read_calibration

IMAGE_SIZE = 384
MAX_SEQ_LEN = 64  # SigLIP 2 was trained with 64-token text sequences
EMBED_DIM = 1152


def _pick_device() -> str:
    """SIGLIP_DEVICE if set and available, else CUDA if present, else CPU."""
    requested = os.getenv("SIGLIP_DEVICE", "").strip().lower()
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested:
        if available.get(requested):
            return requested
        # stderr: in the MCP server, stdout carries the protocol
        print(f"SIGLIP_DEVICE={requested!r} is not available here; choosing automatically", file=sys.stderr)
    return "cuda" if available["cuda"] else "cpu"


class SigLIPScorer:
    """Query-image scorer backed by SigLIP 2."""

    def __init__(self, text_only: bool = False, device: str | None = None):
        """
        Args:
            text_only: Load only the text tower. Image methods will raise.
            device: torch device string; defaults to CUDA if available, else CPU.
        """
        from transformers import AutoTokenizer

        self.device = device or _pick_device()
        self.text_only = text_only
        self._image_cache: dict[str, np.ndarray] = {}

        if text_only:
            from transformers import SiglipTextModel

            calibration = _local_export_calibration()
            if calibration is not None:
                source = str(TEXT_TOWER_DIR)
                self.temperature = float(calibration["temperature"])
                self.bias = float(calibration["bias"])
            else:
                # No local export: pull the text tower out of the full checkpoint.
                source = MODEL_ID
                self.temperature, self.bias = _read_calibration(MODEL_ID)
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.text_model = SiglipTextModel.from_pretrained(source).to(self.device).eval()
            self.model = None
            self.image_processor = None
        else:
            from transformers import AutoImageProcessor, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # Pin the slow processor so embeddings stay identical across transformers versions.
            self.image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
            self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()
            self.text_model = self.model.text_model
            self.temperature = float(self.model.logit_scale.exp().item())
            self.bias = float(self.model.logit_bias.item())

    # ------------------------------------------------------------------ text

    def _tokenize(self, queries: list[str]) -> dict:
        # SigLIP 2 expects lowercased text padded to a fixed 64 tokens.
        return self.tokenizer(
            [q.lower() for q in queries],
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )

    @torch.inference_mode()
    def get_text_embeddings(self, queries: list[str]) -> np.ndarray:
        """Embed several queries. Returns [N, EMBED_DIM], L2-normalised."""
        tokens = self._tokenize(queries)
        input_ids = tokens["input_ids"].to(self.device)
        if self.text_only:
            emb = self.text_model(input_ids=input_ids).pooler_output
        else:
            emb = self.model.get_text_features(input_ids=input_ids)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    def get_text_embedding(self, query: str) -> np.ndarray:
        """Embed one query. Returns [1, EMBED_DIM], L2-normalised."""
        return self.get_text_embeddings([query])

    # ----------------------------------------------------------------- image

    @staticmethod
    def _load_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode in ("RGBA", "LA", "P", "PA") or "transparency" in image.info:
            # Flatten transparency onto white; diagrams are drawn for light backgrounds.
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")

    @torch.inference_mode()
    def get_image_embeddings(
        self, images: list[Image.Image | np.ndarray | str | Path], batch_size: int = 8
    ) -> np.ndarray:
        """Embed several images. Returns [N, EMBED_DIM], L2-normalised."""
        if self.text_only:
            raise RuntimeError("SigLIPScorer was loaded with text_only=True; it cannot embed images.")
        out: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = [self._load_image(img) for img in images[start:start + batch_size]]
            pixels = self.image_processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
            emb = self.model.get_image_features(pixel_values=pixels)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)

    def get_image_embedding(
        self, image: Image.Image | np.ndarray | str | Path, use_cache: bool = True
    ) -> np.ndarray:
        """Embed one image. Returns [1, EMBED_DIM], L2-normalised."""
        cache_key = str(image) if isinstance(image, (str, Path)) else None
        if use_cache and cache_key and cache_key in self._image_cache:
            return self._image_cache[cache_key]
        embedding = self.get_image_embeddings([image])
        if use_cache and cache_key:
            self._image_cache[cache_key] = embedding
        return embedding

    # --------------------------------------------------------------- scoring

    def score_embeddings(self, image_embeddings: np.ndarray, text_embedding: np.ndarray) -> np.ndarray:
        """Sigmoid scores for one text embedding against [N, D] image embeddings."""
        logits = image_embeddings @ text_embedding.reshape(-1) * self.temperature + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def score(self, image: Image.Image | np.ndarray | str | Path, query: str) -> float:
        """Probability-like score that the image matches the query."""
        img = self.get_image_embedding(image)
        txt = self.get_text_embedding(query)
        return float(self.score_embeddings(img, txt)[0])

    def clear_cache(self) -> None:
        self._image_cache.clear()


def _local_export_calibration() -> dict | None:
    """The local text-tower export's calibration, if the export exists and is of MODEL_ID."""
    calibration_file = TEXT_TOWER_DIR / "calibration.json"
    if not (TEXT_TOWER_DIR / "config.json").exists() or not calibration_file.exists():
        return None
    calibration = json.loads(calibration_file.read_text())
    if calibration.get("model_id") != MODEL_ID:
        # The values huggingface_hub itself treats as "offline"
        if os.getenv("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
            raise RuntimeError(
                f"The text-tower export in {TEXT_TOWER_DIR} is of {calibration.get('model_id')!r}, but "
                f"SIGLIP_MODEL_ID is {MODEL_ID!r}, and downloads are disabled (HF_HUB_OFFLINE=1). "
                "Rebuild the image with --build-arg SIGLIP_MODEL_ID=... instead of changing it at runtime."
            )
        print(
            f"Ignoring the text-tower export in {TEXT_TOWER_DIR}: it is of {calibration.get('model_id')!r}, "
            f"not {MODEL_ID!r}. Re-run scripts/export_text_tower.py.",
            file=sys.stderr,
        )
        return None
    return calibration


def main() -> None:
    """Smoke test: embed a query and, if a path is given, score an image against it."""
    import sys

    query = "self-attention mechanism in a transformer"
    if len(sys.argv) > 1:
        scorer = SigLIPScorer()
        print(f"score({sys.argv[1]!r}, {query!r}) = {scorer.score(sys.argv[1], query):.6f}")
    else:
        scorer = SigLIPScorer(text_only=True)
        emb = scorer.get_text_embedding(query)
        print(f"text embedding shape={emb.shape} norm={np.linalg.norm(emb):.4f}")
    print(f"temperature={scorer.temperature:.3f} bias={scorer.bias:.3f}")


if __name__ == "__main__":
    main()
