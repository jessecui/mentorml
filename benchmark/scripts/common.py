"""
Shared loading and embedding code for the retrieval benchmarks.

Two benchmark sets exist:

- "curated" (default): the production corpus in corpus/, built by
  scripts/curate_corpus.py. Each image carries one query written for it
  during curation. This is what the app actually serves.
- "legacy": the original 92 Jay Alammar diagrams in benchmark/corpus/ with
  the queries in benchmark/queries/. Kept so earlier results stay reproducible.

A query counts as answered when its own image is retrieved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
CACHE_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "embeddings"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
DATASET_NAMES = {
    "curated": "mentorml-curated-diagram-retrieval-v2",
    "legacy": "mentorml-diagram-retrieval",
}


@dataclass
class Benchmark:
    name: str
    image_ids: list[str]
    image_paths: list[Path]
    queries: list[dict]  # {"query": str, "relevant_image_id": str}


def _benchmark_image(corpus_dir: Path, item: dict) -> Path:
    lossless = PROJECT_ROOT / "corpus_cache" / "norm" / f"{item['id']}.png"
    return lossless if lossless.exists() else corpus_dir / "images" / item["file"]


def load_benchmark(name: str = "curated") -> Benchmark:
    if name == "curated":
        corpus_dir = PROJECT_ROOT / "corpus"
        metadata_file = corpus_dir / "metadata.json"
        if not metadata_file.exists():
            sys.exit("❌ corpus/metadata.json not found. Run: python scripts/curate_corpus.py")
        items = json.loads(metadata_file.read_text())
        return Benchmark(
            name=name,
            image_ids=[m["id"] for m in items],
            # The lossless images the production embeddings were made from, so CLIP sees
            # the same pixels as SigLIP; the shipped WebPs (some lossy) only on a fresh clone
            image_paths=[_benchmark_image(corpus_dir, m) for m in items],
            queries=[{"query": m["benchmark_query"], "relevant_image_id": m["id"]} for m in items],
        )
    if name == "legacy":
        corpus_dir = PROJECT_ROOT / "benchmark" / "corpus"
        items = json.loads((corpus_dir / "metadata" / "corpus_with_queries.json").read_text())
        items = [m for m in items if (corpus_dir / "images" / m["source"] / m["filename"]).exists()]
        queries = json.loads((PROJECT_ROOT / "benchmark" / "queries" / "benchmark_queries.json").read_text())
        image_ids = [m["id"] for m in items]
        present = set(image_ids)
        kept = [q for q in queries if q["relevant_image_id"] in present]
        if len(kept) < len(queries):
            print(f"⚠️  Skipping {len(queries) - len(kept)} legacy queries whose image file is missing")
        return Benchmark(
            name=name,
            image_ids=image_ids,
            image_paths=[corpus_dir / "images" / m["source"] / m["filename"] for m in items],
            queries=[{"query": q["query"], "relevant_image_id": q["relevant_image_id"]} for q in kept],
        )
    raise ValueError(f"Unknown benchmark set: {name}")


# =============================================================================
# SIGLIP 2
# =============================================================================

def _images_signature(paths: list[Path]) -> str:
    """Hash of the images' bytes, so a cache is never reused for changed images."""
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def siglip_image_embeddings(bench: Benchmark):
    """
    Image embeddings, index-aligned with bench.image_ids, and the full scorer if one
    had to be loaded to compute them (so the caller can reuse it instead of loading
    a second model).
    """
    if bench.name == "curated":
        from model.scorer import MODEL_ID

        data = np.load(PROJECT_ROOT / "corpus" / "embeddings.npz")
        if "model_id" in data and str(data["model_id"]) != MODEL_ID:
            sys.exit(f"❌ corpus/embeddings.npz was built with {data['model_id']}, not {MODEL_ID}.")
        if [str(i) for i in data["ids"]] != bench.image_ids:
            sys.exit("❌ corpus/embeddings.npz is out of step with metadata. Run: python scripts/curate_corpus.py --only build")
        return data["embeddings"].astype(np.float32), None

    from model.scorer import MODEL_ID, SigLIPScorer

    cache = CACHE_DIR / "siglip_embeddings.npz"
    signature = _images_signature(bench.image_paths)
    if cache.exists():
        data = np.load(cache)
        if (str(data.get("model_id", "")) == MODEL_ID and str(data.get("signature", "")) == signature
                and [str(f) for f in data["filenames"]] == [p.name for p in bench.image_paths]):
            return data["embeddings"].astype(np.float32), None
    print(f"Computing SigLIP 2 image embeddings for {len(bench.image_paths)} images...")
    scorer = SigLIPScorer()
    embeddings = scorer.get_image_embeddings(bench.image_paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, embeddings=embeddings, filenames=[p.name for p in bench.image_paths],
             temperature=scorer.temperature, bias=scorer.bias, model_id=MODEL_ID, signature=signature)
    return embeddings, scorer


def make_siglip_ranker(bench: Benchmark):
    """Return rank(query) -> image IDs, best first."""
    from model.scorer import SigLIPScorer

    image_embeddings, scorer = siglip_image_embeddings(bench)
    if scorer is None:
        scorer = SigLIPScorer(text_only=True)

    def rank(query: str) -> list[str]:
        scores = image_embeddings @ scorer.get_text_embedding(query).reshape(-1)
        return [bench.image_ids[i] for i in np.argsort(-scores)]

    return rank


# =============================================================================
# CLIP BASELINE
# =============================================================================

def make_clip_ranker(bench: Benchmark):
    """Return rank(query) -> image IDs, best first, using CLIP ViT-L/14 @ 336px."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {CLIP_MODEL_ID} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()  # type: ignore[arg-type]
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    cache = CACHE_DIR / f"clip_embeddings_{bench.name}.npz"
    signature = _images_signature(bench.image_paths)
    image_embeddings = None
    if cache.exists():
        data = np.load(cache)
        if (str(data.get("model_id", "")) == CLIP_MODEL_ID and str(data.get("signature", "")) == signature
                and [str(i) for i in data["ids"]] == bench.image_ids):
            image_embeddings = data["embeddings"]
    if image_embeddings is None:
        print(f"Computing CLIP image embeddings for {len(bench.image_paths)} images...")
        rows = []
        for start in range(0, len(bench.image_paths), 8):
            images = [Image.open(p).convert("RGB") for p in bench.image_paths[start:start + 8]]
            inputs = processor(images=images, return_tensors="pt").to(device)  # type: ignore[call-arg]
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            rows.append(emb.cpu().numpy())
        image_embeddings = np.concatenate(rows)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=image_embeddings, ids=np.array(bench.image_ids),
                 model_id=CLIP_MODEL_ID, signature=signature)

    def rank(query: str) -> list[str]:
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)  # type: ignore[call-arg]
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        scores = image_embeddings @ emb.cpu().numpy().reshape(-1)
        return [bench.image_ids[i] for i in np.argsort(-scores)]

    return rank
