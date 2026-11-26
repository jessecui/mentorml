"""
MCP server exposing diagram retrieval as the `retrieve_diagram` tool.

Owns the SigLIP 2 text tower and the precomputed image embeddings of the
curated corpus (see scripts/curate_corpus.py). Runs in its own subprocess and
is reached over stdio by the agent's MCP client.

Retrieval is a single text embedding plus a dot product, so a call takes
tens of milliseconds. Everything a caller needs to choose between the
matches (what each image shows, what it teaches, how clear it is) was written
once at curation time and is returned as text; no model looks at the image
at query time.

Run standalone:
    python -m mcp_server.diagram_server
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from model.scorer import MODEL_ID, SigLIPScorer

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
CORPUS_METADATA = CORPUS_DIR / "metadata.json"
EMBEDDINGS_FILE = CORPUS_DIR / "embeddings.npz"

DEFAULT_TOP_K = 8
MAX_TOP_K = 15
# Extra candidates found by keyword match on titles, concepts and descriptions
KEYWORD_MATCHES = 4


def _log(msg: str) -> None:
    # MCP uses stdout for protocol; logs must go to stderr.
    print(msg, file=sys.stderr, flush=True)


if not CORPUS_METADATA.exists() or not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Curated corpus not found in {CORPUS_DIR}\n"
        "Run: python scripts/curate_corpus.py"
    )

_log("📦 Loading SigLIP 2 text tower in MCP server...")
_scorer = SigLIPScorer(text_only=True)

_emb_data = np.load(EMBEDDINGS_FILE)
_image_embeddings: np.ndarray = _emb_data["embeddings"].astype(np.float32)
if "model_id" in _emb_data and str(_emb_data["model_id"]) != MODEL_ID:
    raise RuntimeError(
        f"corpus/embeddings.npz was built with {_emb_data['model_id']}, but the scorer is {MODEL_ID}.\n"
        "Unset SIGLIP_MODEL_ID, or rebuild the corpus: python scripts/curate_corpus.py --only build"
    )
_temperature = float(_emb_data["temperature"])
_bias = float(_emb_data["bias"])
_corpus: list[dict] = json.loads(CORPUS_METADATA.read_text())

if [m["id"] for m in _corpus] != [str(i) for i in _emb_data["ids"]]:
    raise RuntimeError(
        "corpus/metadata.json and corpus/embeddings.npz are out of step.\n"
        "Run: python scripts/curate_corpus.py --only build"
    )
_log(f"   ✅ {len(_corpus)} diagrams, embeddings {_image_embeddings.shape}")

# The first forward pass allocates buffers; do it now, not on a user's request.
_scorer.get_text_embedding("warm-up query")
_log("   ✅ Text tower warmed up")


# --- Keyword index (BM25) over what was written about each diagram at curation time ---

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it its of on or that the their this to vs with "
    "diagram showing shows show each one two using used via between over per".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _document_tokens(item: dict) -> list[str]:
    # Titles and concept keywords count for more than running description text.
    return (
        _tokens(item["title"]) * 3
        + _tokens(" ".join(item.get("concepts", []))) * 2
        + _tokens(item["description"])
        + _tokens(item["teaches"])
    )


_doc_term_counts = [Counter(_document_tokens(item)) for item in _corpus]
_doc_lengths = np.array([sum(c.values()) for c in _doc_term_counts], dtype=np.float32)
_avg_doc_length = float(_doc_lengths.mean()) if len(_doc_lengths) else 1.0
_doc_frequency: Counter = Counter(term for counts in _doc_term_counts for term in counts)


def keyword_scores(query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 score of every diagram for the query's words."""
    n = len(_corpus)
    scores = np.zeros(n, dtype=np.float32)
    for term in set(_tokens(query)):
        df = _doc_frequency.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, counts in enumerate(_doc_term_counts):
            tf = counts.get(term, 0)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * _doc_lengths[i] / _avg_doc_length))
    return scores


def _query_variants(query: str) -> list[str]:
    """
    Phrasings for a structured "concept: parts" query (search() calls this only when
    the caller says the query is structured): the query as written, plus the concept
    name alone, because some diagrams match the short name far better than the long
    description, and others the reverse.
    """
    # Colons in ratios, times or URLs have no space after them; a raw question has no list
    head, _, tail = query.partition(": ")
    head = head.strip()
    if tail and len(head.split()) <= 8 and re.search(r"[A-Za-z]{2,}", head):
        return [query, head, f"a diagram of {head}"]
    return [query]


def _round_robin(rankings: list[np.ndarray], limit: int) -> list[int]:
    """Take the best not-yet-seen item from each ranking in turn until `limit` are chosen."""
    chosen: list[int] = []
    for depth in range(max(len(r) for r in rankings)):
        for ranking in rankings:
            i = int(ranking[depth])
            if i not in chosen:
                chosen.append(i)
                if len(chosen) == limit:
                    return chosen
    return chosen


def search(query: str, top_k: int = DEFAULT_TOP_K, structured: bool = False) -> list[dict]:
    """
    Rank the corpus against a query and return the best matches.

    The ranking is query-to-image similarity. For a structured "concept: parts"
    query (structured=True) it is taken across a few phrasings; anything else is
    searched as written. A few diagrams whose title, keywords or description share the
    query's words are appended after them, which recovers images that are
    right for the topic but drawn in a way the image match does not pick up.
    """
    if not _corpus:
        return []
    top_k = max(1, min(int(top_k), MAX_TOP_K, len(_corpus)))
    text_embs = _scorer.get_text_embeddings(_query_variants(query) if structured else [query])  # [V, D]

    image_sims = text_embs @ _image_embeddings.T                     # [V, N]
    order = _round_robin([np.argsort(-row) for row in image_sims], top_k)

    by_keyword = keyword_scores(query)
    order += [int(i) for i in np.argsort(-by_keyword) if by_keyword[i] > 0 and int(i) not in order][:KEYWORD_MATCHES]

    scores = 1.0 / (1.0 + np.exp(-(image_sims.max(axis=0) * _temperature + _bias)))
    return [
        {
            "id": _corpus[i]["id"],
            "score": round(float(scores[i]), 4),
            "title": _corpus[i]["title"],
            "description": _corpus[i]["description"],
            "teaches": _corpus[i]["teaches"],
            "clarity": _corpus[i]["glanceable"],
            "source_name": _corpus[i]["source_name"],
            "source_title": _corpus[i]["source_title"],
        }
        for i in order
    ]


mcp = FastMCP("mentorml-diagrams")


@mcp.tool()
def retrieve_diagram(query: str, top_k: int = DEFAULT_TOP_K, structured: bool = False) -> dict:
    """
    Search the curated library of AI/ML teaching diagrams.

    Returns the best-matching diagrams for the query, best first. Each one
    comes with a description of what it shows, so you can pick the one that
    fits your explanation, or none if nothing is really about the topic.

    Args:
        query: The picture you are looking for, written as the concept name,
               a colon, then its key parts, e.g. "scaled dot-product attention:
               queries, keys, values, softmax" or "actor-critic: actor picks
               the action, critic estimates its value".
        top_k: How many image matches to return (default 8). Up to four more
               are added when their title or description shares the query's words.
        structured: True only when the query follows the "concept: parts" form
               above; the concept name is then also searched on its own.

    Returns:
        A dictionary with:
        - query: the query that was searched
        - candidates: list of matches, each with
            - id: diagram identifier to cite, e.g. "fig_3fa9c1d2e0"
            - score: query-image match strength from 0 to 1 (not comparable across queries)
            - title: the concept the diagram shows
            - description: what is drawn and how the parts connect
            - teaches: what a learner takes away from it
            - clarity: 1-5, how quickly the main idea lands
            - source_name, source_title: where it comes from
    """
    return {"query": query, "candidates": search(query, top_k, structured)}


if __name__ == "__main__":
    _log("🚀 Starting MentorML diagram MCP server (stdio)...")
    mcp.run(transport="stdio")
