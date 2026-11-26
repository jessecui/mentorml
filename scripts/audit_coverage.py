#!/usr/bin/env python3
"""
Check which topics the diagram corpus can illustrate.

For every topic in scripts/coverage_topics.py:

1. Run the same retrieval the app uses and get the candidates.
2. Ask Gemini which candidates directly depict the topic, judging from their
   descriptions only. This mirrors how the agent picks a diagram.
3. For topics with none, look through the REJECTED images for near-misses:
   images the rubric judged to be real, standalone diagrams that failed only
   on a score.

Usage:
    python scripts/audit_coverage.py
    python scripts/audit_coverage.py --section "Reinforcement learning"

Output:
    corpus/coverage.json    per-topic result, for comparing runs
    corpus_cache/coverage.html   the gaps and their near-misses, with images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from scripts.coverage_topics import TOPICS  # noqa: E402
from scripts.curate_corpus import (  # noqa: E402
    ACCEPTED_KINDS, CACHE_DIR, CANDIDATES_FILE, MIN_GLANCEABLE, MIN_INSIGHT, MIN_VISUAL_QUALITY,
    verdict,
)

COVERAGE_FILE = PROJECT_ROOT / "corpus" / "coverage.json"
COVERAGE_PAGE = CACHE_DIR / "coverage.html"
JUDGE_MODEL = "gemini-3-flash-preview"
NEAR_MISS_CANDIDATES = 6
# Verdicts that mean "a real diagram that only missed on a score"
SCORE_ONLY_REJECTIONS = {"low_insight", "not_glanceable", "low_visual_quality"}


class Judgement(BaseModel):
    depicting_ids: list[str] = Field(
        description="IDs of the candidates that directly depict the topic, best first. Empty if none does."
    )
    reason: str = Field(description="One short sentence.")


JUDGE_PROMPT = """A learner asked an AI/ML teaching assistant about this topic:

TOPIC: {topic}

These diagrams are available, each described in text:

{candidates}

Which of them DIRECTLY depict the topic? A diagram qualifies only if its title or description shows this concept or its named parts. A diagram about a neighbouring concept, a broader architecture that merely contains it, or one that shares a keyword does not qualify.

Return the qualifying IDs, best first, or an empty list."""


def judge(client, topic: str, candidates: list[dict]) -> Judgement:
    from google.genai import types

    if not candidates:
        return Judgement(depicting_ids=[], reason="no candidates")
    block = "\n".join(
        f'- ID {c["id"]} | {c["title"]}\n  Shows: {c["description"]}\n  Teaches: {c["teaches"]}' for c in candidates
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Judgement,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL, contents=[JUDGE_PROMPT.format(topic=topic, candidates=block)], config=config
            )
            parsed = response.parsed if isinstance(response.parsed, Judgement) else Judgement.model_validate_json(response.text)
            known = {c["id"] for c in candidates}
            parsed.depicting_ids = [i for i in parsed.depicting_ids if i in known]
            return parsed
        except Exception as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
    return Judgement(depicting_ids=[], reason=f"judge failed: {last_error!r}"[:200])


def load_near_miss_pool() -> list[dict]:
    """Rejected candidates that are genuine standalone diagrams and failed only on a score."""
    pool = []
    for c in json.loads(CANDIDATES_FILE.read_text()):
        r = c.get("rubric")
        if not r or r["kind"] not in ACCEPTED_KINDS:
            continue
        v = verdict(c, MIN_VISUAL_QUALITY, MIN_GLANCEABLE, MIN_INSIGHT)
        if v in SCORE_ONLY_REJECTIONS and (CACHE_DIR / "norm" / f"{c['id']}.png").exists():
            pool.append({
                "id": c["id"], "title": r["title"], "description": r["description"], "teaches": r["what_it_teaches"],
                "rejected_for": v, "visual_quality": r["visual_quality"], "glanceable": r["glanceable"],
                "insight": r["insight"], "source": c["source"], "page_url": c["page_url"],
            })
    return pool


def write_page(results: list[dict], corpus_by_id: dict[str, dict]) -> None:
    gaps = [r for r in results if not r["covered"]]
    covered = len(results) - len(gaps)
    parts = [f"<h2>Coverage: {covered} of {len(results)} topics have a diagram</h2>"]
    for section in dict.fromkeys(r["section"] for r in results):
        rows = [r for r in results if r["section"] == section]
        n = sum(r["covered"] for r in rows)
        parts.append(f"<h3>{escape(section)} ({n}/{len(rows)})</h3>")
        for r in rows:
            if r["covered"]:
                best = corpus_by_id[r["diagram_ids"][0]]
                parts.append(f'<div class="t ok">✓ {escape(r["topic"])} <span>→ {escape(best["title"])}</span></div>')
                continue
            parts.append(f'<div class="t gap">✗ {escape(r["topic"])}</div>')
            if r["near_misses"]:
                cards = "".join(
                    f'<div class="card"><a href="norm/{m["id"]}.png" target="_blank"><img loading="lazy" src="norm/{m["id"]}.png"></a>'
                    f'<div><b>{escape(m["title"])}</b><br>{escape(m["id"])} · {escape(m["source"])}<br>'
                    f'rejected: {escape(m["rejected_for"])} (quality {m["visual_quality"]}, glance {m["glanceable"]}, insight {m["insight"]})</div></div>'
                    for m in r["near_misses"]
                )
                parts.append(f'<div class="grid">{cards}</div>')
    COVERAGE_PAGE.write_text(f"""<!doctype html><meta charset="utf-8"><title>Corpus coverage</title>
<style>
body{{font:14px system-ui;margin:20px;max-width:1300px}} .t{{padding:3px 0}} .t span{{color:#777}}
.gap{{color:#b00;font-weight:600;margin-top:8px}} .ok{{color:#161}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:6px 0 14px 18px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:8px;font-size:12px;font-weight:400;color:#222}}
.card img{{width:100%;max-height:220px;object-fit:contain}}
</style>{''.join(parts)}""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", help="Audit one section of the topic list only")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("❌ GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    from google import genai

    client = genai.Client(api_key=api_key)

    # Importing the server module loads the corpus and the text tower exactly as the app does.
    from mcp_server import diagram_server

    # The MCP library turns on verbose request logging; the audit only needs its own output.
    import logging

    logging.getLogger().setLevel(logging.WARNING)

    corpus_by_id = {m["id"]: m for m in diagram_server._corpus}
    topics = [(section, t) for section, items in TOPICS.items() for t in items
              if not args.section or section == args.section]
    if not topics:
        sys.exit(f"No topics in section {args.section!r}. Sections: {list(TOPICS)}")
    print(f"Auditing {len(topics)} topics against {len(corpus_by_id)} diagrams...")

    # Topics are written "concept: parts", like the planner's searches
    retrieved = {t: diagram_server.search(t, structured=True) for _, t in topics}
    with ThreadPoolExecutor(max_workers=8) as pool:
        judgements = dict(zip(retrieved, pool.map(lambda t: judge(client, t, retrieved[t]), retrieved)))

    # Near-misses for the gaps: match rejected diagrams' descriptions against the topic.
    gaps = [t for _, t in topics if not judgements[t].depicting_ids]
    near: dict[str, list[dict]] = {t: [] for t in gaps}
    pool_items = load_near_miss_pool()
    if gaps and pool_items:
        scorer = diagram_server._scorer
        texts = [f'{p["title"]}. {p["description"]}' for p in pool_items]
        pool_emb = np.concatenate([scorer.get_text_embeddings(texts[i:i + 32]) for i in range(0, len(texts), 32)])
        shortlists = {}
        for t in gaps:
            order = np.argsort(-(pool_emb @ scorer.get_text_embedding(t).reshape(-1)))[:NEAR_MISS_CANDIDATES]
            shortlists[t] = [pool_items[i] for i in order]
        with ThreadPoolExecutor(max_workers=8) as pool:
            near_judgements = dict(zip(gaps, pool.map(lambda t: judge(client, t, shortlists[t]), gaps)))
        for t in gaps:
            by_id = {p["id"]: p for p in shortlists[t]}
            near[t] = [by_id[i] for i in near_judgements[t].depicting_ids]

    results = [
        {
            "section": section,
            "topic": t,
            "covered": bool(judgements[t].depicting_ids),
            "diagram_ids": judgements[t].depicting_ids,
            "reason": judgements[t].reason,
            "near_misses": near.get(t, []),
        }
        for section, t in topics
    ]

    covered = sum(r["covered"] for r in results)
    print(f"\nCoverage: {covered}/{len(results)} topics ({covered / len(results) * 100:.0f}%)\n")
    for section in dict.fromkeys(r["section"] for r in results):
        rows = [r for r in results if r["section"] == section]
        print(f"{section}: {sum(r['covered'] for r in rows)}/{len(rows)}")
        for r in rows:
            if not r["covered"]:
                hint = f"   [{len(r['near_misses'])} near-miss]" if r["near_misses"] else ""
                print(f"   ✗ {r['topic']}{hint}")

    if not args.section:
        COVERAGE_FILE.write_text(json.dumps(
            {"n_diagrams": len(corpus_by_id), "n_topics": len(results), "n_covered": covered, "topics": results},
            indent=1, ensure_ascii=False,
        ))
    write_page(results, corpus_by_id)
    print(f"\nGaps with images of the near-misses: {COVERAGE_PAGE}")


if __name__ == "__main__":
    main()
