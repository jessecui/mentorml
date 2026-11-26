#!/usr/bin/env python3
"""
Build the production diagram corpus.

Three stages, each resumable:

  fetch   Pull figures from the pages in scripts/corpus_sources.py into a
          local cache, with their captions and surrounding text.
  score   Have Gemini judge every candidate against the rubric (does it
          teach something, visual quality, understandable at a glance) and
          write a description. Scores are kept for rejects too.
  build   Apply the thresholds, drop near-duplicates, embed the survivors
          with SigLIP and write corpus/ (images, metadata, embeddings).
          Also writes a review page showing what passed and what did not.

Usage:
    python scripts/curate_corpus.py                 # all three stages
    python scripts/curate_corpus.py --only fetch
    python scripts/curate_corpus.py --only build --min-glanceable 3
    python scripts/curate_corpus.py --sources d2l,arxiv --limit 40

Changing a threshold only needs `--only build`: no pages are re-fetched and
no images are re-scored.

Outputs:
    corpus/candidates.json   every candidate with its scores (committed)
    corpus/metadata.json     accepted images, index-aligned with embeddings
    corpus/embeddings.npz    SigLIP image embeddings
    corpus/images/           accepted images, resized and compressed
    corpus_cache/            downloads and review.html (not committed)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from scripts.corpus_sources import KNOWN_REJECTS, SOURCES  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================

CORPUS_DIR = PROJECT_ROOT / "corpus"
CANDIDATES_FILE = CORPUS_DIR / "candidates.json"
METADATA_FILE = CORPUS_DIR / "metadata.json"
EMBEDDINGS_FILE = CORPUS_DIR / "embeddings.npz"
IMAGES_DIR = CORPUS_DIR / "images"

CACHE_DIR = PROJECT_ROOT / "corpus_cache"
PAGES_CACHE = CACHE_DIR / "pages"
RAW_CACHE = CACHE_DIR / "raw"
NORM_CACHE = CACHE_DIR / "norm"        # every candidate as an RGB PNG, max side 1600
EMB_CACHE = CACHE_DIR / "embeddings"   # one subfolder per SigLIP model
REVIEW_FILE = CACHE_DIR / "review.html"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; diagram-curation/1.0)"}
REQUEST_TIMEOUT = 30

# Cheap checks, applied before any model call
MIN_LONG_SIDE = 500
MIN_SHORT_SIDE = 220
MAX_ASPECT = 4.5
MAX_SIDE = 1600            # images are downscaled to this before saving
SVG_RENDER_WIDTH = 1600
MAX_FIGURES_PER_PAGE = 30  # papers with hundreds of sample images are not worth the downloads

# Rubric
RUBRIC_MODEL = "gemini-3-flash-preview"
RUBRIC_VERSION = 2         # bump to force re-scoring after changing the prompt
SCORE_WORKERS = 8
ACCEPTED_KINDS = {"diagram", "explanatory_chart"}

# Thresholds (strict by default; override on the command line)
MIN_VISUAL_QUALITY = 4
MIN_GLANCEABLE = 4
MIN_INSIGHT = 4
# Same figure reached twice scores 0.98+. Different concepts drawn in one house style reach
# 0.95-0.97, so a lower threshold would throw away distinct diagrams.
NEAR_DUPLICATE_COSINE = 0.98

SKIP_URL_PARTS = (
    "logo", "icon", "avatar", "badge", "emoji", "sprite", "spinner", "tracking",
    "pixel", "/static/logo", "/static/base/", "/front/assets/", "/outputs/", "thumbnail", "banner",
    "ar5iv.png", "book-cover", "/cover",
)
SKIP_EXTENSIONS = (".gif", ".mp4", ".webm", ".ico")


# =============================================================================
# CANDIDATE STORE
# =============================================================================

_store_lock = threading.Lock()


def load_candidates() -> dict[str, dict]:
    if CANDIDATES_FILE.exists():
        return {c["id"]: c for c in json.loads(CANDIDATES_FILE.read_text())}
    return {}


def save_candidates(candidates: dict[str, dict]) -> None:
    with _store_lock:
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        ordered = sorted(candidates.values(), key=lambda c: (c["source"], c["page_url"], c["order"]))
        tmp = CANDIDATES_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ordered, indent=1, ensure_ascii=False))
        tmp.replace(CANDIDATES_FILE)


def candidate_id(image_url: str) -> str:
    return "fig_" + hashlib.sha1(image_url.encode()).hexdigest()[:10]


# =============================================================================
# STAGE 1: FETCH
# =============================================================================

class HostThrottle:
    """Minimum delay between requests to one host."""

    def __init__(self, delay: float):
        self.delay = delay
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            pause = self._last + self.delay - time.time()
            if pause > 0:
                time.sleep(pause)
            self._last = time.time()


def http_get(url: str, throttle: HostThrottle, retries: int = 2) -> requests.Response | None:
    for attempt in range(retries + 1):
        throttle.wait()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 404, 410):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def fetch_page(url: str, throttle: HostThrottle) -> tuple[str, str] | None:
    """Return (html, final_url), using the page cache when possible."""
    PAGES_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    html_file = PAGES_CACHE / f"{key}.html"
    meta_file = PAGES_CACHE / f"{key}.json"
    if html_file.exists() and meta_file.exists():
        return html_file.read_text(encoding="utf-8"), json.loads(meta_file.read_text())["final_url"]
    resp = http_get(url, throttle)
    if resp is None:
        return None
    resp.encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else "utf-8"
    html_file.write_text(resp.text, encoding="utf-8")
    meta_file.write_text(json.dumps({"url": url, "final_url": resp.url}))
    return resp.text, resp.url


def _clean(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def extract_figures(html: str, page_url: str) -> tuple[str, list[dict]]:
    """Pull every figure-like image out of a page, with whatever describes it."""
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1", class_="ltx_title") or soup.find("title") or soup.find("h1")
    page_title = _clean(title_tag.get_text(" ") if title_tag else "", 200)
    page_title = re.sub(r"^\[[\d.]+\]\s*", "", page_title)       # ar5iv prefixes the arXiv ID
    page_title = re.sub(r"^\d+(\.\d+)*\s+", "", page_title)        # book section numbers
    page_title = re.split(r"\s+(?:[–|·]|--)\s+", page_title)[0].strip()   # drop the site-name suffix

    container = None
    for selector in ("div.ltx_page_content", "#quarto-document-content", "div.prose-doc",
                     "div.post-content", "article", "main"):
        container = soup.select_one(selector)
        if container:
            break
    container = container or soup.body or soup

    figures: list[dict] = []
    seen: set[str] = set()
    for tag in container.find_all(["img", "image", "object"]):
        if tag.name == "object":
            src = tag.get("data", "")
        else:
            src = tag.get("src") or tag.get("data-src") or ""
        src = str(src).strip()
        if not src or src.startswith("data:"):
            continue
        image_url = urljoin(page_url, src)
        path = urlparse(image_url).path.lower()
        if path.endswith(SKIP_EXTENSIONS) or any(part in image_url.lower() for part in SKIP_URL_PARTS):
            continue
        if image_url in seen:
            continue
        seen.add(image_url)

        caption = ""
        figure = tag.find_parent("figure") or tag.find_parent("div", class_="figure")
        # arXiv nests panels inside an outer figure that owns the caption
        while figure is not None and not figure.find(["figcaption"]) and figure.find_parent("figure"):
            figure = figure.find_parent("figure")
        if figure is not None:
            cap = figure.find("figcaption") or figure.find(class_="caption")
            caption = _clean(cap.get_text(" ") if cap else "", 600)
        parent = tag.parent
        if (not caption and parent is not None and parent.name == "div"
                and len(parent.find_all(recursive=False)) <= 4 and len(parent.get_text(" ", strip=True)) <= 600):
            # Jay Alammar puts the caption as bare text after the image, inside a small
            # wrapper div; a large content div's text is not a caption
            caption = _clean(parent.get_text(" "), 600)

        previous_p = tag.find_previous("p")
        heading = tag.find_previous(["h1", "h2", "h3", "h4"])
        figures.append({
            "image_url": image_url,
            "alt": _clean(str(tag.get("alt") or ""), 200),
            "caption": caption,
            "context": _clean(previous_p.get_text(" ") if previous_p else "", 500),
            "section": _clean(heading.get_text(" ") if heading else "", 120),
        })
    return page_title, figures


def collect_pages(source: dict, throttle: HostThrottle) -> list[tuple[str, str]]:
    """Resolve a source to (fetch_url, public_url) pairs."""
    pages: list[tuple[str, str]] = [(u, u) for u in source.get("pages", [])]

    link_re = re.compile(source["link_pattern"]) if source.get("link_pattern") else None
    skip_re = re.compile(source["skip_pattern"]) if source.get("skip_pattern") else None
    for seed in source.get("crawl", []):
        fetched = fetch_page(seed, throttle)
        if fetched is None:
            print(f"  ⚠️  [{source['key']}] could not fetch seed {seed}")
            continue
        html, final_url = fetched
        found = [final_url]
        seed_dir = final_url.rsplit("/", 1)[0] + "/"
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            url = urljoin(final_url, a["href"]).split("#")[0]
            # Stay inside the seed's own chapter/unit; site-wide navigation is ignored.
            if not url.startswith(seed_dir):
                continue
            if link_re and link_re.match(url) and url not in found:
                found.append(url)
        for url in found:
            if skip_re and skip_re.search(url):
                continue
            if (url, url) not in pages:
                pages.append((url, url))

    for arxiv_id in source.get("arxiv_ids", []):
        pages.append((f"arxiv:{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"))
    return pages


def _rasterise_svg(raw: bytes) -> Image.Image:
    """Render an SVG to pixels. resvg is sharper; MuPDF copes with files resvg rejects."""
    try:
        import resvg_py

        png = resvg_py.svg_to_bytes(svg_string=raw.decode("utf-8", errors="replace"),
                                    width=SVG_RENDER_WIDTH, background="white")
        return Image.open(io.BytesIO(bytes(png)))
    except Exception:
        import pymupdf

        page = pymupdf.open(stream=raw, filetype="svg")[0]
        zoom = SVG_RENDER_WIDTH / max(page.rect.width, 1)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return Image.open(io.BytesIO(pixmap.tobytes("png")))


def normalise_image(raw: bytes, url: str) -> tuple[Image.Image | None, str]:
    """Decode (rasterising SVG), flatten onto white and cap the size."""
    try:
        head = raw[:600].lstrip().lower()
        if urlparse(url).path.lower().endswith(".svg") or head.startswith(b"<?xml") or b"<svg" in head:
            image = _rasterise_svg(raw)
        else:
            image = Image.open(io.BytesIO(raw))
            if getattr(image, "is_animated", False):
                return None, "animated"
        image.load()
    except Exception:
        return None, "unreadable"

    rgba = image.convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    if max(flat.size) > MAX_SIDE:
        flat.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return flat, "ok"


def precheck(image: Image.Image) -> str:
    long_side, short_side = max(image.size), min(image.size)
    if long_side < MIN_LONG_SIDE or short_side < MIN_SHORT_SIDE:
        return "too_small"
    if long_side / short_side > MAX_ASPECT:
        return "bad_aspect"
    # Nearly blank images (spacers, empty canvases)
    if np.asarray(image.convert("L").resize((64, 64))).std() < 4:
        return "blank"
    return "ok"


def fetch_source(source: dict, candidates: dict[str, dict], limit: int | None) -> int:
    throttle = HostThrottle(source.get("delay", 0.5))
    image_throttle = HostThrottle(min(source.get("delay", 0.5), 1.0))
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    NORM_CACHE.mkdir(parents=True, exist_ok=True)

    added = 0
    pages = collect_pages(source, throttle)
    print(f"  [{source['key']}] {len(pages)} pages")
    for fetch_url, public_url in pages:
        if limit is not None and added >= limit:
            break
        if fetch_url.startswith("arxiv:"):
            arxiv_id = fetch_url.split(":", 1)[1]
            fetched = (fetch_page(f"https://arxiv.org/html/{arxiv_id}", throttle)
                       or fetch_page(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", throttle))
        else:
            fetched = fetch_page(fetch_url, throttle)
        if fetched is None:
            print(f"  ⚠️  [{source['key']}] no page for {public_url}")
            continue
        html, final_url = fetched
        page_title, figures = extract_figures(html, final_url)

        for order, fig in enumerate(figures[:MAX_FIGURES_PER_PAGE]):
            if limit is not None and added >= limit:
                break
            cid = candidate_id(fig["image_url"])
            norm_file = NORM_CACHE / f"{cid}.png"
            if cid in candidates and (norm_file.exists() or candidates[cid].get("precheck") != "ok"):
                continue

            raw_file = next(iter(RAW_CACHE.glob(f"{cid}.*")), None)
            if raw_file is not None:
                raw = raw_file.read_bytes()
            else:
                resp = http_get(fig["image_url"], image_throttle)
                if resp is None:
                    continue
                raw = resp.content
                ext = Path(urlparse(fig["image_url"]).path).suffix.lower() or ".img"
                (RAW_CACHE / f"{cid}{ext}").write_bytes(raw)

            image, status = normalise_image(raw, fig["image_url"])
            # A fresh dict, stored under the lock: another thread may be serialising
            # the candidates while this one works
            with _store_lock:
                record = dict(candidates.get(cid, {}))
            record.update({
                "id": cid,
                "source": source["key"],
                "source_name": source["name"],
                "page_url": public_url,
                "page_title": page_title,
                "order": order,
                **fig,
            })
            if image is None:
                record["precheck"] = status
            else:
                record["width"], record["height"] = image.size
                record["precheck"] = precheck(image)
                record["content_hash"] = hashlib.sha1(image.tobytes()).hexdigest()[:16]
                if record["precheck"] == "ok":
                    image.save(norm_file, optimize=True)
            record.setdefault("fetched_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            with _store_lock:
                candidates[cid] = record
            added += 1
    return added


def stage_fetch(candidates: dict[str, dict], source_keys: set[str] | None, limit: int | None) -> None:
    print("=" * 60 + "\nStage 1: fetch\n" + "=" * 60)
    sources = [s for s in SOURCES if source_keys is None or s["key"] in source_keys]
    # One thread per source: each source is a different host with its own delay.
    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as pool:
        futures = {pool.submit(fetch_source, s, candidates, limit): s for s in sources}
        for future in as_completed(futures):
            source = futures[future]
            try:
                print(f"  ✅ [{source['key']}] {future.result()} new candidates")
            except Exception as exc:
                print(f"  ❌ [{source['key']}] failed: {exc!r}")
            save_candidates(candidates)

    # Exact duplicates (same pixels reached through different URLs)
    first_seen: dict[str, str] = {}
    for c in sorted(candidates.values(), key=lambda c: (c["source"], c["page_url"], c["order"])):
        c.pop("duplicate_of", None)
        h = c.get("content_hash")
        if c.get("precheck") != "ok" or not h:
            continue
        if h in first_seen:
            c["duplicate_of"] = first_seen[h]
        else:
            first_seen[h] = c["id"]
    save_candidates(candidates)

    ok = sum(1 for c in candidates.values() if c.get("precheck") == "ok" and not c.get("duplicate_of"))
    print(f"\n{len(candidates)} candidates in total, {ok} pass the cheap checks")


# =============================================================================
# STAGE 2: SCORE
# =============================================================================

class Rubric(BaseModel):
    kind: Literal[
        "diagram", "explanatory_chart", "results_plot", "table", "text_or_equation",
        "code_or_screenshot", "example_output", "photo", "decorative", "other",
    ] = Field(description="What sort of image this is.")
    teaches_something: bool = Field(description="True only if it shows a structure, a process or a relationship.")
    what_it_teaches: str = Field(description="One sentence; empty if teaches_something is false.")
    standalone: bool = Field(description="Explains a general concept without needing the article it came from.")
    cropped: bool = Field(description="Content is visibly cut off at an edge.")
    decoration: Literal["none", "minor", "dominant"] = Field(
        description="Cartoon characters, mascots, clip-art or jokes that carry no technical information."
    )
    insight: int = Field(ge=1, le=5)
    visual_quality: int = Field(ge=1, le=5)
    glanceable: int = Field(ge=1, le=5)
    title: str = Field(description="At most 8 words naming the concept shown.")
    description: str = Field(description="1-2 concrete sentences for a learner.")
    concepts: list[str] = Field(description="3-8 lowercase keywords.")
    benchmark_query: str = Field(description="10-25 word search query that only this image answers.")
    reason: str = Field(description="One sentence justifying the scores.")


RUBRIC_PROMPT = """You are curating images for an AI/ML teaching assistant. The image will be shown inline in a chat answer, on its own, to a learner who has NOT read the page it came from. The assistant writes its own explanation; the image has to earn its place next to it.

Judge the IMAGE ITSELF. The context below is only there to help you identify the topic and describe the image accurately. Do not give the image credit for information that appears only in the context.

Context from the source page:
- Source: {source_name}
- Page title: {page_title}
- Section: {section}
- Caption: {caption}
- Text just before the image: {context}

Fill in every field.

kind:
- "diagram": a schematic of an architecture, a data flow, an algorithm's steps, or a concept.
- "explanatory_chart": a plot drawn to explain a concept (an activation function, a learning-rate schedule, a loss surface).
- "results_plot": experimental results, benchmark comparisons or training curves reported by a paper.
- "table", "text_or_equation", "code_or_screenshot": mostly text, numbers, maths or UI.
- "example_output": samples produced by a model (generated images, example prompts and completions, game frames).
- "photo", "decorative": photographs, mascots, cartoons, title cards, logos, book covers, memes.
- "other": anything else.

teaches_something: true ONLY if, after looking at the image, a learner would understand a structure, a process, or a relationship between parts. It is false for decorative or mascot illustrations, title cards, images that only name things without showing how they work or relate, in-jokes, and bare lists of labels.

what_it_teaches: one sentence starting "A learner sees". Leave it empty if teaches_something is false. If the best you can write is a list of what is pictured, then teaches_something is false.

standalone: true only if the image explains a GENERAL concept and would fit a fresh explanation written by someone else. It is false when the image:
- is one step of a walkthrough and needs the earlier steps;
- is tied to one tutorial's specifics: code variable names (such as "last_hidden_states[0]"), a particular dataset's rows or sizes, a particular library call;
- refers to things defined elsewhere, or uses symbols it does not label;
- is a fragment or a zoomed-in piece of a larger figure.

cropped: true if boxes, text or arrows are visibly cut off at an edge of the image.

decoration: how much of the image is cartoon characters, mascots, clip-art, emoji or jokes that carry no technical information.
- "none": everything drawn carries information.
- "minor": a small flourish that does not get in the way.
- "dominant": the characters or artwork are a main visual element, or a component is represented by a character or face instead of a labelled shape.

insight (1-5): how much understanding the image adds beyond a sentence of text.
5 = reveals a mechanism: how the parts interact, what happens at each step, why it works.
4 = shows a real structure or process with its important parts and connections.
3 = a simple overview: a few boxes and arrows that a sentence could replace.
2 = little more than labels, or a before/after with nothing in between.
1 = no information.

visual_quality (1-5), judged at a chat width of about 700 pixels:
5 = crisp and professional, all text easily legible, consistent styling.
4 = clean, text legible, minor roughness.
3 = readable but rough: low resolution, compression artefacts, cramped or tiny text, clashing colours, sloppy drawing, large empty areas.
2 = hard to read.
1 = blurry, illegible or watermarked.

glanceable (1-5): how quickly the main idea lands without any surrounding text.
5 = one clear idea, understood in about five seconds: a few labelled parts and an obvious flow.
4 = a clear main idea; some details need a second look.
3 = understandable with effort: many parts or panels, small annotations, or notation that is not explained in the image.
2 = dense or overwhelming.
1 = incomprehensible without the article.

title: at most 8 words naming the concept shown.

description: 1-2 concrete sentences for a learner, naming the parts and how they connect. Do not start with "This image" or "This diagram".

concepts: 3-8 lowercase keywords a learner might search for.

benchmark_query: a search query of 10-25 words that this image, and no similar image, would answer. Include specific labels or symbols visible in the image, the relationship or process it shows, and the technical concept. Write about content, not appearance: "encoder stack with Nx layers containing self-attention and feed-forward sublayers" is good; "green and pink boxes with arrows" and "transformer architecture" are bad.

reason: one sentence justifying the scores.

Be strict. Most images scraped from articles are not good enough: expect to give 4 or 5 on any scale to well under half of what you see. When unsure between two scores, choose the lower one, and when unsure about standalone, answer false."""


def _image_bytes_for_model(path: Path) -> bytes:
    image = Image.open(path).convert("RGB")
    if max(image.size) > 1280:
        image.thumbnail((1280, 1280), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def score_one(client, candidate: dict) -> dict | None:
    from google.genai import types

    prompt = RUBRIC_PROMPT.format(
        source_name=candidate.get("source_name", ""),
        page_title=candidate.get("page_title", "") or "(none)",
        section=candidate.get("section", "") or "(none)",
        caption=candidate.get("caption", "") or candidate.get("alt", "") or "(none)",
        context=candidate.get("context", "") or "(none)",
    )
    image_part = types.Part.from_bytes(
        data=_image_bytes_for_model(NORM_CACHE / f"{candidate['id']}.png"), mime_type="image/png"
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Rubric,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=RUBRIC_MODEL, contents=[image_part, prompt], config=config
            )
            parsed = response.parsed
            if isinstance(parsed, Rubric):
                return parsed.model_dump()
            return Rubric.model_validate_json(response.text).model_dump()
        except Exception as exc:  # rate limits, transient errors, malformed JSON
            last_error = exc
            time.sleep(3 * (attempt + 1))
    print(f"    ⚠️  {candidate['id']}: scoring failed: {last_error!r}"[:300])
    return None


def stage_score(candidates: dict[str, dict], source_keys: set[str] | None, limit: int | None) -> None:
    print("=" * 60 + "\nStage 2: score\n" + "=" * 60)
    todo = [
        c for c in candidates.values()
        if c.get("precheck") == "ok"
        and not c.get("duplicate_of")
        and (source_keys is None or c["source"] in source_keys)
        and (c.get("rubric") is None or c.get("rubric_version") != RUBRIC_VERSION)
        and (NORM_CACHE / f"{c['id']}.png").exists()
    ]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} candidates to score with {RUBRIC_MODEL}")
    if not todo:
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("❌ GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    from google import genai

    client = genai.Client(api_key=api_key)
    done = 0
    with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
        futures = {pool.submit(score_one, client, c): c for c in todo}
        for future in as_completed(futures):
            candidate = futures[future]
            rubric = future.result()
            if rubric is not None:
                with _store_lock:
                    candidate["rubric"] = rubric
                    candidate["rubric_version"] = RUBRIC_VERSION
                    candidate["rubric_model"] = RUBRIC_MODEL
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} scored")
                save_candidates(candidates)
    save_candidates(candidates)


# =============================================================================
# STAGE 3: BUILD
# =============================================================================

def verdict(candidate: dict, min_quality: int, min_glanceable: int, min_insight: int) -> str:
    """Why a candidate is in or out. 'accepted' means it passed every check."""
    if candidate.get("precheck") != "ok":
        return f"precheck:{candidate.get('precheck')}"
    if candidate.get("duplicate_of"):
        return "duplicate"
    rubric = candidate.get("rubric")
    if not rubric:
        return "unscored"
    if not rubric["teaches_something"] or not rubric["what_it_teaches"].strip():
        return "teaches_nothing"
    if rubric["kind"] not in ACCEPTED_KINDS:
        return f"kind:{rubric['kind']}"
    if not rubric["standalone"]:
        return "not_standalone"
    if rubric["cropped"]:
        return "cropped"
    if rubric["decoration"] == "dominant":
        return "decorated"
    if rubric["insight"] < min_insight:
        return "low_insight"
    if rubric["visual_quality"] < min_quality:
        return "low_visual_quality"
    if rubric["glanceable"] < min_glanceable:
        return "not_glanceable"
    return "accepted"


def embed_candidates(items: list[dict], device: str | None) -> tuple[np.ndarray, float, float, str]:
    from model.scorer import MODEL_ID, SigLIPScorer

    # Cached vectors are only valid for the model that made them
    cache = EMB_CACHE / MODEL_ID.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    def cached(c: dict) -> Path:
        # Keyed by the image's content as well: a changed image at the same URL is re-embedded
        return cache / f"{c['id']}_{c['content_hash']}.npy"

    missing = [c for c in items if not cached(c).exists()]
    scorer = None
    if missing:
        print(f"Embedding {len(missing)} images with SigLIP 2 ({device or 'default device'})...")
        scorer = SigLIPScorer(device=device)
        for start in range(0, len(missing), 8):
            batch = missing[start:start + 8]
            embs = scorer.get_image_embeddings([NORM_CACHE / f"{c['id']}.png" for c in batch])
            for c, emb in zip(batch, embs):
                np.save(cached(c), emb.astype(np.float32))
            print(f"  {min(start + 8, len(missing))}/{len(missing)} embedded")
        temperature, bias = scorer.temperature, scorer.bias
    else:
        from model.scorer import _read_calibration

        temperature, bias = _read_calibration(MODEL_ID)
    embeddings = (np.stack([np.load(cached(c)) for c in items])
                  if items else np.zeros((0, 1152), dtype=np.float32))
    return embeddings, temperature, bias, MODEL_ID


def drop_near_duplicates(items: list[dict], embeddings: np.ndarray) -> tuple[list[dict], np.ndarray, dict[str, str]]:
    """Keep the best of each cluster of look-alike images."""
    def quality_key(c: dict) -> tuple:
        r = c["rubric"]
        return (r["glanceable"] + r["visual_quality"] + r["insight"], c["width"] * c["height"])

    order = sorted(range(len(items)), key=lambda i: quality_key(items[i]), reverse=True)
    kept: list[int] = []
    dropped: dict[str, str] = {}
    for i in order:
        if kept:
            sims = embeddings[kept] @ embeddings[i]
            j = int(np.argmax(sims))
            if sims[j] >= NEAR_DUPLICATE_COSINE:
                dropped[items[i]["id"]] = items[kept[j]]["id"]
                continue
        kept.append(i)
    kept.sort()
    return [items[i] for i in kept], embeddings[kept], dropped


def save_corpus_image(candidate: dict) -> str:
    """Write the accepted image as WebP: lossless when small, high-quality lossy otherwise."""
    image = Image.open(NORM_CACHE / f"{candidate['id']}.png").convert("RGB")
    target = IMAGES_DIR / f"{candidate['id']}.webp"
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True, method=6)
    if buffer.tell() > 350_000:
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=90, method=6)
    target.write_bytes(buffer.getvalue())
    return target.name


def write_review_page(candidates: dict[str, dict], verdicts: dict[str, str]) -> None:
    rows = []
    for c in sorted(candidates.values(), key=lambda c: (verdicts[c["id"]] != "accepted", c["source"], c["page_url"], c["order"])):
        v = verdicts[c["id"]]
        if v.startswith("precheck") or v == "duplicate":
            continue
        r = c.get("rubric") or {}
        rows.append(
            f'<div class="card {"ok" if v == "accepted" else "no"}" data-verdict="{escape(v)}" data-source="{escape(c["source"])}">'
            f'<a href="norm/{c["id"]}.png" target="_blank"><img loading="lazy" src="norm/{c["id"]}.png"></a>'
            f'<div class="meta"><b>{escape(v)}</b> · {escape(c["id"])} · {escape(c["source"])}<br>'
            f'quality {r.get("visual_quality", "?")} · glance {r.get("glanceable", "?")} · insight {r.get("insight", "?")}'
            f' · {escape(str(r.get("kind", "?")))} · standalone {r.get("standalone", "?")}'
            f' · decoration {escape(str(r.get("decoration", "?")))}<br>'
            f'<i>{escape(r.get("title", ""))}</i><br>{escape(r.get("what_it_teaches", ""))}<br>'
            f'<span class="why">{escape(r.get("reason", ""))}</span><br>'
            f'<a href="{escape(c["page_url"])}" target="_blank">source page</a></div></div>'
        )
    counts: dict[str, int] = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    summary = " · ".join(f"{escape(k)}: {n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    REVIEW_FILE.write_text(f"""<!doctype html><meta charset="utf-8"><title>Corpus review</title>
<style>
body{{font:14px system-ui;margin:20px;background:#f6f6f6}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.card{{background:#fff;border-radius:8px;padding:10px;border-top:5px solid #c33}} .card.ok{{border-top-color:#2a2}}
.card img{{width:100%;max-height:260px;object-fit:contain;background:#fff}} .meta{{font-size:12px;line-height:1.45;margin-top:6px}}
.why{{color:#666}} button{{margin:0 6px 12px 0;padding:5px 12px}} .hide{{display:none}}
</style>
<h2>Corpus review</h2><p>{summary}</p>
<button onclick="show('all')">All</button><button onclick="show('ok')">Accepted</button><button onclick="show('no')">Rejected</button>
<div class="grid">{''.join(rows)}</div>
<script>function show(w){{document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hide',w!=='all'&&!c.classList.contains(w)))}}</script>
""", encoding="utf-8")


def stage_build(candidates: dict[str, dict], min_quality: int, min_glanceable: int, min_insight: int,
                device: str | None) -> None:
    print("=" * 60 + "\nStage 3: build\n" + "=" * 60)
    verdicts = {cid: verdict(c, min_quality, min_glanceable, min_insight) for cid, c in candidates.items()}
    accepted = [c for c in candidates.values() if verdicts[c["id"]] == "accepted"]
    missing = [c["id"] for c in accepted if not (NORM_CACHE / f"{c['id']}.png").exists()]
    if missing:
        sys.exit(
            f"❌ {len(missing)} accepted images are not in {NORM_CACHE} (e.g. {missing[0]}).\n"
            "   The download cache is not in git, so on a fresh clone run the fetch stage first:\n"
            "   python scripts/curate_corpus.py --only fetch   (then --only build)\n"
            "   Nothing was written; the existing corpus is unchanged."
        )
    if not accepted:
        sys.exit("❌ No candidate passes the current thresholds. Nothing was written; the existing corpus is unchanged.")
    accepted.sort(key=lambda c: (c["source"], c["page_url"], c["order"]))

    by_url = {c["image_url"]: c for c in candidates.values()}
    for url in KNOWN_REJECTS:
        c = by_url.get(url)
        if c is None:
            print(f"  (known reject not among candidates: {url})")
        elif verdicts[c["id"]] == "accepted":
            sys.exit(f"❌ Known reject was accepted: {url}\n   The rubric or thresholds have drifted; nothing was written.")
        else:
            print(f"  ✅ known reject correctly rejected ({verdicts[c['id']]}): {url}")

    embeddings, temperature, bias, model_id = embed_candidates(accepted, device)
    accepted, embeddings, near_dupes = drop_near_duplicates(accepted, embeddings)
    for cid, kept_id in near_dupes.items():
        verdicts[cid] = "near_duplicate"

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for c in accepted:
        r = c["rubric"]
        metadata.append({
            "id": c["id"],
            "file": save_corpus_image(c),
            "width": c["width"],
            "height": c["height"],
            "title": r["title"],
            "description": r["description"],
            "teaches": r["what_it_teaches"],
            "concepts": r["concepts"],
            "kind": r["kind"],
            "visual_quality": r["visual_quality"],
            "glanceable": r["glanceable"],
            "insight": r["insight"],
            "benchmark_query": r["benchmark_query"],
            "source": c["source"],
            "source_name": c["source_name"],
            "source_title": c["page_title"],
            "source_url": c["page_url"],
            "image_url": c["image_url"],
        })
    # New files go to temporary names and are renamed into place, so an interrupted build
    # leaves the old corpus intact; stale images are removed only after both are written
    metadata_tmp = METADATA_FILE.with_suffix(".json.tmp")
    embeddings_tmp = EMBEDDINGS_FILE.with_name("embeddings.tmp.npz")
    metadata_tmp.write_text(json.dumps(metadata, indent=1, ensure_ascii=False))
    np.savez(
        embeddings_tmp,
        embeddings=embeddings.astype(np.float32),
        ids=np.array([m["id"] for m in metadata]),
        temperature=temperature,
        bias=bias,
        model_id=model_id,
    )
    embeddings_tmp.replace(EMBEDDINGS_FILE)
    metadata_tmp.replace(METADATA_FILE)

    keep_files = {m["file"] for m in metadata}
    for stale in IMAGES_DIR.glob("*.webp"):
        if stale.name not in keep_files:
            stale.unlink()
    write_review_page(candidates, verdicts)

    counts: dict[str, int] = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    print("\nVerdicts:")
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {v}")
    by_source: dict[str, int] = {}
    for m in metadata:
        by_source[m["source"]] = by_source.get(m["source"], 0) + 1
    size_mb = sum((IMAGES_DIR / m["file"]).stat().st_size for m in metadata) / 1e6
    print(f"\n✅ {len(metadata)} images in corpus ({size_mb:.1f} MB): "
          + ", ".join(f"{k} {n}" for k, n in sorted(by_source.items())))
    print(f"   Review page: {REVIEW_FILE}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["fetch", "score", "build"], help="Run a single stage")
    parser.add_argument("--sources", help="Comma-separated source keys (default: all)")
    parser.add_argument("--limit", type=int, help="Max new candidates per source (fetch) or images to score")
    parser.add_argument("--min-visual-quality", type=int, default=MIN_VISUAL_QUALITY)
    parser.add_argument("--min-glanceable", type=int, default=MIN_GLANCEABLE)
    parser.add_argument("--min-insight", type=int, default=MIN_INSIGHT)
    parser.add_argument("--device", default=None, help="torch device for embedding (cpu, mps, cuda)")
    args = parser.parse_args()

    source_keys = set(args.sources.split(",")) if args.sources else None
    if source_keys:
        unknown = source_keys - {s["key"] for s in SOURCES}
        if unknown:
            sys.exit(f"Unknown source keys: {sorted(unknown)}")

    candidates = load_candidates()
    if args.only in (None, "fetch"):
        stage_fetch(candidates, source_keys, args.limit)
    if args.only in (None, "score"):
        stage_score(candidates, source_keys, args.limit)
    if args.only in (None, "build"):
        stage_build(candidates, args.min_visual_quality, args.min_glanceable, args.min_insight, args.device)


if __name__ == "__main__":
    main()
