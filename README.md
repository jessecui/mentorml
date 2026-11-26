# MentorML

A multimodal AI/ML teaching assistant that explains concepts with diagrams. A Plan-and-Execute agent turns a question into a teaching plan, retrieves candidate diagrams from a curated corpus with SigLIP 2, and writes an explanation around the diagram that fits best.

## Features

- 🤖 **LangGraph Agent**: Plan-and-Execute architecture: one structured planning call, tool-based retrieval, one streamed answer
- 🔍 **SigLIP 2 Retrieval**: Query-image retrieval over precomputed embeddings (83% top-1, 97% top-5 on the curated corpus)
- 🔌 **MCP Tool Server**: Diagram retrieval runs as a separate MCP subprocess (stdio); the agent calls it like any LangChain tool
- 🎨 **Curated Corpus**: 620 diagrams from eleven sources, each one vetted by a Gemini rubric for whether it teaches something, how clear it is and how good it looks
- 📊 **LangSmith Eval**: Traced retrieval (SigLIP vs CLIP) and end-to-end agent eval against a versioned dataset
- 💬 **Conversation Memory**: Redis-backed checkpointing with 24h TTL
- ⚡ **SSE Streaming**: Real-time token streaming; the first words arrive about two seconds after the question
- 🖥️ **React Frontend**: Chat UI with inline diagrams, source links and collapsible teaching plans

## Quick Start

```bash
git clone https://github.com/jessecui/mentor-ml.git
cd mentor-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Then edit with your keys

# Start Redis with the search module (required for conversation memory)
docker run -p 6379:6379 redis/redis-stack-server  # or: redis-stack-server

# Install frontend dependencies
cd frontend && npm install && cd ..
```

The SigLIP 2 checkpoint (about 4.5 GB) downloads from Hugging Face on first start. To keep only the part the app needs at query time (the text tower, about 2.8 GB), run `python scripts/export_text_tower.py` once; the Docker build does this automatically.

### Development (two servers)

```bash
# Terminal 1: Backend
uvicorn main:app --reload --port 8080

# Terminal 2: Frontend (with hot reload)
cd frontend && npm run dev
```
Open **http://localhost:5173**

### Production (single server)

```bash
# Build frontend
cd frontend && npm run build && cd ..

# Run server (serves API + frontend)
uvicorn main:app --port 8080
```
Open **http://localhost:8080**

## Architecture

```
User question
     │
     ▼
 Plan node ── one structured call: topic, steps, diagram searches, follow-up or not
     │
     ▼
 Tools node ── retrieve_diagram ←── MCP stdio ──→ diagram_server.py
     │           (searches run in parallel)        (SigLIP 2 text tower +
     ▼                                              precomputed image embeddings)
 Answer node ── one streamed call: writes the explanation and picks the
     │          best candidate (or none); may search once more itself
     ▼
 Response
```

Each search returns about a dozen candidates, each with a description of what the image shows, what it teaches and how clear it is. Those descriptions were written once, when the corpus was curated, so no model looks at an image while a learner waits.

Retrieval ranks images by SigLIP 2 query-image similarity, using the query as written plus the bare concept name, since some diagrams match one phrasing far better than the other. A keyword match over titles and descriptions then adds a few more candidates, which catches diagrams the image match misses. All of this takes tens of milliseconds.

If none of the candidates is about the topic, the answer goes out without a diagram and the question is logged as a miss.

## Diagram Corpus

The corpus the app serves lives in `corpus/` and is built by one script:

```bash
pip install -r requirements-dev.txt

python scripts/curate_corpus.py                 # fetch, score and build
python scripts/curate_corpus.py --only build    # re-apply thresholds, no downloads, no model calls
python scripts/curate_corpus.py --only build --min-glanceable 3
```

It runs in three stages:

1. **Fetch**: pulls figures, captions and surrounding text from the pages listed in `scripts/corpus_sources.py`, and drops thumbnails, extreme aspect ratios and blank images.
2. **Score**: Gemini judges each image on its own, as a learner would see it:
   - **Teaches something** (pass/fail): it must show a structure, a process or a relationship. Decorative illustrations, title cards and mascots fail.
   - **Standalone** (pass/fail): it explains a general concept and is not one step of a walkthrough or tied to one tutorial's variable names.
   - **Insight** (1-5): how much understanding it adds beyond a sentence of text.
   - **Visual quality** (1-5): legibility and rendering at chat width.
   - **Glanceable** (1-5): how quickly the main idea lands without surrounding text.

   The same call writes the description the agent later uses to choose between candidates.
3. **Build**: keeps images that pass every check (scores of 4 or more by default), removes duplicates, embeds the rest with SigLIP 2 and writes `corpus/`.

Scores are kept for every candidate, including rejects, in `corpus/candidates.json`, so changing a threshold is instant. The build also writes `corpus_cache/review.html`, a page showing what passed and what was rejected, with scores and reasons.

To widen coverage, add pages or arXiv IDs to `scripts/corpus_sources.py` and re-run the script. Pages already fetched and images already scored are skipped.

### Sources

| Source | Diagrams accepted |
|--------|-------------------|
| [Dive into Deep Learning](https://d2l.ai/) | 285 |
| [Lil'Log](https://lilianweng.github.io/) (Lilian Weng) | 124 |
| [Jay Alammar](https://jalammar.github.io/)'s Illustrated posts | 78 |
| arXiv papers (54 landmark papers, from word2vec to ReAct) | 42 |
| [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) | 28 |
| [colah's blog](https://colah.github.io/) (Christopher Olah) | 19 |
| [Stanford CS231n notes](https://cs231n.github.io/) | 16 |
| [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) | 14 |
| [Distill](https://distill.pub/) | 10 |
| [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) | 3 |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) | 1 |

620 of 2,308 candidate images made it in. The most common reasons for rejection were being a results plot (404), being too small (267), adding little insight (222) and not being understandable at a glance (191).

### Coverage audit

`scripts/audit_coverage.py` checks the corpus against a list of topics a learner is likely to ask about (`scripts/coverage_topics.py`). For each topic it runs the app's own retrieval and has Gemini judge, from the descriptions alone, whether any candidate directly depicts it. Topics with no diagram are listed together with any rejected images that came close.

```bash
python scripts/audit_coverage.py
```

Currently 133 of 138 topics (96%) have a diagram. The audit is how new sources get chosen: run it, add pages for the gaps, re-run the curation.

Every diagram is shown with a link to the page it came from.

### Misses

Questions where the plan asked for a diagram but none of the candidates was good enough to show are logged to Redis, so they survive redeploys. They show where the corpus needs more sources.

The log holds other people's questions, so reading it needs a separate admin token. Set `ADMIN_TOKEN` in `.env`, then:

```bash
curl http://localhost:8080/misses -H "X-Admin-Token: $ADMIN_TOKEN"
```

With `ADMIN_TOKEN` unset the endpoint is switched off.

## API Usage

The chat endpoints need a session. Log in once with the app password and keep the cookie. A thread ID must be `thread-` followed by 32 random hex characters (it is the only key to a conversation, so it must be unguessable):

```bash
curl -c cookies.txt -X POST http://localhost:8080/validate-password \
  -H "Content-Type: application/json" \
  -d '{"password": "your app password"}'
```

### Streaming (Recommended)

```bash
# Stream chat responses with SSE
curl -N -b cookies.txt -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How do transformers work?", "thread_id": "thread-3f9a1c2e0b7d4e6f8a5c1b2d3e4f5a6b"}'
```

SSE Events:
- `plan` - The teaching plan
- `diagram` - A candidate diagram with its metadata (rendered once the text cites it)
- `token` - Response text tokens
- `reset` - The answer restarted (a fallback model took over); discard the text so far
- `done` - Final state with the diagrams the answer cites
- `error` - Something failed, or no answer was produced

### Non-Streaming

```bash
# Chat with the agent (blocking)
curl -b cookies.txt -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do transformers work?", "thread_id": "thread-3f9a1c2e0b7d4e6f8a5c1b2d3e4f5a6b"}'
```

### Response Format

```json
{
  "response": "Transformers use self-attention to process sequences...\n\n[diagram: fig_3a236b414c]\n\n...",
  "diagrams": [
    {
      "id": "fig_3a236b414c",
      "image_url": "/corpus/images/fig_3a236b414c.webp",
      "title": "Creating Query Key and Value Vectors",
      "description": "Each input embedding x is multiplied by three shared weight matrices...",
      "teaches": "A learner sees how input token embeddings are multiplied by weight matrices...",
      "source_name": "Jay Alammar",
      "source_title": "The Illustrated GPT-2 (Visualizing Transformer Language Models)",
      "source_url": "https://jalammar.github.io/illustrated-gpt2/",
      "score": 0.99,
      "query": "self-attention: queries, keys, values, softmax, weighted sum"
    }
  ],
  "plan": {
    "topic": "Transformers",
    "steps": ["Explain self-attention", "Describe Q, K, V matrices", "..."],
    "diagrams_needed": ["self-attention: queries, keys, values, softmax, weighted sum"]
  }
}
```

## SigLIP Scorer

The scorer uses [SigLIP 2](https://arxiv.org/abs/2502.14786) (So400m/14 @ 384px), Google's contrastive vision-language model, through Hugging Face `transformers`.

```python
from model.scorer import SigLIPScorer

# Full model: needed to embed images (curation, benchmarks)
scorer = SigLIPScorer()
score = scorer.score("diagram.png", "transformer attention mechanism")
image_embeddings = scorer.get_image_embeddings(["img1.png", "img2.png"])

# Text tower only: all the app needs at query time
scorer = SigLIPScorer(text_only=True)
text_embedding = scorer.get_text_embedding("self-attention layer")
```

| Component | Specification |
|-----------|---------------|
| **Model** | `google/siglip2-so400m-patch14-384` |
| **Image Size** | 384×384 |
| **Embedding** | 1152 dimensions |
| **Text length** | 64 tokens, lowercased |
| **Framework** | PyTorch (Hugging Face `transformers`) |
| **Query time** | about 60 ms per query on CPU |

## Benchmark

SigLIP 2 against CLIP ViT-L/14 on diagram retrieval. Every image in the corpus has one query written for it; a query counts as answered when its own image is retrieved. Top-5 is closer to what the app does than top-1, since the agent retrieves several candidates and chooses among them.

### Results

**Curated corpus** (620 queries over 620 images; both models embed the same lossless images):

| Model | Top-1 Accuracy | Top-5 Accuracy | MRR |
|-------|----------------|----------------|-----|
| **SigLIP 2** | **82.6%** (512/620) | **97.3%** (603/620) | 0.891 |
| CLIP ViT-L/14 | 45.0% (279/620) | 66.0% (409/620) | 0.552 |

**+37.6 percentage points** top-1, **+31.3** top-5. The benchmark measures pure query-image retrieval with the query as written; the app's extra phrasings and keyword matches are not used.

The LangSmith experiment on the same 620 queries (dataset `mentorml-curated-diagram-retrieval-v2`) gave SigLIP 82.6% and CLIP 45.2% top-1 (+37.4 pp). It was run before the benchmark was changed to give CLIP the lossless images instead of the shipped WebPs, which explains the one-query difference.

**Legacy set** (92 queries over 92 Jay Alammar diagrams):

| Model | Top-1 Accuracy | Top-5 Accuracy |
|-------|----------------|----------------|
| **SigLIP 2** (PyTorch, current) | **73.9%** (68/92) | **98.9%** (91/92) |
| SigLIP 2 (JAX, original implementation) | 76.1% (70/92) | not recorded |
| CLIP ViT-L/14 | 46.7% (43/92) | 85.9% (79/92) |

The PyTorch port produces text embeddings identical to the JAX original and image embeddings within 0.5% (cosine similarity 0.99 or higher); the two-query difference on the legacy set comes from how the two implementations resize images.

### Run Benchmark

```bash
pip install -r requirements-dev.txt

# Offline, self-contained
python benchmark/scripts/evaluate.py                   # curated corpus
python benchmark/scripts/evaluate.py --corpus legacy   # original 92 diagrams
```

### LangSmith Eval

The retrieval benchmark is also runnable via LangSmith for traced runs, a versioned dataset, and a side-by-side comparison UI. End-to-end agent quality (does retrieval find the expected diagram, does the answer cite it, plus an LLM judge on explanation quality) ships as a separate script.

```bash
# Requires LANGSMITH_API_KEY in .env

# One-time: upload benchmark queries as a LangSmith dataset
python benchmark/scripts/upload_to_langsmith.py

# Retrieval eval (SigLIP + CLIP as evaluate() targets; top-1 and top-5)
python benchmark/scripts/langsmith_evaluate_retrieval.py

# Full agent eval (LangGraph end-to-end)
python benchmark/scripts/langsmith_evaluate_agent.py --limit 5   # smoke test
python benchmark/scripts/langsmith_evaluate_agent.py             # full run
```

The curated corpus changes whenever it is rebuilt. After a rebuild, upload it under a new `--dataset-name` before evaluating.

## Directory Structure

```
agent/
├── graph.py                      # LangGraph agent (plan → retrieve → answer)
└── tools.py                      # MCP client: persistent stdio session, wrapped tools

mcp_server/
└── diagram_server.py             # MCP server: SigLIP text tower + retrieve_diagram tool

model/
├── scorer.py                     # SigLIP 2 scorer (PyTorch)
└── siglip2_text/                 # Exported text tower (gitignored)

corpus/                           # The corpus the app serves
├── images/                       # Accepted diagrams (WebP)
├── metadata.json                 # Descriptions, scores and sources
├── embeddings.npz                # SigLIP 2 image embeddings
├── candidates.json               # Every candidate with its rubric scores
└── coverage.json                 # Latest coverage audit

scripts/
├── corpus_sources.py             # Pages and papers to pull figures from
├── curate_corpus.py              # Fetch → score → build
├── audit_coverage.py             # Which topics have a diagram, which do not
├── coverage_topics.py            # Topic list for the audit
└── export_text_tower.py          # Save the text tower on its own

benchmark/
├── corpus/, queries/             # Legacy benchmark: 92 Jay Alammar diagrams
├── results/                      # Evaluation results
└── scripts/
    ├── common.py                    # Benchmark loading, SigLIP and CLIP rankers
    ├── evaluate.py                  # Offline SigLIP-vs-CLIP evaluation
    ├── upload_to_langsmith.py       # Upload dataset to LangSmith
    ├── langsmith_evaluate_retrieval.py  # Retrieval eval via LangSmith
    ├── langsmith_evaluate_agent.py  # End-to-end agent eval via LangSmith
    ├── scrape_ai_ml_diagrams.py     # Legacy: scrape the 92 diagrams
    └── generate_queries.py          # Legacy: write their queries

frontend/src/
├── components/
│   ├── Chat.tsx         # Main container
│   ├── ChatInput.tsx    # Input with send/stop/clear
│   ├── DiagramCard.tsx  # Diagram display with source link
│   ├── Message.tsx      # Message bubble + teaching plan
│   └── MessageList.tsx  # Message list + empty state
├── hooks/
│   └── useStreamChat.ts # SSE streaming hook
└── types.ts             # TypeScript interfaces
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key   # GEMINI_API_KEY also works
APP_PASSWORD=password_for_the_app
AUTH_SECRET=a_long_random_string     # Optional: signs login sessions (defaults to a key derived from the password and API key)
TRUST_PROXY=1                        # Behind a proxy (the Docker image sets it for Railway): count wrong passwords per visitor

# Optional - enables GET /misses (send it as the X-Admin-Token header)
ADMIN_TOKEN=a_long_random_string

# Optional
REDIS_URL=redis://localhost:6379     # Default
# Model chains: the first is used, the rest are fallbacks if a call errors
MENTORML_PLAN_MODELS=gemini-3.5-flash-lite,gemini-3.6-flash
MENTORML_ANSWER_MODELS=gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash
MENTORML_PLAN_THINKING=minimal       # Thinking level for the planning call
MENTORML_ANSWER_THINKING=low         # Thinking level for the answer (3.8 does not accept "minimal")

# Optional - LangSmith tracing & eval
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true            # Auto-trace live agent runs
LANGSMITH_PROJECT=mentorml-prod   # Default project: "default"
```

## Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- Redis with the search module (Redis Stack), for conversation memory
- ~5GB disk space for the SigLIP 2 checkpoint (about 3GB in the Docker image, which keeps only the text tower)
- Gemini API key
