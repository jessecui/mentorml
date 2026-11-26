"""
MentorML FastAPI Server

Provides:
- POST /chat: Chat with the LangGraph agent (requires thread_id for conversation state)
- POST /chat/stream: The same, streamed as Server-Sent Events
- GET /diagrams/{diagram_id}: Serve a diagram image
- GET /misses: Questions that got no diagram
- GET /: Health check
"""

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Before importing the agent: agent/graph.py reads its MENTORML_* settings at import
load_dotenv()

from agent.graph import DIAGRAM_REF, content_text, rendered_ids, replay_shown  # noqa: E402

# Python's built-in type table has no .webp; without /etc/mime.types, corpus images
# would be served as text/plain
mimetypes.add_type("image/webp", ".webp")

logger = logging.getLogger("mentorml.server")

# Paths
PROJECT_ROOT = Path(__file__).parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
CORPUS_IMAGES_DIR = CORPUS_DIR / "images"
CORPUS_METADATA = CORPUS_DIR / "metadata.json"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# index.html must always be re-fetched, or browsers keep loading an old frontend build after a rebuild.
NO_CACHE = {"Cache-Control": "no-cache"}


# Login session (same design as project-philo): an HttpOnly cookie holding an expiry
# time, signed so it cannot be forged or set from the browser. The frontend's own
# "mentor_auth" cookie is only a hint for whether to show the sign-in prompt.
SESSION_COOKIE = "mentor_session"
SESSION_TTL_S = 24 * 3600

# Redis list used as an append-only log. It survives redeploys, unlike the container's disk.
MISSES_KEY = "mentorml:diagram_misses"
LOG_MAX_ENTRIES = 5000

# Globals for context-manager cleanup
_async_redis_cm = None
_mcp_exit_stack: AsyncExitStack | None = None


def _load_corpus() -> dict[str, dict]:
    if not CORPUS_METADATA.exists():
        return {}
    return {item["id"]: item for item in json.loads(CORPUS_METADATA.read_text())}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Spawns the diagram MCP server (which owns the SigLIP text tower and the
    corpus embeddings) via a persistent stdio session, loads its tools,
    creates the async Redis checkpointer, and initializes the agent.
    """
    global _async_redis_cm, _mcp_exit_stack

    print("🚀 Starting MentorML server...")

    # Imported here, not at module level, so `import main` does not start the Redis
    # checkpointer or MCP client machinery (agent.graph itself is imported at the top)
    import redis.asyncio as aioredis
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    from agent.graph import create_agent
    from agent.tools import load_persistent_mcp_tools

    app.state.corpus = _load_corpus()
    print(f"🖼️  {len(app.state.corpus)} diagrams in corpus")

    print("🔌 Spawning diagram MCP server (stdio, persistent session)...")
    _mcp_exit_stack = AsyncExitStack()
    tools = await load_persistent_mcp_tools(_mcp_exit_stack)
    print(f"   ✅ Loaded {len(tools)} tool(s) from MCP server: {[t.name for t in tools]}")

    # Setup async Redis checkpointer with 1 day TTL
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print("🔗 Connecting to Redis...")
    ttl_config = {
        "default_ttl": 1440,     # 24 hours in minutes
        "refresh_on_read": True,  # Reset TTL when conversation is accessed
    }

    _async_redis_cm = AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_config)
    async_checkpointer = await _async_redis_cm.__aenter__()
    print("📝 Setting up Redis checkpoint indices...")
    await async_checkpointer.asetup()
    app.state.async_checkpointer = async_checkpointer
    # Short timeouts, so a stalled Redis cannot pile up background log writes
    app.state.redis = aioredis.from_url(
        redis_url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)

    print("🤖 Creating LangGraph agent...")
    agent = create_agent(tools, checkpointer=async_checkpointer, known_diagram_ids=set(app.state.corpus))
    app.state.agent = agent
    app.state.async_agent = agent

    print("✅ MentorML ready!")
    yield

    print("👋 Shutting down MentorML server...")
    if _background_tasks:
        # Let in-flight miss-log writes finish (briefly) before closing Redis
        await asyncio.wait(_background_tasks, timeout=3)
    await app.state.redis.aclose()
    if _async_redis_cm:
        await _async_redis_cm.__aexit__(None, None, None)
    if _mcp_exit_stack:
        await _mcp_exit_stack.aclose()


app = FastAPI(
    title="MentorML API",
    description="A multimodal AI/ML teaching assistant with diagram retrieval",
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None,      # Disable /docs
    redoc_url=None,     # Disable /redoc
    openapi_url=None,   # Disable /openapi.json
)

# Diagram images from the curated corpus. Public on purpose, like /diagrams/{id}: they
# are published figures shown with links to their sources, so there is nothing to protect.
if CORPUS_IMAGES_DIR.exists():
    app.mount("/corpus/images", StaticFiles(directory=str(CORPUS_IMAGES_DIR)), name="corpus-images")

# Serve frontend static files in production (after `npm run build`)
if FRONTEND_DIST.exists():
    # Serve static assets (JS, CSS, etc.)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")


# --- Request/Response Models ---

MAX_MESSAGE_CHARS = 4000


class ChatRequest(BaseModel):
    """Chat request with message and thread ID for conversation state."""
    # Every turn is replayed into later prompts, so messages are bounded
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, pattern=r"\S")  # not just whitespace
    # The thread ID is the only key to a conversation, so only the unguessable random
    # format is accepted. Older timestamp-based IDs could be guessed and are rejected on
    # purpose; a tab still running the old frontend needs one reload after this deploy
    # (it would fail anyway, since it also lacks the new session cookie).
    thread_id: str = Field(pattern=r"^thread-[0-9a-f]{32}$")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Explain the attention mechanism in transformers",
                    "thread_id": "thread-3f9a1c2e0b7d4e6f8a5c1b2d3e4f5a6b"
                }
            ]
        }
    }


class ValidatePasswordRequest(BaseModel):
    """Password validation request."""
    password: str = Field(max_length=200)


class DiagramReference(BaseModel):
    """A diagram shown in a response."""
    id: str
    image_url: str
    title: str
    description: str   # What the diagram shows, written at curation time
    teaches: str       # What a learner takes away from it
    source_name: str   # Site or publication it comes from
    source_title: str  # Page or paper title
    source_url: str
    score: float = 0.0  # Retrieval score for the query that found it
    query: str = ""     # The search that found it


class TeachingPlanResponse(BaseModel):
    """The agent's teaching plan."""
    topic: str
    steps: list[str]
    diagrams_needed: list[str]


class ChatResponse(BaseModel):
    """Chat response with agent's message and any retrieved diagrams."""
    response: str
    diagrams: list[DiagramReference] = []
    plan: TeachingPlanResponse | None = None


# --- Helpers ---

def _diagram_payload(diagram_id: str, found: dict | None = None) -> dict | None:
    """Everything the frontend needs to show a diagram, or None for an unknown ID."""
    item = app.state.corpus.get(diagram_id)
    if item is None:
        return None
    found = found or {}
    return DiagramReference(
        id=diagram_id,
        image_url=f"/corpus/images/{item['file']}",
        title=item.get("title", ""),
        description=item.get("description", ""),
        teaches=item.get("teaches", ""),
        source_name=item.get("source_name", ""),
        source_title=item.get("source_title", ""),
        source_url=item.get("source_url", ""),
        score=float(found.get("score", 0.0)),
        query=found.get("query", ""),
    ).model_dump()


def _candidates_from_tool_output(content) -> dict[str, dict]:
    """Map candidate ID -> {score, query} from one retrieve_diagram result."""
    try:
        result = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(result, dict):
        return {}
    return {
        c["id"]: {"score": c.get("score", 0.0), "query": result.get("query", "")}
        for c in result.get("candidates", [])
        if isinstance(c, dict) and "id" in c
    }


def _shown_ids(messages) -> set[str]:
    """Diagram IDs the user was actually shown in a stretch of conversation."""
    return replay_shown(messages, set(app.state.corpus))[0]


def _referenced_diagrams(text: str, candidates: dict[str, dict], earlier_ids: set[str]) -> list[dict]:
    """
    Diagrams the answer shows, in order of appearance (see agent.graph.rendered_ids):
    only IDs this turn's searches returned, or IDs already shown earlier in the
    conversation, at most MAX_DIAGRAMS_PER_ANSWER. Anything else was never offered to
    the model and is not shown.
    """
    ids = rendered_ids(text, set(candidates) | earlier_ids, set(app.state.corpus))
    return [p for p in (_diagram_payload(i, candidates.get(i)) for i in ids) if p]


async def _append_log(key: str, entry: dict) -> None:
    """Append to a capped Redis list. Logging must never break a response."""
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    try:
        redis = app.state.redis
        await redis.rpush(key, json.dumps(entry, ensure_ascii=False))
        await redis.ltrim(key, -LOG_MAX_ENTRIES, -1)
    except Exception as exc:
        print(f"⚠️  Could not write to {key}: {exc!r}")


# Background log writes; a reference is kept so a task is not collected before it finishes
_background_tasks: set[asyncio.Task] = set()


def _log_miss_in_background(*args) -> None:
    """Write the miss log without making the client wait for Redis."""
    task = asyncio.create_task(_log_miss_if_any(*args))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


KEEPALIVE = object()
KEEPALIVE_INTERVAL_S = 15.0


async def _with_keepalive(events, interval: float = KEEPALIVE_INTERVAL_S):
    """
    Pass events through, yielding KEEPALIVE whenever none has arrived for `interval`
    seconds, so the stream can send a comment line and proxies do not close an idle
    connection while a model is thinking or a fallback is starting.
    """
    queue: asyncio.Queue = asyncio.Queue()
    finished = object()

    async def pump():
        try:
            async for event in events:
                await queue.put(event)
            await queue.put(finished)
        except BaseException as exc:  # handed to the consumer, whatever kind it is
            queue.put_nowait(exc)
            if not isinstance(exc, Exception):
                raise  # let cancellation and exits propagate in this task

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield KEEPALIVE
                continue
            if item is finished:
                return
            if isinstance(item, Exception):
                raise item
            if isinstance(item, BaseException):
                # Cancellation or an exception group from a subprocess session: surface it as
                # an ordinary error so the client gets an error event instead of a hang
                raise RuntimeError(f"agent stream stopped: {type(item).__name__}") from item
            yield item
    finally:
        task.cancel()  # the client went away, or the stream ended
        # Let the run unwind before returning; asyncio.wait never raises the task's
        # CancelledError, so an outer cancellation of this stream is not swallowed
        await asyncio.wait([task], timeout=5)


async def _log_miss_if_any(request: ChatRequest, plan, diagrams: list[dict], candidates: dict[str, dict]) -> None:
    """Record questions where a diagram was wanted but none was good enough to show."""
    # A fallback plan searches on the raw message (even "thanks"), so its misses mean nothing.
    # No candidates at all means the search itself failed, which is an outage, not a gap.
    if (diagrams or plan is None or not plan.diagrams_needed or getattr(plan, "is_fallback", False)
            or not candidates):
        return
    # Scores are not comparable across searches, so keep the best few of each search
    by_search: dict[str, list[tuple[str, float]]] = {}
    for cid, info in candidates.items():
        by_search.setdefault(info["query"], []).append((cid, info["score"]))
    await _append_log(MISSES_KEY, {
        "question": request.message[:500],
        "topic": plan.topic,
        "searches": [
            {"query": query, "best_candidates": [
                {"id": cid, "score": score} for cid, score in sorted(found, key=lambda x: -x[1])[:3]]}
            for query, found in by_search.items()
        ],
    })


# --- Auth Dependency ---

def _session_secret() -> bytes:
    """
    Key for signing session cookies. AUTH_SECRET if set; otherwise derived from the
    server's own secrets (the password and the Gemini API key) so sessions survive
    restarts and the key cannot be guessed; otherwise random (sessions then reset when
    the server restarts).
    """
    explicit = os.getenv("AUTH_SECRET")
    if explicit:
        return explicit.encode()
    password = os.getenv("APP_PASSWORD", "")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    if password and api_key:
        return hashlib.sha256(f"mentorml-session|{password}|{api_key}".encode()).digest()
    return secrets.token_bytes(32)


SESSION_SECRET = _session_secret()
_SESSION_TOKEN = re.compile(r"(\d{1,12})\.([0-9a-f]{64})")


def _sign(expires_at: int) -> str:
    return hmac.new(SESSION_SECRET, str(expires_at).encode(), hashlib.sha256).hexdigest()


def _new_session_token() -> str:
    expires_at = int(time.time()) + SESSION_TTL_S
    return f"{expires_at}.{_sign(expires_at)}"


def _session_is_valid(token: str | None) -> bool:
    match = _SESSION_TOKEN.fullmatch(token or "")
    if not match:
        return False
    expires_at = int(match.group(1))
    if expires_at < time.time():
        return False
    return hmac.compare_digest(match.group(2).encode(), _sign(expires_at).encode())


async def require_auth(mentor_session: str | None = Cookie(default=None)):
    """Require a valid session cookie (issued by /validate-password)."""
    if not _session_is_valid(mentor_session):
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="Chat with MentorML", dependencies=[Depends(require_auth)])
async def chat(request: ChatRequest):
    """
    Send a message to the MentorML agent.

    The agent can retrieve relevant diagrams and provide explanations.
    Use the same thread_id across requests to maintain conversation context.
    """
    from langchain_core.messages import HumanMessage

    agent = app.state.agent

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        messages = result.get("messages", [])

        # The answer is the last AI message with text (not a bare tool call)
        response_text = ""
        for msg in reversed(messages):
            if msg.type == "human":
                break  # only this turn's messages; never fall back to an earlier answer
            if msg.type == "ai" and not getattr(msg, "tool_calls", None):
                text = content_text(msg.content)
                if text.strip():
                    response_text = text
                    break
        if not response_text:
            return ChatResponse(response="No response generated.", diagrams=[])

        # Candidates retrieved during this turn (everything after the last human message)
        last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
        candidates: dict[str, dict] = {}
        for msg in messages[last_human:]:
            if msg.type == "tool" and msg.name == "retrieve_diagram":
                for diagram_id, info in _candidates_from_tool_output(msg.content).items():
                    candidates.setdefault(diagram_id, info)  # first search wins, as in the stream

        diagrams = _referenced_diagrams(response_text, candidates, _shown_ids(messages[:last_human]))

        plan = result.get("plan")
        plan_response = None
        if plan:
            plan_response = TeachingPlanResponse(
                topic=plan.topic, steps=plan.steps, diagrams_needed=plan.diagrams_needed
            )
        _log_miss_in_background(request, plan, diagrams, candidates)

        return ChatResponse(
            response=response_text,
            diagrams=[DiagramReference(**d) for d in diagrams],
            plan=plan_response,
        )

    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="The answer could not be generated. Please try again.") from e


# --- Streaming Endpoint ---

@app.post("/chat/stream", summary="Stream chat with MentorML (SSE)", dependencies=[Depends(require_auth)])
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response using Server-Sent Events.

    Events emitted:
    - plan: The agent's teaching plan
    - diagram: A candidate diagram with metadata (shown once the text cites it)
    - token: A text token from the response
    - done: Final signal with the diagrams the answer actually cites
    - error: Any error that occurred
    """
    from fastapi.responses import StreamingResponse
    from langchain_core.messages import HumanMessage

    agent = app.state.async_agent

    async def event_generator():
        """Generate SSE events from agent stream."""
        candidates: dict[str, dict] = {}
        text_buffer = ""
        plan = None
        config = {"configurable": {"thread_id": request.thread_id}}
        sent_earlier: set[str] = set()
        events = None

        try:
            # Diagrams already shown in this conversation may be cited again
            snapshot = await agent.aget_state(config)
            earlier_ids = _shown_ids((snapshot.values or {}).get("messages", []))

            events = _with_keepalive(agent.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                version="v2",
            ))
            async for event in events:
                if event is KEEPALIVE:
                    yield ": keep-alive\n\n"  # an SSE comment; clients ignore it
                    continue
                event_type = event.get("event")
                node = event.get("metadata", {}).get("langgraph_node", "")

                # Plan complete - emit teaching plan
                if event_type == "on_chain_end" and plan is None and event.get("name") == "plan":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("plan"):
                        plan = output["plan"]
                        plan_data = {
                            "topic": plan.topic,
                            "steps": plan.steps,
                            "diagrams_needed": plan.diagrams_needed,
                        }
                        yield f"event: plan\ndata: {json.dumps(plan_data)}\n\n"

                # Retrieval finished - send every candidate so the one the text cites can render at once
                elif event_type == "on_tool_end" and event.get("name") == "retrieve_diagram":
                    tool_output = event.get("data", {}).get("output")
                    found = _candidates_from_tool_output(getattr(tool_output, "content", tool_output))
                    for diagram_id, info in found.items():
                        if diagram_id in candidates:
                            continue
                        candidates[diagram_id] = info
                        payload = _diagram_payload(diagram_id, info)
                        if payload:
                            yield f"event: diagram\ndata: {json.dumps(payload)}\n\n"

                # A new answer generation (a fallback model, or the second call after an
                # extra search) replaces whatever the previous one streamed
                elif event_type == "on_chat_model_start" and node == "answer":
                    if text_buffer:
                        text_buffer = ""
                        yield "event: reset\ndata: {}\n\n"

                # Answer tokens
                elif event_type == "on_chat_model_stream" and node == "answer":
                    chunk = event.get("data", {}).get("chunk")
                    text = content_text(chunk.content) if chunk is not None else ""
                    if text:
                        text_buffer += text
                        yield f"event: token\ndata: {json.dumps(text)}\n\n"
                        # A diagram from an earlier turn is not among this turn's search
                        # results; send it as soon as the text cites it
                        # Only the new text and a short overlap: a reference can be split
                        # across chunks, and rescanning everything would be quadratic
                        scan_from = max(0, len(text_buffer) - len(text) - 64)
                        for diagram_id in DIAGRAM_REF.findall(text_buffer, scan_from):
                            if (diagram_id in earlier_ids and diagram_id not in candidates
                                    and diagram_id not in sent_earlier):
                                payload = _diagram_payload(diagram_id)
                                if payload:
                                    sent_earlier.add(diagram_id)
                                    yield f"event: diagram\ndata: {json.dumps(payload)}\n\n"

            if not text_buffer.strip():
                yield f"event: error\ndata: {json.dumps({'error': 'No response generated.'})}\n\n"
                return

            final_diagrams = _referenced_diagrams(text_buffer, candidates, earlier_ids)
            # Started before "done" (a client that disconnects on "done" closes this
            # generator) but not awaited, so a slow Redis cannot hold the answer open
            _log_miss_in_background(request, plan, final_diagrams, candidates)
            yield f"event: done\ndata: {json.dumps({'diagrams': final_diagrams})}\n\n"

        except Exception:
            logger.exception("Streamed chat request failed")
            yield f"event: error\ndata: {json.dumps({'error': 'The answer could not be generated. Please try again.'})}\n\n"
        finally:
            # Close the wrapper explicitly: when the client disconnects this generator is
            # closed, and the agent run must stop now rather than at garbage collection
            if events is not None:
                await events.aclose()

    streaming = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
    return streaming


@app.get("/diagrams/{diagram_id}", summary="Get a diagram image")
async def get_diagram(diagram_id: str):
    """Retrieve a diagram image by ID (e.g. "fig_3fa9c1d2e0")."""
    item = app.state.corpus.get(diagram_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Diagram not found: {diagram_id}")
    filepath = CORPUS_IMAGES_DIR / item["file"]
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagram file missing: {diagram_id}")
    return FileResponse(filepath)


@app.get("/misses", summary="Questions that got no diagram")
async def get_misses(x_admin_token: str | None = Header(default=None)):
    """
    Questions where a diagram was wanted but nothing in the corpus was good enough to show.

    These are other people's questions, so the app's shared login is not enough: the
    request must carry an X-Admin-Token header matching the ADMIN_TOKEN env var. With
    ADMIN_TOKEN unset the endpoint is off.
    """
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token or not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), admin_token.encode()):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        entries = await app.state.redis.lrange(MISSES_KEY, 0, -1)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Miss log unavailable: {type(exc).__name__}") from exc
    misses = []
    for entry in entries:
        try:
            misses.append(json.loads(entry))
        except json.JSONDecodeError:
            continue  # skip a corrupt entry rather than fail the whole listing
    return {"misses": misses}


# The password is a name, and names are easy to guess, so wrong answers are limited
# per client address (as in project-philo).
MAX_FAILED_ATTEMPTS = 10
FAILED_ATTEMPT_WINDOW_S = 15 * 60
# Addresses tracked before stale entries are swept, so many one-off addresses
# cannot grow the table without bound
MAX_TRACKED_CLIENTS = 10_000
_failed_attempts: dict[str, list[float]] = {}


def _client_address(http_request: Request) -> str:
    """
    The visitor's address: the socket address, unless TRUST_PROXY is set. Behind Railway's
    proxy every request comes from the proxy, which adds the visitor's address as the LAST
    X-Forwarded-For entry (a client can only forge entries before it), so set TRUST_PROXY=1
    there. Off by default, because a client reaching the app directly could forge the header;
    if it is forgotten behind a proxy, visitors share one lockout rather than escape it.
    """
    forwarded = http_request.headers.get("x-forwarded-for", "")
    if forwarded and os.getenv("TRUST_PROXY", "").strip().lower() in ("1", "true", "yes"):
        return forwarded.split(",")[-1].strip()
    return http_request.client.host if http_request.client else "unknown"


def _recent_failures(client: str, now: float) -> list[float]:
    if len(_failed_attempts) > MAX_TRACKED_CLIENTS:
        for address in list(_failed_attempts):
            _failed_attempts[address] = [t for t in _failed_attempts[address] if now - t < FAILED_ATTEMPT_WINDOW_S]
            if not _failed_attempts[address]:
                del _failed_attempts[address]
        # Still full of recent failures: forget the oldest addresses first
        while len(_failed_attempts) > MAX_TRACKED_CLIENTS:
            del _failed_attempts[next(iter(_failed_attempts))]
    # Re-inserted rather than updated in place, so the dict stays ordered by each
    # address's latest attempt and the sweep above drops the stalest first
    recent = [t for t in _failed_attempts.pop(client, []) if now - t < FAILED_ATTEMPT_WINDOW_S]
    if recent:
        _failed_attempts[client] = recent
    return recent


@app.post("/validate-password", summary="Validate app password")
async def validate_password(request: ValidatePasswordRequest, http_request: Request, response: Response):
    """
    Validate the password and, if it matches, issue the signed session cookie that
    protected endpoints check.

    Password is stored in APP_PASSWORD environment variable.
    Comparison is case-insensitive with all whitespace removed.
    Returns {"valid": true/false}; 429 after MAX_FAILED_ATTEMPTS wrong answers from
    one address within FAILED_ATTEMPT_WINDOW_S.
    """
    app_password = "".join(os.getenv("APP_PASSWORD", "").split()).lower()
    user_password = "".join(request.password.split()).lower()

    if not app_password:
        raise HTTPException(
            status_code=500,
            detail="APP_PASSWORD environment variable not configured"
        )

    client = _client_address(http_request)
    now = time.time()
    if len(_recent_failures(client, now)) >= MAX_FAILED_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")

    valid = hmac.compare_digest(user_password.encode(), app_password.encode())
    if not valid:
        # Appended at the end of the dict's order (newest failure last)
        _failed_attempts[client] = _failed_attempts.pop(client, []) + [now]
    if valid:
        # A successful sign-in clears this visitor's earlier typos
        _failed_attempts.pop(client, None)
        forwarded_proto = http_request.headers.get("x-forwarded-proto", "")
        response.set_cookie(
            SESSION_COOKIE,
            _new_session_token(),
            max_age=SESSION_TTL_S,
            httponly=True,
            samesite="lax",
            secure=http_request.url.scheme == "https" or forwarded_proto == "https",
        )
    return {"valid": valid}


@app.get("/", summary="Health check / Frontend")
async def root():
    """Serve frontend index.html in production, health check in dev."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers=NO_CACHE)
    # Fallback: health check for dev/API-only mode
    return {
        "status": "healthy",
        "service": "MentorML API",
        "version": "3.0.0",
    }


# Catch-all for SPA routing (must be last)
@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Serve files at the root of the frontend build (favicon and the like); otherwise
    index.html for any unmatched route (SPA client-side routing)."""
    try:
        static_file = (FRONTEND_DIST / path).resolve()
        is_static = bool(path) and static_file.is_file() and static_file.is_relative_to(FRONTEND_DIST.resolve())
    except (ValueError, OSError):  # NUL bytes, over-long names
        is_static = False
    if is_static:
        return FileResponse(static_file, headers=NO_CACHE if static_file.name == "index.html" else None)
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers=NO_CACHE)
    raise HTTPException(status_code=404, detail="Not found")


# --- Local Development ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
