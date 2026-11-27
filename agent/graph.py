"""
LangGraph agent for MentorML.

A Plan-and-Execute agent:

1. PLAN     One structured call turns the learner's message into a teaching
            plan: the topic, the steps to cover, and the diagrams to look for.
            The same call decides whether the message is a follow-up.
2. RETRIEVE The plan's diagram searches go straight to the `retrieve_diagram`
            MCP tool, in parallel. Each returns several candidates with
            descriptions.
3. ANSWER   One streamed call writes the explanation and picks the candidate
            that adds most to it (or none, if nothing fits). It may run one
            more search of its own if the candidates miss the point.

Flow: plan → tools → answer (⇄ tools, at most once) → end

Conversation state is checkpointed (Redis in production).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
import weakref
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator

# Each step has a model chain: the first is used, the rest are fallbacks tried in
# order if a call errors (model retired, overloaded, rate limited).
#
# Planning is a short structured call where speed matters most, so it runs on a
# small fast model. The answer is what the learner reads, so it gets the strongest.
def _model_chain(env_var: str, default: str) -> list[str]:
    """Comma-separated model names from env_var; the default when it is unset or empty."""
    return [m.strip() for m in (os.getenv(env_var) or default).split(",") if m.strip()] or default.split(",")


PLAN_MODELS = _model_chain("MENTORML_PLAN_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash")
ANSWER_MODELS = _model_chain("MENTORML_ANSWER_MODELS", "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash")
# Thinking levels. Note gemini-3.8-flash does not accept "minimal".
PLAN_THINKING = os.getenv("MENTORML_PLAN_THINKING", "minimal")
ANSWER_THINKING = os.getenv("MENTORML_ANSWER_THINKING", "low")

# The planning call usually takes 1-2 seconds but occasionally stalls. Each plan
# model gets PLAN_MODEL_TIMEOUT_S before the next one in the chain is tried; past
# PLAN_TIMEOUT_S in total the agent stops waiting and searches on the question itself.
# Both are enforced in the app: Gemini rejects request deadlines under 10 seconds.
# By default the total allows every plan model its full share, plus a second of slack.
PLAN_MODEL_TIMEOUT_S = float(os.getenv("MENTORML_PLAN_MODEL_TIMEOUT_S") or 4)
PLAN_TIMEOUT_S = float(os.getenv("MENTORML_PLAN_TIMEOUT_S") or len(PLAN_MODELS) * PLAN_MODEL_TIMEOUT_S + 1)

logger = logging.getLogger("mentorml.agent")

# Answer models get one attempt each (no SDK retry backoff) and a total deadline,
# enforced in the app, so an error or a stall hands over to the next model in the
# chain. The deadline covers the whole streamed answer (typically about 10 seconds),
# so it sits far above the longest real answer and only catches a model that has hung.
ANSWER_MODEL_TIMEOUT_S = float(os.getenv("MENTORML_ANSWER_MODEL_TIMEOUT_S") or 180)

# How an answer cites a diagram. The server and the eval import this, so all agree.
DIAGRAM_REF = re.compile(r"\[diagram:\s*([A-Za-z0-9_-]+)\]")

RETRIEVE_TOOL = "retrieve_diagram"
MAX_DIAGRAM_QUERIES = 2
MAX_TERMS = 10
MAX_FOLLOWUPS = 3
# Earlier question-and-answer pairs sent to the answer model along with the new question
MAX_HISTORY_EXCHANGES = 8
# Tool calls issued by the planner carry this prefix so they can be told apart
# from searches the answering model decides to run itself.
PLAN_CALL_PREFIX = "plan_"


# --- State Schema ---

class TeachingPlan(BaseModel):
    """Structured teaching plan produced by the planner."""
    topic: str = Field(description="The main topic being taught")
    steps: list[str] = Field(description="Ordered list of teaching steps")
    diagrams_needed: list[str] = Field(default_factory=list, description="Diagram searches to run")
    is_followup: bool = Field(default=False, description="Whether this continues the previous topic")
    is_fallback: bool = Field(default=False, description="Planning failed; this plan just searches on the question")
    terms_to_explain: list[str] = Field(default_factory=list, description="Terms the answer must explain in plain words")
    followups: list[str] = Field(default_factory=list, description="Questions the learner might ask next")


class AgentState(BaseModel):
    """State for the Plan-and-Execute teaching agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    plan: TeachingPlan | None = None

    @field_validator("plan", mode="before")
    @classmethod
    def _plan_from_checkpoint(cls, value):
        # Newer checkpoint libraries hand a saved plan back in its serialised form
        # ({"type": "constructor", "kwargs": {...}}) instead of rebuilding the TeachingPlan.
        if isinstance(value, dict) and value.get("type") == "constructor" and isinstance(value.get("kwargs"), dict):
            return value["kwargs"]
        return value

    class Config:
        arbitrary_types_allowed = True


class PlanOutput(BaseModel):
    """What the planner model returns."""
    is_followup: bool = Field(
        description="True if the message continues the topic of the conversation so far; false for a new topic or a first message."
    )
    topic: str = Field(description="The topic to teach, in a few words.")
    steps: list[str] = Field(description="Ordered teaching steps, each one short sentence.")
    diagram_queries: list[str] = Field(
        description="Searches for diagrams that would help, most important first. Each names a concept and its key parts."
    )
    terms_to_explain: list[str] = Field(
        description="Technical terms and acronyms the answer will need that this reader may not know."
    )
    followups: list[str] = Field(
        description="2-3 short questions the learner might ask next, in the learner's own voice."
    )


# --- Prompts ---

# Who the answers are written for. Shared by the planner and the answer prompt.
READER = (
    "The reader is one person: smart, motivated, comfortable with basic programming, and knows roughly what a "
    "neural network is (layers, weights, training by adjusting the weights to reduce error). Assume nothing "
    "beyond that: no reinforcement learning, statistics, probability or linear algebra vocabulary, and no "
    "familiarity with research acronyms. They would much rather have something briefly explained that they "
    "already knew than be lost halfway through. They can ask follow-up questions, so one answer does not have "
    "to cover everything."
)

PLANNER_PROMPT = """You are the planner for MentorML, a teaching assistant for AI and machine learning that explains concepts with diagrams.

""" + READER + """

Turn the learner's latest message into a teaching plan.

steps: the points to cover, in order. Use 2-4 steps for a simple question and up to 6 for a broad topic. For a follow-up, use 1-3 steps that answer just what was asked.
- The first step is always the everyday intuition: an analogy or plain picture of what is going on and why it matters. For a comparison, plan one analogy that can carry every item.
- Later steps cover how it works, in plain words.
- Do not plan a step for formulas or derivations unless the learner asked for the maths.

terms_to_explain: the technical terms and acronyms the answer will need that this reader may not know, up to 10. Include ones that feel basic to an expert ("policy", "reward", "gradient", "loss", "sampling"). The answer must explain each in plain words when it first appears.

followups: 2-3 short questions (under 12 words each) the learner might want to ask next, written in the learner's own voice and plain words. They should go beyond what this answer will cover: a concrete example, a comparison, why something works, or the maths behind one part. Give none for greetings, thanks or off-topic messages.

diagram_queries: searches to run against a curated library of ML teaching diagrams (architectures, training procedures, attention, reinforcement learning, diffusion, agents and more). The answer should normally include a diagram, so:
- Give 1 search for most questions, and 2 only when the answer covers two distinct things worth seeing.
- Write each search as the concept's usual name, then a colon, then the key parts the picture should show, 6-15 words in total. For example "scaled dot-product attention: queries, keys, values, softmax, weighted sum" or "actor-critic: actor picks action, critic estimates value, environment returns reward". Keep the part before the colon short and standard ("dropout", "LSTM cell", "LoRA"), because it is also searched on its own. These searches use the standard technical names, unlike the answer.
- For a follow-up, search for what the follow-up is about, not the earlier topic again.
- Give no searches only when there is nothing to draw: greetings, thanks, or questions that are not about a technical concept.

is_followup: true when the message continues the conversation's current topic."""

ANSWER_PROMPT = """You are MentorML, a teaching assistant that explains AI and machine learning concepts with diagrams.

THE READER
""" + READER + """

HOW TO EXPLAIN
1. Intuition first, and stay with it. Open with an everyday picture or analogy for what is going on and why it matters, told in a few full sentences of prose (not a bullet list) before any technical name appears. Then keep using it: begin each part of the answer by saying what happens in the analogy, and only then say what that corresponds to in the real system. When comparing several things, carry ONE analogy through all of them, so each difference shows up as a change to the same picture.
2. Meaning first, name second. Say the idea in plain words, then attach the technical name to it. Never use a technical term, acronym or symbol the reader has not already been given in plain words, including terms that feel basic to an expert (policy, reward, gradient, loss, distribution, sampling, baseline). The first time an acronym appears, spell it out and say what it means. If a term is not worth explaining, leave it out and say the plain thing.
3. Everyday words. Write "parameter count", not "parameter footprint"; "extra", not "auxiliary"; "the usual reinforcement learning setup", not "reinforcement learning machinery".
4. No symbols in prose. Write "the winning response", not a variable for it. Leave formulas out by default and describe what the calculation does in words. Include a formula only when the learner asked for the maths, and then explain every symbol in words right beside it.
5. Understanding over coverage. Explain the core idea of each thing well instead of listing every component. Leave derivations, implementation details and edge cases for follow-up questions.
6. Tables, lists and diagram walkthroughs follow the same rules: plain words in every cell and bullet.

EXAMPLES of rules 2-4. Copy the pattern, not the topics.
Instead of: "A frozen copy of the original model used to calculate a Kullback-Leibler (KL) penalty."
Write: "A frozen copy of the original model. The model being trained is penalised for drifting too far from how this copy behaves, which stops it degenerating into gibberish. That penalty is known as a KL penalty."
Instead of: "It then updates the actor using a clipped surrogate objective."
Write: "It then limits how large each update is allowed to be, so one lucky or unlucky example cannot throw the model off course. That limit is called a clipped surrogate objective."
Instead of: "a prompt x, a winning response y_w and a losing response y_l. The loss increases the likelihood of y_w while decreasing the likelihood of y_l."
Write: "a prompt, a response people preferred and one they did not. Training nudges the model to make the preferred response more likely and the other one less likely."
Instead of: "There is no dynamic rollout generation during training."
Write: "The model writes no new responses during training. It learns only from examples collected beforehand."
Instead of: "Softmax converts the logits into a distribution."
Write: "The raw scores are then squashed into percentages that add up to 100%, so each can be read as how likely that option is. This step is called softmax."

ENDING
Stop when the explanation is done. Do not end with a question, an offer to explain more, or a recap such as "in conclusion". Suggested follow-up questions are shown to the reader separately.

DIAGRAMS
Candidate diagrams have already been retrieved for you and are listed below with a description of what each one shows. You cannot see the images, so rely on those descriptions.
- First decide which candidates DIRECTLY depict a concept you are explaining: the candidate's title or description must name that concept or its parts. A candidate about a different concept does not qualify, however loosely related.
- If one qualifies, include the best one. Include a second only if it shows something different and equally useful. The whole answer must contain at most two [diagram: ...] references. Among qualifying candidates, prefer higher clarity.
- If none qualifies, include no diagram and do not mention that one is missing. A missing diagram is better than an unrelated one. Never use a diagram as an analogy or a "conceptual bridge" to a different topic, and never write things like "while this diagram is about X".
- Cite only IDs from the candidate list, or IDs you used earlier in this conversation. Never invent an ID.
- When you walk the reader through a diagram, stick to what its description says and do not invent visual details. The descriptions are written in technical language: restate them in the reader's plain words, explaining any term or acronym the diagram uses.
{tool_rule}
FORMAT
Reference a diagram as [diagram: ID] using the exact ID. Put the reference on its own line with a blank line before and after, then explain what it shows AFTER it. Never refer to a diagram inline.

CORRECT:
Here's how attention works:

[diagram: fig_3fa9c1d2e0]

The diagram above shows...

WRONG:
As you see in [diagram: fig_3fa9c1d2e0], it shows the attention mechanism...
"""

TOOL_RULE_ALLOWED = (
    "- Only if no candidate fits AND a differently worded search would clearly find a better picture, "
    "you may call retrieve_diagram once. Otherwise do not call it; just write the answer.\n"
)
TOOL_RULE_USED = ("- You already ran an extra search; any new candidates it found are included in the list "
                  "below. Do not search again, and use only the candidates listed.\n")


# --- Helpers ---

# Model clients, reused across requests. Keyed by event loop because an async client
# is tied to the loop it first ran on (scripts that call asyncio.run twice get a new one).
_model_cache: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict]" = weakref.WeakKeyDictionary()


def _chat_models(models: list[str], thinking_level: str, **kwargs) -> list[ChatGoogleGenerativeAI]:
    """Chat models for a chain, built once per configuration (and event loop) and reused."""
    def build() -> list[ChatGoogleGenerativeAI]:
        return [ChatGoogleGenerativeAI(model=m, temperature=1.0, thinking_level=thinking_level, **kwargs)
                for m in models]

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return build()  # not inside a loop: nothing to reuse against
    per_loop = _model_cache.setdefault(loop, {})
    key = (tuple(models), thinking_level, tuple(sorted(kwargs.items())))
    if key not in per_loop:
        per_loop[key] = build()
    return list(per_loop[key])


def message_text(message: BaseMessage) -> str:
    """Plain text of a message. Gemini 3 returns content as a list of blocks."""
    return content_text(message.content)


def content_text(content) -> str:
    """Plain text of message content, whether a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def _last_human_index(messages: list[BaseMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return -1


def _is_plan_call(message: BaseMessage) -> bool:
    calls = getattr(message, "tool_calls", None) or []
    return bool(calls) and all(str(c.get("id", "")).startswith(PLAN_CALL_PREFIX) for c in calls)


def _conversation_digest(messages: list[BaseMessage], upto: int, max_turns: int = 2, max_chars: int = 400) -> str:
    """The last few exchanges before the current message, trimmed, for the planner."""
    lines: list[str] = []
    for m in messages[:upto]:
        if isinstance(m, HumanMessage):
            lines.append(f"Learner: {message_text(m)[:max_chars]}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            text = message_text(m).strip()
            if text:
                lines.append(f"MentorML: {text[:max_chars]}{'...' if len(text) > max_chars else ''}")
    return "\n".join(lines[-2 * max_turns:])


def _format_candidates(tool_messages: list[ToolMessage]) -> str:
    """Render retrieval results as a text block the answering model can choose from."""
    blocks: list[str] = []
    seen: set[str] = set()
    for tm in tool_messages:
        try:
            result = json.loads(tm.content) if isinstance(tm.content, str) else tm.content
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        lines = [f'Search: "{result.get("query", "")}"']
        for c in result.get("candidates", []):
            if not isinstance(c, dict) or "id" not in c or c["id"] in seen:
                continue
            seen.add(c["id"])
            lines.append(
                f'- ID {c["id"]} | {c.get("title", "")} | clarity {c.get("clarity", "?")}/5\n'
                f'  Shows: {c.get("description", "")}\n'
                f'  Teaches: {c.get("teaches", "")}'
            )
        if len(lines) > 1:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no candidates were retrieved)"


# --- Which diagrams render (shared with main.py, so the agent and the server agree) ---

MAX_DIAGRAMS_PER_ANSWER = 2


def search_result_ids(content) -> list[str]:
    """Candidate IDs from one retrieve_diagram result, in rank order."""
    try:
        result = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(result, dict):
        return []
    return [c["id"] for c in result.get("candidates", []) if isinstance(c, dict) and "id" in c]


def rendered_ids(text: str, offered: set[str], known_ids: set[str] | None) -> list[str]:
    """
    The diagrams an answer actually shows: its references to offered IDs (this turn's
    search results, or diagrams shown earlier), each once, in order, at most
    MAX_DIAGRAMS_PER_ANSWER, skipping IDs not in the corpus when known_ids is given.
    """
    shown: list[str] = []
    for diagram_id in DIAGRAM_REF.findall(text):
        if diagram_id in shown or diagram_id not in offered:
            continue
        if known_ids is not None and diagram_id not in known_ids:
            continue
        shown.append(diagram_id)
        if len(shown) == MAX_DIAGRAMS_PER_ANSWER:
            break
    return shown


def replay_shown(messages: list[BaseMessage], known_ids: set[str] | None) -> tuple[set[str], dict[int, list[str]]]:
    """
    Replay a conversation through rendered_ids: the set of diagram IDs the user was
    shown, and for each final answer (by index) the IDs it showed. A message that also
    made a tool call is skipped, since the answer after it replaced it.
    """
    shown: set[str] = set()
    per_message: dict[int, list[str]] = {}
    turn_results: set[str] = set()
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            turn_results = set()
        elif isinstance(msg, ToolMessage) and getattr(msg, "name", "") == RETRIEVE_TOOL:
            turn_results.update(search_result_ids(msg.content))
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            ids = rendered_ids(message_text(msg), turn_results | shown, known_ids)
            per_message[i] = ids
            shown.update(ids)
    return shown, per_message


# --- Node Functions ---

async def _plan_with_fallbacks(messages: list[BaseMessage]):
    """
    Run the planning call on each plan model in turn until one answers in time.

    Each model gets one attempt (in this library max_retries=1 means a single try;
    0 means "use the SDK default" of several) and PLAN_MODEL_TIMEOUT_S, so a model
    that errors or stalls hands over to the next instead of using up the budget.
    """
    last_error: Exception | None = None
    for llm in _chat_models(PLAN_MODELS, PLAN_THINKING, max_retries=1):
        planner = llm.with_structured_output(PlanOutput, method="json_schema")
        try:
            raw = await asyncio.wait_for(planner.ainvoke(messages), timeout=PLAN_MODEL_TIMEOUT_S)
            # An empty or unparseable plan is a failure too (None fails validation)
            return raw if isinstance(raw, PlanOutput) else PlanOutput.model_validate(raw)
        except Exception as exc:  # includes TimeoutError and validation errors
            logger.warning("Plan model %s failed (%s: %.300s); trying the next one", llm.model, type(exc).__name__, exc)
            last_error = exc
    raise last_error or RuntimeError("no plan models configured")


async def plan_node(state: AgentState) -> dict:
    """
    Decompose the learner's message into a teaching plan and issue its diagram
    searches as tool calls. One model call, whether or not it is a follow-up.
    """
    idx = _last_human_index(state.messages)
    if idx < 0:
        return {}
    question = message_text(state.messages[idx])

    context = ""
    digest = _conversation_digest(state.messages, idx)
    if digest:
        previous_topic = state.plan.topic if state.plan else "(unknown)"
        context = f"Current topic of the conversation: {previous_topic}\n\nConversation so far:\n{digest}\n\n"

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"{context}Learner's latest message: {question}"),
    ]
    try:
        output = await asyncio.wait_for(_plan_with_fallbacks(messages), timeout=PLAN_TIMEOUT_S)
        plan = TeachingPlan(
            topic=output.topic.strip() or question[:60],
            steps=[s for s in output.steps if s.strip()] or [f"Explain: {question}"],
            diagrams_needed=[q.strip() for q in output.diagram_queries if q.strip()][:MAX_DIAGRAM_QUERIES],
            is_followup=output.is_followup and bool(digest),
            terms_to_explain=[t.strip() for t in output.terms_to_explain if t.strip()][:MAX_TERMS],
            followups=[f.strip() for f in output.followups if f.strip()][:MAX_FOLLOWUPS],
        )
    except Exception as exc:
        # Planning must never block an answer: on an error or a timeout, search on the question itself.
        logger.warning("Planning failed (%s: %s); searching on the question instead", type(exc).__name__, exc)
        plan = TeachingPlan(topic=question[:60], steps=[f"Explain: {question}"],
                            diagrams_needed=[question[:200]], is_fallback=True)

    update: dict = {"plan": plan}
    if plan.diagrams_needed:
        update["messages"] = [AIMessage(
            content="",
            tool_calls=[
                # The planner writes "concept: parts"; a fallback plan's query is the raw question
                {"name": RETRIEVE_TOOL, "args": {"query": q, "structured": not plan.is_fallback},
                 "id": f"{PLAN_CALL_PREFIX}{uuid.uuid4().hex[:12]}",
                 "type": "tool_call"}
                for q in plan.diagrams_needed
            ],
        )]
    return update


async def _answer_with_fallbacks(runnables: list, names: list[str], messages: list[BaseMessage]):
    """
    Try each answer model in turn. Each gets ANSWER_MODEL_TIMEOUT_S for its whole
    (streamed) answer; an error or a timeout moves on to the next. Each new attempt
    starts a new chat-model run, which the server turns into a "reset" for the client.
    """
    last_error: Exception | None = None
    empty_response = None
    for runnable, name in zip(runnables, names):
        try:
            response = await asyncio.wait_for(runnable.ainvoke(messages), timeout=ANSWER_MODEL_TIMEOUT_S)
        except Exception as exc:  # includes TimeoutError
            logger.warning("Answer model %s failed (%s: %.300s); trying the next one", name, type(exc).__name__, exc)
            last_error = exc
            continue
        if content_text(response.content).strip() or getattr(response, "tool_calls", None):
            return response
        # No text and no tool call (a safety block, a malformed call): a failure too
        logger.warning("Answer model %s returned an empty answer; trying the next one", name)
        empty_response = response
    if empty_response is not None:
        return empty_response  # every model came back empty; the server reports "No response"
    raise last_error or RuntimeError("no answer models configured")


def _make_answer_node(tools: list[BaseTool], known_diagram_ids: set[str] | None):
    """Factory for the answering node, with the retrieval tool available for one extra search."""

    async def answer_node(state: AgentState) -> dict:
        """
        Write the explanation in one streamed call, choosing among the
        diagrams the plan retrieved.
        """
        if not state.plan:
            return {"messages": [AIMessage(content="I'd be happy to help! What would you like to learn about?")]}
        plan = state.plan

        idx = _last_human_index(state.messages)
        this_turn = state.messages[idx + 1:] if idx >= 0 else []

        # Every search result from this turn, the planner's and any the answering model
        # ran itself, goes into the prompt as text. Tool-call messages are never replayed:
        # the second answer call runs without tools bound, and Gemini can reject a
        # transcript containing function calls it has no declaration for.
        turn_results = [m for m in this_turn if isinstance(m, ToolMessage)]
        own_calls = [m for m in this_turn if isinstance(m, AIMessage) and m.tool_calls and not _is_plan_call(m)]
        may_search = not own_calls and bool(tools)
        if own_calls:
            tool_rule = TOOL_RULE_USED
        elif tools:
            tool_rule = TOOL_RULE_ALLOWED
        else:
            tool_rule = ""  # no search tool: say nothing about searching

        system = (
            ANSWER_PROMPT.format(tool_rule=tool_rule)
            + f'\nTEACHING PLAN\nTopic: "{plan.topic}"'
            + (" (a follow-up: answer what was asked, do not repeat the earlier explanation)" if plan.is_followup else "")
            + "\nSteps to cover:\n"
            + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan.steps))
            + ("\nTerms the reader needs explained in plain words when they first appear (leave out any you "
               "do not need): " + ", ".join(plan.terms_to_explain) if plan.terms_to_explain else "")
            + "\n\nCANDIDATE DIAGRAMS\n"
            + _format_candidates(turn_results)
        )

        messages_for_llm: list[BaseMessage] = [SystemMessage(content=system)]
        # Conversation history as plain text turns. Earlier tool exchanges are left out:
        # their results belonged to earlier questions. Each earlier answer keeps only the
        # diagram references the user actually saw (not ones cut by the two-diagram limit,
        # never offered, or gone from the corpus), so the model cannot refer back to an
        # image that is not on screen.
        history = state.messages[:idx + 1]
        _, shown_per_message = replay_shown(history, known_diagram_ids)
        # Only the most recent exchanges are sent, so a long conversation does not grow
        # the prompt without limit. The full history stays in the checkpoint.
        questions = [i for i, m in enumerate(history) if isinstance(m, HumanMessage)]
        first_sent = questions[-(MAX_HISTORY_EXCHANGES + 1)] if len(questions) > MAX_HISTORY_EXCHANGES + 1 else 0
        for i, m in enumerate(history):
            if i < first_sent:
                continue
            if isinstance(m, HumanMessage):
                if isinstance(messages_for_llm[-1], HumanMessage):
                    # The previous question got no answer (stopped, failed, or empty):
                    # say so, rather than send two user turns in a row
                    messages_for_llm.append(AIMessage(content="(This question was not answered.)"))
                messages_for_llm.append(m)
            elif isinstance(m, AIMessage) and not m.tool_calls:
                keep = set(shown_per_message.get(i, []))

                def scrub(ref: re.Match, keep: set[str] = keep) -> str:
                    if ref.group(1) in keep:
                        keep.discard(ref.group(1))  # first occurrence only, as rendered
                        return ref.group(0)
                    return ""

                original = message_text(m)
                text = DIAGRAM_REF.sub(scrub, original)
                if text.strip():
                    messages_for_llm.append(AIMessage(content=text))
                elif original.strip():
                    # The answer was only a diagram that is no longer shown; Gemini rejects
                    # empty turns, and this question was answered, so say that instead
                    messages_for_llm.append(AIMessage(content="(Answered with a diagram that is no longer available.)"))

        llms = _chat_models(ANSWER_MODELS, ANSWER_THINKING, max_retries=1)
        runnables = [llm.bind_tools(tools) for llm in llms] if may_search else llms
        response = await _answer_with_fallbacks(runnables, [llm.model for llm in llms], messages_for_llm)
        if len(getattr(response, "tool_calls", None) or []) > 1:
            response.tool_calls = response.tool_calls[:1]  # the prompt allows one extra search
        return {"messages": [response]}

    return answer_node


def _route_after_plan(state: AgentState) -> Literal["tools", "answer"]:
    last = state.messages[-1] if state.messages else None
    return "tools" if last is not None and getattr(last, "tool_calls", None) else "answer"


def _route_after_answer(state: AgentState) -> Literal["tools", "end"]:
    last = state.messages[-1] if state.messages else None
    return "tools" if last is not None and getattr(last, "tool_calls", None) else "end"


# --- Graph Construction ---

def create_agent(
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    known_diagram_ids: set[str] | None = None,
):
    """
    Create the Plan-and-Execute LangGraph agent.

    Flow: plan → tools → answer (⇄ tools, at most once) → end

    Args:
        tools: LangChain tools the agent can call (loaded from the MCP server).
        checkpointer: Optional checkpoint saver for conversation state.
                     If None, uses InMemorySaver (state lost on restart).
        known_diagram_ids: IDs in the current corpus, used when deciding which
                     diagram references in saved history the user actually saw.
                     References the user did not see are removed before the
                     history is shown to the model.

    Returns:
        Compiled LangGraph agent
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    # A failed search becomes an error ToolMessage (skipped when candidates are
    # listed) instead of failing the whole turn
    workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    workflow.add_node("answer", _make_answer_node(tools, known_diagram_ids))

    workflow.add_edge(START, "plan")
    workflow.add_conditional_edges("plan", _route_after_plan, {"tools": "tools", "answer": "answer"})
    workflow.add_edge("tools", "answer")
    workflow.add_conditional_edges("answer", _route_after_answer, {"tools": "tools", "end": END})

    return workflow.compile(checkpointer=checkpointer)
