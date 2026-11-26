"""
End-to-end agent eval against the LangSmith retrieval dataset.

Target: invokes the full LangGraph agent (plan -> MCP-backed retrieve_diagram
-> answer) on each query and returns the agent's response, the diagram IDs
its searches retrieved, and the IDs the answer actually cites.

Evaluators:
- tool_call_correctness  (well-grounded): is the dataset's relevant_image_id
                          among the candidates the agent retrieved?
- cited_expected         did the answer cite that diagram?
- cited_any              did the answer include any diagram at all?
- explanation_quality    (LLM judge, ungrounded — no reference answers):
                          accuracy + accessibility, rated 1-5 by Gemini Flash.

Usage:
    export LANGSMITH_API_KEY=...
    export GOOGLE_API_KEY=...
    python benchmark/scripts/langsmith_evaluate_agent.py --limit 3
    python benchmark/scripts/langsmith_evaluate_agent.py --no-judge
    python benchmark/scripts/langsmith_evaluate_agent.py
"""

import argparse
import asyncio
import json
import sys
import uuid
from contextlib import AsyncExitStack

from dotenv import load_dotenv

from common import DATASET_NAMES, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")



# =============================================================================
# AGENT TARGET
# =============================================================================

def make_agent_target(agent):
    """Build an async target function that invokes the agent end-to-end."""
    from langchain_core.messages import HumanMessage

    async def target(inputs: dict) -> dict:
        thread_id = f"eval-{uuid.uuid4()}"
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=inputs["query"])]},
            config={"configurable": {"thread_id": thread_id}},
        )

        msgs = result.get("messages", [])

        # Final AI text response, read the way main.chat does: this turn only, and
        # never a message that also made a tool call
        response_text = ""
        for msg in reversed(msgs):
            if msg.type == "human":
                break
            if msg.type != "ai" or getattr(msg, "tool_calls", None):
                continue
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if isinstance(content, str) and content.strip():
                response_text = content
                break

        # Diagram IDs the agent's searches returned (in rank order, deduped)
        retrieved: list[str] = []
        for msg in msgs:
            if msg.type == "tool" and msg.name == "retrieve_diagram":
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    for candidate in (tool_result or {}).get("candidates", []):
                        if candidate["id"] not in retrieved:
                            retrieved.append(candidate["id"])
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        # Only IDs a search returned count: the app shows nothing for any other ID
        from agent.graph import rendered_ids

        # What the app would show: retrieved IDs only, each once, at most two
        cited = rendered_ids(response_text, set(retrieved), None)

        plan = result.get("plan")
        plan_topic = plan.topic if plan else None

        return {
            "response": response_text,
            "retrieved_diagram_ids": retrieved,
            "cited_diagram_ids": cited,
            "plan_topic": plan_topic,
        }

    return target


# =============================================================================
# EVALUATOR A: tool-call correctness
# =============================================================================

def tool_call_correctness(run, example) -> dict:
    """1 if the dataset's relevant_image_id appears in the agent's retrieval set."""
    expected = (example.outputs or {}).get("relevant_image_id")
    retrieved = (run.outputs or {}).get("retrieved_diagram_ids", []) or []
    return {
        "key": "tool_call_correctness",
        "score": 1 if expected in retrieved else 0,
        "comment": f"expected={expected} retrieved={retrieved}",
    }


def cited_expected(run, example) -> dict:
    """1 if the answer cites the dataset's relevant diagram."""
    expected = (example.outputs or {}).get("relevant_image_id")
    cited = (run.outputs or {}).get("cited_diagram_ids", []) or []
    return {"key": "cited_expected", "score": 1 if expected in cited else 0, "comment": f"cited={cited}"}


def cited_any(run, example) -> dict:
    """1 if the answer includes at least one diagram."""
    cited = (run.outputs or {}).get("cited_diagram_ids", []) or []
    return {"key": "cited_any", "score": 1 if cited else 0}


# =============================================================================
# EVALUATOR B: LLM judge for explanation quality
# =============================================================================

JUDGE_PROMPT = """You are evaluating an AI/ML teaching response.

Question asked: {query}

Response:
{response}

Rate the response on two dimensions, each 1-5 (5 = excellent, 1 = poor):
- accuracy: is the technical ML content factually correct?
- accessibility: is it clear and well-paced for a learner with some ML background?

Respond with ONLY a JSON object on a single line, no other text, no markdown:
{{"accuracy": <int 1-5>, "accessibility": <int 1-5>, "reasoning": "<one short sentence>"}}"""


def make_quality_judge():
    """Build an async LLM-judge evaluator. Returns a function suitable for evaluators=[...]."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    from agent.graph import _model_chain

    # A fixed judge, separate from the models under test and deterministic, so scores
    # are comparable across runs and across changes to MENTORML_ANSWER_MODELS
    judge_models = _model_chain("MENTORML_JUDGE_MODELS", "gemini-3.6-flash,gemini-3.5-flash")

    class Judge:
        async def ainvoke(self, messages):
            last_error: Exception | None = None
            for model in judge_models:
                try:
                    llm = ChatGoogleGenerativeAI(model=model, temperature=0.0, thinking_level="low", max_retries=1)
                    return await llm.ainvoke(messages)
                except Exception as exc:
                    last_error = exc
            raise last_error or RuntimeError("no judge models configured")

    llm = Judge()

    async def explanation_quality(run, example) -> list[dict]:
        response = (run.outputs or {}).get("response", "")
        query = (example.inputs or {}).get("query", "")

        if not response:
            return [
                {"key": "accuracy", "score": 0, "comment": "no response"},
                {"key": "accessibility", "score": 0, "comment": "no response"},
            ]

        judgment = await llm.ainvoke([
            SystemMessage(content="You are a strict but fair AI/ML educator."),
            HumanMessage(content=JUDGE_PROMPT.format(query=query, response=response)),
        ])

        text = getattr(judgment, "text", None) or judgment.content
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        text = (text or "").strip()

        # Strip markdown fencing if the model added it despite instructions
        if "```" in text:
            chunks = text.split("```")
            if len(chunks) >= 2:
                inner = chunks[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                text = inner.strip()

        try:
            parsed = json.loads(text)
            acc = int(parsed.get("accuracy", 0))
            access = int(parsed.get("accessibility", 0))
            reasoning = str(parsed.get("reasoning", ""))[:200]
            return [
                {"key": "accuracy", "score": acc / 5.0, "comment": reasoning},
                {"key": "accessibility", "score": access / 5.0},
            ]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return [
                {"key": "accuracy", "score": 0, "comment": f"judge parse error: {text[:120]!r}"},
                {"key": "accessibility", "score": 0},
            ]

    return explanation_quality


# =============================================================================
# MAIN
# =============================================================================

async def amain():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["curated"], default="curated",
                        help="Only the curated dataset works here: the agent retrieves from the curated corpus, "
                             "so the legacy dataset's diagram IDs can never match.")
    parser.add_argument("--dataset-name", default=None, help="Defaults to a name derived from --corpus")
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-judge evaluator (only run the grounded checks).")
    parser.add_argument("--max-concurrency", type=int, default=2,
                        help="Parallel agent invocations. Higher = faster but risks Gemini rate limits.")
    args = parser.parse_args()

    args.dataset_name = args.dataset_name or DATASET_NAMES[args.corpus]

    from langgraph.checkpoint.memory import InMemorySaver
    from langsmith import Client, aevaluate

    from agent.graph import create_agent
    from agent.tools import load_persistent_mcp_tools

    client = Client()

    # Resolve the dataset, and make sure it describes the corpus the agent retrieves from
    from common import load_benchmark

    if not list(client.list_datasets(dataset_name=args.dataset_name)):
        print(f"❌ Dataset '{args.dataset_name}' not found. Run upload_to_langsmith.py first.")
        sys.exit(1)
    all_examples = list(client.list_examples(dataset_name=args.dataset_name))
    corpus_ids = set(load_benchmark("curated").image_ids)
    stale = [e for e in all_examples if (e.outputs or {}).get("relevant_image_id") not in corpus_ids]
    if stale:
        print(f"❌ {len(stale)} of {len(all_examples)} examples in '{args.dataset_name}' name diagrams that are "
              "not in the current corpus (it was probably rebuilt). Upload it again under a new --dataset-name.")
        sys.exit(1)
    if args.limit is not None:
        data = all_examples[:args.limit]
        print(f"Running against {len(data)}/{len(all_examples)} examples (--limit {args.limit})")
    else:
        data = args.dataset_name
        print(f"Running against full dataset '{args.dataset_name}'")

    # Build evaluator list
    evaluators: list = [tool_call_correctness, cited_expected, cited_any]
    if not args.no_judge:
        evaluators.append(make_quality_judge())
    print(f"Evaluators: {[getattr(e, '__name__', repr(e)) for e in evaluators]}")
    print(f"Max concurrency: {args.max_concurrency}")
    print()

    async with AsyncExitStack() as stack:
        print("🔌 Spawning diagram MCP server (stdio, persistent session)...")
        tools = await load_persistent_mcp_tools(stack)
        print(f"   ✅ Loaded {len(tools)} tool(s) from MCP server")

        agent = create_agent(tools, checkpointer=InMemorySaver())
        target = make_agent_target(agent)

        print("\n" + "=" * 60)
        print("Evaluating full LangGraph agent end-to-end")
        print("=" * 60)

        results = await aevaluate(
            target,
            data=data,
            evaluators=evaluators,
            experiment_prefix="agent-e2e",
            max_concurrency=args.max_concurrency,
        )
        print(f"\n✅ Agent eval complete: {results.experiment_name}")
        print("View results in the LangSmith UI under your dataset.")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
