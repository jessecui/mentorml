"""
LangSmith-backed retrieval eval: SigLIP 2 vs CLIP, top-1 and top-5 accuracy.

Runs each model as a target against the LangSmith dataset uploaded by
upload_to_langsmith.py. Same comparison as evaluate.py, but with traces,
versioned datasets, and a side-by-side comparison UI.

Usage:
    export LANGSMITH_API_KEY=...
    python benchmark/scripts/langsmith_evaluate_retrieval.py                  # curated corpus
    python benchmark/scripts/langsmith_evaluate_retrieval.py --corpus legacy
    python benchmark/scripts/langsmith_evaluate_retrieval.py --limit 10
    python benchmark/scripts/langsmith_evaluate_retrieval.py --skip-clip
    python benchmark/scripts/langsmith_evaluate_retrieval.py --dataset-name custom-name
"""

import argparse
import sys

from dotenv import load_dotenv

from common import DATASET_NAMES, PROJECT_ROOT, load_benchmark, make_clip_ranker, make_siglip_ranker

load_dotenv(PROJECT_ROOT / ".env")


def make_target(rank):
    """Wrap a ranker as a LangSmith target returning its five best diagram IDs."""
    def target(inputs: dict) -> dict:
        top = rank(inputs["query"])[:5]
        return {"diagram_id": top[0], "top_5_diagram_ids": top}

    return target


def top_1_accuracy(run, example) -> dict:
    """1 if the top-ranked diagram is the reference, else 0."""
    predicted = (run.outputs or {}).get("diagram_id")
    expected = (example.outputs or {}).get("relevant_image_id")
    return {"key": "top_1_accuracy", "score": 1 if predicted == expected else 0}


def top_5_accuracy(run, example) -> dict:
    """1 if the reference diagram is among the five best, else 0."""
    predicted = (run.outputs or {}).get("top_5_diagram_ids") or []
    expected = (example.outputs or {}).get("relevant_image_id")
    return {"key": "top_5_accuracy", "score": 1 if expected in predicted else 0}


def _mean_scores(results) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in results:
        for r in row["evaluation_results"]["results"]:
            if r.score is not None:
                totals.setdefault(r.key, []).append(float(r.score))
    return {k: sum(v) / len(v) for k, v in totals.items() if v}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["curated", "legacy"], default="curated")
    parser.add_argument("--dataset-name", default=None, help="Defaults to a name derived from --corpus")
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-siglip", action="store_true")
    args = parser.parse_args()

    from langsmith import Client, evaluate

    from model.siglip_config import MODEL_ID as SIGLIP_MODEL_ID

    client = Client()
    bench = load_benchmark(args.corpus)
    dataset_name = args.dataset_name or DATASET_NAMES[args.corpus]

    existing = list(client.list_datasets(dataset_name=dataset_name))
    if not existing:
        print(f"❌ Dataset '{dataset_name}' not found. Run upload_to_langsmith.py --corpus {args.corpus} first.")
        sys.exit(1)
    examples = list(client.list_examples(dataset_name=dataset_name))
    known_ids = set(bench.image_ids)
    stale = [e for e in examples if (e.outputs or {}).get("relevant_image_id") not in known_ids]
    if stale or len(examples) != len(bench.queries):
        print(f"❌ Dataset '{dataset_name}' ({len(examples)} examples) does not match the local "
              f"'{bench.name}' corpus ({len(bench.queries)} queries, {len(stale)} unknown IDs).")
        print("   The corpus was probably rebuilt. Upload it again under a new --dataset-name.")
        sys.exit(1)
    limited = args.limit is not None
    data = examples[:args.limit] if limited else dataset_name
    print(f"Running against {'the first ' + str(args.limit) + ' examples of' if limited else 'all of'} '{dataset_name}'")

    evaluators = [top_1_accuracy, top_5_accuracy]
    metadata = {"corpus": bench.name, "n_images": len(bench.image_ids)}
    summary: dict[str, dict[str, float]] = {}

    if not args.skip_siglip:
        print("\n" + "=" * 60 + "\nEvaluating SigLIP 2\n" + "=" * 60)
        results = evaluate(
            make_target(make_siglip_ranker(bench)), data=data, evaluators=evaluators,
            experiment_prefix="siglip-v2", metadata={**metadata, "model": SIGLIP_MODEL_ID},
            max_concurrency=1,
        )
        summary["SigLIP 2"] = _mean_scores(results)
        print(f"\n✅ SigLIP eval complete: {results.experiment_name}")

    if not args.skip_clip:
        print("\n" + "=" * 60 + "\nEvaluating CLIP baseline\n" + "=" * 60)
        results = evaluate(
            make_target(make_clip_ranker(bench)), data=data, evaluators=evaluators,
            experiment_prefix="clip-vit-l14", metadata={**metadata, "model": "clip-vit-large-patch14-336"},
            max_concurrency=1,
        )
        summary["CLIP"] = _mean_scores(results)
        print(f"\n✅ CLIP eval complete: {results.experiment_name}")

    print("\n" + "=" * 60)
    for model, scores in summary.items():
        print(f"{model:<10} top-1 {scores.get('top_1_accuracy', 0) * 100:5.1f}%   top-5 {scores.get('top_5_accuracy', 0) * 100:5.1f}%")
    if "SigLIP 2" in summary and "CLIP" in summary:
        gap = (summary["SigLIP 2"].get("top_1_accuracy", 0) - summary["CLIP"].get("top_1_accuracy", 0)) * 100
        print(f"SigLIP 2 minus CLIP: {gap:+.1f} pp top-1")
    print("View results & side-by-side comparison in the LangSmith UI under your dataset.")


if __name__ == "__main__":
    main()
