"""
Offline benchmark: SigLIP 2 vs CLIP on diagram retrieval.

For each query, both models rank every image in the corpus. Reports top-1 and
top-5 accuracy (is the query's own image ranked first / in the first five)
and mean reciprocal rank. Top-5 is closer to what the app does than top-1,
since the app retrieves several candidates and lets the answering model choose.

Usage:
    python benchmark/scripts/evaluate.py                    # curated corpus
    python benchmark/scripts/evaluate.py --corpus legacy    # original 92 diagrams
    python benchmark/scripts/evaluate.py --skip-clip
    python benchmark/scripts/evaluate.py --limit 20

Output: benchmark/results/<corpus>_evaluation_results.json
        benchmark/results/<corpus>_evaluation_summary.txt
"""

import argparse
import json

from common import RESULTS_DIR, load_benchmark, make_clip_ranker, make_siglip_ranker


def evaluate_ranker(rank, queries: list[dict]) -> dict:
    ranks: list[int] = []
    for q in queries:
        ranking = rank(q["query"])
        ranks.append(ranking.index(q["relevant_image_id"]) + 1)
    n = len(ranks) or 1  # avoid dividing by zero when there are no queries
    return {
        "top1_correct": sum(r == 1 for r in ranks),
        "top5_correct": sum(r <= 5 for r in ranks),
        "top1_accuracy": sum(r == 1 for r in ranks) / n,
        "top5_accuracy": sum(r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SigLIP 2 vs CLIP on diagram retrieval")
    parser.add_argument("--corpus", choices=["curated", "legacy"], default="curated")
    parser.add_argument("--skip-clip", action="store_true", help="Skip the CLIP baseline")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Use only the first N queries")
    args = parser.parse_args()

    bench = load_benchmark(args.corpus)
    queries = bench.queries[:args.limit] if args.limit is not None else bench.queries
    print(f"Benchmark '{bench.name}': {len(queries)} queries over {len(bench.image_ids)} images\n")

    results: dict = {"corpus": bench.name, "n_queries": len(queries), "n_images": len(bench.image_ids)}

    print("Evaluating SigLIP 2...")
    results["siglip"] = evaluate_ranker(make_siglip_ranker(bench), queries)
    if not args.skip_clip:
        print("Evaluating CLIP...")
        results["clip"] = evaluate_ranker(make_clip_ranker(bench), queries)

    results["details"] = [
        {
            "query": q["query"],
            "target_id": q["relevant_image_id"],
            "siglip_rank": results["siglip"]["ranks"][i],
            "clip_rank": results["clip"]["ranks"][i] if "clip" in results else None,
        }
        for i, q in enumerate(queries)
    ]

    n = len(queries)
    lines = [
        f"SigLIP 2 vs CLIP on the '{bench.name}' benchmark",
        "=" * 48,
        f"Queries: {n}",
        f"Images:  {len(bench.image_ids)}",
        "",
        f"{'Model':<10}{'Top-1':>20}{'Top-5':>20}{'MRR':>8}",
    ]
    for key, label in (("siglip", "SigLIP 2"), ("clip", "CLIP")):
        if key not in results:
            continue
        r = results[key]
        del r["ranks"]
        top1 = f"{r['top1_accuracy'] * 100:.1f}% ({r['top1_correct']}/{n})"
        top5 = f"{r['top5_accuracy'] * 100:.1f}% ({r['top5_correct']}/{n})"
        lines.append(f"{label:<10}{top1:>20}{top5:>20}{r['mrr']:>8.3f}")
    if "clip" in results:
        gap1 = (results["siglip"]["top1_accuracy"] - results["clip"]["top1_accuracy"]) * 100
        gap5 = (results["siglip"]["top5_accuracy"] - results["clip"]["top5_accuracy"]) * 100
        results["top1_gap_pp"] = gap1
        results["top5_gap_pp"] = gap5
        lines += ["", f"SigLIP 2 minus CLIP: {gap1:+.1f} pp top-1, {gap5:+.1f} pp top-5"]
    lines += [
        "",
        "Models:",
        "  - SigLIP 2 So400m/14 @ 384px (PyTorch, Hugging Face transformers)",
        "  - CLIP ViT-L/14 @ 336px (Hugging Face transformers)",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)

    # Only a full run replaces the saved results; a --limit run is a quick check
    if args.limit is not None or args.skip_clip:
        print("\n(Partial run: saved results left unchanged)")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{bench.name}_evaluation_results.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / f"{bench.name}_evaluation_summary.txt").write_text(summary + "\n")
    print(f"\nSaved to {RESULTS_DIR.relative_to(RESULTS_DIR.parent.parent)}/{bench.name}_evaluation_*")


if __name__ == "__main__":
    main()
