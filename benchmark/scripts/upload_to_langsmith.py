"""
One-time uploader: benchmark queries -> LangSmith dataset.

Each query becomes one example with:
    inputs  = {"query": "<query string>"}
    outputs = {"relevant_image_id": "<diagram id>"}

Usage:
    export LANGSMITH_API_KEY=...
    python benchmark/scripts/upload_to_langsmith.py                  # curated corpus
    python benchmark/scripts/upload_to_langsmith.py --corpus legacy  # original 92 diagrams
    python benchmark/scripts/upload_to_langsmith.py --dataset-name custom-name
    python benchmark/scripts/upload_to_langsmith.py --dry-run

The curated corpus changes whenever scripts/curate_corpus.py is re-run. To
benchmark a rebuilt corpus, upload it under a new --dataset-name (or delete
the old dataset in the LangSmith UI first).
"""

import argparse

from dotenv import load_dotenv

from common import DATASET_NAMES, PROJECT_ROOT, load_benchmark

load_dotenv(PROJECT_ROOT / ".env")

DATASET_DESCRIPTION = (
    "Diagram retrieval benchmark for MentorML ({corpus} corpus, {n} images). Each example pairs a "
    "natural-language query with the ID of the one diagram it was written for. Used to evaluate "
    "embedding-based retrieval models (SigLIP 2 vs CLIP)."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["curated", "legacy"], default="curated")
    parser.add_argument("--dataset-name", default=None, help="Defaults to a name derived from --corpus")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded; do not call LangSmith.")
    args = parser.parse_args()

    bench = load_benchmark(args.corpus)
    dataset_name = args.dataset_name or DATASET_NAMES[args.corpus]
    queries = bench.queries
    print(f"Loaded {len(queries)} queries from the '{bench.name}' benchmark")

    inputs = [{"query": q["query"]} for q in queries]
    outputs = [{"relevant_image_id": q["relevant_image_id"]} for q in queries]

    if args.dry_run:
        print("\nDRY RUN — first 3 examples that would be uploaded:")
        for i in range(min(3, len(queries))):
            print(f"  inputs:  {inputs[i]}")
            print(f"  outputs: {outputs[i]}")
            print()
        print(f"Dataset name: {dataset_name}")
        return

    from langsmith import Client

    client = Client()
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        dataset = existing[0]
        print(f"⚠️  Dataset '{dataset_name}' already exists (id={dataset.id})")
        existing_count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        print(f"   It has {existing_count} examples; skipping upload to avoid duplicates.")
        print("   Delete it in the LangSmith UI first, or pass a new --dataset-name.")
        return

    print(f"Creating dataset '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=DATASET_DESCRIPTION.format(corpus=bench.name, n=len(bench.image_ids)),
    )
    print(f"Uploading {len(queries)} examples...")
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    print(f"✅ Uploaded {len(queries)} examples to dataset '{dataset_name}' (id={dataset.id})")


if __name__ == "__main__":
    main()
