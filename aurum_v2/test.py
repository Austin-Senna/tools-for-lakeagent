from aurum_v2.profiler.source_readers import discover_sources, SourceConfig
from aurum_v2.profiler.column_profiler import Profiler
from aurum_v2.config import AurumConfig
# from aurum_v2.config import load_config

from typing import List
import json
import random

from typing import List
import json
import random

BUCKET = "your-bucket-name"  # <-- set this properly


def generate_paths(
    dataset_ids: List[str],
    n: int,
    manifest_path: str = "file_listing.jsonl",
) -> List[str]:
    """
    Randomly sample n S3 paths from a static JSONL manifest,
    filtering only by provided dataset_ids.
    """

    if not isinstance(dataset_ids, list) or not dataset_ids:
        raise ValueError("dataset_ids must be a non-empty list")

    dataset_ids_set = set(dataset_ids)
    matching_items = []

    # Read manifest line-by-line (memory safe)
    with open(manifest_path, "r") as f:
        for line in f:
            item = json.loads(line)
            key = item.get("key", "")

            # key format: folder/dataset_id/file
            parts = key.split("/", 2)
            if len(parts) < 3:
                continue

            folder, dataset_id, _ = parts

            if dataset_id in dataset_ids_set:
                matching_items.append(key)

    if not matching_items:
        raise ValueError("No files found for given dataset_ids.")

    n = min(n, len(matching_items))
    sampled_keys = random.sample(matching_items, n)

    s3_paths = [f"s3://{BUCKET}/{key}" for key in sampled_keys]

    return s3_paths


def main():
    paths = generate_paths(10)
    source_configs = [SourceConfig("s3", "s3", {"s3_paths": paths})]
    source_readers = discover_sources(source_configs)
    profiler = Profiler(AurumConfig)
    profiler.run(source_readers, run_ner=True)
    profiler.store_profiles()

if __name__ == "__main__":
    main()