import random
import sys

DATASOURCES_PATH = "verified_datasources_for_tests.txt"
METADATA_PATH = "verified_s3_tables_metadata.txt"
OUTPUT_PATH = "combined_dataset.txt"
TARGET_TOTAL = 10_000


def load_lines(path):
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main(seed=None):
    datasources = load_lines(DATASOURCES_PATH)
    metadata = load_lines(METADATA_PATH)

    print(f"Loaded {len(datasources)} lines from {DATASOURCES_PATH}")
    print(f"Loaded {len(metadata)} lines from {METADATA_PATH}")

    if len(datasources) >= TARGET_TOTAL:
        print(f"WARNING: datasources alone ({len(datasources)}) meets/exceeds target {TARGET_TOTAL}. No metadata will be sampled.")
        combined = datasources
    else:
        n_sample = TARGET_TOTAL - len(datasources)
        if n_sample > len(metadata):
            print(f"WARNING: requested {n_sample} samples but metadata only has {len(metadata)}. Using all metadata.")
            n_sample = len(metadata)

        rng = random.Random(seed)
        sampled_metadata = rng.sample(metadata, n_sample)
        combined = datasources + sampled_metadata

    rng_shuffle = random.Random(seed)
    rng_shuffle.shuffle(combined)

    with open(OUTPUT_PATH, "w") as f:
        for line in combined:
            f.write(line + "\n")

    print(f"Written {len(combined)} lines to {OUTPUT_PATH}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed=seed)
