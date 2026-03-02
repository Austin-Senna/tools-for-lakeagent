"""
filter_metadata.py

Reads verified_s3_tables.txt and writes verified_s3_tables_metadata.txt
containing only entries that appear to be real data files (not metadata).

Metadata patterns detected from analysis of verified_s3_tables.txt
and cross-referenced with s3_list_all.py ignore rules.
"""

import re
import sys
from pathlib import Path

# ── Exact filename stems that are always metadata ──────────────────────────
_METADATA_EXACT = {
    # License / legal
    'cc-zero', 'cc-by', 'legalcode', 'open-licenses', 'government-works',
    'odc-odbl', 'license',
    # Navigation / catalog
    'index', 'metadata', 'catalog', 'dcat-us', 'iso',
    # Socrata / generic export names
    'edit',
    'search',
    'contact',
    'resolve',
    'request',
    # Format / protocol metadata
    'wmsserver', 'policyinformation', 'gmxcodelists', 'gmi',
    # Documentation
    'readme', 'headers', 'signed-metadata',
    # Misc portal metadata
    'bios', 'hires', 'cwhr',
    # Versioned stubs – v1/v2 files are almost always ICPSR/dataverse stubs
    'v1', 'v2',
}

# ── Regex patterns for metadata filenames ──────────────────────────────────
# Socrata dataset ID  e.g. tsqz-67gi.txt
_SOCRATA_ID_RE = re.compile(r'^[a-z0-9]{4}-[a-z0-9]{4}$')

# Pure number (NOAA coastal DEM ids, Atlas IDs, etc.)  e.g. 34104.txt, 6.txt
_PURE_NUMBER_RE = re.compile(r'^\d+(-\d+)?$')

# Filename already ends with a random 6-char suffix  e.g. index-DKB9SW.txt
# This catches variants like "index-yPXves", "data-ghDEVP", etc.
_INDEX_VARIANT_RE = re.compile(r'^index-[A-Za-z0-9]{6}$')

# Dataset name reused as filename (long slug repeated as the file name)
# e.g. "10-year-comparison-of-taxpayer-income.txt" inside that same dataset folder
# Detected heuristic: filename stem == dataset slug from the path
_SLUG_REPEAT_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+){3,}$')   # 4+ hyphen-separated tokens

# DOI / URL fragments that end up as filenames
_DOI_RE = re.compile(r'^doi-[a-z0-9]+$')
_URL_FRAGMENT_RE = re.compile(r'^www_[a-z0-9_]+$')


def _dataset_slug(uri: str) -> str:
    """Extract the dataset folder name from the URI."""
    # s3://bucket/folder/DATASET_SLUG/files/filename.ext
    parts = uri.split('/')
    try:
        files_idx = parts.index('files')
        return parts[files_idx - 1].lower()
    except ValueError:
        return ''


def is_metadata(uri: str) -> bool:
    """Return True if the URI looks like a metadata file that should be excluded."""
    filename = uri.split('/')[-1]
    stem = filename.rsplit('.', 1)[0].lower()

    # 1. Exact match
    if stem in _METADATA_EXACT:
        return True

    # 2. Socrata ID pattern
    if _SOCRATA_ID_RE.match(stem):
        return True

    # 3. Pure number
    if _PURE_NUMBER_RE.match(stem):
        return True

    # 4. index-XXXXXX variant
    if _INDEX_VARIANT_RE.match(stem):
        return True

    # 5. DOI / URL fragment filenames
    if _DOI_RE.match(stem):
        return True
    if _URL_FRAGMENT_RE.match(stem):
        return True

    # 6. Filename stem == dataset slug (dataset name repeated as file)
    dataset_slug = _dataset_slug(uri)
    if dataset_slug and stem == dataset_slug:
        return True

    return False


def filter_file(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    lines = [l.strip() for l in input_file.read_text().splitlines() if l.strip()]
    total = len(lines)

    kept = [l for l in lines if not is_metadata(l)]
    removed = total - len(kept)

    Path(output_path).write_text('\n'.join(kept) + '\n', encoding='utf-8')

    print(f"Input lines  : {total:,}")
    print(f"Removed (meta): {removed:,}  ({100*removed/total:.1f}%)")
    print(f"Kept (data)  : {len(kept):,}  ({100*len(kept)/total:.1f}%)")
    print(f"Output written to: {output_path}")


if __name__ == '__main__':
    input_txt  = sys.argv[1] if len(sys.argv) > 1 else 'verified_s3_tables.txt'
    output_txt = sys.argv[2] if len(sys.argv) > 2 else 'verified_s3_tables_metadata.txt'
    filter_file(input_txt, output_txt)
