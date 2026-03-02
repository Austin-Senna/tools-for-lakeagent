import os
import sys
import re
import boto3
import concurrent.futures
from botocore.config import Config
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
REGION = "us-east-1"
MAX_THREADS = 50  # Process 50 datasets at the exact same time

def _get_s3_client():
    # ✨ CRITICAL: Increase the connection pool size so threads don't bottleneck
    boto_config = Config(max_pool_connections=MAX_THREADS)
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        config=boto_config
    )


# Regex patterns for metadata filenames detected from verified_datasources_for_tests.txt
_SOCRATA_ID_RE = re.compile(r'^[a-z0-9]{4}-[a-z0-9]{4}$')          # e.g. tsqz-67gi
_PURE_NUMBER_RE = re.compile(r'^\d+(-\d+)?$')                        # e.g. 0, 1, 4, 1-0, 15901
_RANDOM_SUFFIX_RE = re.compile(r'^.+-[A-Za-z0-9]{6}$')              # e.g. data-ghDEVP, SG610-...-kDO3fv

# Exact metadata filenames (lowercased for comparison)
_METADATA_EXACT = {
    'data', 'rows', 'columns', 'metadata',
    'gmi', 'open-licenses', 'legalcode', 'government-works',
    'index', 'odc-odbl', 'wmsserver', 'resolve', 'request',
    'edit', 'search', 'contact', 'policyinformation', 'gmxcodelists',
    'bios', 'hires', 'cwhr', 'license', 'readme'
    # legacy entries kept from original ignore_list
    'signed-metadata', 'headers', 'dcat-us', 'catalog',
    'readme', 'iso', 'cc-zero', 'cc-by'
}

def _is_metadata_filename(filename: str) -> bool:
    """Returns True if the filename (without extension) looks like metadata, not real data."""
    stem = filename.rsplit('.', 1)[0].lower()
    if stem in _METADATA_EXACT:
        return True
    if _SOCRATA_ID_RE.match(stem):
        return True
    if _PURE_NUMBER_RE.match(stem):
        return True
    if _RANDOM_SUFFIX_RE.match(stem):
        return True
    return False

def list_files(s3, dataset_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    all_files = []

    # Iterate over both folders dynamically
    for folder in ["wikipedia", "datagov"]:
        prefix = f"{folder}/{dataset_id}/"
        paginator = s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                # Get the filename or relative path
                relative_path = key.split(prefix, 1)[-1]

                if not relative_path or key.endswith('/'):
                    continue

                # Skip metadata files identified by name alone
                fname = relative_path.split('/')[-1]
                if _is_metadata_filename(fname):
                    continue
                    
                all_files.append({
                    'path': relative_path,
                    'dataset_id': dataset_id,
                    'folder': folder,
                    'full_key': key
                })
    return all_files


def _is_table_by_peeking(s3, file_path: str) -> bool:
    """Downloads the first 2KB of a file from S3 to verify structural integrity."""
    try:
        response = s3.get_object(Bucket=BUCKET, Key=file_path, Range='bytes=0-2048')
        content = response['Body'].read().decode('utf-8', errors='ignore')
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:
            return False

        json_lines = sum(1 for line in lines[:5] if line.startswith('{') and line.endswith('}'))
        if json_lines >= 3:
            return True
            
        first_char = content.strip()[0]
        if first_char in ['{', '[']:
            return False

        delimiters = [',', '\t', '|', ';']
        for delim in delimiters:
            counts = []
            for line in lines[:5]:
                clean_line = re.sub(r'"[^"]*"', '', line)
                counts.append(clean_line.count(delim))
            
            valid_counts = [c for c in counts if c > 0]
            if len(valid_counts) >= 3 and len(set(valid_counts[:3])) == 1:
                if valid_counts[0] >= 1:
                    return True
                
        return False
        
    except Exception:
        return False

def process_single_dataset(s3, dataset_id: str, table_only: bool) -> tuple:
    """Worker function for a single thread. Returns (dataset_id, list_of_uris)."""
    files_data = list_files(s3, dataset_id)
    s3_uris = []
    
    for f in files_data:
        lower_path = f['path'].lower()
        if lower_path.endswith(('.jpg', '.png', '.pdf')):
            continue
            
        uri = f"s3://{BUCKET}/{f['full_key']}"

        if not table_only or _is_table_by_peeking(s3, f['full_key']):
            s3_uris.append(uri)
            
    return dataset_id, s3_uris

def extract_and_verify(input_txt, output_txt, table_only, log_txt="processed_datasets.log"):
    if not Path(input_txt).exists():
        print(f"Error: Could not find {input_txt}")
        return

    # 1. Load the state (Checkpointing)
    processed_datasets = set()
    if Path(log_txt).exists():
        with open(log_txt, 'r') as log:
            processed_datasets = set(line.strip() for line in log if line.strip())
            
    # 2. Figure out what is left to do
    with open(input_txt, "r") as f:
        all_datasets = [line.strip() for line in f if line.strip()]
        
    pending_datasets = [ds for ds in all_datasets if ds not in processed_datasets]
    
    print(f"Total datasets: {len(all_datasets)}")
    print(f"Already processed: {len(processed_datasets)}")
    print(f"Pending tasks: {len(pending_datasets)}")
    
    if not pending_datasets:
        print("Everything is up to date!")
        return

    s3_client = _get_s3_client()
    
    # 3. Stream writing: Open files in Append mode
    with open(output_txt, 'a', encoding='utf-8') as out_file, \
         open(log_txt, 'a', encoding='utf-8') as log_file:
             
        # 4. Multithreading execution
        print(f"\nSpinning up {MAX_THREADS} threads... Hold onto your terminal.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            
            # Submit all tasks to the queue
            future_to_dataset = {
                executor.submit(process_single_dataset, s3_client, ds, table_only): ds 
                for ds in pending_datasets
            }
            
            completed_count = 0
            # As threads finish (in whatever order), catch their results instantly
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                completed_count += 1
                
                try:
                    _, verified_uris = future.result()
                    
                    # Stream writes to disk instantly
                    for uri in verified_uris:
                        out_file.write(f"{uri}\n")
                        print(f"  ✅ VERIFIED: {uri}")
                    
                    # Mark dataset as done so we don't repeat it on crash
                    log_file.write(f"{dataset_id}\n")
                    
                    # Force Python to actually write the buffer to the OS right now
                    out_file.flush()
                    log_file.flush()
                    
                    if completed_count % 50 == 0:
                        print(f"Progress: {completed_count} / {len(pending_datasets)} datasets scanned...")
                        
                except Exception as exc:
                    print(f"❌ Dataset {dataset_id} generated an exception: {exc}")

    print("\nDone! 🎉")


if __name__ == "__main__":
    load_dotenv()
    input_txt = sys.argv[1] if len(sys.argv) > 1 else "all_datasets_complete.txt"
    output_txt = sys.argv[2] if len(sys.argv) > 2 else "verified_s3_tables.txt"
    table_only = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    extract_and_verify(input_txt, output_txt, table_only=table_only)