import os
import json
import sys
import re
import boto3
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
REGION = "us-east-1"

def _get_s3_client():
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def list_files(dataset_ids: List[str], limit: int = 100) -> Dict[str, Any]:
    s3 = _get_s3_client()
    all_files = []
    
    for dataset_id in dataset_ids:
        folder = "datagov" 
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{folder}/{dataset_id}/", MaxKeys=limit)

        if 'Contents' in response:
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    all_files.append({
                        'path': obj['Key'].split(f"{folder}/{dataset_id}/", 1)[-1],
                        'dataset_id': dataset_id,
                        'folder': folder,
                        'full_key': obj['Key'] # Store the full S3 key for easy access
                    })
    return {'files': all_files}

def _is_table_by_peeking(s3, file_path: str) -> bool:
    """
    Downloads the first 2KB of a file from S3. 
    Uses consistency heuristics and blocklists to isolate true tables.
    """
    lower_path = file_path.lower()
    ignore_list = ['metadata.json', 'signed-metadata', 'catalog.', 'cc-by', 'iso.xml', 'iso.txt', 'readme']
    if any(skip in lower_path for skip in ignore_list):
        return False

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
    
def get_s3_uris_for_tables(dataset_ids: List[str], limit_per_dataset: int = 100) -> List[str]:
    """Returns a flat list of s3:// URIs ready for DuckDB to consume."""
    print(f"Listing files for {len(dataset_ids)} datasets...")
    files_data = list_files(dataset_ids, limit=limit_per_dataset).get('files', [])
    s3 = _get_s3_client()
    s3_uris = []
    
    print(f"Found {len(files_data)} candidate files. Filtering for tables...")
    for f in files_data:
        if f['path'].lower().endswith(('.jpg', '.png', '.pdf', '.zip')):
            continue
            
        # Pass the exact full S3 key to the peek function
        if _is_table_by_peeking(s3, f['full_key']):
            uri = f"s3://{BUCKET}/{f['full_key']}"
            s3_uris.append(uri)
            print(f"  ✅ VERIFIED: {uri}")
            
    return s3_uris

def extract_and_verify_tables(input_txt, output_txt):
    if not Path(input_txt).exists():
        print(f"Error: Could not find {input_txt}")
        return

    # Read ALL datasets into a list and strip newlines
    with open(input_txt, "r") as f:
        datasets = [line.strip() for line in f if line.strip()]
        
    if not datasets:
        print("No datasets found in input file.")
        return

    # Process all datasets to get verified URIs
    verified_tables = get_s3_uris_for_tables(datasets)

    print(f"\nWriting {len(verified_tables)} verified tables to {output_txt}...")
    # Write them out line-by-line
    with open(output_txt, "w") as w:
        for table in verified_tables:
            w.write(f"{table}\n")
            
    print("Done!")

if __name__ == "__main__":
    load_dotenv()
    
    input_txt = sys.argv[1] if len(sys.argv) > 1 else "all_datasets_complete.txt"
    output_txt = sys.argv[2] if len(sys.argv) > 2 else "verified_s3_tables.txt"
    
    extract_and_verify_tables(input_txt, output_txt)