import os
import json
import sys
import re
import boto3
from pathlib import Path
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
REGION = "us-east-1"

def _get_s3_client():
    """Get authenticated S3 client using your .env credentials."""
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def is_actual_table(s3, bucket: str, key: str) -> bool:
    """
    Downloads the first 2KB of the file from S3.
    Runs the regex consistency check to verify it is a true table.
    """
    try:
        response = s3.get_object(Bucket=bucket, Key=key, Range='bytes=0-2048')
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
        
    except Exception as e:
        return False

def extract_and_verify_tables(input_file: str, output_file: str):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Could not find {input_file}")
        return

    # 1. Setup filters and S3 (Includes .zip)
    target_extensions = ('.txt', '.csv', '.tsv', '.jsonl', '.zip')
    ignore_files = ('metadata', 'catalog.', 'cc-by', 'readme', 'iso.', 'index.')
    s3 = _get_s3_client()
    
    valid_s3_uris = []

    print(f"1. Scanning {input_file} for candidate files...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip(): continue
            
            try:
                record = json.loads(line)
                key = record.get("Key", "")
                key_lower = key.lower()
                
                # FILTER 1: Must have a table-like extension or be a zip
                if not key_lower.endswith(target_extensions):
                    continue
                    
                # FILTER 2: Must not be a known junk/administrative file
                filename = key_lower.split('/')[-1]
                if any(junk in filename for junk in ignore_files):
                    continue
                
                full_uri = f"s3://{BUCKET}/{key}"

                # Only process text files with the S3 peek
                print(f"  -> Peeking at candidate: {filename}...")
                if is_actual_table(s3, BUCKET, key):
                    valid_s3_uris.append(full_uri)
                    print(f"     ✅ VERIFIED: {full_uri}")
                else:
                    print("     ❌ Rejected (Failed structural check)")
                    
            except json.JSONDecodeError:
                continue

    # Deduplicate and sort
    sorted_uris = sorted(list(set(valid_s3_uris)))

    print(f"\n2. Found {len(sorted_uris)} VERIFIED tables out of the whole data lake.")
    print(f"3. Saving DuckDB URIs to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as out:
        for uri in sorted_uris:
            out.write(f"{uri}\n")
            
    print("Done! You can now feed this text file directly into your DuckDB S3Reader. 🎉")

if __name__ == "__main__":
    load_dotenv()
    
    input_jsonl = sys.argv[1] if len(sys.argv) > 1 else "file_listing.jsonl"
    output_txt = sys.argv[2] if len(sys.argv) > 2 else "verified_s3_tables.txt"
    
    extract_and_verify_tables(input_jsonl, output_txt)