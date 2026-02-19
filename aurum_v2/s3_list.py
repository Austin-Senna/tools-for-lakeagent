import os
import re
import sys
from typing import Optional, Dict, Any, List
import boto3
import requests
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
FOLDERS = ["datagov"]
REGION = "us-east-1"

def _get_s3_client():
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower() if text else "")

def _score_by_query(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text: return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens: return 0.0
    query_set = set(query_tokens)
    common = query_set.intersection(text_tokens)
    if not common: return 0.0
    return (len(common) / len(query_set) * 0.8) + (len(common) / len(text_tokens) * 0.2)

def _dataset_exists(s3, folder: str, dataset_id: str) -> bool:
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{folder}/{dataset_id}/", MaxKeys=1)
    return "Contents" in response or "CommonPrefixes" in response

def _search_datagov_packages(query: str) -> List[Dict[str, Any]]:
    url = "https://catalog.data.gov/api/3/action/package_search"
    response = requests.get(url, params={"q": query}, headers={"User-Agent": "DataLakeAgentTools/1.0"}, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("result", {}).get("results", [])

def search_keyword(keywords: List[str], limit: Optional[int] = None) -> Dict[str, Any]:
    s3 = _get_s3_client()
    results, seen_ids = [], set()

    for keyword in keywords:
        query_tokens = _tokenize(keyword)
        try: datagov_hits = _search_datagov_packages(keyword)
        except Exception: datagov_hits = []

        for item in datagov_hits:
            name = item.get("name") or ""
            title = item.get("title") or name
            if name and _dataset_exists(s3, "datagov", name):
                if name not in seen_ids:
                    seen_ids.add(name)
                    results.append({
                        "title": title,
                        "dataset_id": name,
                        "score": _score_by_query(query_tokens, f"{title} {name}"),
                        "folder": "datagov"
                    })

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return {"results": results[:limit] if limit else results}

def list_files(dataset_ids: List[str], limit: int = 100) -> Dict[str, Any]:
    s3 = _get_s3_client()
    all_files = []
    
    for dataset_id in dataset_ids:
        # Assuming datagov based on strict FOLDERS config
        folder = "datagov" 
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{folder}/{dataset_id}/", MaxKeys=limit)

        if 'Contents' in response:
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    all_files.append({
                        'path': obj['Key'].split(f"{folder}/{dataset_id}/", 1)[-1],
                        'dataset_id': dataset_id,
                        'folder': folder
                    })
    return {'files': all_files}
import re
def _is_table_by_peeking(s3, folder: str, dataset_id: str, file_path: str) -> bool:
    """
    Downloads the first 2KB of a file from S3. 
    Uses consistency heuristics and blocklists to isolate true tables.
    """
    # DEFENSE 1: The Artifact Blocklist
    # Ignore standard CKAN/data.gov administrative files
    lower_path = file_path.lower()
    ignore_list = ['metadata.json', 'signed-metadata', 'catalog.', 'cc-by', 'iso.xml', 'iso.txt', 'readme']
    if any(skip in lower_path for skip in ignore_list):
        return False

    try:
        s3_key = f"{folder}/{dataset_id}/{file_path}"
        response = s3.get_object(Bucket=BUCKET, Key=s3_key, Range='bytes=0-2048')
        content = response['Body'].read().decode('utf-8', errors='ignore')
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:
            return False

        # DEFENSE 2: Strict JSONL Check
        # Does every line independently start and end with brackets?
        json_lines = sum(1 for line in lines[:5] if line.startswith('{') and line.endswith('}'))
        if json_lines >= 3:
            return True
            
        # DEFENSE 3: The Pretty-Printed JSON Trap
        # If it starts with a bracket but wasn't caught by the JSONL check above, 
        # it is a pretty-printed metadata file. Reject it.
        first_char = content.strip()[0]
        if first_char in ['{', '[']:
            return False

        # DEFENSE 4: 3-Line Delimiter Consistency (CSV / TSV / Pipes)
        delimiters = [',', '\t', '|', ';']
        for delim in delimiters:
            counts = []
            for line in lines[:5]:
                # Strip out text inside double quotes so we only count structural delimiters
                clean_line = re.sub(r'"[^"]*"', '', line)
                counts.append(clean_line.count(delim))
            
            valid_counts = [c for c in counts if c > 0]
            
            # Require at least 3 lines with the exact same number of delimiters
            if len(valid_counts) >= 3 and len(set(valid_counts[:3])) == 1:
                # Ensure it has at least 1 delimiter (meaning 2+ columns)
                if valid_counts[0] >= 1:
                    return True
                
        return False
        
    except Exception:
        # Ignore binary files or access errors
        return False
    
def get_s3_uris_for_tables(dataset_ids: List[str], limit_per_dataset: int = 100) -> List[str]:
    """Returns a flat list of s3:// URIs ready for DuckDB to consume."""
    files_data = list_files(dataset_ids, limit=limit_per_dataset).get('files', [])
    s3 = _get_s3_client()
    s3_uris = []
    
    for f in files_data:
        # Skip known non-tables
        if f['path'].lower().endswith(('.jpg', '.png', '.pdf', '.zip')):
            continue
            
        if _is_table_by_peeking(s3, f['folder'], f['dataset_id'], f['path']):
            # Format the exact URI DuckDB needs
            uri = f"s3://{BUCKET}/{f['folder']}/{f['dataset_id']}/{f['path']}"
            s3_uris.append(uri)
            
    return s3_uris

if __name__ == "__main__":
    load_dotenv()
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    
    print(f"1. Searching data.gov for '{query}'...")
    search_results = search_keyword([query], limit=3)
    dataset_ids = [res['dataset_id'] for res in search_results.get('results', [])]
    
    print(f"2. Peeking inside {len(dataset_ids)} datasets to find tables...")
    table_uris = get_s3_uris_for_tables(dataset_ids)
    
    print("\n--- FINAL S3 URIs TO FEED THE PROFILER ---")
    for uri in table_uris:
        print(uri)