import os
import re
import sys
import tempfile
import traceback
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any, List
import boto3
import requests
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
# CHANGED: We now strictly limit the search to datagov
FOLDERS = ["datagov"]
REGION = "us-east-1"

SANDBOX_BASE_DIR = Path(__file__).resolve().parent / ".sandbox"
_SANDBOX_DIR = None
_SANDBOX_OVERRIDE = None


def _resolve_dataset_folder(dataset_id: str) -> Optional[str]:
    """Resolve dataset folder for a dataset_id."""
    if not dataset_id:
        return None
    s3 = _get_s3_client()
    matches = []
    for folder in FOLDERS:
        if _dataset_exists(s3, folder, dataset_id):
            matches.append(folder)
    if len(matches) == 1:
        return matches[0]
    return None

def _dataset_exists(s3, folder: str, dataset_id: str) -> bool:
    """Check whether a dataset exists under a given folder."""
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"{folder}/{dataset_id}/",
        MaxKeys=1
    )
    return "Contents" in response or "CommonPrefixes" in response

def _get_sandbox_dir() -> Path:
    """Get or create the sandbox directory for downloaded files."""
    global _SANDBOX_DIR, _SANDBOX_OVERRIDE
    if _SANDBOX_OVERRIDE is not None:
        _SANDBOX_DIR = _SANDBOX_OVERRIDE
        return _SANDBOX_DIR

    if _SANDBOX_DIR is None or not _SANDBOX_DIR.exists():
        SANDBOX_BASE_DIR.mkdir(parents=True, exist_ok=True)
        _SANDBOX_DIR = Path(tempfile.mkdtemp(prefix="task_", dir=SANDBOX_BASE_DIR))
    return _SANDBOX_DIR

def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text) if text else ""
    return re.findall(r"[a-z0-9]+", text.lower())

def _get_s3_client():
    """Get authenticated S3 client."""
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def _score_by_query(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    common = query_set.intersection(text_tokens)
    if not common:
        return 0.0
    coverage = len(common) / len(query_set)
    density = len(common) / len(text_tokens)
    return (coverage * 0.8) + (density * 0.2)

def _search_datagov_packages(query: str) -> List[Dict[str, Any]]:
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": query}
    headers = {"User-Agent": "DataLakeAgentTools/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError("data.gov search failed")
    return data.get("result", {}).get("results", [])

def search_keyword(keywords: List[str], limit: Optional[int] = None) -> Dict[str, Any]:
    """Tag-style keyword search filtered by S3 existence (Datagov ONLY)."""
    if not isinstance(keywords, list):
        return {'error': "keywords must be a list of strings."}
    
    for kw in keywords:
        if not kw or not kw.strip():
            return {'error': "All keywords must be non-empty."}

    s3 = _get_s3_client()
    results = []
    seen_ids = set()
    results_by_keyword = {}

    for keyword in keywords:
        query_tokens = _tokenize(keyword)
        keyword_results = []

        # CHANGED: Removed Wikipedia search block. Now Datagov only.
        try:
            datagov_hits = _search_datagov_packages(keyword)
        except Exception:
            datagov_hits = []

        for item in datagov_hits:
            name = item.get("name") or ""
            title = item.get("title") or name
            if name and _dataset_exists(s3, "datagov", name):
                score_text = f"{title} {name}".strip()
                result_entry = {
                    "title": title,
                    "dataset_id": name,
                    "score": _score_by_query(query_tokens, score_text),
                }
                keyword_results.append(result_entry)
                if name not in seen_ids:
                    seen_ids.add(name)
                    results.append(result_entry)

        keyword_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        results_by_keyword[keyword] = [{"title": r.get("title", ""), "dataset_id": r.get("dataset_id", "")} for r in keyword_results]

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    if limit is not None and limit < len(results):
        results = results[:limit]

    cleaned = [{"title": r.get("title", ""), "dataset_id": r.get("dataset_id", "")} for r in results]

    return {
        "results": cleaned,
        "results_by_keyword": results_by_keyword,
        "count": len(results),
        "keywords": keywords
    }

def list_files(dataset_ids: List[str], limit: int = 100) -> Dict[str, Any]:
    """List files within one or more datasets/directories."""
    if not isinstance(dataset_ids, list):
        return {'error': "dataset_ids must be a list of strings."}

    s3 = _get_s3_client()
    all_files = []
    results_by_dataset = {}
    any_truncated = False

    for dataset_id in dataset_ids:
        folder = _resolve_dataset_folder(dataset_id)
        if folder is None:
            results_by_dataset[dataset_id] = {'error': f"Dataset not found: {dataset_id}"}
            continue

        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f"{folder}/{dataset_id}/",
            MaxKeys=limit
        )

        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    relative_path = obj['Key'].split(f"{folder}/{dataset_id}/", 1)[-1]
                    file_entry = {
                        'path': relative_path,
                        'size': obj['Size'],
                        'dataset_id': dataset_id,
                        'folder': folder  # Track folder for peeking later
                    }
                    files.append(file_entry)
                    all_files.append(file_entry)

        results_by_dataset[dataset_id] = {
            'files': files,
            'count': len(files),
            'truncated': response.get('IsTruncated', False)
        }
        if response.get('IsTruncated', False):
            any_truncated = True

    return {
        'files': all_files,
        'count': len(all_files),
        'dataset_ids': dataset_ids,
        'by_dataset': results_by_dataset,
        'truncated': any_truncated
    }

def _is_table_by_peeking(s3, folder: str, dataset_id: str, file_path: str) -> bool:
    """
    Downloads just the first 1KB of a file from S3. 
    Checks if the first few lines have common table delimiters.
    """
    # return True
    try:
        s3_key = f"{folder}/{dataset_id}/{file_path}"
        
        # Use HTTP Range to fetch only the first 1024 bytes (extremely fast)
        response = s3.get_object(Bucket=BUCKET, Key=s3_key, Range='bytes=0-1024')
        content = response['Body'].read().decode('utf-8', errors='ignore')
        
        # Look at the first 5 lines (sometimes files have a couple lines of metadata at top)
        lines = content.split('\n')
        for line in lines[:5]:
            # If a single line has > 1 comma, tab, or pipe, it is likely a data row/header
            if any(line.count(delim) > 1 for delim in [',', '\t', '|', ';']):
                return True
                
        # Also check for JSON Lines or JSON Array of objects
        if content.strip().startswith('[{') or content.strip().startswith('{"'):
            return True
            
        return False
    except Exception as e:
        # Ignore files we can't read (like true binary files)
        return False

def get_tables_from_datasets(dataset_ids: List[str], limit_per_dataset: int = 100) -> Dict[str, Any]:
    """Finds and filters only the table files by peeking at their contents."""
    file_response = list_files(dataset_ids, limit=limit_per_dataset)
    
    if 'error' in file_response:
        return file_response
        
    all_files = file_response.get('files', [])
    table_files = []
    
    s3 = _get_s3_client()
    
    print(f"\n--- Scanning {len(all_files)} files to see if they are tables... ---")
    
    for file_obj in all_files:
        path = file_obj['path']
        dataset_id = file_obj['dataset_id']
        folder = file_obj['folder']
        
        # We know images aren't tables, skip peeking at them to save time
        if path.lower().endswith(('.jpg', '.png', '.pdf', '.zip')):
            continue
            
        # PEEK AT THE FILE!
        if _is_table_by_peeking(s3, folder, dataset_id, path):
            table_files.append({
                'dataset_id': dataset_id,
                'file_path': path,
                'size': file_obj['size']
            })
            
    return {
        'table_files': table_files,
        'count': len(table_files),
        'dataset_ids_scanned': dataset_ids
    }

def download(files: List[Dict[str, str]]) -> Dict[str, Any]:
    """Download one or more files from S3 to the local sandbox directory."""
    if not isinstance(files, list):
        return {'error': "files must be a list of {dataset_id, file_path} objects"}
    if len(files) > 50:
        return {'error': "Maximum 5 files per download call"}
    if len(files) == 0:
        return {'error': "files list cannot be empty"}

    s3 = _get_s3_client()
    sandbox = _get_sandbox_dir()
    downloaded, errors = [], []

    for file_spec in files:
        dataset_id = file_spec.get('dataset_id', '')
        file_path = file_spec.get('file_path', '')

        if not dataset_id or not file_path:
            errors.append({'error': "dataset_id and file_path are required", 'file_spec': file_spec})
            continue

        folder = _resolve_dataset_folder(dataset_id)
        if folder is None:
            errors.append({'error': f"Dataset not found: {dataset_id}", 'file_path': file_path})
            continue

        s3_key = f"{folder}/{dataset_id}/{file_path.lstrip('/')}"
        local_path = sandbox / dataset_id / file_path.lstrip('/')
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(BUCKET, s3_key, str(local_path))
            downloaded.append({
                'local_path': str(local_path),
                'file_path': file_path,
                'dataset_id': dataset_id,
                'size': local_path.stat().st_size,
                'status': 'downloaded'
            })
        except Exception as e:
            errors.append({'error': str(e), 'dataset_id': dataset_id, 'file_path': file_path})

    result = {'downloaded': downloaded, 'download_count': len(downloaded), 'sandbox_dir': str(sandbox)}
    if errors: result['errors'] = errors
    return result

# =============================================================================
# Example Usage: End-to-End Pipeline
# =============================================================================
if __name__ == "__main__":
    load_dotenv()
    
    query = sys.argv[1] if len(sys.argv) > 1 else "climate"
    print(f"Searching for datasets matching: '{query}'...")
    
    # 1. Search datagov
    search_results = search_keyword([query], limit=3)
    found_datasets = [res['dataset_id'] for res in search_results.get('results', [])]
    
    if found_datasets:
        print(f"Found datasets: {found_datasets}")
        
        # 2. Filter JUST the tables by peering inside them
        tables_response = get_tables_from_datasets(found_datasets)
        tables_to_download = tables_response.get('table_files', [])
        
        print(f"\nFound {tables_response['count']} true tables by reading their headers.")
        
        # 3. Download the first batch of true tables
        if tables_to_download:
            batch_to_download = tables_to_download
            download_response = download(batch_to_download)
            
            print("\nDownload Results:")
            for success in download_response.get('downloaded', []):
                print(f" - Saved: {success['local_path']}")
    else:
        print("No datasets found.")