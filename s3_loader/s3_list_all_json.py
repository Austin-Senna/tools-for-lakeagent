import os
import sys
import boto3
import concurrent.futures
from botocore.config import Config
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
REGION = "us-east-1"
MAX_THREADS = 50 

def _get_s3_client():
    boto_config = Config(max_pool_connections=MAX_THREADS)
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        config=boto_config
    )

def list_files(s3, dataset_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    all_files = []
    folder = "datagov" 
    prefix = f"{folder}/{dataset_id}/"
    
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=limit)

    if 'Contents' in response:
        for obj in response['Contents']:
            if not obj['Key'].endswith('/'):
                all_files.append({
                    'path': obj['Key'].split(prefix, 1)[-1],
                    'dataset_id': dataset_id,
                    'folder': folder,
                    'full_key': obj['Key']
                })
    return all_files

def _is_valid_json_table(s3, file_path: str) -> bool:
    """
    Downloads the first 50KB of a JSON file.
    Uses structural fingerprints to identify plain arrays, GeoJSON, and Socrata JSON.
    """
    lower_path = file_path.lower()
    ignore_list = ['metadata.json', 'signed-metadata']
    if any(skip in lower_path for skip in ignore_list):
        return False
    
    try:
        # 50KB is large enough to catch Socrata metadata and GeoJSON headers, 
        # but small enough to remain incredibly cheap/fast.
        response = s3.get_object(Bucket=BUCKET, Key=file_path, Range='bytes=0-51200')
        content = response['Body'].read().decode('utf-8', errors='ignore').strip()
        
        if not content:
            return False

        # RULE 1: Plain JSON array
        if content.startswith('['):
            return True

        if content.startswith('{'):
            # RULE 2: Socrata JSON format (checks for meta.view.columns)
            if '"meta"' in content and '"view"' in content and '"columns"' in content:
                return True
            
            # RULE 3: GeoJSON FeatureCollection
            if '"FeatureCollection"' in content and '"features"' in content:
                return True
                
        return False
        
    except Exception:
        return False

def process_single_dataset(s3, dataset_id: str) -> tuple:
    files_data = list_files(s3, dataset_id)
    s3_uris = []
    
    for f in files_data:
        lower_path = f['path'].lower()
        
        # We only care about .json files for this specific script
        if not lower_path.endswith('.json'):
            continue
            
        uri = f"s3://{BUCKET}/{f['full_key']}"
            
        if _is_valid_json_table(s3, f['full_key']):
            s3_uris.append(uri)
            
    return dataset_id, s3_uris

def extract_and_verify_json(input_txt, output_txt, log_txt="processed_datasets_json.log"):
    if not Path(input_txt).exists():
        print(f"Error: Could not find {input_txt}")
        return

    processed_datasets = set()
    if Path(log_txt).exists():
        with open(log_txt, 'r') as log:
            processed_datasets = set(line.strip() for line in log if line.strip())
            
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
    
    with open(output_txt, 'a', encoding='utf-8') as out_file, \
         open(log_txt, 'a', encoding='utf-8') as log_file:
             
        print(f"\nSpinning up {MAX_THREADS} threads to hunt for JSON tables...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            
            future_to_dataset = {
                executor.submit(process_single_dataset, s3_client, ds): ds 
                for ds in pending_datasets
            }
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                completed_count += 1
                
                try:
                    _, verified_uris = future.result()
                    
                    for uri in verified_uris:
                        out_file.write(f"{uri}\n")
                        print(f"  ✅ VERIFIED JSON: {uri}")
                    
                    log_file.write(f"{dataset_id}\n")
                    
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
    # Outputting to a dedicated JSON file so your CSV list stays safe
    output_txt = sys.argv[2] if len(sys.argv) > 2 else "verified_s3_json_tables.txt"
    
    extract_and_verify_json(input_txt, output_txt)