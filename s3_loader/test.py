import os
import boto3
from dotenv import load_dotenv

BUCKET = "lakeqa-yc4103-datalake"
REGION = "us-east-1"
# ⚠️ REPLACE THIS WITH A REAL DATASET ID FROM YOUR TEXT FILE ⚠️
TEST_DATASET_ID = "0-2-second-spectral-response-acceleration-5-of-critical-damping-with-a-1-probability-of-ex-0397e" 

def test_single_dataset():
    load_dotenv()
    s3 = boto3.client('s3', region_name=REGION, 
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    
    # Check if the prefix is right
    folder = "datagov" 
    # folder = "harvard-lil/gov-data/collections/data_gov" # Try this if 'datagov' returns 0 files
    
    prefix = f"{folder}/{TEST_DATASET_ID}/"
    print(f"🔍 Searching S3 for Prefix: {prefix}")
    
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    
    if 'Contents' not in response:
        print("❌ S3 RETURNED 0 FILES. Your prefix is wrong!")
        return

    files = response['Contents']
    print(f"✅ Found {len(files)} files. Let's look at them:\n")
    
    for obj in files:
        key = obj['Key']
        print(f"📄 File: {key}")
        
        if key.endswith('/'):
            print("   -> Ignored (It's a directory)")
        elif not key.endswith('.json'):
            print("   -> Ignored (Not a .json file)")
        else:
            print("   -> 🟢 IT'S A JSON! Let's peek...")
            peek = s3.get_object(Bucket=BUCKET, Key=key, Range='bytes=0-500')
            content = peek['Body'].read().decode('utf-8', errors='ignore').strip()
            print(f"   -> Starts with: {content[:50]}...")
            
test_single_dataset()