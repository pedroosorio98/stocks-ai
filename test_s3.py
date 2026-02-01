# diagnose_credentials.py
import boto3
import os

print("="*70)
print("🔍 DIAGNOSING AWS CREDENTIALS")
print("="*70)

# Get credentials from environment
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_REGION')

print(f"\n📋 Using credentials:")
print(f"   Access Key: {access_key}")
print(f"   Secret Key: {secret_key[:8]}... (hidden)")
print(f"   Region: {region}")

print("\n" + "="*70)

# Test 1: Check which IAM user these credentials belong to
print("TEST 1: Identifying IAM User")
print("-"*70)
try:
    sts = boto3.client(
        'sts',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    identity = sts.get_caller_identity()
    
    print(f"✅ Credentials are VALID")
    print(f"   User ARN: {identity['Arn']}")
    print(f"   Account: {identity['Account']}")
    print(f"   User ID: {identity['UserId']}")
    
    # Extract username from ARN
    arn_parts = identity['Arn'].split('/')
    username = arn_parts[-1] if len(arn_parts) > 1 else "Unknown"
    print(f"   Username: {username}")
    
    if username == 'rag-s3-reader':
        print("\n   ✅ CORRECT! Using rag-s3-reader credentials")
    else:
        print(f"\n   ⚠️  WARNING! Using '{username}' NOT 'rag-s3-reader'")
        print("   You need to create new access keys for rag-s3-reader!")
    
except Exception as e:
    print(f"❌ CREDENTIALS INVALID: {e}")
    print("\n💡 Your AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is wrong!")
    print("   Generate new credentials in IAM Console")

print("\n" + "="*70)

# Test 2: List buckets
print("TEST 2: Listing S3 Buckets")
print("-"*70)
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    response = s3.list_buckets()
    print(f"✅ Can list buckets ({len(response['Buckets'])} found):")
    for bucket in response['Buckets']:
        print(f"   - {bucket['Name']}")
        
except Exception as e:
    print(f"❌ Cannot list buckets: {e}")

print("\n" + "="*70)

# Test 3: Check specific bucket
print("TEST 3: Accessing osorio-stocks-research Bucket")
print("-"*70)
bucket_name = 'osorio-stocks-research'
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    # Try to list objects
    response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
    
    if 'Contents' in response:
        print(f"✅ Can access bucket! Found {len(response['Contents'])} objects:")
        for obj in response['Contents'][:5]:
            print(f"   - {obj['Key']} ({obj['Size']} bytes)")
    else:
        print(f"⚠️  Bucket is empty or no objects found")
        
except Exception as e:
    print(f"❌ CANNOT ACCESS BUCKET: {e}")
    print("\n💡 Possible reasons:")
    print("   1. Bucket policy blocks this IAM user")
    print("   2. IAM user doesn't have s3:ListBucket permission")
    print("   3. Bucket doesn't exist or is in different region")

print("\n" + "="*70)

# Test 4: Try to get the specific file
print("TEST 4: Accessing internal.faiss File")
print("-"*70)
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    # Check if file exists
    response = s3.head_object(Bucket=bucket_name, Key='internal.faiss')
    
    print(f"✅ File exists!")
    print(f"   Size: {response['ContentLength']} bytes")
    print(f"   Last Modified: {response['LastModified']}")
    print(f"   ETag: {response['ETag']}")
    
    # Try to download just 1 byte to test permissions
    try:
        response = s3.get_object(Bucket=bucket_name, Key='internal.faiss', Range='bytes=0-0')
        print(f"\n✅ CAN DOWNLOAD! Permissions are correct!")
    except Exception as e:
        print(f"\n❌ CANNOT DOWNLOAD: {e}")
        
except Exception as e:
    print(f"❌ CANNOT ACCESS FILE: {e}")
    print("\n💡 Possible reasons:")
    print("   1. File doesn't exist at 'internal.faiss'")
    print("   2. File is in a subfolder (try 'folder/internal.faiss')")
    print("   3. IAM user doesn't have s3:GetObject permission")

print("\n" + "="*70)
print("📊 DIAGNOSIS COMPLETE")
print("="*70)
