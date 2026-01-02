#!/bin/bash
set -e

# ============================================
# iTACO Dataset Preparation Script
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "iTACO Dataset Preparation Script"
echo "============================================"

# Install huggingface_hub if needed
echo "Checking huggingface_hub..."
pip install -q huggingface_hub

# ============================================
# 1. Download Synthetic Dataset from HuggingFace
# ============================================
echo ""
echo "[1/3] Downloading synthetic dataset from HuggingFace..."
echo "Dataset: https://huggingface.co/datasets/3dlg-hcvc/video2articulation"
echo ""
echo "NOTE: This dataset requires authentication."
echo "If you haven't logged in, please:"
echo "  1. Visit https://huggingface.co/datasets/3dlg-hcvc/video2articulation"
echo "  2. Click 'Request access' and accept terms"
echo "  3. Run: python -c \"from huggingface_hub import login; login()\""
echo "  4. Or set HF_TOKEN environment variable"
echo ""

# Create directories
mkdir -p sim_data
mkdir -p real_data

# Download the dataset using python
python -c "
import os
from huggingface_hub import snapshot_download, get_token

# Check if logged in
token = os.environ.get('HF_TOKEN') or get_token()
if not token:
    print('ERROR: Not logged in to HuggingFace!')
    print('Please run: python -c \"from huggingface_hub import login; login()\"')
    print('Or set HF_TOKEN environment variable')
    exit(1)

print('Downloading dataset (this may take a while)...')
snapshot_download(
    repo_id='3dlg-hcvc/video2articulation',
    repo_type='dataset',
    local_dir='./hf_dataset',
    token=token
)
print('Download completed!')
"

# Move files to correct locations
if [ -d "./hf_dataset/sim_data" ]; then
    cp -r ./hf_dataset/sim_data/* ./sim_data/
fi

if [ -d "./hf_dataset/real_data" ]; then
    cp -r ./hf_dataset/real_data/* ./real_data/
fi

echo "Synthetic dataset downloaded successfully!"

# ============================================
# 2. Download PartNet-Mobility Dataset
# ============================================
echo ""
echo "[2/3] PartNet-Mobility Dataset"
echo "============================================"

# Check if partnet-mobility-v0 exists
if [ -d "partnet-mobility-v0" ]; then
    echo "PartNet-Mobility dataset already exists!"
else
    echo "PartNet-Mobility dataset not found. Attempting to download..."
    echo ""
    
    # Check for SAPIEN_TOKEN (default token available)
    SAPIEN_TOKEN="${SAPIEN_TOKEN:-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjIxMDMyMDgyNkBzdHUuaGl0LmVkdS5jbiIsImlwIjoiMTcyLjIwLjAuMSIsInByaXZpbGVnZSI6MSwiZmlsZU9ubHkiOnRydWUsImlhdCI6MTc2NzMzNzQwNywiZXhwIjoxNzk4ODczNDA3fQ.DDrq2p4DY22P7AZOdJhp3kxJsgGMt4cAzQaUXLo5tyA}"
    
    if [ -z "$SAPIEN_TOKEN" ]; then
        echo "SAPIEN_TOKEN not set."
        echo ""
        echo "To download PartNet-Mobility Dataset:"
        echo "  1. Register at https://sapien.ucsd.edu/downloads (use .edu email)"
        echo "  2. Wait for verification"
        echo "  3. Get your API token from the website"
        echo "  4. Set: export SAPIEN_TOKEN='your_token_here'"
        echo "  5. Re-run this script"
        echo ""
        echo "Or manually download and extract to: $SCRIPT_DIR/partnet-mobility-v0/"
        echo ""
    else
        echo "Found SAPIEN_TOKEN. Downloading PartNet-Mobility dataset..."
        pip install -q sapien
        
        python -c "
import sapien
import os

token = os.environ.get('SAPIEN_TOKEN')
if not token:
    print('ERROR: SAPIEN_TOKEN not set')
    exit(1)

# List of common model IDs to download (you can modify this list)
# Full dataset has ~2000+ models, here we download essential ones
model_ids = [
    # Microwave
    7265, 7263, 7236,
    # Refrigerator
    10638, 10905, 11231,
    # Dishwasher
    12085, 12349,
    # Oven
    7119, 7128, 7167,
    # Cabinet/Storage
    45245, 45332, 46123,
    # Door
    8867, 8877, 8893,
    # Drawer
    45623, 46440, 46877,
    # Laptop
    10211, 10239, 10306,
    # Washing Machine
    7236, 7310,
]

print(f'Downloading {len(model_ids)} models...')
os.makedirs('partnet-mobility-v0', exist_ok=True)

for i, model_id in enumerate(model_ids):
    try:
        print(f'[{i+1}/{len(model_ids)}] Downloading model {model_id}...')
        urdf_file = sapien.asset.download_partnet_mobility(model_id, token)
        print(f'  Downloaded to: {urdf_file}')
    except Exception as e:
        print(f'  Failed to download model {model_id}: {e}')

print('PartNet-Mobility download completed!')
"
    fi
fi

# ============================================
# 3. Verify Directory Structure
# ============================================
echo ""
echo "[3/3] Verifying directory structure..."
echo "============================================"

check_dir() {
    if [ -d "$1" ]; then
        echo "[OK] $1"
    else
        echo "[MISSING] $1"
    fi
}

echo ""
echo "Expected directories:"
check_dir "docs"
check_dir "partnet-mobility-v0"
check_dir "sim_data"
check_dir "sim_data/partnet_mobility"
check_dir "sim_data/exp_results"
check_dir "sim_data/exp_results/preprocessing"
check_dir "real_data"
check_dir "real_data/raw_data"
check_dir "real_data/exp_results"
check_dir "real_data/exp_results/preprocessing"

echo ""
echo "============================================"
echo "Dataset preparation completed!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. If PartNet-Mobility is missing, download from https://sapien.ucsd.edu/downloads"
echo "2. Run preprocessing (optional) or use pre-processed data from HuggingFace"
echo "3. Follow README.md for coarse prediction and refinement"

