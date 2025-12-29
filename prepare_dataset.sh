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
echo "Please manually download PartNet-Mobility Dataset from:"
echo "  https://sapien.ucsd.edu/downloads"
echo ""
echo "After downloading, extract it to: $SCRIPT_DIR/partnet-mobility-v0/"
echo ""
echo "Expected structure:"
echo "  partnet-mobility-v0/"
echo "      |__148/"
echo "      |__149/"
echo "      ..."
echo ""

# Check if partnet-mobility-v0 exists
if [ -d "partnet-mobility-v0" ]; then
    echo "PartNet-Mobility dataset found!"
else
    echo "WARNING: PartNet-Mobility dataset not found at $SCRIPT_DIR/partnet-mobility-v0/"
    echo "Please download it manually."
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

