#!/usr/bin/env python3
"""
iTACO Dataset Preparation Script

This script downloads and prepares the dataset for iTACO project.
- Downloads synthetic dataset from HuggingFace
- Provides instructions for PartNet-Mobility dataset
- Verifies directory structure
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def install_huggingface_hub():
    """Install huggingface_hub if not present."""
    try:
        from huggingface_hub import snapshot_download
        return True
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub[cli]")
        return True

def download_hf_dataset(local_dir: Path):
    """Download dataset from HuggingFace."""
    from huggingface_hub import snapshot_download
    
    print("\n[1/3] Downloading synthetic dataset from HuggingFace...")
    print("Dataset: https://huggingface.co/datasets/3dlg-hcvc/video2articulation")
    
    hf_cache_dir = local_dir / "hf_dataset"
    
    snapshot_download(
        repo_id="3dlg-hcvc/video2articulation",
        repo_type="dataset",
        local_dir=str(hf_cache_dir),
        local_dir_use_symlinks=False,
    )
    
    # Move files to correct locations
    sim_data_src = hf_cache_dir / "sim_data"
    real_data_src = hf_cache_dir / "real_data"
    
    if sim_data_src.exists():
        sim_data_dst = local_dir / "sim_data"
        sim_data_dst.mkdir(parents=True, exist_ok=True)
        for item in sim_data_src.iterdir():
            dst = sim_data_dst / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        print("  - sim_data downloaded successfully!")
    
    if real_data_src.exists():
        real_data_dst = local_dir / "real_data"
        real_data_dst.mkdir(parents=True, exist_ok=True)
        for item in real_data_src.iterdir():
            dst = real_data_dst / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        print("  - real_data downloaded successfully!")
    
    print("Synthetic dataset downloaded successfully!")

def check_partnet_mobility(local_dir: Path):
    """Check if PartNet-Mobility dataset exists."""
    print("\n[2/3] PartNet-Mobility Dataset")
    print("=" * 50)
    
    partnet_dir = local_dir / "partnet-mobility-v0"
    
    if partnet_dir.exists():
        num_objects = len([d for d in partnet_dir.iterdir() if d.is_dir()])
        print(f"PartNet-Mobility dataset found! ({num_objects} objects)")
        return True
    else:
        print("Please manually download PartNet-Mobility Dataset from:")
        print("  https://sapien.ucsd.edu/downloads")
        print()
        print(f"After downloading, extract it to: {partnet_dir}/")
        print()
        print("Expected structure:")
        print("  partnet-mobility-v0/")
        print("      |__148/")
        print("      |__149/")
        print("      ...")
        print()
        print("WARNING: PartNet-Mobility dataset not found!")
        return False

def verify_structure(local_dir: Path):
    """Verify the directory structure."""
    print("\n[3/3] Verifying directory structure...")
    print("=" * 50)
    
    expected_dirs = [
        "docs",
        "partnet-mobility-v0",
        "sim_data",
        "sim_data/partnet_mobility",
        "sim_data/exp_results",
        "sim_data/exp_results/preprocessing",
        "real_data",
        "real_data/raw_data",
        "real_data/exp_results",
        "real_data/exp_results/preprocessing",
    ]
    
    print("\nExpected directories:")
    all_ok = True
    for dir_path in expected_dirs:
        full_path = local_dir / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}")
        else:
            print(f"  [MISSING] {dir_path}")
            all_ok = False
    
    return all_ok

def main():
    parser = argparse.ArgumentParser(description="iTACO Dataset Preparation Script")
    parser.add_argument("--skip-download", action="store_true", 
                        help="Skip downloading HuggingFace dataset")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify directory structure")
    args = parser.parse_args()
    
    print("=" * 50)
    print("iTACO Dataset Preparation Script")
    print("=" * 50)
    
    # Get project root directory
    script_dir = Path(__file__).parent.resolve()
    
    if args.verify_only:
        verify_structure(script_dir)
        return
    
    # Create necessary directories
    (script_dir / "sim_data").mkdir(exist_ok=True)
    (script_dir / "real_data").mkdir(exist_ok=True)
    
    # Download HuggingFace dataset
    if not args.skip_download:
        install_huggingface_hub()
        download_hf_dataset(script_dir)
    
    # Check PartNet-Mobility
    check_partnet_mobility(script_dir)
    
    # Verify structure
    verify_structure(script_dir)
    
    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. If PartNet-Mobility is missing, download from https://sapien.ucsd.edu/downloads")
    print("2. Run preprocessing (optional) or use pre-processed data from HuggingFace")
    print("3. Follow README.md for coarse prediction and refinement")

if __name__ == "__main__":
    main()

