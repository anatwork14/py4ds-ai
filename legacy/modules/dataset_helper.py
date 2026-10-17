from pathlib import Path
import zipfile
import shutil
import random
import json
import pandas as pd

def zip_extraction(project_root: Path):
    dataset_dir = project_root / "dataset"
    zip_files = list(dataset_dir.glob("*.zip"))
    zip_path = zip_files[0] if zip_files else dataset_dir / "archive.zip"

    # FIXED: We are now targeting the clean, single-level directories
    seg_test_dir = dataset_dir / "seg_test"
    seg_train_dir = dataset_dir / "seg_train"
    seg_val_dir = dataset_dir / "seg_val"

    if seg_val_dir.exists() and seg_train_dir.exists() and seg_test_dir.exists():
        print("[INFO] Dataset is already extracted, flattened, and split. Skipping re-extraction.")
        return {
            "dataset_dir": dataset_dir,
            "zip_path": zip_path,
            "seg_test_dir": seg_test_dir,
            "seg_train_dir": seg_train_dir,
            "seg_val_dir": seg_val_dir
        }

    # 1. Check zip file
    if not zip_path.exists():
        print(f"[ERROR] No zip file found in {dataset_dir}")
        print("[EDA] canceled.")
        return None

    # 2. Extract zip
    try:
        print(f"[INFO] Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"[INFO] Extraction completed into: {dataset_dir}")

    except zipfile.BadZipFile:
        print(f"[ERROR] '{zip_path.name}' is not a valid ZIP file or is corrupted.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to extract zip file: {e}")
        return None

    # 3. Extracting information from the nested structure
    print("[INFO] Flattening nested directory structures...")
    for split_name in ["seg_train", "seg_test", "seg_pred"]:
        outer_dir = dataset_dir / split_name
        inner_dir = outer_dir / split_name # This is the pesky double folder
        
        if inner_dir.exists() and inner_dir.is_dir():
            # Move every class folder (buildings, forest, etc.) up one level
            for item in inner_dir.iterdir():
                shutil.move(str(item), str(outer_dir / item.name))
            
            # Delete the empty inner folder shell
            inner_dir.rmdir()

    # 4. Cleanup and Split
    # Delete seg_pred completely since we don't need it for this EDA
    seg_pred_base = dataset_dir / "seg_pred"
    if seg_pred_base.exists():
        shutil.rmtree(str(seg_pred_base))

    if not seg_val_dir.exists():
        print(f"[INFO] Creating validation set (15% of train) at {seg_val_dir}...")
        seg_val_dir.mkdir(parents=True, exist_ok=True)
        random.seed(42)  # For reproducible splits
        
        for class_dir in seg_train_dir.iterdir():
            if class_dir.is_dir():
                val_class_dir = seg_val_dir / class_dir.name
                val_class_dir.mkdir(parents=True, exist_ok=True)
                
                images = list(class_dir.glob("*.jpg"))
                images.sort()  # Sort to ensure reproducibility before shuffle
                random.shuffle(images)
                
                split_idx = int(len(images) * 0.15)
                val_images = images[:split_idx]
                
                for img_path in val_images:
                    shutil.move(str(img_path), str(val_class_dir / img_path.name))
        print("[INFO] Stratified validation set created.")

    return {
        "dataset_dir": dataset_dir,
        "zip_path": zip_path,
        "seg_test_dir": seg_test_dir,
        "seg_train_dir": seg_train_dir,
        "seg_val_dir": seg_val_dir
    }

def save_clean_dataset_and_metadata(clean_df, project_root, norm_mean_r, norm_mean_g, norm_mean_b, norm_std_r, norm_std_g, norm_std_b):
    # Bundle the metadata into a clean dictionary
    dataset_metadata = {
        "normalization": {
            "mean": [norm_mean_r, norm_mean_g, norm_mean_b],
            "std": [norm_std_r, norm_std_g, norm_std_b]
        },
        "input_size": [150, 150],
        "classes": sorted(clean_df['class'].unique().tolist()),
        "total_clean_images": len(clean_df),
        "splits": {
            "train": "seg_train",
            "val": "seg_val",
            "test": "seg_test"
        }
    }

    # Put into the dataset folder for better organization
    # dataset_path = project_root / "dataset"
    metadata_filename = "dataset_metadata.json"
    metadata_path = project_root / metadata_filename

    # Export metadata to JSON and save alongside the dataset
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=4)
        
    print(f"Metadata (with split paths) successfully saved to {metadata_path}")

def load_dataset_from_directories(project_root: Path) -> pd.DataFrame:
    """Reads the dataset splits directly from directories to build a lightweight DataFrame."""
    dataset_path = project_root / "dataset"
    metadata_path = project_root / "dataset_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        splits = metadata.get("splits", {"train": "seg_train", "val": "seg_val", "test": "seg_test"})
    else:
        splits = {"train": "seg_train", "val": "seg_val", "test": "seg_test"}
        
    data = []
    for split_name, split_folder in splits.items():
        split_dir = dataset_path / split_folder
        if not split_dir.exists():
            continue
            
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for filepath in class_dir.glob("*"):
                if filepath.is_file() and filepath.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    data.append({
                        "filepath": str(filepath),
                        "split": split_name,
                        "class": class_dir.name
                    })
                    
    df = pd.DataFrame(data)
    print(f"Loaded dataset with {len(df)} images.")
    return df
