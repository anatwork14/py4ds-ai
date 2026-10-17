from pathlib import Path
import os
import pandas as pd
from PIL import Image, ImageStat
import imagehash

def process_single_image(filepath):
    file_bytes = os.path.getsize(filepath)
    if file_bytes == 0:
        return (filepath, None, None, None, None, file_bytes, None, None, None, None, None, None, None)
        
    try:
        # Pass 1 & Only: Pillow extracts dimensions, channels, hash, AND pixel stats
        with Image.open(filepath) as img:
            width, height = img.size
            channels = len(img.getbands())
            dtype = 'uint8' if img.mode in ('RGB', 'L', 'RGBA') else img.mode
            phash = str(imagehash.phash(img))
            
            # Use Pillow's native stat engine
            stats = ImageStat.Stat(img)
            
            if len(stats.mean) >= 3:
                mean_r, mean_g, mean_b = stats.mean[:3]
                std_r, std_g, std_b = stats.stddev[:3]
            elif len(stats.mean) == 1: # Grayscale fallback
                mean_r = mean_g = mean_b = stats.mean[0]
                std_r = std_g = std_b = stats.stddev[0]
            else:
                mean_r = mean_g = mean_b = std_r = std_g = std_b = None

        return (filepath, width, height, channels, dtype, file_bytes, phash, 
                mean_r, mean_g, mean_b, std_r, std_g, std_b)
        
    except Exception:
        return (filepath, None, None, None, None, file_bytes, None, None, None, None, None, None, None)

def build_master_dataframe(dataset_info):
    # 1. Map data into a basic list of dictionaries
    data = []
    for split_key, split_path in dataset_info.items():
        if not isinstance(split_path, Path) or not split_path.is_dir() or split_key in ['dataset_dir']:
            continue
        split_name = split_key.replace('_dir', '').replace('seg_', '')
        for filepath in split_path.rglob("*"):
            if filepath.is_file():
                data.append({
                    "filepath": str(filepath),
                    "split": split_name,
                    "class": filepath.parent.name if filepath.parent != split_path else 'Unknown',
                    "filename": filepath.name
                })

    df = pd.DataFrame(data)
    print(f"Extracting metrics for {len(df)} images.")
    results = [process_single_image(f) for f in df['filepath']]

    # 4. Merge back into the DataFrame
    cols = ['filepath', 'width', 'height', 'channels', 'dtype', 'file_bytes', 'phash', 
            'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b']
    stats_df = pd.DataFrame(results, columns=cols)

    df = df.merge(stats_df.drop(columns=['filepath']), left_index=True, right_index=True)
    return df

def clean_dataset(df):
    from IPython.display import display
    print(f"Original dataset size: {len(df)}\n")

    # Step 1: Resolution Analysis & Cleaning
    valid_images = df.dropna(subset=['width', 'height']).copy()
    valid_images['aspect_ratio'] = valid_images['width'] / valid_images['height']

    total_valid = len(valid_images)
    perfect_150 = valid_images[(valid_images['width'] == 150) & (valid_images['height'] == 150) & (valid_images['class'] != 'Unknown')]
    outliers = valid_images[(valid_images['aspect_ratio'] != 1.0) & (valid_images['class'] != 'Unknown')]

    # 1A. Breakdown
    print(f"Total Valid Images: {total_valid}")
    print(f"Images at exactly 150x150: {len(perfect_150)} ({(len(perfect_150)/total_valid) * 100:.2f}%)")
    print(f"Total Non-Square Images Detected: {len(outliers)}\n")

    # Calculate counts as Pandas Series
    counts_150 = perfect_150['class'].value_counts().rename("150x150 Count")
    counts_outlier = outliers['class'].value_counts().rename("Outlier Count")

    # Combine them side-by-side, fill missing outlier counts with 0, and convert to integers
    combined_breakdown = pd.concat([counts_150, counts_outlier], axis=1).fillna(0).astype(int)

    print("Class Breakdown Summary:")
    display(combined_breakdown)

    # 1B. Action (Filter down to only perfect squares)
    clean_df = perfect_150.copy() 
    print(f"-> Removed {len(df) - len(clean_df)} invalid, unknown, or non-square images.\n")

    # Step 2: Duplication Analysis & Cleaning
    # 2A. Breakdown (Count ONLY the redundant copies that need to be deleted)
    redundant_count = clean_df.duplicated(subset=['phash'], keep='first').sum()
    print(f"Detected {redundant_count} redundant duplicate copies in the dataset.")

    if redundant_count > 0:
        # We use keep=False here just to build the visual leakage table
        all_duplicates = clean_df[clean_df.duplicated(subset=['phash'], keep=False)]
        pivot_dup = all_duplicates.pivot_table(index='phash', columns='split', aggfunc='size', fill_value=0)
        
        print("Cross-Split Leakage (Images appearing in multiple splits):")
        display(pivot_dup[pivot_dup.sum(axis=1) > 1].head(10))

    # 2B. Action (Prioritize Test/Val, then drop the redundant copies)
    split_priority = {'test': 1, 'val': 2, 'train': 3}
    clean_df['priority'] = clean_df['split'].map(split_priority)
    clean_df = clean_df.sort_values('priority')

    clean_df = clean_df.drop_duplicates(subset=['phash'], keep='first').drop(columns=['priority'])

    print(f"-> Removed {redundant_count} redundant images (Prioritized keeping test/val instances).\n")
    print(f"Final Cleaned Dataset Size: {len(clean_df)}\n")

    return clean_df
