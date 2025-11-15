"""
GBIF Plant Dataset Processing Pipeline

Processes GBIF occurrence and multimedia data to create a stratified dataset
for training a Vision Transformer on plant genus classification.

Target: 500 genera x 800 images each = ~400,000 images at 224x224 pixels
"""

import pandas as pd
import numpy as np
import json
import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict
import time
import concurrent.futures
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration parameters for the data processing pipeline"""

    # Paths
    DATA_DIR = Path("broader_transformer/data")
    OUTPUT_DIR = Path("broader_transformer/processed_data")
    OCCURRENCE_FILE = DATA_DIR / "occurrence.txt"
    MULTIMEDIA_FILE = DATA_DIR / "multimedia.txt"

    # Dataset parameters
    NUM_GENERA = 500
    IMAGES_PER_GENUS = 800
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Image parameters
    IMAGE_SIZE = 224
    JPEG_QUALITY = 95

    # Download parameters
    MAX_WORKERS = 15  # Parallel download workers
    DOWNLOAD_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    # Column indices (0-based)
    OCC_GBIFID_COL = 0
    OCC_GENUS_COL = 166
    OCC_SPECIES_COL = 204
    OCC_SCIENTIFICNAME_COL = 149
    OCC_ISSUE_COL = 188
    OCC_MEDIATYPE_COL = 190

    MUL_GBIFID_COL = 0
    MUL_TYPE_COL = 1
    MUL_IDENTIFIER_COL = 3  # Image URL
    MUL_DESCRIPTION_COL = 6  # Contains organ info
    MUL_LICENSE_COL = 13

    # Output structure
    IMAGE_DIR = OUTPUT_DIR / "images"
    METADATA_DIR = OUTPUT_DIR / "metadata"
    TRAIN_DIR = IMAGE_DIR / "train"
    VAL_DIR = IMAGE_DIR / "val"
    TEST_DIR = IMAGE_DIR / "test"


# ============================================================================
# Step 1: Parse and Filter Occurrence Data
# ============================================================================

def parse_occurrence_data(config: Config) -> pd.DataFrame:
    """
    Parse occurrence.txt and filter out records with taxonomic issues.

    Filters applied:
    - Exclude TAXON_MATCH_FUZZY
    - Exclude TAXON_MATCH_HIGHERRANK
    - Require mediaType = 'StillImage'

    Returns:
        DataFrame with columns: gbifID, genus, species, scientificName
    """
    print("=" * 80)
    print("STEP 1: Parsing and Filtering Occurrence Data")
    print("=" * 80)

    print(f"\nReading occurrence file: {config.OCCURRENCE_FILE}")
    print("This may take several minutes for 2.6M records...")

    # Read the file in chunks to manage memory
    chunk_size = 100000
    filtered_chunks = []
    total_rows = 0
    filtered_rows = 0

    # Column names we need (use names instead of indices to avoid confusion)
    usecols = ['gbifID', 'genus', 'species', 'scientificName', 'issue', 'mediaType']

    for chunk in tqdm(pd.read_csv(
        config.OCCURRENCE_FILE,
        sep='\t',
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False
    ), desc="Processing chunks"):

        total_rows += len(chunk)

        # Filter out taxonomic issues
        if 'issue' in chunk.columns:
            mask = ~chunk['issue'].astype(str).str.contains('TAXON_MATCH_FUZZY', na=False)
            chunk = chunk[mask]

            mask = ~chunk['issue'].astype(str).str.contains('TAXON_MATCH_HIGHERRANK', na=False)
            chunk = chunk[mask]

        # Filter for StillImage media type
        if 'mediaType' in chunk.columns:
            chunk = chunk[chunk['mediaType'] == 'StillImage']

        # Keep only records with valid genus
        chunk = chunk[chunk['genus'].notna()]
        chunk = chunk[chunk['genus'] != '']

        filtered_rows += len(chunk)

        # Drop issue and mediaType columns
        chunk = chunk[['gbifID', 'genus', 'species', 'scientificName']]

        filtered_chunks.append(chunk)

    # Combine all chunks
    print("\nCombining filtered chunks...")
    df = pd.concat(filtered_chunks, ignore_index=True)

    print(f"\n[OK] Occurrence data parsed:")
    print(f"  Total records read: {total_rows:,}")
    print(f"  Records after filtering: {filtered_rows:,}")
    print(f"  Records with valid genus: {len(df):,}")
    print(f"  Data loss: {((total_rows - len(df)) / total_rows * 100):.2f}%")
    print(f"  Unique genera: {df['genus'].nunique():,}")

    return df


# ============================================================================
# Step 2: Parse Multimedia Data
# ============================================================================

def parse_multimedia_data(config: Config) -> pd.DataFrame:
    """
    Parse multimedia.txt to extract image URLs and plant organ information.

    Returns:
        DataFrame with columns: gbifID, image_url, organ, license
    """
    print("\n" + "=" * 80)
    print("STEP 2: Parsing Multimedia Data")
    print("=" * 80)

    print(f"\nReading multimedia file: {config.MULTIMEDIA_FILE}")
    print("Extracting image URLs and plant organ information...")

    # Columns we need (use column names directly)
    usecols = ['gbifID', 'type', 'identifier', 'description', 'license']

    df = pd.read_csv(
        config.MULTIMEDIA_FILE,
        sep='\t',
        usecols=usecols,
        low_memory=False
    )

    # Rename identifier to image_url for clarity
    df = df.rename(columns={'identifier': 'image_url'})

    # Filter for StillImage type
    df = df[df['type'] == 'StillImage']

    # Extract plant organ from description
    # Format: "Scientific Name: organ"
    def extract_organ(desc):
        if pd.isna(desc):
            return 'unknown'
        parts = str(desc).split(':')
        if len(parts) >= 2:
            organ = parts[-1].strip().lower()
            return organ
        return 'unknown'

    df['organ'] = df['description'].apply(extract_organ)

    # Keep only necessary columns
    df = df[['gbifID', 'image_url', 'organ', 'license']]

    # Remove records without image URL
    df = df[df['image_url'].notna()]

    print(f"\n[OK] Multimedia data parsed:")
    print(f"  Total image records: {len(df):,}")
    print(f"  Unique gbifIDs: {df['gbifID'].nunique():,}")
    print(f"\n  Organ distribution:")
    organ_counts = df['organ'].value_counts()
    for organ, count in organ_counts.head(10).items():
        print(f"    {organ}: {count:,} ({count/len(df)*100:.1f}%)")

    return df


# ============================================================================
# Step 3: Join Occurrence and Multimedia Data
# ============================================================================

def join_data(occurrence_df: pd.DataFrame, multimedia_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join occurrence and multimedia data on gbifID.

    Returns:
        DataFrame with columns: gbifID, genus, species, scientificName, image_url, organ, license
    """
    print("\n" + "=" * 80)
    print("STEP 3: Joining Occurrence and Multimedia Data")
    print("=" * 80)

    print("\nPerforming inner join on gbifID...")

    df = pd.merge(
        occurrence_df,
        multimedia_df,
        on='gbifID',
        how='inner'
    )

    print(f"\n[OK] Data joined:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique genera: {df['genus'].nunique():,}")
    print(f"  Unique species: {df['species'].nunique():,}")

    return df


# ============================================================================
# Step 4: Identify Top N Genera
# ============================================================================

def identify_top_genera(df: pd.DataFrame, config: Config) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify top N genera by image count.

    Returns:
        Tuple of (list of genus names, DataFrame with genus statistics)
    """
    print("\n" + "=" * 80)
    print(f"STEP 4: Identifying Top {config.NUM_GENERA} Genera")
    print("=" * 80)

    print("\nCounting images per genus...")

    genus_counts = df.groupby('genus').size().reset_index(name='image_count')
    genus_counts = genus_counts.sort_values('image_count', ascending=False)

    # Select top N genera
    top_genera = genus_counts.head(config.NUM_GENERA)
    top_genus_list = top_genera['genus'].tolist()

    print(f"\n[OK] Top {config.NUM_GENERA} genera identified:")
    print(f"\nTop 10 genera:")
    for idx, row in top_genera.head(10).iterrows():
        print(f"  {row['genus']}: {row['image_count']:,} images")

    print(f"\nRanks around target threshold:")
    for rank in [100, 200, 300, 400, 500]:
        if rank <= len(top_genera):
            row = top_genera.iloc[rank - 1]
            print(f"  Rank {rank} ({row['genus']}): {row['image_count']:,} images")

    # Check if any genera have < target images
    below_target = top_genera[top_genera['image_count'] < config.IMAGES_PER_GENUS]
    if len(below_target) > 0:
        print(f"\n[WARNING] Warning: {len(below_target)} genera have < {config.IMAGES_PER_GENUS} images:")
        for idx, row in below_target.head(10).iterrows():
            print(f"    {row['genus']}: {row['image_count']:,} images")

    return top_genus_list, top_genera


# ============================================================================
# Step 5: Stratified Sampling
# ============================================================================

def stratified_sampling(
    df: pd.DataFrame,
    top_genera: List[str],
    config: Config
) -> pd.DataFrame:
    """
    Perform stratified sampling to select images for each genus.

    Stratification levels:
    1. Genus
    2. Species within genus
    3. Organ type within species

    Returns:
        DataFrame with sampled images and train/val/test split assignments
    """
    print("\n" + "=" * 80)
    print("STEP 5: Stratified Sampling")
    print("=" * 80)

    print(f"\nSampling {config.IMAGES_PER_GENUS} images per genus...")
    print("Stratification: genus -> species -> organ")

    # Filter to top genera only
    df_filtered = df[df['genus'].isin(top_genera)].copy()

    sampled_dfs = []
    genus_stats = []

    for genus in tqdm(top_genera, desc="Sampling genera"):
        genus_df = df_filtered[df_filtered['genus'] == genus].copy()
        total_images = len(genus_df)

        # If fewer images than target, use all
        if total_images <= config.IMAGES_PER_GENUS:
            sampled = genus_df
            n_sampled = total_images
        else:
            # Stratified sampling by species and organ
            # Calculate sampling proportions
            species_organ_counts = genus_df.groupby(['species', 'organ']).size()
            total_count = species_organ_counts.sum()

            # Sample proportionally from each species-organ combination
            sampled_parts = []

            for (species, organ), count in species_organ_counts.items():
                n_to_sample = int(np.ceil(count / total_count * config.IMAGES_PER_GENUS))

                subset = genus_df[
                    (genus_df['species'] == species) &
                    (genus_df['organ'] == organ)
                ]

                # Sample (with replacement if necessary)
                if len(subset) < n_to_sample:
                    sampled_subset = subset
                else:
                    sampled_subset = subset.sample(n=n_to_sample, random_state=42)

                sampled_parts.append(sampled_subset)

            # Combine and limit to target
            sampled = pd.concat(sampled_parts, ignore_index=True)
            if len(sampled) > config.IMAGES_PER_GENUS:
                sampled = sampled.sample(n=config.IMAGES_PER_GENUS, random_state=42)

            n_sampled = len(sampled)

        # Assign train/val/test splits
        n_train = int(n_sampled * config.TRAIN_RATIO)
        n_val = int(n_sampled * config.VAL_RATIO)
        n_test = n_sampled - n_train - n_val

        splits = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
        np.random.seed(42)
        np.random.shuffle(splits)
        sampled['split'] = splits

        sampled_dfs.append(sampled)

        # Collect statistics
        n_species = sampled['species'].nunique()
        organ_dist = sampled['organ'].value_counts().to_dict()

        genus_stats.append({
            'genus': genus,
            'total_available': total_images,
            'sampled': n_sampled,
            'n_species': n_species,
            'organ_distribution': organ_dist,
            'train': n_train,
            'val': n_val,
            'test': n_test
        })

    # Combine all sampled data
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)

    print(f"\n[OK] Sampling complete:")
    print(f"  Total images sampled: {len(df_sampled):,}")
    print(f"  Train: {len(df_sampled[df_sampled['split']=='train']):,}")
    print(f"  Val: {len(df_sampled[df_sampled['split']=='val']):,}")
    print(f"  Test: {len(df_sampled[df_sampled['split']=='test']):,}")

    # Save sampling statistics
    stats_path = config.METADATA_DIR / "sampling_statistics.json"
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(genus_stats, f, indent=2)
    print(f"\n  Statistics saved to: {stats_path}")

    return df_sampled


# ============================================================================
# Step 6: Download and Process Images
# ============================================================================

def center_crop_and_resize(image: Image.Image, size: int) -> Image.Image:
    """
    Center crop image to square and resize to target size.

    Args:
        image: PIL Image
        size: Target size (width and height)

    Returns:
        Processed PIL Image
    """
    width, height = image.size

    # Calculate crop dimensions (square centered)
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop to square
    image = image.crop((left, top, right, bottom))

    # Resize to target size
    image = image.resize((size, size), Image.Resampling.LANCZOS)

    return image


def download_and_process_image(
    url: str,
    save_path: Path,
    config: Config
) -> Tuple[bool, str]:
    """
    Download an image from URL, process it, and save to disk.

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    for attempt in range(config.MAX_RETRIES):
        try:
            # Download image
            response = requests.get(
                url,
                timeout=config.DOWNLOAD_TIMEOUT,
                headers={'User-Agent': 'GBIF-Dataset-Processor/1.0'}
            )
            response.raise_for_status()

            # Open image
            image = Image.open(BytesIO(response.content))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process: center crop and resize
            image = center_crop_and_resize(image, config.IMAGE_SIZE)

            # Save
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path, 'JPEG', quality=config.JPEG_QUALITY)

            return True, ""

        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{config.MAX_RETRIES} failed: {str(e)}"
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
            else:
                return False, error_msg

    return False, "Max retries exceeded"


def download_images(df: pd.DataFrame, config: Config, genus_to_id: Dict[str, int]) -> pd.DataFrame:
    """
    Download and process all images in parallel.

    Returns:
        DataFrame with successfully downloaded images
    """
    print("\n" + "=" * 80)
    print("STEP 6: Downloading and Processing Images")
    print("=" * 80)

    print(f"\nDownloading {len(df):,} images with {config.MAX_WORKERS} parallel workers...")
    print(f"Images will be processed to {config.IMAGE_SIZE}x{config.IMAGE_SIZE} pixels")
    print("\nThis will take several hours. Progress will be saved periodically.")

    # Prepare download tasks
    tasks = []
    for idx, row in df.iterrows():
        genus_id = genus_to_id[row['genus']]
        genus_name = row['genus']
        split = row['split']

        # Create filename
        image_id = f"{genus_id:03d}_{idx:06d}"
        filename = f"{image_id}.jpg"

        # Determine save directory
        split_dir = config.IMAGE_DIR / split / f"genus_{genus_id:03d}_{genus_name}"
        save_path = split_dir / filename

        # Skip if already exists
        if save_path.exists():
            tasks.append({
                'idx': idx,
                'url': row['image_url'],
                'save_path': save_path,
                'skipped': True,
                'success': True,
                'error': ""
            })
        else:
            tasks.append({
                'idx': idx,
                'url': row['image_url'],
                'save_path': save_path,
                'skipped': False,
                'success': None,
                'error': ""
            })

    # Download in parallel
    successful_indices = []
    failed_downloads = []
    skipped_count = sum(1 for t in tasks if t['skipped'])

    print(f"Found {skipped_count:,} already downloaded images (skipping)")

    # Process only non-skipped tasks
    tasks_to_process = [t for t in tasks if not t['skipped']]

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_task = {
            executor.submit(
                download_and_process_image,
                task['url'],
                task['save_path'],
                config
            ): task
            for task in tasks_to_process
        }

        # Process completed downloads
        with tqdm(total=len(tasks_to_process), desc="Downloading") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                success, error = future.result()

                task['success'] = success
                task['error'] = error

                if success:
                    successful_indices.append(task['idx'])
                else:
                    failed_downloads.append({
                        'url': task['url'],
                        'save_path': str(task['save_path']),
                        'error': error
                    })

                pbar.update(1)

    # Add skipped indices to successful
    successful_indices.extend([t['idx'] for t in tasks if t['skipped']])

    # Filter DataFrame to successful downloads
    df_success = df.loc[successful_indices].copy()

    # Add file paths to DataFrame
    file_paths = []
    for idx, row in df_success.iterrows():
        genus_id = genus_to_id[row['genus']]
        genus_name = row['genus']
        split = row['split']
        image_id = f"{genus_id:03d}_{idx:06d}"
        filename = f"{image_id}.jpg"

        rel_path = f"{split}/genus_{genus_id:03d}_{genus_name}/{filename}"
        file_paths.append(rel_path)

    df_success['file_path'] = file_paths

    # Save failed downloads log
    if failed_downloads:
        failed_path = config.METADATA_DIR / "failed_downloads.json"
        with open(failed_path, 'w') as f:
            json.dump(failed_downloads, f, indent=2)
        print(f"\n[WARNING] {len(failed_downloads):,} downloads failed - see {failed_path}")

    print(f"\n[OK] Image download complete:")
    print(f"  Successful: {len(df_success):,}")
    print(f"  Failed: {len(failed_downloads):,}")
    print(f"  Success rate: {len(df_success)/(len(df_success)+len(failed_downloads))*100:.1f}%")

    return df_success


# ============================================================================
# Step 7: Generate Metadata Files
# ============================================================================

def generate_metadata(
    df: pd.DataFrame,
    top_genera_df: pd.DataFrame,
    config: Config
) -> None:
    """
    Generate all metadata files and statistics.
    """
    print("\n" + "=" * 80)
    print("STEP 7: Generating Metadata Files")
    print("=" * 80)

    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Label mapping (genus_id → genus_name)
    print("\n1. Creating label mapping...")
    genus_to_id = {genus: idx for idx, genus in enumerate(sorted(df['genus'].unique()))}
    id_to_genus = {idx: genus for genus, idx in genus_to_id.items()}

    label_mapping = {
        'genus_to_id': genus_to_id,
        'id_to_genus': id_to_genus,
        'num_classes': len(genus_to_id)
    }

    label_path = config.METADATA_DIR / "label_mapping.json"
    with open(label_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"   [OK] Saved to: {label_path}")

    # 2. Create manifest CSVs for each split
    print("\n2. Creating data manifests...")

    for split in ['train', 'val', 'test']:
        df_split = df[df['split'] == split].copy()
        df_split['genus_id'] = df_split['genus'].map(genus_to_id)

        # Select columns for manifest
        manifest = df_split[[
            'file_path',
            'genus_id',
            'genus',
            'species',
            'organ',
            'image_url',
            'license'
        ]].copy()

        manifest_path = config.METADATA_DIR / f"{split}_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"   [OK] {split}: {len(manifest):,} images → {manifest_path}")

    # 3. Dataset statistics
    print("\n3. Generating dataset statistics...")

    stats = {
        'overview': {
            'total_images': len(df),
            'num_genera': len(genus_to_id),
            'num_species': df['species'].nunique(),
            'train_images': len(df[df['split'] == 'train']),
            'val_images': len(df[df['split'] == 'val']),
            'test_images': len(df[df['split'] == 'test']),
        },
        'per_genus_stats': {},
        'organ_distribution': df['organ'].value_counts().to_dict(),
        'split_distribution': {
            'train': len(df[df['split'] == 'train']),
            'val': len(df[df['split'] == 'val']),
            'test': len(df[df['split'] == 'test'])
        }
    }

    # Per-genus statistics
    for genus in sorted(genus_to_id.keys()):
        genus_df = df[df['genus'] == genus]
        stats['per_genus_stats'][genus] = {
            'genus_id': genus_to_id[genus],
            'total_images': len(genus_df),
            'num_species': genus_df['species'].nunique(),
            'train': len(genus_df[genus_df['split'] == 'train']),
            'val': len(genus_df[genus_df['split'] == 'val']),
            'test': len(genus_df[genus_df['split'] == 'test']),
            'organ_distribution': genus_df['organ'].value_counts().to_dict()
        }

    stats_path = config.METADATA_DIR / "dataset_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   [OK] Saved to: {stats_path}")

    # 4. Quality report
    print("\n4. Creating quality report...")

    report_lines = [
        "=" * 80,
        "GBIF PLANT DATASET - PROCESSING REPORT",
        "=" * 80,
        "",
        f"Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Target: {config.NUM_GENERA} genera x {config.IMAGES_PER_GENUS} images = {config.NUM_GENERA * config.IMAGES_PER_GENUS:,} total",
        "",
        "=" * 80,
        "DATASET OVERVIEW",
        "=" * 80,
        f"Total Images: {stats['overview']['total_images']:,}",
        f"Number of Genera: {stats['overview']['num_genera']}",
        f"Number of Species: {stats['overview']['num_species']:,}",
        "",
        "Split Distribution:",
        f"  Training:   {stats['overview']['train_images']:,} ({stats['overview']['train_images']/stats['overview']['total_images']*100:.1f}%)",
        f"  Validation: {stats['overview']['val_images']:,} ({stats['overview']['val_images']/stats['overview']['total_images']*100:.1f}%)",
        f"  Test:       {stats['overview']['test_images']:,} ({stats['overview']['test_images']/stats['overview']['total_images']*100:.1f}%)",
        "",
        "=" * 80,
        "ORGAN DISTRIBUTION",
        "=" * 80,
    ]

    for organ, count in sorted(stats['organ_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats['overview']['total_images'] * 100
        report_lines.append(f"  {organ:12s}: {count:7,} ({pct:5.1f}%)")

    report_lines.extend([
        "",
        "=" * 80,
        "TOP 20 GENERA (by image count)",
        "=" * 80,
    ])

    genus_image_counts = [(g, s['total_images']) for g, s in stats['per_genus_stats'].items()]
    genus_image_counts.sort(key=lambda x: x[1], reverse=True)

    for idx, (genus, count) in enumerate(genus_image_counts[:20], 1):
        genus_stats = stats['per_genus_stats'][genus]
        report_lines.append(
            f"  {idx:3d}. {genus:20s}: {count:4,} images "
            f"({genus_stats['num_species']:3d} species)"
        )

    report_lines.extend([
        "",
        "=" * 80,
        "GENERA WITH < TARGET IMAGES",
        "=" * 80,
    ])

    below_target = [(g, s['total_images']) for g, s in stats['per_genus_stats'].items()
                    if s['total_images'] < config.IMAGES_PER_GENUS]

    if below_target:
        below_target.sort(key=lambda x: x[1])
        for genus, count in below_target[:20]:
            report_lines.append(f"  {genus:20s}: {count:4,} images")
        if len(below_target) > 20:
            report_lines.append(f"  ... and {len(below_target) - 20} more")
    else:
        report_lines.append("  None - all genera have >= target images!")

    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    report_path = config.METADATA_DIR / "quality_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"   [OK] Saved to: {report_path}")

    print("\n[OK] All metadata files generated successfully!")

    return genus_to_id


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main execution pipeline"""

    config = Config()

    print("\n" + "=" * 80)
    print("GBIF PLANT DATASET PROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nTarget: {config.NUM_GENERA} genera x {config.IMAGES_PER_GENUS} images each")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE} pixels")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("\n" + "=" * 80)

    # Step 1: Parse occurrence data
    occurrence_df = parse_occurrence_data(config)

    # Step 2: Parse multimedia data
    multimedia_df = parse_multimedia_data(config)

    # Step 3: Join data
    combined_df = join_data(occurrence_df, multimedia_df)

    # Step 4: Identify top genera
    top_genera_list, top_genera_df = identify_top_genera(combined_df, config)

    # Step 5: Stratified sampling
    sampled_df = stratified_sampling(combined_df, top_genera_list, config)

    # Step 6: Create preliminary genus mapping for download paths
    genus_to_id = {genus: idx for idx, genus in enumerate(sorted(top_genera_list))}

    # Step 7: Download and process images
    final_df = download_images(sampled_df, config, genus_to_id)

    # Step 8: Generate metadata
    genus_to_id = generate_metadata(final_df, top_genera_df, config)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nFinal dataset:")
    print(f"  Total images: {len(final_df):,}")
    print(f"  Classes: {len(genus_to_id)}")
    print(f"  Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"\nOutput location: {config.OUTPUT_DIR}")
    print(f"  Images: {config.IMAGE_DIR}")
    print(f"  Metadata: {config.METADATA_DIR}")
    print("\nReady for training!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
