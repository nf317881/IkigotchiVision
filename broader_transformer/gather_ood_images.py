"""
Script to gather Out-of-Distribution (OOD) plant images from GBIF data files.

This script reads the same occurrence.txt and multimedia.txt files used for training,
but selects images from genera that are NOT in the training data to test how
confidence thresholding handles unknown/unseen genera.

Usage:
    python gather_ood_images.py --num_images 1000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
from io import BytesIO
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures


def load_trained_genera(label_mapping_path):
    """Load the list of genera in the training data."""
    with open(label_mapping_path, 'r') as f:
        data = json.load(f)
    return set(data['genus_to_id'].keys())


def parse_gbif_data(occurrence_file, multimedia_file):
    """
    Parse GBIF occurrence and multimedia files.

    Returns:
        DataFrame with columns: gbifID, genus, species, image_url
    """
    print("\n" + "="*80)
    print("Parsing GBIF Data Files")
    print("="*80)

    # Parse occurrence data
    print(f"\nReading occurrence file: {occurrence_file}")
    print("This may take several minutes...")

    usecols = ['gbifID', 'genus', 'species', 'scientificName', 'issue', 'mediaType']

    chunk_size = 100000
    filtered_chunks = []

    for chunk in tqdm(pd.read_csv(
        occurrence_file,
        sep='\t',
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False
    ), desc="Processing occurrence chunks"):

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

        # Drop unnecessary columns
        chunk = chunk[['gbifID', 'genus', 'species', 'scientificName']]
        filtered_chunks.append(chunk)

    occurrence_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"  Loaded {len(occurrence_df):,} occurrence records with {occurrence_df['genus'].nunique():,} unique genera")

    # Parse multimedia data
    print(f"\nReading multimedia file: {multimedia_file}")

    usecols = ['gbifID', 'type', 'identifier']

    multimedia_df = pd.read_csv(
        multimedia_file,
        sep='\t',
        usecols=usecols,
        low_memory=False
    )

    # Filter for StillImage
    multimedia_df = multimedia_df[multimedia_df['type'] == 'StillImage']
    multimedia_df = multimedia_df[multimedia_df['identifier'].notna()]
    multimedia_df = multimedia_df.rename(columns={'identifier': 'image_url'})
    multimedia_df = multimedia_df[['gbifID', 'image_url']]

    print(f"  Loaded {len(multimedia_df):,} image records")

    # Join occurrence and multimedia
    print("\nJoining data...")
    df = pd.merge(occurrence_df, multimedia_df, on='gbifID', how='inner')

    print(f"  Combined: {len(df):,} records")

    return df


def identify_ood_genera(df, trained_genera, min_images=50):
    """
    Identify OOD genera (not in training) with sufficient images.

    Returns:
        List of (genus, count) tuples sorted by count descending
    """
    print("\n" + "="*80)
    print("Identifying Out-of-Distribution (OOD) Genera")
    print("="*80)

    # Count images per genus
    genus_counts = df['genus'].value_counts()

    # Filter to OOD genera with sufficient images
    ood_genera = []
    for genus, count in genus_counts.items():
        if genus not in trained_genera and count >= min_images:
            ood_genera.append((genus, count))

    # Sort by count descending
    ood_genera.sort(key=lambda x: x[1], reverse=True)

    print(f"\nFound {len(ood_genera)} OOD genera with >= {min_images} images")
    print(f"\nTop 20 OOD genera:")
    for idx, (genus, count) in enumerate(ood_genera[:20], 1):
        print(f"  {idx:3d}. {genus:20s}: {count:,} images")

    return ood_genera


def sample_ood_images(df, ood_genera, target_images):
    """
    Sample images from OOD genera.

    Strategy:
    - Sample proportionally from each genus (more images from abundant genera)
    - Within each genus, sample randomly

    Returns:
        DataFrame with sampled images
    """
    print("\n" + "="*80)
    print("Sampling OOD Images")
    print("="*80)

    total_ood_images = sum(count for _, count in ood_genera)
    print(f"\nTotal OOD images available: {total_ood_images:,}")
    print(f"Target to sample: {target_images:,}")

    sampled_dfs = []
    genus_samples = {}

    for genus, count in ood_genera:
        # Calculate proportional sample size
        proportion = count / total_ood_images
        n_to_sample = int(np.ceil(proportion * target_images))

        # Don't sample more than available
        n_to_sample = min(n_to_sample, count)

        # Sample from this genus
        genus_df = df[df['genus'] == genus]
        sampled = genus_df.sample(n=n_to_sample, random_state=42)
        sampled_dfs.append(sampled)

        genus_samples[genus] = n_to_sample

    # Combine all samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # If we have too many, randomly sample down to target
    if len(sampled_df) > target_images:
        sampled_df = sampled_df.sample(n=target_images, random_state=42)

    print(f"\nSampled {len(sampled_df):,} images from {len(genus_samples)} genera")
    print(f"\nImages per genus:")

    # Count actual samples (after potential downsampling)
    actual_counts = sampled_df['genus'].value_counts()
    for genus in sorted(actual_counts.index):
        print(f"  {genus:20s}: {actual_counts[genus]:4d} images")

    return sampled_df


def center_crop_and_resize(image, size=224):
    """Center crop image to square and resize to target size."""
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


def download_and_process_image(url, save_path, timeout=30, max_retries=3):
    """
    Download an image from URL, process it, and save to disk.

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    for attempt in range(max_retries):
        try:
            # Download image
            response = requests.get(
                url,
                timeout=timeout,
                headers={'User-Agent': 'GBIF-Dataset-Processor/1.0'}
            )
            response.raise_for_status()

            # Open image
            image = Image.open(BytesIO(response.content))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Check minimum size
            if image.size[0] < 100 or image.size[1] < 100:
                return False, "Image too small"

            # Process: center crop and resize
            image = center_crop_and_resize(image, size=224)

            # Save
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path, 'JPEG', quality=95)

            return True, ""

        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return False, error_msg

    return False, "Max retries exceeded"


def download_images(df, output_dir, max_workers=15):
    """
    Download and process all images in parallel.

    Returns:
        List of metadata for successfully downloaded images
    """
    print("\n" + "="*80)
    print("Downloading and Processing Images")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nDownloading {len(df):,} images with {max_workers} parallel workers...")
    print(f"Images will be processed to 224x224 pixels")

    # Prepare download tasks
    tasks = []
    for idx, row in df.iterrows():
        genus = row['genus']
        filename = f"{genus}_{idx:06d}.jpg"
        save_path = output_dir / filename

        # Skip if already exists
        if save_path.exists():
            tasks.append({
                'idx': idx,
                'genus': genus,
                'species': row['species'],
                'url': row['image_url'],
                'filename': filename,
                'save_path': save_path,
                'skipped': True,
                'success': True
            })
        else:
            tasks.append({
                'idx': idx,
                'genus': genus,
                'species': row['species'],
                'url': row['image_url'],
                'filename': filename,
                'save_path': save_path,
                'skipped': False,
                'success': None
            })

    # Count skipped
    skipped_count = sum(1 for t in tasks if t['skipped'])
    print(f"Found {skipped_count:,} already downloaded images (skipping)")

    # Process only non-skipped tasks
    tasks_to_process = [t for t in tasks if not t['skipped']]

    successful_metadata = []
    failed_count = 0

    if len(tasks_to_process) > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_task = {
                executor.submit(
                    download_and_process_image,
                    task['url'],
                    task['save_path']
                ): task
                for task in tasks_to_process
            }

            # Process completed downloads
            with tqdm(total=len(tasks_to_process), desc="Downloading") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    success, error = future.result()

                    task['success'] = success

                    if success:
                        successful_metadata.append({
                            'filename': task['filename'],
                            'genus': task['genus'],
                            'species': task['species'],
                            'url': task['url']
                        })
                    else:
                        failed_count += 1

                    pbar.update(1)

    # Add skipped tasks to successful
    for task in tasks:
        if task['skipped']:
            successful_metadata.append({
                'filename': task['filename'],
                'genus': task['genus'],
                'species': task['species'],
                'url': task['url']
            })

    print(f"\n[OK] Image download complete:")
    print(f"  Successful: {len(successful_metadata):,}")
    print(f"  Failed: {failed_count:,}")
    if len(successful_metadata) + failed_count > 0:
        success_rate = 100.0 * len(successful_metadata) / (len(successful_metadata) + failed_count)
        print(f"  Success rate: {success_rate:.1f}%")

    return successful_metadata


def main():
    parser = argparse.ArgumentParser(description='Gather OOD plant images from GBIF data')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Target number of OOD images to download')
    parser.add_argument('--output_dir', type=str, default='ood_test_images',
                        help='Directory to save OOD images')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing GBIF data files')
    parser.add_argument('--label_mapping', type=str,
                        default='processed_data/metadata/label_mapping.json',
                        help='Path to label mapping file')
    parser.add_argument('--min_images_per_genus', type=int, default=50,
                        help='Minimum images required for OOD genus')
    parser.add_argument('--max_workers', type=int, default=15,
                        help='Number of parallel download workers')
    args = parser.parse_args()

    print("="*80)
    print("OOD Image Gathering from GBIF Data")
    print("="*80)

    # Check if data files exist
    data_dir = Path(args.data_dir)
    occurrence_file = data_dir / "occurrence.txt"
    multimedia_file = data_dir / "multimedia.txt"

    if not occurrence_file.exists():
        print(f"\n[ERROR] Occurrence file not found: {occurrence_file}")
        print("Please download GBIF data to the data/ directory")
        return

    if not multimedia_file.exists():
        print(f"\n[ERROR] Multimedia file not found: {multimedia_file}")
        print("Please download GBIF data to the data/ directory")
        return

    # Load trained genera
    print("\nLoading trained genera...")
    trained_genera = load_trained_genera(args.label_mapping)
    print(f"  Found {len(trained_genera)} genera in training data")

    # Parse GBIF data
    df = parse_gbif_data(occurrence_file, multimedia_file)

    # Identify OOD genera
    ood_genera = identify_ood_genera(df, trained_genera, min_images=args.min_images_per_genus)

    if len(ood_genera) == 0:
        print("\n[ERROR] No OOD genera found with sufficient images!")
        return

    # Sample images from OOD genera
    sampled_df = sample_ood_images(df, ood_genera, args.num_images)

    # Download images
    metadata = download_images(sampled_df, args.output_dir, max_workers=args.max_workers)

    # Save metadata
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / 'metadata.json'

    ood_genera_list = sorted(set(item['genus'] for item in metadata))

    metadata_dict = {
        'num_images': len(metadata),
        'ood_genera': ood_genera_list,
        'images': metadata
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"  Successfully downloaded: {len(metadata)} images")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata saved: {metadata_path}")

    # Count by genus
    genus_counts = defaultdict(int)
    for item in metadata:
        genus_counts[item['genus']] += 1

    print(f"\n  Images per genus:")
    for genus in sorted(genus_counts.keys()):
        count = genus_counts[genus]
        print(f"    {genus}: {count} images")

    print("\n" + "="*80)
    print("Done! Ready to test OOD confidence thresholds.")
    print("="*80)


if __name__ == '__main__':
    main()
