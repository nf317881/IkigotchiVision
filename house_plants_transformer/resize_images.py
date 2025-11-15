"""
Image Resizing Script for House Plants Dataset

Resizes all images in the house_plant_species directory to 224x224 pixels.
Preserves the original directory structure:
    house_plant_species/
        Species_name/
            1.jpg
            2.jpg
            ...

Creates a resized version:
    house_plant_species_224/
        Species_name/
            1.jpg
            2.jpg
            ...
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse


def resize_images(
    input_dir='house_plant_species',
    output_dir='house_plant_species_224',
    target_size=224,
    quality=95
):
    """
    Resize all images in the house plants dataset.

    Args:
        input_dir: Source directory with original images
        output_dir: Target directory for resized images
        target_size: Target size for both width and height (default: 224)
        quality: JPEG quality (default: 95)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Get all species directories
    species_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    print(f"Resizing images from {input_dir} to {output_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Found {len(species_dirs)} species")
    print("=" * 60)

    total_images = 0
    successful = 0
    failed = 0
    skipped = 0

    # Process each species directory
    for species_dir in tqdm(species_dirs, desc="Processing species"):
        species_name = species_dir.name

        # Create output species directory
        output_species_dir = output_path / species_name
        output_species_dir.mkdir(exist_ok=True)

        # Get all image files (support multiple formats)
        # Use a set to avoid duplicates from case-insensitive filesystems
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.jpe']
        image_files_set = set()
        for ext in image_extensions:
            image_files_set.update(species_dir.glob(ext))
        image_files = list(image_files_set)

        # Process each image
        for img_path in image_files:
            total_images += 1

            # Determine output filename (always save as .jpg for consistency)
            output_filename = img_path.stem + '.jpg'
            output_img_path = output_species_dir / output_filename

            # Skip if already exists
            if output_img_path.exists():
                skipped += 1
                continue

            try:
                # Open image
                with Image.open(img_path) as img:
                    # Convert to RGB (handles RGBA, grayscale, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize with high-quality resampling
                    # Use LANCZOS for best quality
                    img_resized = img.resize(
                        (target_size, target_size),
                        Image.Resampling.LANCZOS
                    )

                    # Save resized image
                    img_resized.save(output_img_path, 'JPEG', quality=quality)

                successful += 1

            except Exception as e:
                failed += 1
                print(f"\nError processing {img_path}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("Resizing Complete!")
    print("=" * 60)
    print(f"Total images found: {total_images}")
    print(f"Successfully resized: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\n⚠ Warning: {failed} images failed to process")

    print(f"\n✓ Resized images saved to: {output_path}")


def count_images(data_dir='house_plant_species'):
    """
    Count images per species in the dataset.

    Args:
        data_dir: Directory containing species folders
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' does not exist!")
        return

    species_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    print(f"\nImage count per species in {data_dir}:")
    print("=" * 60)

    total_images = 0
    species_counts = []

    for species_dir in species_dirs:
        species_name = species_dir.name

        # Get all image files
        # Use a set to avoid duplicates from case-insensitive filesystems
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.jpe']
        image_files_set = set()
        for ext in image_extensions:
            image_files_set.update(species_dir.glob(ext))
        image_files = list(image_files_set)

        count = len(image_files)
        species_counts.append((species_name, count))
        total_images += count

    # Sort by count (descending)
    species_counts.sort(key=lambda x: x[1], reverse=True)

    # Print counts
    for species_name, count in species_counts:
        print(f"  {species_name}: {count}")

    print("=" * 60)
    print(f"Total species: {len(species_dirs)}")
    print(f"Total images: {total_images}")
    print(f"Average images per species: {total_images / len(species_dirs):.1f}")
    print(f"Min images: {min(c for _, c in species_counts)}")
    print(f"Max images: {max(c for _, c in species_counts)}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Resize images for house plants dataset'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='house_plant_species',
        help='Input directory with original images (default: house_plant_species)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='house_plant_species_224',
        help='Output directory for resized images (default: house_plant_species_224)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target image size (default: 224)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality (default: 95)'
    )
    parser.add_argument(
        '--count_only',
        action='store_true',
        help='Only count images without resizing'
    )

    args = parser.parse_args()

    if args.count_only:
        count_images(args.input_dir)
    else:
        # First, show image counts
        count_images(args.input_dir)

        # Then resize
        print("\n")
        resize_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=args.size,
            quality=args.quality
        )


if __name__ == "__main__":
    main()
