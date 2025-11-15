"""
Calculate average number of images per species with minimum threshold filtering.

This script analyzes the plant_data directory and calculates:
1. Total images per species (across all plant parts)
2. Average images per species when filtering out species with fewer than N images
"""

import os
from pathlib import Path
from collections import defaultdict


def count_images_per_species(plant_data_dir):
    """
    Count the total number of images for each species.

    Args:
        plant_data_dir: Path to the plant_data directory

    Returns:
        Dictionary mapping species names to image counts
    """
    species_counts = defaultdict(int)
    plant_data_path = Path(plant_data_dir)

    # Valid image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Iterate through each species directory
    for species_dir in plant_data_path.iterdir():
        if not species_dir.is_dir():
            continue

        species_name = species_dir.name

        # Count images in all subdirectories (flower, leaf, bark, fruit, whole_plant)
        for subdir in species_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Count image files in this subdirectory
            for file in subdir.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    species_counts[species_name] += 1

    return dict(species_counts)


def calculate_average_with_threshold(species_counts, min_images):
    """
    Calculate average images per species after filtering by minimum threshold.

    Args:
        species_counts: Dictionary mapping species names to image counts
        min_images: Minimum number of images required (species with fewer are excluded)

    Returns:
        Tuple of (average, filtered_counts, removed_species)
    """
    # Filter species with at least min_images
    filtered_counts = {
        species: count
        for species, count in species_counts.items()
        if count >= min_images
    }

    # Calculate average
    if filtered_counts:
        average = sum(filtered_counts.values()) / len(filtered_counts)
    else:
        average = 0

    # Get list of removed species
    removed_species = [
        (species, count)
        for species, count in species_counts.items()
        if count < min_images
    ]
    removed_species.sort(key=lambda x: x[1])  # Sort by count

    return average, filtered_counts, removed_species


def main():
    """Main function to run the analysis."""
    # Set up paths
    script_dir = Path(__file__).parent
    plant_data_dir = script_dir / "plant_data"

    if not plant_data_dir.exists():
        print(f"Error: plant_data directory not found at {plant_data_dir}")
        return

    print("Counting images per species...")
    species_counts = count_images_per_species(plant_data_dir)

    total_species = len(species_counts)
    total_images = sum(species_counts.values())
    overall_avg = total_images / total_species if total_species > 0 else 0

    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total species: {total_species}")
    print(f"Total images: {total_images}")
    print(f"Overall average: {overall_avg:.2f} images per species")

    # Get user input for threshold
    print(f"\n{'='*70}")

    while True:
        try:
            min_images_input = input("\nEnter minimum number of images (N) [or 'q' to quit]: ").strip()

            if min_images_input.lower() == 'q':
                break

            min_images = int(min_images_input)

            if min_images < 0:
                print("Please enter a non-negative number.")
                continue

            # Calculate with threshold
            average, filtered_counts, removed_species = calculate_average_with_threshold(
                species_counts, min_images
            )

            print(f"\n{'='*70}")
            print(f"RESULTS FOR N = {min_images}")
            print(f"{'='*70}")
            print(f"Species retained: {len(filtered_counts)}")
            print(f"Species removed: {len(removed_species)}")
            print(f"Total images (retained species): {sum(filtered_counts.values())}")
            print(f"Average images per species: {average:.2f}")

            if removed_species:
                print(f"\nRemoved species (showing up to 10):")
                for species, count in removed_species[:10]:
                    print(f"  - {species}: {count} images")
                if len(removed_species) > 10:
                    print(f"  ... and {len(removed_species) - 10} more")

            # Show distribution stats for retained species
            if filtered_counts:
                counts_list = list(filtered_counts.values())
                print(f"\nRetained species distribution:")
                print(f"  Min images: {min(counts_list)}")
                print(f"  Max images: {max(counts_list)}")
                print(f"  Median images: {sorted(counts_list)[len(counts_list)//2]}")

        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
