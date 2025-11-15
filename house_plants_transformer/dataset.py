"""
House Plant Dataset Loader

Loads house plant images organized by species from the folder structure:
    house_plant_species_224/
        African Violet (Saintpaulia ionantha)/
            1.jpg
            2.jpg
            ...
        Aloe Vera/
            1.jpg
            2.jpg
            ...
        ...

Supports:
- Automatic train/val/test splitting
- Data augmentation
- Class balancing
- Species classification (no organ types - just species)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict
import json


class HousePlantDataset(Dataset):
    """
    Dataset for house plant images with species labels.

    Simple species-only classification (unlike the broad transformer which has organ types).
    """

    def __init__(
        self,
        data_dir,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        transform=None,
        seed=42,
        min_images_per_species=20
    ):
        """
        Args:
            data_dir: Path to house_plant_species_224 directory
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            transform: Image transformations
            seed: Random seed for reproducible splits
            min_images_per_species: Minimum images required per species (default: 20)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.min_images_per_species = min_images_per_species

        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"

        # Load all image paths and create labels
        self.samples = []
        self.species_to_idx = {}

        # Scan directory structure
        self._load_dataset(train_ratio, val_ratio, test_ratio, seed)

        print(f"\n{split.upper()} Dataset:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of species: {len(self.species_to_idx)}")

    def _load_dataset(self, train_ratio, val_ratio, test_ratio, seed):
        """Load and split dataset."""
        random.seed(seed)

        # Get all species directories
        species_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        # First pass: count images per species to filter if needed
        species_image_counts = {}

        if self.min_images_per_species > 0:
            for species_dir in species_dirs:
                species_name = species_dir.name

                # Get all image files
                # Use a set to avoid duplicates from case-insensitive filesystems
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                image_files_set = set()
                for ext in image_extensions:
                    image_files_set.update(species_dir.glob(ext))

                species_image_counts[species_name] = len(image_files_set)

        # Second pass: collect all species that meet the threshold
        species_idx_counter = 0
        filtered_species_count = 0

        for species_dir in species_dirs:
            species_name = species_dir.name

            # Filter species by minimum image count
            if self.min_images_per_species > 0:
                if species_image_counts[species_name] < self.min_images_per_species:
                    filtered_species_count += 1
                    continue

            self.species_to_idx[species_name] = species_idx_counter
            species_idx = species_idx_counter
            species_idx_counter += 1

            # Get all image files in this species directory
            # Use a set to avoid duplicates from case-insensitive filesystems
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files_set = set()
            for ext in image_extensions:
                image_files_set.update(species_dir.glob(ext))
            image_files = list(image_files_set)

            if len(image_files) == 0:
                continue

            # Shuffle images for this species
            random.shuffle(image_files)

            # Calculate split indices
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # Split images
            if self.split == 'train':
                selected_files = image_files[:n_train]
            elif self.split == 'val':
                selected_files = image_files[n_train:n_train + n_val]
            else:  # test
                selected_files = image_files[n_train + n_val:]

            # Add to samples
            for img_path in selected_files:
                self.samples.append((img_path, species_idx))

        # Shuffle all samples
        random.shuffle(self.samples)

        # Report filtering statistics
        if self.min_images_per_species > 0 and self.split == 'train':
            print(f"\n  Filtered out {filtered_species_count} species with < {self.min_images_per_species} images")
            print(f"  Kept {len(self.species_to_idx)} species with >= {self.min_images_per_species} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image: (3, img_size, img_size) tensor
            label: species index (int)
        """
        img_path, species_idx = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, species_idx

    def get_class_weights(self):
        """
        Calculate class weights for balanced training.

        Returns:
            Tensor of shape (num_classes,) with weights inversely proportional to class frequency
        """
        num_classes = len(self.species_to_idx)
        class_counts = torch.zeros(num_classes)

        for _, species_idx in self.samples:
            class_counts[species_idx] += 1

        # Calculate weights (inverse frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (class_counts + 1e-6)

        # Clamp weights to avoid extreme values
        min_weight = class_weights[class_counts > 0].min()
        max_weight = min_weight * 10.0
        class_weights = torch.clamp(class_weights, min=1.0, max=max_weight)

        # Normalize so average weight is 1.0
        class_weights = class_weights / class_weights.mean()

        return class_weights

    def save_label_mapping(self, filepath):
        """Save species label mappings to JSON."""
        mapping = {
            'species_to_idx': self.species_to_idx,
            'idx_to_species': {v: k for k, v in self.species_to_idx.items()},
            'num_species': len(self.species_to_idx),
            'min_images_per_species': self.min_images_per_species,
        }

        with open(filepath, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"Label mapping saved to {filepath}")


def get_train_transforms(img_size=224):
    """
    Data augmentation for training.

    Includes:
    - Random resized crop
    - Random horizontal flip
    - Color jitter
    - Random rotation
    - Normalization

    Args:
        img_size: Target image size (default: 224)
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(img_size=224):
    """
    Transforms for validation/test (no augmentation).

    Args:
        img_size: Target image size (default: 224)
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=4,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    img_size=224,
    seed=42,
    min_images_per_species=20
):
    """
    Create train, validation, and test dataloaders for house plants.

    Args:
        data_dir: Path to house_plant_species_224 directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        img_size: Image size (default: 224)
        seed: Random seed
        min_images_per_species: Minimum images required per species (default: 20)

    Returns:
        train_loader, val_loader, test_loader, species_to_idx
    """
    # Create datasets
    train_dataset = HousePlantDataset(
        data_dir=data_dir,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=get_train_transforms(img_size=img_size),
        seed=seed,
        min_images_per_species=min_images_per_species
    )

    val_dataset = HousePlantDataset(
        data_dir=data_dir,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=get_val_transforms(img_size=img_size),
        seed=seed,
        min_images_per_species=min_images_per_species
    )

    test_dataset = HousePlantDataset(
        data_dir=data_dir,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=get_val_transforms(img_size=img_size),
        seed=seed,
        min_images_per_species=min_images_per_species
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Save label mapping from train dataset
    train_dataset.save_label_mapping('house_plants_label_mapping.json')

    return train_loader, val_loader, test_loader, train_dataset.species_to_idx


if __name__ == "__main__":
    # Test the dataset
    print("Testing HousePlantDataset...")

    data_dir = Path("house_plant_species_224")

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist!")
        print("Please run resize_images.py first to create the 224x224 dataset.")
    else:
        # Create dataloaders
        train_loader, val_loader, test_loader, species_to_idx = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            num_workers=0,  # Use 0 for testing on Windows
            min_images_per_species=20
        )

        print(f"\nDataLoader test:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")

        print("\n✓ Dataset test completed successfully!")
