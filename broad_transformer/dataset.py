"""
Plant Dataset Loader

Loads plant images organized by species and organ type from the folder structure:
    plant_data/
        Species_name/
            flower/
                plantnet_0.jpg
                plantnet_1.jpg
                ...
            leaf/
                plantnet_0.jpg
                ...
            bark/
                ...
            fruit/
                ...
            whole_plant/
                harvard_0.jpg
                ...

Supports:
- Automatic train/val/test splitting
- Data augmentation
- Class balancing
- Multi-label classification (species + organ type)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict
import json


class PlantDataset(Dataset):
    """
    Dataset for plant images with species and organ type labels.

    Supports two classification modes:
    1. Species-only: Classify plant species (ignore organ type)
    2. Joint: Classify both species and organ type
    """

    def __init__(
        self,
        data_dir,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        transform=None,
        classification_mode='species',  # 'species' or 'joint'
        organ_types=['flower', 'leaf', 'bark', 'fruit', 'whole_plant'],
        seed=42,
        min_images_per_species=0
    ):
        """
        Args:
            data_dir: Path to plant_data directory
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            transform: Image transformations
            classification_mode: 'species' (species only) or 'joint' (species + organ)
            organ_types: List of organ types to include
            seed: Random seed for reproducible splits
            min_images_per_species: Minimum total images required per species (0 = no filtering)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.classification_mode = classification_mode
        self.organ_types = organ_types
        self.min_images_per_species = min_images_per_species

        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"

        # Load all image paths and create labels
        self.samples = []
        self.species_to_idx = {}
        self.organ_to_idx = {organ: idx for idx, organ in enumerate(organ_types)}

        # Scan directory structure
        self._load_dataset(train_ratio, val_ratio, test_ratio, seed)

        print(f"\n{split.upper()} Dataset:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of species: {len(self.species_to_idx)}")
        print(f"  Number of organ types: {len(self.organ_to_idx)}")

        # Print distribution by organ type
        organ_counts = defaultdict(int)
        for _, _, organ_idx in self.samples:
            organ_name = self.organ_types[organ_idx]
            organ_counts[organ_name] += 1

        print(f"  Distribution by organ type:")
        for organ, count in sorted(organ_counts.items()):
            print(f"    {organ}: {count}")

    def _load_dataset(self, train_ratio, val_ratio, test_ratio, seed):
        """Load and split dataset."""
        random.seed(seed)

        # First pass: count total images per species to filter if needed
        species_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        species_image_counts = {}

        if self.min_images_per_species > 0:
            for species_dir in species_dirs:
                species_name = species_dir.name
                total_images = 0

                for organ_type in self.organ_types:
                    organ_dir = species_dir / organ_type
                    if organ_dir.exists():
                        image_files = list(organ_dir.glob('*.jpg')) + list(organ_dir.glob('*.png'))
                        total_images += len(image_files)

                species_image_counts[species_name] = total_images

        # Second pass: collect all species that meet the threshold
        species_idx_counter = 0
        filtered_species_count = 0

        for species_dir in species_dirs:
            species_name = species_dir.name

            # Filter species by minimum image count
            if self.min_images_per_species > 0:
                if species_image_counts[species_name] <= self.min_images_per_species:
                    filtered_species_count += 1
                    continue

            self.species_to_idx[species_name] = species_idx_counter
            species_idx = species_idx_counter
            species_idx_counter += 1

            # Collect images from each organ type
            for organ_type in self.organ_types:
                organ_dir = species_dir / organ_type

                if not organ_dir.exists():
                    continue

                # Get all images in this organ directory
                image_files = sorted(list(organ_dir.glob('*.jpg')) + list(organ_dir.glob('*.png')))

                if len(image_files) == 0:
                    continue

                organ_idx = self.organ_to_idx[organ_type]

                # Shuffle images for this species-organ combination
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
                    self.samples.append((img_path, species_idx, organ_idx))

        # Shuffle all samples
        random.shuffle(self.samples)

        # Report filtering statistics
        if self.min_images_per_species > 0 and self.split == 'train':
            print(f"\n  Filtered out {filtered_species_count} species with <= {self.min_images_per_species} images")
            print(f"  Kept {len(self.species_to_idx)} species with > {self.min_images_per_species} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image: (3, img_size, img_size) tensor
            label: species index (int) if mode='species',
                   or (species_idx, organ_idx) tuple if mode='joint'
        """
        img_path, species_idx, organ_idx = self.samples[idx]

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

        # Return based on classification mode
        if self.classification_mode == 'species':
            return image, species_idx
        else:  # joint classification
            # Combine species and organ into single label
            # Total classes = num_species * num_organs
            combined_label = species_idx * len(self.organ_types) + organ_idx
            return image, combined_label

    def get_class_weights(self):
        """
        Calculate class weights for balanced training.

        Returns:
            Tensor of shape (num_classes,) with weights inversely proportional to class frequency
        """
        if self.classification_mode == 'species':
            num_classes = len(self.species_to_idx)
            class_counts = torch.zeros(num_classes)

            for _, species_idx, _ in self.samples:
                class_counts[species_idx] += 1
        else:
            num_classes = len(self.species_to_idx) * len(self.organ_types)
            class_counts = torch.zeros(num_classes)

            for _, species_idx, organ_idx in self.samples:
                combined_label = species_idx * len(self.organ_types) + organ_idx
                class_counts[combined_label] += 1

        # Calculate weights (inverse frequency) with better stability
        # Avoid extreme weights by using sqrt of inverse frequency
        total_samples = class_counts.sum()
        class_weights = total_samples / (class_counts + 1e-6)

        # Clamp weights to avoid extreme values that destabilize training
        # Max weight should not exceed 10x the minimum non-zero weight
        min_weight = class_weights[class_counts > 0].min()
        max_weight = min_weight * 10.0
        class_weights = torch.clamp(class_weights, min=1.0, max=max_weight)

        # Normalize so average weight is 1.0
        class_weights = class_weights / class_weights.mean()

        return class_weights

    def save_label_mapping(self, filepath):
        """Save species and organ label mappings to JSON."""
        mapping = {
            'species_to_idx': self.species_to_idx,
            'idx_to_species': {v: k for k, v in self.species_to_idx.items()},
            'organ_to_idx': self.organ_to_idx,
            'idx_to_organ': {v: k for k, v in self.organ_to_idx.items()},
            'classification_mode': self.classification_mode,
            'num_species': len(self.species_to_idx),
            'num_organs': len(self.organ_to_idx),
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
    - Mild color jitter
    - Normalization

    Args:
        img_size: Target image size (default: 224)
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),  # Resize with small margin
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
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
        transforms.Resize(int(img_size * 1.05)),  # Resize with small margin
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(
    data_dir,
    batch_size=64,
    num_workers=4,
    classification_mode='species',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    img_size=224,
    seed=42,
    min_images_per_species=0
):
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to plant_data directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        classification_mode: 'species' or 'joint'
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        img_size: Image size (default: 224)
        seed: Random seed
        min_images_per_species: Minimum total images required per species (0 = no filtering)

    Returns:
        train_loader, val_loader, test_loader, label_mapping
    """
    # Create datasets
    train_dataset = PlantDataset(
        data_dir=data_dir,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=get_train_transforms(img_size=img_size),
        classification_mode=classification_mode,
        seed=seed,
        min_images_per_species=min_images_per_species
    )

    val_dataset = PlantDataset(
        data_dir=data_dir,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=get_val_transforms(img_size=img_size),
        classification_mode=classification_mode,
        seed=seed,
        min_images_per_species=min_images_per_species
    )

    test_dataset = PlantDataset(
        data_dir=data_dir,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=get_val_transforms(img_size=img_size),
        classification_mode=classification_mode,
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
    train_dataset.save_label_mapping('label_mapping.json')

    return train_loader, val_loader, test_loader, train_dataset.species_to_idx


if __name__ == "__main__":
    # Test the dataset
    print("Testing PlantDataset...")

    data_dir = Path("plant_data")

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist!")
        print("Please run the scraper first to collect plant images.")
    else:
        # Create dataloaders
        train_loader, val_loader, test_loader, species_to_idx = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            num_workers=0,  # Use 0 for testing on Windows
            classification_mode='species'
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
