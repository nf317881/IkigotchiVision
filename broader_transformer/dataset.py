"""
Plant Genus Dataset Loader

Loads plant genus images from the processed GBIF dataset structure:
    broader_transformer/processed_data/
        images/
            train/
                genus_001_Prunus/
                    000001.jpg
                    000002.jpg
                    ...
                genus_002_Geranium/
                    ...
            val/
                genus_001_Prunus/
                    ...
            test/
                genus_001_Prunus/
                    ...
        metadata/
            label_mapping.json
            train_manifest.csv
            val_manifest.csv
            test_manifest.csv

Supports:
- Pre-split train/val/test from data processing
- Data augmentation
- Genus classification (500 classes)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import pandas as pd
import numpy as np


class PlantGenusDataset(Dataset):
    """
    Dataset for plant genus images with genus labels.

    Reads from pre-processed and pre-split GBIF data.
    """

    def __init__(
        self,
        data_dir,
        split='train',
        transform=None,
        use_manifest=True
    ):
        """
        Args:
            data_dir: Path to processed_data directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
            use_manifest: Whether to use CSV manifests (recommended for faster loading)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_manifest = use_manifest

        # Load label mapping
        label_mapping_path = self.data_dir / "metadata" / "label_mapping.json"
        with open(label_mapping_path, 'r') as f:
            label_data = json.load(f)

        self.genus_to_id = label_data['genus_to_id']
        self.id_to_genus = {int(k): v for k, v in label_data['id_to_genus'].items()}
        self.num_classes = label_data['num_classes']

        # Load samples
        self.samples = []

        if use_manifest:
            self._load_from_manifest()
        else:
            self._load_from_directory()

        print(f"\n{split.upper()} Dataset:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of genera: {self.num_classes}")

    def _load_from_manifest(self):
        """Load samples from CSV manifest (faster)."""
        manifest_path = self.data_dir / "metadata" / f"{self.split}_manifest.csv"

        if not manifest_path.exists():
            print(f"Warning: Manifest {manifest_path} not found, falling back to directory scan")
            self._load_from_directory()
            return

        # Read manifest
        df = pd.read_csv(manifest_path)

        # Build sample list
        for _, row in df.iterrows():
            img_path = self.data_dir / "images" / row['file_path']
            genus_id = int(row['genus_id'])

            if img_path.exists():
                self.samples.append((img_path, genus_id))

    def _load_from_directory(self):
        """Load samples by scanning directory structure (fallback method)."""
        image_dir = self.data_dir / "images" / self.split

        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")

        # Scan all genus directories
        for genus_dir in sorted(image_dir.iterdir()):
            if not genus_dir.is_dir():
                continue

            # Extract genus name from directory name (genus_XXX_GenusName)
            dir_name = genus_dir.name
            if not dir_name.startswith('genus_'):
                continue

            # Parse genus name from directory
            genus_name = '_'.join(dir_name.split('_')[2:])

            if genus_name not in self.genus_to_id:
                print(f"Warning: Unknown genus {genus_name}, skipping")
                continue

            genus_id = self.genus_to_id[genus_name]

            # Get all images in this genus directory
            for img_path in genus_dir.glob('*.jpg'):
                self.samples.append((img_path, genus_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image: (3, img_size, img_size) tensor
            label: genus index (int)
        """
        img_path, genus_id = self.samples[idx]

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

        return image, genus_id

    def get_class_weights(self):
        """
        Calculate class weights for balanced training.

        Returns:
            Tensor of shape (num_classes,) with weights inversely proportional to class frequency
        """
        class_counts = torch.zeros(self.num_classes)

        for _, genus_id in self.samples:
            class_counts[genus_id] += 1

        # Calculate weights (inverse frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (class_counts + 1e-6)

        # Normalize so that the minimum weight among existing classes is 1.0
        # This prevents extreme weights for rare classes
        min_weight = class_weights[class_counts > 0].min()
        class_weights = class_weights / min_weight

        # Cap maximum weight to prevent domination by very rare classes
        max_weight = 10.0
        class_weights = torch.clamp(class_weights, min=1.0, max=max_weight)

        return class_weights


class Mixup:
    """
    Mixup augmentation: https://arxiv.org/abs/1710.09412

    Mixes two samples by interpolating images and labels.
    """
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Mixup parameter. Higher = more mixing. Typical values: 0.1-0.4
        """
        self.alpha = alpha

    def __call__(self, images, labels, num_classes):
        """
        Apply mixup to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            num_classes: Number of classes

        Returns:
            mixed_images: Mixed images (B, C, H, W)
            labels_a: Original labels (B,)
            labels_b: Permuted labels (B,)
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """
    CutMix augmentation: https://arxiv.org/abs/1905.04899

    Cuts and pastes patches between images and mixes labels accordingly.
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: CutMix parameter. Higher = more mixing. Typical values: 0.5-1.0
        """
        self.alpha = alpha

    def __call__(self, images, labels, num_classes):
        """
        Apply CutMix to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            num_classes: Number of classes

        Returns:
            mixed_images: Mixed images (B, C, H, W)
            labels_a: Original labels (B,)
            labels_b: Permuted labels (B,)
            lam: Mixing coefficient (area ratio)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        # Get random box
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    Compute loss for mixup/cutmix.

    Args:
        criterion: Loss function
        pred: Model predictions
        labels_a: First set of labels
        labels_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def get_train_transforms(img_size=224, stronger_aug=False):
    """
    Data augmentation for training.

    Includes:
    - Random resized crop
    - Random horizontal flip
    - Color jitter
    - Random rotation
    - RandomErasing (if stronger_aug=True)
    - RandAugment (if stronger_aug=True)
    - Normalization

    Note: Images are already resized to 224x224, so we just apply augmentations

    Args:
        img_size: Target image size (default: 224)
        stronger_aug: Use stronger augmentation for better regularization
    """
    if stronger_aug:
        # Stronger augmentation to reduce overfitting
        aug_list = [
            # More aggressive crop
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Larger rotation range
            transforms.RandomRotation(degrees=20),
            # RandAugment for automated strong augmentation
            transforms.RandAugment(num_ops=2, magnitude=9),
            # Stronger color jitter
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.ToTensor(),
            # Random erasing after converting to tensor
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        # Original moderate augmentation
        aug_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.08
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

    return transforms.Compose(aug_list)


def get_val_transforms(img_size=224):
    """
    Transforms for validation/test (no augmentation).

    Args:
        img_size: Target image size (default: 224)
    """
    return transforms.Compose([
        # Images are already exactly 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(
    data_dir='processed_data',
    batch_size=32,
    num_workers=4,
    img_size=224,
    use_manifest=True,
    stronger_aug=False
):
    """
    Create train, validation, and test dataloaders for plant genus classification.

    Args:
        data_dir: Path to processed_data directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Image size (default: 224, matches preprocessed images)
        use_manifest: Use CSV manifests for faster loading (default: True)
        stronger_aug: Use stronger augmentation for training (default: False)

    Returns:
        train_loader, val_loader, test_loader, genus_to_id
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = PlantGenusDataset(
        data_dir=data_dir,
        split='train',
        transform=get_train_transforms(img_size=img_size, stronger_aug=stronger_aug),
        use_manifest=use_manifest
    )

    val_dataset = PlantGenusDataset(
        data_dir=data_dir,
        split='val',
        transform=get_val_transforms(img_size=img_size),
        use_manifest=use_manifest
    )

    test_dataset = PlantGenusDataset(
        data_dir=data_dir,
        split='test',
        transform=get_val_transforms(img_size=img_size),
        use_manifest=use_manifest
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader, train_dataset.genus_to_id


if __name__ == "__main__":
    # Test the dataset
    print("Testing PlantGenusDataset...")

    data_dir = Path("processed_data")

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist!")
        print("Please run process_gbif_data.py first to create the processed dataset.")
    else:
        # Create dataloaders
        train_loader, val_loader, test_loader, genus_to_id = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            num_workers=0,  # Use 0 for testing on Windows
            use_manifest=True
        )

        print(f"\nDataLoader test:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Number of genera: {len(genus_to_id)}")

        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label range: [{labels.min().item()}, {labels.max().item()}]")

        print("\n✓ Dataset test completed successfully!")
