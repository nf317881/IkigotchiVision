"""
Training Script for Plant Genus Vision Transformer

Features:
- Automatic mixed precision training (AMP) for faster training
- Learning rate scheduling with cosine annealing
- Early stopping
- Model checkpointing with automatic resume support
- Training metrics tracking
- Support for class-weighted loss
- Transfer learning with pretrained models
- Optimized for 500 plant genera classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import sys
import ttach as tta
import numpy as np

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent))

# Suppress pydantic warnings from timm library
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

from model import create_genus_vit_small, create_genus_vit_base, create_genus_vit_pretrained
from dataset import create_dataloaders, Mixup, CutMix, mixup_criterion


class Trainer:
    """
    Trainer class for Plant Genus Vision Transformer.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir='checkpoints',
        use_amp=True,
        early_stopping_patience=15,
        config=None,
        use_tta=False,
        tta_mode='lightweight',
        use_mixup=False,
        mixup_alpha=0.2,
        use_cutmix=False,
        cutmix_alpha=1.0,
        mixup_prob=0.5,
        num_classes=500
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp
        self.early_stopping_patience = early_stopping_patience
        self.config = config
        self.use_tta = use_tta
        self.tta_mode = tta_mode

        # Mixup/CutMix augmentation
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_prob = mixup_prob
        self.num_classes = num_classes
        if use_mixup:
            self.mixup = Mixup(alpha=mixup_alpha)
        if use_cutmix:
            self.cutmix = CutMix(alpha=cutmix_alpha)

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup/CutMix randomly
            use_mixing = False
            if (self.use_mixup or self.use_cutmix) and np.random.rand() < self.mixup_prob:
                use_mixing = True
                if self.use_mixup and self.use_cutmix:
                    # Randomly choose between Mixup and CutMix
                    if np.random.rand() < 0.5:
                        images, labels_a, labels_b, lam = self.mixup(images, labels, self.num_classes)
                    else:
                        images, labels_a, labels_b, lam = self.cutmix(images, labels, self.num_classes)
                elif self.use_mixup:
                    images, labels_a, labels_b, lam = self.mixup(images, labels, self.num_classes)
                else:
                    images, labels_a, labels_b, lam = self.cutmix(images, labels, self.num_classes)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    if use_mixing:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mixing:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            num_batches += 1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # Use original labels for accuracy approximation (even with mixing)
            if use_mixing:
                # For mixed samples, use labels_a as approximation
                correct += predicted.eq(labels_a).sum().item()
            else:
                correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        epoch_loss = total_loss / num_batches
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        """Validate on validation set with optional Test-Time Augmentation."""
        self.model.eval()

        # Wrap model with TTA if enabled
        if self.use_tta:
            if self.tta_mode == 'lightweight':
                # Lightweight: Horizontal + Vertical flip (2x augmentations)
                tta_transforms = tta.Compose([
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                ])
                print(f"Using TTA: Lightweight (H+V flip, 4x augmentations)")
            elif self.tta_mode == 'moderate':
                # Moderate: H+V flip + 4 rotations (8x augmentations)
                tta_transforms = tta.Compose([
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 90, 180, 270]),
                ])
                print(f"Using TTA: Moderate (H+V flip + rotations, 16x augmentations)")
            else:
                raise ValueError(f"Unknown TTA mode: {self.tta_mode}")

            tta_model = tta.ClassificationTTAWrapper(self.model, tta_transforms)
        else:
            tta_model = self.model

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        desc = 'Validation (TTA)' if self.use_tta else 'Validation'
        pbar = tqdm(self.val_loader, desc=desc)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (with or without TTA)
            outputs = tta_model(images)
            loss = self.criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            num_batches += 1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        epoch_loss = total_loss / num_batches
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def test(self):
        """Test on test set with optional Test-Time Augmentation."""
        self.model.eval()

        # Wrap model with TTA if enabled
        if self.use_tta:
            if self.tta_mode == 'lightweight':
                # Lightweight: Horizontal + Vertical flip (2x augmentations)
                tta_transforms = tta.Compose([
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                ])
                print(f"Using TTA: Lightweight (H+V flip, 4x augmentations)")
            elif self.tta_mode == 'moderate':
                # Moderate: H+V flip + 4 rotations (8x augmentations)
                tta_transforms = tta.Compose([
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 90, 180, 270]),
                ])
                print(f"Using TTA: Moderate (H+V flip + rotations, 16x augmentations)")
            else:
                raise ValueError(f"Unknown TTA mode: {self.tta_mode}")

            tta_model = tta.ClassificationTTAWrapper(self.model, tta_transforms)
        else:
            tta_model = self.model

        total_loss = 0.0
        correct = 0
        total = 0
        top5_correct = 0
        num_batches = 0

        all_predictions = []
        all_labels = []

        desc = 'Testing (TTA)' if self.use_tta else 'Testing'
        pbar = tqdm(self.test_loader, desc=desc)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (with or without TTA)
            outputs = tta_model(images)
            loss = self.criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            num_batches += 1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Store for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        test_loss = total_loss / num_batches
        test_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        return test_loss, test_acc, top5_acc, all_predictions, all_labels

    def save_checkpoint(self, epoch, is_best=False, config=None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': config
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            print(f"  Saved best model with val_acc={self.best_val_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def train(self, num_epochs, start_epoch=1):
        """Train for multiple epochs."""
        if start_epoch > 1:
            print(f"\nResuming training from epoch {start_epoch}...")
        else:
            print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Check for improvement
            is_best = val_acc > self.best_val_acc

            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best, config=self.config)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"No improvement for {self.early_stopping_patience} epochs")
                break

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        # Plot training curves
        self.plot_training_curves()

        return self.history

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], label='Train')
        axes[1].plot(epochs, self.history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        # Learning rate
        axes[2].plot(epochs, self.history['learning_rates'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        print(f"Training curves saved to {self.save_dir / 'training_curves.png'}")


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_dir': 'processed_data',
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': None,  # Will be set based on pretrained vs scratch
        'weight_decay': 1e-4,
        'num_workers': 4,
        'img_size': 224,
        'model_size': 'small',  # 'small', 'base', or 'pretrained'
        'use_pretrained': True,  # Use pretrained weights (highly recommended!)
        'pretrained_model_name': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        'use_amp': True,
        'use_checkpoint': False,
        'use_class_weights': True,
        'label_smoothing': 0.2,  # Increased from 0.1 for stronger regularization
        'early_stopping_patience': 15,
        'save_dir': 'checkpoints_v2',  # New directory to keep old training separate
        'resume_from': 'C:/Users/betud/Documents/IkigotchiVision/broader_transformer/checkpoints_v2/latest_checkpoint.pth',  # Start fresh training with new regularization
        # Stronger augmentation settings
        'stronger_aug': True,  # Use RandAugment, RandomErasing, stronger rotations
        'use_mixup': True,  # Use Mixup augmentation
        'mixup_alpha': 0.2,  # Mixup parameter
        'use_cutmix': True,  # Use CutMix augmentation
        'cutmix_alpha': 1.0,  # CutMix parameter
        'mixup_prob': 0.5,  # Probability of applying Mixup/CutMix per batch
    }

    print("Plant Genus Vision Transformer Training")
    print("="*60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, genus_to_id = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        stronger_aug=config.get('stronger_aug', False)
    )

    num_classes = len(genus_to_id)
    print(f"\nNumber of classes: {num_classes}")

    # Create model
    print(f"\nCreating Vision Transformer ({config['model_size']})...")

    pretrained_loaded = False
    if config['use_pretrained'] and config['model_size'] in ['small', 'base']:
        # Use pretrained model from timm
        try:
            model = create_genus_vit_pretrained(
                model_name=config['pretrained_model_name'],
                num_classes=num_classes,
                pretrained=True
            )
            print("✓ Successfully loaded pretrained weights!")
            print(f"Model: {config['pretrained_model_name']}")
            pretrained_loaded = True
        except Exception as e:
            print(f"\n⚠ Warning: Could not load pretrained model: {e}")
            print("This is likely because 'timm' library is not installed.")
            print("Install with: pip install timm")
            print("\nFalling back to training from scratch...")
            config['use_pretrained'] = False

    if not config['use_pretrained']:
        # Train from scratch
        print("Training from scratch (no pretrained weights)")
        if config['model_size'] == 'small':
            model = create_genus_vit_small(
                num_classes=num_classes,
                img_size=config['img_size'],
                use_checkpoint=config['use_checkpoint']
            )
        else:  # base
            model = create_genus_vit_base(
                num_classes=num_classes,
                img_size=config['img_size'],
                use_checkpoint=True
            )

    # Set learning rate
    if config['learning_rate'] is None:
        if pretrained_loaded:
            config['learning_rate'] = 1e-4  # Lower LR for better generalization with stronger augmentation
            print(f"Using fine-tuning learning rate: {config['learning_rate']}")
        else:
            config['learning_rate'] = 1e-3  # Training from scratch LR
            print(f"Using training-from-scratch learning rate: {config['learning_rate']}")

    model = model.to(device)

    # Loss function
    if config['use_class_weights']:
        print("Calculating class weights...")
        class_weights = train_loader.dataset.get_class_weights()
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config['label_smoothing']
        )
        print(f"Using weighted CrossEntropyLoss with label_smoothing={config['label_smoothing']}")
        print(f"Weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        print(f"Using standard CrossEntropyLoss with label_smoothing={config['label_smoothing']}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )

    print(f"Using CosineAnnealingLR scheduler (initial_lr={config['learning_rate']}, eta_min=1e-6)")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config['save_dir'],
        use_amp=config['use_amp'],
        early_stopping_patience=config['early_stopping_patience'],
        config=config,
        use_mixup=config.get('use_mixup', False),
        mixup_alpha=config.get('mixup_alpha', 0.2),
        use_cutmix=config.get('use_cutmix', False),
        cutmix_alpha=config.get('cutmix_alpha', 1.0),
        mixup_prob=config.get('mixup_prob', 0.5),
        num_classes=num_classes
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if config['resume_from'] is not None:
        checkpoint_path = Path(config['resume_from'])
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            last_epoch = trainer.load_checkpoint(checkpoint_path)
            start_epoch = last_epoch + 1
            print(f"Will resume from epoch {start_epoch}")
        else:
            print(f"\nWarning: Checkpoint not found at {checkpoint_path}")
            print("Starting training from scratch...")

    # Train
    history = trainer.train(num_epochs=config['num_epochs'], start_epoch=start_epoch)

    # Test on best model
    print("\n" + "="*60)
    print("Testing best model...")
    checkpoint_path = Path(config['save_dir']) / 'best_checkpoint.pth'
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
        test_loss, test_acc, top5_acc, predictions, labels = trainer.test()

        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy (Top-1): {test_acc:.2f}%")
        print(f"  Test Accuracy (Top-5): {top5_acc:.2f}%")

        # Save test results
        test_results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'top5_acc': top5_acc
        }

        with open(Path(config['save_dir']) / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
