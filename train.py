"""
Training Script for Plant Vision Transformer

Features:
- Automatic mixed precision training (AMP) for faster training
- Learning rate scheduling with warmup
- Early stopping
- Model checkpointing
- Training metrics tracking
- Support for class-weighted loss
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

from vision_transformer import create_vit_small, create_vit_tiny, create_vit_base
from dataset import create_dataloaders


class Trainer:
    """
    Trainer class for Vision Transformer.
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
        early_stopping_patience=10
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

        pbar = tqdm(self.train_loader, desc='Training')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc='Validation')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def test(self):
        """Test on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        top5_correct = 0

        all_predictions = []
        all_labels = []

        pbar = tqdm(self.test_loader, desc='Testing')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * images.size(0)
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

        test_loss = total_loss / total
        test_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        return test_loss, test_acc, top5_acc, all_predictions, all_labels

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history
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

    def train(self, num_epochs):
        """Train for multiple epochs."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
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
            self.save_checkpoint(epoch, is_best=is_best)

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
        'data_dir': 'plant_data',
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'classification_mode': 'species',  # 'species' or 'joint'
        'model_size': 'small',  # 'tiny', 'small', or 'base'
        'use_amp': True,
        'use_checkpoint': False,
        'use_class_weights': False,
        'early_stopping_patience': 15,
        'save_dir': 'checkpoints'
    }

    print("Plant Vision Transformer Training")
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
    train_loader, val_loader, test_loader, species_to_idx = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        classification_mode=config['classification_mode']
    )

    # Calculate number of classes
    if config['classification_mode'] == 'species':
        num_classes = len(species_to_idx)
    else:
        # Joint classification: species * organs
        num_classes = len(species_to_idx) * 5  # 5 organ types

    print(f"\nNumber of classes: {num_classes}")

    # Create model
    print(f"\nCreating Vision Transformer ({config['model_size']})...")
    if config['model_size'] == 'tiny':
        model = create_vit_tiny(
            num_classes=num_classes,
            img_size=56,
            use_checkpoint=config['use_checkpoint']
        )
    elif config['model_size'] == 'small':
        model = create_vit_small(
            num_classes=num_classes,
            img_size=56,
            use_checkpoint=config['use_checkpoint']
        )
    else:  # base
        model = create_vit_base(
            num_classes=num_classes,
            img_size=56,
            use_checkpoint=True  # Always use checkpointing for base
        )

    model = model.to(device)

    # Loss function
    if config['use_class_weights']:
        print("Calculating class weights...")
        class_weights = train_loader.dataset.get_class_weights()
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = len(train_loader) * 5  # 5 epochs warmup

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        early_stopping_patience=config['early_stopping_patience']
    )

    # Train
    history = trainer.train(num_epochs=config['num_epochs'])

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
