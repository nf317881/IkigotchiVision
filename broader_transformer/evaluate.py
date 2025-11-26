"""
Evaluate Test-Time Augmentation (TTA) Performance

This script evaluates the trained model with different TTA configurations:
- Baseline (no TTA)
- Lightweight TTA (H+V flip, 4x augmentations)
- Moderate TTA (H+V flip + rotations, 16x augmentations)

It compares performance on the validation set and saves results to JSON.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model import create_genus_vit_pretrained
from dataset import create_dataloaders
from train import Trainer


def evaluate_with_tta(checkpoint_path, data_dir, tta_mode=None, dataset='val'):
    """
    Evaluate model with specified TTA mode.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to processed data directory
        tta_mode: None (no TTA), 'lightweight', or 'moderate'
        dataset: 'val' or 'test'

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    if tta_mode is None:
        print(f"Evaluating: BASELINE (No TTA)")
    else:
        print(f"Evaluating: TTA Mode = {tta_mode.upper()}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'='*60}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, genus_to_id = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,  # Reduced batch size for TTA to avoid OOM
        num_workers=4,
        img_size=224
    )

    num_classes = len(genus_to_id)
    print(f"Number of classes: {num_classes}")

    # Create model
    print(f"\nCreating model...")
    pretrained_model_name = config.get('pretrained_model_name', 'vit_small_patch16_224.augreg_in21k_ft_in1k')
    model = create_genus_vit_pretrained(
        model_name=pretrained_model_name,
        num_classes=num_classes,
        pretrained=False  # We'll load weights from checkpoint
    )
    model = model.to(device)

    # Load model weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']})")
    print(f"Checkpoint best val acc: {checkpoint['best_val_acc']:.2f}%")

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create trainer with TTA settings
    use_tta = tta_mode is not None
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=None,  # Not needed for evaluation
        scheduler=None,  # Not needed for evaluation
        device=device,
        use_amp=False,  # Disable AMP for evaluation consistency
        use_tta=use_tta,
        tta_mode=tta_mode if use_tta else 'lightweight'
    )

    # Run evaluation
    print(f"\nStarting evaluation...")
    start_time = time.time()

    if dataset == 'val':
        # Use validation set
        loss, acc = trainer.validate()
        top5_acc = None  # validate() doesn't compute top-5
        predictions = None
        labels = None
    else:
        # Use test set
        loss, acc, top5_acc, predictions, labels = trainer.test()

    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.2f}%")
    if top5_acc is not None:
        print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Evaluation time: {elapsed_time:.1f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}\n")

    # Return results
    results = {
        'tta_mode': tta_mode,
        'dataset': dataset,
        'loss': loss,
        'accuracy': acc,
        'top5_accuracy': top5_acc,
        'evaluation_time_seconds': elapsed_time,
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_best_val_acc': checkpoint['best_val_acc']
    }

    return results


def main():
    """Main evaluation function."""
    # Configuration
    checkpoint_path = Path('checkpoints/best_checkpoint.pth')
    data_dir = 'processed_data'
    dataset = 'val'  # Use validation set for faster evaluation

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have trained a model first.")
        return

    print("="*60)
    print("TEST-TIME AUGMENTATION (TTA) EVALUATION")
    print("="*60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Evaluation dataset: {dataset.upper()}")

    # Evaluate with different TTA modes
    all_results = []

    # 1. Baseline (no TTA)
    print("\n\n" + "="*60)
    print("PHASE 1/3: BASELINE EVALUATION (No TTA)")
    print("="*60)
    baseline_results = evaluate_with_tta(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        tta_mode=None,
        dataset=dataset
    )
    all_results.append(baseline_results)

    # 2. Lightweight TTA
    print("\n\n" + "="*60)
    print("PHASE 2/3: LIGHTWEIGHT TTA EVALUATION")
    print("="*60)
    lightweight_results = evaluate_with_tta(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        tta_mode='lightweight',
        dataset=dataset
    )
    all_results.append(lightweight_results)

    # 3. Moderate TTA
    print("\n\n" + "="*60)
    print("PHASE 3/3: MODERATE TTA EVALUATION")
    print("="*60)
    moderate_results = evaluate_with_tta(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        tta_mode='moderate',
        dataset=dataset
    )
    all_results.append(moderate_results)

    # Summary comparison
    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Mode':<20} {'Accuracy':<12} {'Time (min)':<12} {'Speedup':<10}")
    print("-"*60)

    baseline_acc = baseline_results['accuracy']
    baseline_time = baseline_results['evaluation_time_seconds']

    for result in all_results:
        mode_name = result['tta_mode'] if result['tta_mode'] else 'Baseline'
        acc = result['accuracy']
        time_min = result['evaluation_time_seconds'] / 60
        speedup = baseline_time / result['evaluation_time_seconds']
        acc_diff = acc - baseline_acc

        print(f"{mode_name:<20} {acc:>6.2f}% ({acc_diff:+.2f}%)  {time_min:>6.2f}      {speedup:>6.2f}x")

    # Save results to JSON
    output_path = Path('checkpoints/tta_evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'baseline_accuracy': baseline_acc,
                'lightweight_accuracy': lightweight_results['accuracy'],
                'moderate_accuracy': moderate_results['accuracy'],
                'lightweight_improvement': lightweight_results['accuracy'] - baseline_acc,
                'moderate_improvement': moderate_results['accuracy'] - baseline_acc,
            },
            'detailed_results': all_results
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
