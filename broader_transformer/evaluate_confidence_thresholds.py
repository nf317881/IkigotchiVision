"""
Evaluate model performance at different confidence thresholds.

This script tests how the model performs when rejecting low-confidence predictions
(marking them as "unsure" instead of making a potentially wrong prediction).
This is useful when the model might encounter out-of-distribution genera not in the training set.

Usage:
    python evaluate_confidence_thresholds.py --checkpoint_dir checkpoints
    python evaluate_confidence_thresholds.py --checkpoint_dir checkpoints_v2 --dataset test
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from model import create_genus_vit_pretrained
from dataset import create_dataloaders


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in eval mode and config dict
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    num_classes = checkpoint.get('num_classes', 500)

    # Get model name (use pretrained model if available)
    model_name = config.get('pretrained_model_name', 'vit_small_patch16_224.augreg_in21k_ft_in1k')

    print(f"Model: {model_name}")
    print(f"Number of classes: {num_classes}")

    # Create model
    model = create_genus_vit_pretrained(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False  # We're loading our own weights
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, config


@torch.no_grad()
def evaluate_with_thresholds(model, dataloader, device, thresholds):
    """
    Evaluate model at different confidence thresholds.

    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader for validation/test set
        device: Device to run on
        thresholds: List of confidence thresholds to test

    Returns:
        Dictionary with results for each threshold
    """
    model.eval()

    # Collect all predictions and labels
    all_predictions = []
    all_top5_predictions = []
    all_labels = []
    all_confidences = []

    print("\nCollecting predictions...")
    pbar = tqdm(dataloader, desc='Evaluating')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)

        # Get top-1 predictions and confidence scores
        confidences, predictions = probabilities.max(dim=1)

        # Get top-5 predictions
        _, top5_predictions = probabilities.topk(5, dim=1)

        all_predictions.append(predictions.cpu())
        all_top5_predictions.append(top5_predictions.cpu())
        all_labels.append(labels.cpu())
        all_confidences.append(confidences.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_top5_predictions = torch.cat(all_top5_predictions)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)

    total_samples = len(all_labels)

    print(f"\nTotal samples: {total_samples}")
    print(f"Confidence range: [{all_confidences.min():.4f}, {all_confidences.max():.4f}]")
    print(f"Mean confidence: {all_confidences.mean():.4f}")
    print(f"Median confidence: {all_confidences.median():.4f}")

    # Evaluate at each threshold
    results = {}

    print("\nEvaluating thresholds...")
    for threshold in tqdm(thresholds, desc='Thresholds'):
        # Create mask for predictions above threshold
        confident_mask = all_confidences >= threshold

        # Count samples
        num_confident = confident_mask.sum().item()
        num_unsure = total_samples - num_confident

        # Calculate top-1 accuracy on confident predictions
        if num_confident > 0:
            confident_predictions = all_predictions[confident_mask]
            confident_top5_predictions = all_top5_predictions[confident_mask]
            confident_labels = all_labels[confident_mask]

            # Top-1 accuracy
            num_correct = (confident_predictions == confident_labels).sum().item()
            confident_accuracy = 100.0 * num_correct / num_confident

            # Top-5 accuracy
            labels_expanded = confident_labels.unsqueeze(1).expand_as(confident_top5_predictions)
            num_correct_top5 = (confident_top5_predictions == labels_expanded).any(dim=1).sum().item()
            confident_top5_accuracy = 100.0 * num_correct_top5 / num_confident
        else:
            num_correct = 0
            num_correct_top5 = 0
            confident_accuracy = 0.0
            confident_top5_accuracy = 0.0

        # Calculate overall accuracy (treating unsure as wrong)
        overall_accuracy = 100.0 * num_correct / total_samples
        overall_top5_accuracy = 100.0 * num_correct_top5 / total_samples

        # Calculate coverage (percentage of samples above threshold)
        coverage = 100.0 * num_confident / total_samples

        # Store results
        results[f'{threshold:.4f}'] = {
            'threshold': threshold,
            'total_samples': total_samples,
            'num_confident': num_confident,
            'num_unsure': num_unsure,
            'num_correct': num_correct,
            'num_correct_top5': num_correct_top5,
            'confident_accuracy': confident_accuracy,
            'confident_top5_accuracy': confident_top5_accuracy,
            'overall_accuracy': overall_accuracy,
            'overall_top5_accuracy': overall_top5_accuracy,
            'coverage': coverage,
            'unsure_rate': 100.0 - coverage
        }

    return results, all_confidences.numpy()


def generate_report(results, confidence_stats, output_path):
    """
    Generate a text report with threshold analysis.

    Args:
        results: Dictionary of results for each threshold
        confidence_stats: Numpy array of all confidence scores
        output_path: Path to save report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONFIDENCE THRESHOLD ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write("This report analyzes model performance at different confidence thresholds.\n")
        f.write("When the model's confidence is below the threshold, the prediction is marked\n")
        f.write("as 'unsure' rather than making a potentially incorrect prediction.\n\n")

        # Confidence statistics
        f.write("-"*80 + "\n")
        f.write("CONFIDENCE DISTRIBUTION STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {results[list(results.keys())[0]]['total_samples']}\n")
        f.write(f"Min confidence: {confidence_stats.min():.4f}\n")
        f.write(f"Max confidence: {confidence_stats.max():.4f}\n")
        f.write(f"Mean confidence: {confidence_stats.mean():.4f}\n")
        f.write(f"Median confidence: {np.median(confidence_stats):.4f}\n")
        f.write(f"Std deviation: {confidence_stats.std():.4f}\n\n")

        # Percentiles
        f.write("Confidence Percentiles:\n")
        for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            value = np.percentile(confidence_stats, percentile)
            f.write(f"  {percentile:2d}th percentile: {value:.4f}\n")
        f.write("\n")

        # Threshold results table - TOP-1 ACCURACY
        f.write("-"*80 + "\n")
        f.write("THRESHOLD ANALYSIS - TOP-1 ACCURACY\n")
        f.write("-"*80 + "\n\n")

        f.write("Legend:\n")
        f.write("  - Threshold: Minimum confidence required to make a prediction\n")
        f.write("  - Coverage: % of samples with confidence above threshold\n")
        f.write("  - Unsure Rate: % of samples marked as unsure (below threshold)\n")
        f.write("  - Confident Accuracy: Top-1 accuracy on samples above threshold\n")
        f.write("  - Overall Accuracy: Top-1 accuracy treating unsure as incorrect\n\n")

        # Table header
        f.write(f"{'Threshold':>10} | {'Coverage':>9} | {'Unsure':>9} | ")
        f.write(f"{'Confident':>9} | {'Confident':>9} | {'Overall':>9}\n")
        f.write(f"{'':>10} | {'(%)':>9} | {'Rate (%)':>9} | ")
        f.write(f"{'Samples':>9} | {'Acc (%)':>9} | {'Acc (%)':>9}\n")
        f.write("-"*80 + "\n")

        # Sort results by threshold
        sorted_results = sorted(results.items(), key=lambda x: x[1]['threshold'])

        for _, data in sorted_results:
            f.write(f"{data['threshold']:10.4f} | ")
            f.write(f"{data['coverage']:9.2f} | ")
            f.write(f"{data['unsure_rate']:9.2f} | ")
            f.write(f"{data['num_confident']:9d} | ")
            f.write(f"{data['confident_accuracy']:9.2f} | ")
            f.write(f"{data['overall_accuracy']:9.2f}\n")

        f.write("\n")

        # Threshold results table - TOP-5 ACCURACY
        f.write("-"*80 + "\n")
        f.write("THRESHOLD ANALYSIS - TOP-5 ACCURACY\n")
        f.write("-"*80 + "\n\n")

        f.write("Legend:\n")
        f.write("  - Threshold: Minimum confidence required to make a prediction\n")
        f.write("  - Coverage: % of samples with confidence above threshold\n")
        f.write("  - Confident Top-5 Acc: Top-5 accuracy on samples above threshold\n")
        f.write("  - Overall Top-5 Acc: Top-5 accuracy treating unsure as incorrect\n\n")

        # Table header for top-5
        f.write(f"{'Threshold':>10} | {'Coverage':>9} | {'Unsure':>9} | ")
        f.write(f"{'Confident':>9} | {'Confident':>11} | {'Overall':>11}\n")
        f.write(f"{'':>10} | {'(%)':>9} | {'Rate (%)':>9} | ")
        f.write(f"{'Samples':>9} | {'Top-5 (%)':>11} | {'Top-5 (%)':>11}\n")
        f.write("-"*80 + "\n")

        for _, data in sorted_results:
            f.write(f"{data['threshold']:10.4f} | ")
            f.write(f"{data['coverage']:9.2f} | ")
            f.write(f"{data['unsure_rate']:9.2f} | ")
            f.write(f"{data['num_confident']:9d} | ")
            f.write(f"{data['confident_top5_accuracy']:11.2f} | ")
            f.write(f"{data['overall_top5_accuracy']:11.2f}\n")

        f.write("\n")

        # Recommendations
        f.write("-"*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n\n")

        # Find best thresholds for different use cases
        # 1. Maximum overall accuracy
        best_overall = max(sorted_results, key=lambda x: x[1]['overall_accuracy'])
        f.write(f"1. MAXIMUM OVERALL ACCURACY (treating unsure as wrong):\n")
        f.write(f"   Threshold: {best_overall[1]['threshold']:.4f}\n")
        f.write(f"   Overall Accuracy: {best_overall[1]['overall_accuracy']:.2f}%\n")
        f.write(f"   Coverage: {best_overall[1]['coverage']:.2f}%\n")
        f.write(f"   Unsure Rate: {best_overall[1]['unsure_rate']:.2f}%\n\n")

        # 2. High precision (95%+ confident accuracy)
        high_precision = [x for x in sorted_results if x[1]['confident_accuracy'] >= 95.0]
        if high_precision:
            best_high_precision = max(high_precision, key=lambda x: x[1]['coverage'])
            f.write(f"2. HIGH PRECISION (95%+ confident accuracy, max coverage):\n")
            f.write(f"   Threshold: {best_high_precision[1]['threshold']:.4f}\n")
            f.write(f"   Confident Accuracy: {best_high_precision[1]['confident_accuracy']:.2f}%\n")
            f.write(f"   Coverage: {best_high_precision[1]['coverage']:.2f}%\n")
            f.write(f"   Unsure Rate: {best_high_precision[1]['unsure_rate']:.2f}%\n\n")
        else:
            f.write(f"2. HIGH PRECISION (95%+ confident accuracy):\n")
            f.write(f"   Not achievable with this model/dataset\n\n")

        # 3. Balanced (90%+ coverage with best accuracy)
        balanced = [x for x in sorted_results if x[1]['coverage'] >= 90.0]
        if balanced:
            best_balanced = max(balanced, key=lambda x: x[1]['confident_accuracy'])
            f.write(f"3. BALANCED (90%+ coverage, max confident accuracy):\n")
            f.write(f"   Threshold: {best_balanced[1]['threshold']:.4f}\n")
            f.write(f"   Confident Accuracy: {best_balanced[1]['confident_accuracy']:.2f}%\n")
            f.write(f"   Coverage: {best_balanced[1]['coverage']:.2f}%\n")
            f.write(f"   Unsure Rate: {best_balanced[1]['unsure_rate']:.2f}%\n\n")

        # 4. Conservative (reject more to ensure high accuracy)
        conservative = [x for x in sorted_results if x[1]['confident_accuracy'] >= 90.0]
        if conservative:
            best_conservative = max(conservative, key=lambda x: x[1]['confident_accuracy'])
            f.write(f"4. CONSERVATIVE (max confident accuracy >= 90%):\n")
            f.write(f"   Threshold: {best_conservative[1]['threshold']:.4f}\n")
            f.write(f"   Confident Accuracy: {best_conservative[1]['confident_accuracy']:.2f}%\n")
            f.write(f"   Coverage: {best_conservative[1]['coverage']:.2f}%\n")
            f.write(f"   Unsure Rate: {best_conservative[1]['unsure_rate']:.2f}%\n\n")

        # Usage notes
        f.write("-"*80 + "\n")
        f.write("USAGE NOTES\n")
        f.write("-"*80 + "\n\n")

        f.write("For deployment where the model might encounter out-of-distribution genera:\n\n")

        f.write("1. Use a HIGHER threshold (0.7-0.9) if:\n")
        f.write("   - Misidentification has serious consequences\n")
        f.write("   - You prefer 'I don't know' over wrong answers\n")
        f.write("   - User can provide more images for unsure cases\n\n")

        f.write("2. Use a MODERATE threshold (0.5-0.7) if:\n")
        f.write("   - You want balanced precision and coverage\n")
        f.write("   - Most inputs are from the 500 known genera\n")
        f.write("   - Some uncertainty is acceptable\n\n")

        f.write("3. Use a LOWER threshold (0.3-0.5) if:\n")
        f.write("   - Coverage is critical (must provide some answer)\n")
        f.write("   - Errors can be corrected by user\n")
        f.write("   - This is for suggestions, not final decisions\n\n")

        f.write("4. Use NO threshold (0.0) if:\n")
        f.write("   - All inputs guaranteed to be from the 500 known genera\n")
        f.write("   - User understands model limitations\n")
        f.write("   - Display top-k predictions instead of just top-1\n\n")

        f.write("="*80 + "\n")

    print(f"\n✓ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model at different confidence thresholds')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints_v2',
        help='Directory containing checkpoint files (default: checkpoints)'
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='best_checkpoint.pth',
        help='Name of checkpoint file to evaluate (default: best_checkpoint.pth)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Dataset to evaluate on (default: val)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='processed_data',
        help='Directory containing processed data (default: processed_data)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (default: auto - cuda if available, else cpu)'
    )

    args = parser.parse_args()

    # Convert paths
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("="*80)
    print("CONFIDENCE THRESHOLD EVALUATION")
    print("="*80)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    model = model.to(device)

    # Create dataloaders
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, genus_to_id = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=config.get('img_size', 224)
    )

    # Select dataloader
    dataloader = val_loader if args.dataset == 'val' else test_loader
    print(f"Dataset size: {len(dataloader.dataset)} samples")

    # Define thresholds to test
    # Include common thresholds and percentile-based thresholds
    thresholds = [
        0.0,   # No threshold
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  # 0.1 increments
        0.95, 0.99  # High thresholds
    ]

    # Evaluate at each threshold
    results, confidence_stats = evaluate_with_thresholds(
        model, dataloader, device, thresholds
    )

    # Save results as JSON
    json_output_path = checkpoint_dir / 'confidence_threshold_results.json'
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ JSON results saved to: {json_output_path}")

    # Generate text report
    text_output_path = checkpoint_dir / 'confidence_threshold_report.txt'
    generate_report(results, confidence_stats, text_output_path)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Show a few key thresholds - TOP-1
    print(f"\nKey threshold results on {args.dataset} set (TOP-1):\n")
    print(f"{'Threshold':>10} | {'Coverage':>9} | {'Confident Acc':>13} | {'Overall Acc':>12}")
    print("-"*60)

    for threshold in [0.0, 0.5, 0.7, 0.9]:
        key = f'{threshold:.4f}'
        if key in results:
            data = results[key]
            print(f"{data['threshold']:10.4f} | {data['coverage']:8.1f}% | "
                  f"{data['confident_accuracy']:12.2f}% | {data['overall_accuracy']:11.2f}%")

    # Show a few key thresholds - TOP-5
    print(f"\nKey threshold results on {args.dataset} set (TOP-5):\n")
    print(f"{'Threshold':>10} | {'Coverage':>9} | {'Confident Top-5':>14} | {'Overall Top-5':>13}")
    print("-"*65)

    for threshold in [0.0, 0.5, 0.7, 0.9]:
        key = f'{threshold:.4f}'
        if key in results:
            data = results[key]
            print(f"{data['threshold']:10.4f} | {data['coverage']:8.1f}% | "
                  f"{data['confident_top5_accuracy']:13.2f}% | {data['overall_top5_accuracy']:12.2f}%")

    print("\n✓ Evaluation complete!")
    print(f"\nOutput files in {checkpoint_dir}:")
    print(f"  - {json_output_path.name}")
    print(f"  - {text_output_path.name}")


if __name__ == "__main__":
    main()
