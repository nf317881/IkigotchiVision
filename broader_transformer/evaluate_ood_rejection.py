"""
Evaluate Out-of-Distribution (OOD) rejection using confidence thresholds.

This script tests how well confidence thresholding detects and rejects
plant images from genera that were NOT in the training data.

Usage:
    python evaluate_ood_rejection.py --checkpoint_dir checkpoints_v2 --ood_dir ood_test_images
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_genus_vit_pretrained
from dataset import get_val_transforms


def load_model(checkpoint_path, device):
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_genus_vit_pretrained(
        model_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
        num_classes=500,
        pretrained=False
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded (epoch {checkpoint['epoch']})")
    # print(f"  Best val accuracy: {checkpoint['best_val_accuracy']:.2f}%")

    return model


def load_ood_images(ood_dir, transform):
    """Load OOD images and their metadata."""
    ood_dir = Path(ood_dir)

    # Load metadata
    metadata_path = ood_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nLoading OOD images from {ood_dir}...")
    print(f"  Expected genera: {', '.join(sorted(metadata['ood_genera']))}")

    # Load images
    images = []
    genera = []
    filenames = []

    for item in tqdm(metadata['images'], desc="Loading images"):
        img_path = ood_dir / item['filename']

        if not img_path.exists():
            continue

        try:
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)

            images.append(img_tensor)
            genera.append(item['genus'])
            filenames.append(item['filename'])

        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            continue

    print(f"  Loaded {len(images)} OOD images")

    return torch.stack(images), genera, filenames, metadata['ood_genera']


def evaluate_ood_rejection(model, ood_images, device, batch_size=64):
    """
    Run inference on OOD images and collect confidence scores.

    Returns:
        confidences: Array of max confidence scores
        predictions: Array of predicted class indices
        all_probs: Array of all class probabilities (for analysis)
    """
    model.eval()

    all_confidences = []
    all_predictions = []
    all_probs = []

    num_images = len(ood_images)
    num_batches = (num_images + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Running inference"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_images)

            batch = ood_images[start_idx:end_idx].to(device)

            # Forward pass
            outputs = model(batch)
            probabilities = F.softmax(outputs, dim=1)

            # Get predictions and confidences
            confidences, predictions = probabilities.max(dim=1)

            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_probs.append(probabilities.cpu())

    # Concatenate all batches
    confidences = torch.cat(all_confidences).numpy()
    predictions = torch.cat(all_predictions).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return confidences, predictions, all_probs


def analyze_threshold_performance(confidences, thresholds):
    """
    Analyze how many OOD samples are rejected at different thresholds.

    For OOD data, we WANT high rejection rates - that's the point!
    """
    results = {}

    for threshold in thresholds:
        num_rejected = (confidences < threshold).sum()
        num_accepted = (confidences >= threshold).sum()
        rejection_rate = 100.0 * num_rejected / len(confidences)
        acceptance_rate = 100.0 - rejection_rate

        results[threshold] = {
            'threshold': threshold,
            'num_rejected': int(num_rejected),
            'num_accepted': int(num_accepted),
            'rejection_rate': rejection_rate,
            'acceptance_rate': acceptance_rate,
            'mean_confidence_accepted': confidences[confidences >= threshold].mean() if num_accepted > 0 else 0.0,
            'mean_confidence_rejected': confidences[confidences < threshold].mean() if num_rejected > 0 else 0.0
        }

    return results


def plot_confidence_distribution(confidences, genera, ood_genera, output_dir):
    """Plot confidence distribution for OOD images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Overall distribution
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram
    axes[0].hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('OOD Confidence Distribution (All Genera)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CDF
    sorted_conf = np.sort(confidences)
    cdf = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
    axes[1].plot(sorted_conf, cdf * 100, linewidth=2)
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[1].axhline(50, color='gray', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Confidence Threshold')
    axes[1].set_ylabel('% of OOD Images Accepted')
    axes[1].set_title('Cumulative Distribution: OOD Acceptance Rate vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ood_confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Per-genus boxplot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Group confidences by genus
    genus_confidences = {}
    for conf, genus in zip(confidences, genera):
        if genus not in genus_confidences:
            genus_confidences[genus] = []
        genus_confidences[genus].append(conf)

    # Sort by median confidence
    sorted_genera = sorted(genus_confidences.keys(),
                          key=lambda g: np.median(genus_confidences[g]))

    # Create boxplot
    boxplot_data = [genus_confidences[g] for g in sorted_genera]
    positions = np.arange(len(sorted_genera))

    bp = ax.boxplot(boxplot_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    # Add threshold line
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')

    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_genera, rotation=45, ha='right')
    ax.set_ylabel('Confidence Score')
    ax.set_title('OOD Confidence Distribution by Genus (sorted by median)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'ood_confidence_by_genus.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to {output_dir}/")


def save_results(results, confidences, output_dir):
    """Save detailed results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save detailed results as JSON (convert numpy types to Python types)
    results_json = {}
    for k, v in results.items():
        results_json[str(k)] = {
            key: float(val) if isinstance(val, (np.floating, np.integer)) else val
            for key, val in v.items()
        }

    with open(output_dir / 'ood_rejection_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save summary report
    with open(output_dir / 'ood_rejection_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Out-of-Distribution (OOD) Rejection Analysis\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total OOD samples: {len(confidences)}\n")
        f.write(f"Mean confidence: {confidences.mean():.4f}\n")
        f.write(f"Median confidence: {np.median(confidences):.4f}\n")
        f.write(f"Std confidence: {confidences.std():.4f}\n\n")

        f.write("Rejection Rates at Different Thresholds:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Threshold':<12} {'Rejected':<12} {'Accepted':<12} "
                f"{'Rejection %':<15} {'Acceptance %':<15}\n")
        f.write("-"*80 + "\n")

        for threshold in sorted(results.keys()):
            r = results[threshold]
            f.write(f"{r['threshold']:<12.2f} {r['num_rejected']:<12} {r['num_accepted']:<12} "
                   f"{r['rejection_rate']:<15.2f} {r['acceptance_rate']:<15.2f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Key Finding for threshold = 0.5:\n")
        f.write("="*80 + "\n")
        r = results[0.5]
        f.write(f"  Rejection rate: {r['rejection_rate']:.2f}%\n")
        f.write(f"  Acceptance rate: {r['acceptance_rate']:.2f}%\n")
        f.write(f"  Mean confidence (accepted): {r['mean_confidence_accepted']:.4f}\n")
        f.write(f"  Mean confidence (rejected): {r['mean_confidence_rejected']:.4f}\n\n")

        f.write("Interpretation:\n")
        f.write(f"  - The model correctly identifies {r['rejection_rate']:.1f}% of OOD samples as uncertain\n")
        f.write(f"  - However, {r['acceptance_rate']:.1f}% of OOD samples have confidence >= 0.5\n")
        f.write(f"  - These are false positives (OOD samples misclassified as known genera)\n")

    print(f"\n  Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OOD rejection using confidence thresholds'
    )
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v2',
                        help='Directory containing model checkpoint')
    parser.add_argument('--ood_dir', type=str, default='ood_test_images',
                        help='Directory containing OOD images')
    parser.add_argument('--output_dir', type=str, default='ood_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    args = parser.parse_args()

    print("="*80)
    print("OOD Rejection Evaluation")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint_dir) / 'best_checkpoint.pth'
    model = load_model(checkpoint_path, device)

    # Load OOD images
    transform = get_val_transforms(img_size=224)
    ood_images, genera, filenames, ood_genera_list = load_ood_images(args.ood_dir, transform)

    # Run inference
    print("\n" + "="*80)
    print("Running inference on OOD images...")
    print("="*80)
    confidences, predictions, all_probs = evaluate_ood_rejection(
        model, ood_images, device, batch_size=args.batch_size
    )

    # Analyze thresholds
    print("\n" + "="*80)
    print("Analyzing threshold performance...")
    print("="*80)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    results = analyze_threshold_performance(confidences, thresholds)

    # Print summary
    print("\nRejection rates at different thresholds:")
    print("-"*80)
    print(f"{'Threshold':<12} {'Rejected':<12} {'Accepted':<12} {'Rejection %':<15} {'Acceptance %':<15}")
    print("-"*80)
    for threshold in thresholds:
        r = results[threshold]
        print(f"{r['threshold']:<12.2f} {r['num_rejected']:<12} {r['num_accepted']:<12} "
              f"{r['rejection_rate']:<15.2f} {r['acceptance_rate']:<15.2f}")

    # Highlight 0.5 threshold
    print("\n" + "="*80)
    print("Key Finding at threshold = 0.5:")
    print("="*80)
    r = results[0.5]
    print(f"  Rejection rate: {r['rejection_rate']:.2f}%")
    print(f"  Acceptance rate: {r['acceptance_rate']:.2f}%")
    print(f"  Mean confidence (accepted): {r['mean_confidence_accepted']:.4f}")
    print(f"  Mean confidence (rejected): {r['mean_confidence_rejected']:.4f}")
    print("\nInterpretation:")
    print(f"  - The model correctly identifies {r['rejection_rate']:.1f}% of OOD samples as uncertain")
    print(f"  - However, {r['acceptance_rate']:.1f}% of OOD samples have confidence >= 0.5")
    print(f"  - These {r['num_accepted']} samples are false positives")

    # Plot distributions
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    plot_confidence_distribution(confidences, genera, ood_genera_list, args.output_dir)

    # Save results
    save_results(results, confidences, args.output_dir)

    # Statistics by genus
    print("\n" + "="*80)
    print("OOD Statistics by Genus:")
    print("="*80)

    genus_stats = {}
    for conf, pred, genus in zip(confidences, predictions, genera):
        if genus not in genus_stats:
            genus_stats[genus] = {
                'confidences': [],
                'predictions': []
            }
        genus_stats[genus]['confidences'].append(conf)
        genus_stats[genus]['predictions'].append(pred)

    print(f"{'Genus':<20} {'Count':<8} {'Mean Conf':<12} {'Rejected@0.5':<15} {'Rejection %':<15}")
    print("-"*80)

    for genus in sorted(genus_stats.keys()):
        stats = genus_stats[genus]
        confs = np.array(stats['confidences'])
        count = len(confs)
        mean_conf = confs.mean()
        rejected = (confs < 0.5).sum()
        rejection_pct = 100.0 * rejected / count

        print(f"{genus:<20} {count:<8} {mean_conf:<12.4f} {rejected:<15} {rejection_pct:<15.1f}")

    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    print(f"\nResults saved to: {Path(args.output_dir).absolute()}")


if __name__ == '__main__':
    main()
