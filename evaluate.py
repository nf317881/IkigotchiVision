"""
Evaluation and Visualization Tools

Features:
1. Confusion Matrix - See which species get confused
2. Attention Map Visualization - See what the model focuses on
3. Per-Class Accuracy - Track accuracy for each species/organ
4. Misclassification Gallery - Show incorrectly classified images
5. Top-K Accuracy - Check if correct answer is in top-K predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

from vision_transformer import create_vit_small, create_vit_tiny, create_vit_base
from dataset import create_dataloaders, get_val_transforms


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """

    def __init__(self, model, test_loader, device, label_mapping_path='label_mapping.json'):
        self.model = model
        self.test_loader = test_loader
        self.device = device

        # Load label mappings
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)

        self.idx_to_species = self.label_mapping['idx_to_species']
        self.idx_to_organ = self.label_mapping.get('idx_to_organ', {})

    @torch.no_grad()
    def get_predictions(self):
        """
        Get all predictions and ground truth labels.

        Returns:
            predictions: List of predicted class indices
            labels: List of ground truth class indices
            probabilities: List of prediction probabilities
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        print("Collecting predictions...")
        for images, labels in tqdm(self.test_loader):
            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probs.cpu().numpy())

        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

    def plot_confusion_matrix(self, save_path='confusion_matrix.png', normalize=True, top_k=None):
        """
        Plot confusion matrix.

        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
            top_k: Only show top K most common classes (None = all)
        """
        print("\nGenerating confusion matrix...")

        predictions, labels, _ = self.get_predictions()

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)

        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

        # If top_k is specified, only show most common classes
        if top_k is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            top_indices = np.argsort(counts)[-top_k:]
            top_labels = unique_labels[top_indices]

            # Filter confusion matrix
            cm = cm[top_labels][:, top_labels]
            class_names = [self.idx_to_species[str(i)] for i in top_labels]
        else:
            class_names = [self.idx_to_species[str(i)] for i in range(len(self.idx_to_species))]

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=False,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names if len(class_names) <= 30 else False,
            yticklabels=class_names if len(class_names) <= 30 else False,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        title = 'Confusion Matrix'
        if top_k:
            title += f' (Top {top_k} Classes)'
        if normalize:
            title += ' (Normalized)'
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()

    def plot_per_class_accuracy(self, save_path='per_class_accuracy.png', top_k=30):
        """
        Plot per-class accuracy bar chart.

        Args:
            save_path: Path to save the plot
            top_k: Show top K classes by sample count
        """
        print("\nCalculating per-class accuracy...")

        predictions, labels, _ = self.get_predictions()

        # Calculate per-class accuracy
        unique_labels = np.unique(labels)
        accuracies = []
        class_names = []
        counts = []

        for label in unique_labels:
            mask = labels == label
            class_predictions = predictions[mask]
            class_labels = labels[mask]

            accuracy = (class_predictions == class_labels).mean() * 100
            accuracies.append(accuracy)
            counts.append(len(class_labels))

            # Get class name
            class_name = self.idx_to_species[str(label)]
            # Truncate long names
            if len(class_name) > 30:
                class_name = class_name[:27] + '...'
            class_names.append(class_name)

        # Sort by count and take top K
        sorted_indices = np.argsort(counts)[-top_k:]
        accuracies = [accuracies[i] for i in sorted_indices]
        class_names = [class_names[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(class_names) * 0.3)))

        y_pos = np.arange(len(class_names))
        bars = ax.barh(y_pos, accuracies, color='skyblue')

        # Color bars by accuracy
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc >= 80:
                bar.set_color('green')
            elif acc >= 60:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name} (n={count})" for name, count in zip(class_names, counts)])
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(f'Per-Class Accuracy (Top {top_k} Classes by Sample Count)')
        ax.set_xlim([0, 100])
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to {save_path}")
        plt.close()

        # Print statistics
        print(f"\nAccuracy Statistics:")
        print(f"  Mean accuracy: {np.mean(accuracies):.2f}%")
        print(f"  Median accuracy: {np.median(accuracies):.2f}%")
        print(f"  Best class: {max(zip(accuracies, class_names))[1]} ({max(accuracies):.2f}%)")
        print(f"  Worst class: {min(zip(accuracies, class_names))[1]} ({min(accuracies):.2f}%)")

    def plot_top_k_accuracy(self, save_path='topk_accuracy.png', max_k=10):
        """
        Plot Top-K accuracy curve.

        Args:
            save_path: Path to save the plot
            max_k: Maximum K to evaluate
        """
        print("\nCalculating Top-K accuracy...")

        predictions, labels, probabilities = self.get_predictions()

        # Calculate Top-K accuracy for each K
        topk_accuracies = []
        k_values = range(1, min(max_k + 1, probabilities.shape[1] + 1))

        for k in k_values:
            # Get top K predictions
            topk_pred = np.argsort(probabilities, axis=1)[:, -k:]

            # Check if true label is in top K
            correct = np.any(topk_pred == labels[:, np.newaxis], axis=1)
            accuracy = correct.mean() * 100

            topk_accuracies.append(accuracy)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, topk_accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('K')
        plt.ylabel('Accuracy (%)')
        plt.title('Top-K Accuracy')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        plt.ylim([0, 105])

        # Annotate values
        for k, acc in zip(k_values, topk_accuracies):
            if k in [1, 3, 5, max_k]:
                plt.annotate(f'{acc:.1f}%',
                           xy=(k, acc),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Top-K accuracy plot saved to {save_path}")
        plt.close()

        # Print results
        print("\nTop-K Accuracy:")
        for k, acc in zip(k_values, topk_accuracies):
            print(f"  Top-{k}: {acc:.2f}%")

    def visualize_attention_maps(self, image_path, save_dir='attention_maps', layer_idx=-1):
        """
        Visualize attention maps for a single image.

        Args:
            image_path: Path to input image
            save_dir: Directory to save attention visualizations
            layer_idx: Which transformer layer to visualize (-1 = last layer)
        """
        print(f"\nVisualizing attention for {image_path}...")

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Load and preprocess image
        transform = get_val_transforms()
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Get attention maps
        self.model.eval()
        attention_maps = self.model.get_attention_maps(image_tensor)

        # Select layer
        attn_map = attention_maps[layer_idx][0]  # (num_heads, num_patches+1, num_patches+1)

        # Focus on attention from CLS token to patches
        cls_attention = attn_map[:, 0, 1:].cpu().numpy()  # (num_heads, num_patches)

        # Reshape to spatial grid (8x8 for 56x56 images with 7x7 patches)
        grid_size = int(np.sqrt(cls_attention.shape[1]))
        cls_attention_grid = cls_attention.reshape(-1, grid_size, grid_size)

        # Plot original image and attention for each head
        num_heads = cls_attention_grid.shape[0]
        fig, axes = plt.subplots(2, (num_heads + 1) // 2 + 1, figsize=(15, 6))
        axes = axes.flatten()

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention maps
        for head_idx in range(num_heads):
            ax = axes[head_idx + 1]

            # Overlay attention on image
            attn = cls_attention_grid[head_idx]

            # Upsample attention to image size
            from scipy.ndimage import zoom
            attn_upsampled = zoom(attn, (56 / grid_size, 56 / grid_size), order=1)

            ax.imshow(image)
            ax.imshow(attn_upsampled, alpha=0.6, cmap='jet')
            ax.set_title(f'Head {head_idx + 1}')
            ax.axis('off')

        # Hide extra subplots
        for idx in range(num_heads + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = save_dir / f"{Path(image_path).stem}_attention.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention maps saved to {save_path}")
        plt.close()

    def show_misclassifications(self, save_path='misclassifications.png', num_samples=20):
        """
        Show gallery of misclassified images.

        Args:
            save_path: Path to save the gallery
            num_samples: Number of misclassifications to show
        """
        print("\nFinding misclassifications...")

        self.model.eval()

        misclassified_samples = []

        for images, labels in tqdm(self.test_loader):
            images_device = images.to(self.device)

            # Forward pass
            outputs = self.model(images_device)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Find misclassifications
            mask = predicted.cpu() != labels
            misclassified_indices = torch.where(mask)[0]

            for idx in misclassified_indices:
                if len(misclassified_samples) >= num_samples:
                    break

                img = images[idx]
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()
                confidence = probs[idx, pred_label].item()

                misclassified_samples.append({
                    'image': img,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence
                })

            if len(misclassified_samples) >= num_samples:
                break

        if len(misclassified_samples) == 0:
            print("No misclassifications found!")
            return

        # Plot gallery
        rows = (len(misclassified_samples) + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes

        for idx, sample in enumerate(misclassified_samples):
            ax = axes[idx]

            # Denormalize image
            img = sample['image'].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)

            # Get class names
            true_name = self.idx_to_species[str(sample['true_label'])]
            pred_name = self.idx_to_species[str(sample['pred_label'])]

            # Truncate long names
            if len(true_name) > 20:
                true_name = true_name[:17] + '...'
            if len(pred_name) > 20:
                pred_name = pred_name[:17] + '...'

            title = f"True: {true_name}\nPred: {pred_name}\n({sample['confidence']*100:.1f}%)"
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        # Hide extra subplots
        for idx in range(len(misclassified_samples), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassifications gallery saved to {save_path}")
        plt.close()

    def generate_classification_report(self, save_path='classification_report.txt'):
        """
        Generate detailed classification report.

        Args:
            save_path: Path to save the report
        """
        print("\nGenerating classification report...")

        predictions, labels, _ = self.get_predictions()

        # Generate report
        target_names = [self.idx_to_species[str(i)] for i in range(len(self.idx_to_species))]

        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=4,
            zero_division=0
        )

        # Save to file
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)

        print(f"Classification report saved to {save_path}")
        print("\nSample from report:")
        print(report[:500] + "...")


def main():
    """
    Run comprehensive evaluation on a trained model.
    """
    print("Plant Vision Transformer Evaluation")
    print("=" * 60)

    # Configuration
    checkpoint_path = 'checkpoints/best_checkpoint.pth'
    data_dir = 'plant_data'
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load label mapping
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    num_classes = label_mapping['num_species']

    # Load model
    print("Loading model...")
    model = create_vit_small(num_classes=num_classes, img_size=56)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%\n")

    # Load test data
    print("Loading test dataset...")
    _, _, test_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4,
        classification_mode='species'
    )

    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device)

    # Run evaluations
    print("\n" + "=" * 60)
    print("Running Evaluations...")
    print("=" * 60)

    evaluator.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png',
        top_k=30
    )

    evaluator.plot_per_class_accuracy(
        save_path=output_dir / 'per_class_accuracy.png',
        top_k=30
    )

    evaluator.plot_top_k_accuracy(
        save_path=output_dir / 'topk_accuracy.png',
        max_k=10
    )

    evaluator.show_misclassifications(
        save_path=output_dir / 'misclassifications.png',
        num_samples=20
    )

    evaluator.generate_classification_report(
        save_path=output_dir / 'classification_report.txt'
    )

    print("\n" + "=" * 60)
    print(f"✓ Evaluation complete! Results saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
