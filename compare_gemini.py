"""
Compare Vision Transformer with Gemini 2.5 Flash

This script compares the accuracy of a trained Vision Transformer model
against Google's Gemini 2.5 Flash on a stratified sample of plant images.

Features:
- Stratified sampling (proportional to species frequency)
- Structured JSON output from Gemini (forced valid species selection)
- Configurable sample size for cost management
- Side-by-side accuracy comparison

Requirements:
    pip install google-generativeai python-dotenv

Setup:
    Create a .env file in the project directory with:
    GOOGLE_API_KEY=your_api_key_here
"""

import torch
import numpy as np
from pathlib import Path
import json
import os
from tqdm import tqdm
from collections import defaultdict
import random
import time
from PIL import Image
import base64
import io

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")
    print("Falling back to system environment variables")

# Google AI imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")

from vision_transformer import create_vit_small, create_vit_pretrained
from dataset import create_dataloaders, get_val_transforms


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Sample configuration
    'sample_size': 200,  # Number of images to test (reduce to 10-20 for testing)
    'random_seed': 42,

    # Model paths
    'checkpoint_path': 'checkpoints/best_checkpoint.pth',
    'data_dir': 'plant_data',
    'label_mapping_path': 'label_mapping.json',

    # Gemini configuration - compare multiple models
    'gemini_models': {
        'flash': 'gemini-2.5-flash-lite',           # Gemini 2.5 Flash (fast, cheap)
        'pro': 'gemini-2.5-pro'  # Gemini 2.5 Pro (slower, better)
    },
    'api_delay': 0.1,  # Delay between API calls (seconds) to avoid rate limits

    # Output
    'output_file': 'gemini_comparison_results.json',
    'save_intermediate': True,  # Save results after each prediction

    # API Key (loaded from .env file)
    'api_key': os.getenv('GOOGLE_API_KEY', None),  # Reads from .env file after load_dotenv()
}


# ============================================================================
# Stratified Sampler
# ============================================================================

class StratifiedSampler:
    """
    Sample images proportionally based on species frequency.

    Species with more images in the test set get proportionally more samples.
    """

    def __init__(self, test_loader, sample_size, seed=42):
        """
        Args:
            test_loader: DataLoader for test set
            sample_size: Total number of images to sample
            seed: Random seed for reproducibility
        """
        self.test_loader = test_loader
        self.sample_size = sample_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def sample(self):
        """
        Sample images proportionally from test set.

        Returns:
            List of (image_tensor, label, image_path) tuples
        """
        # Collect all samples organized by species
        species_samples = defaultdict(list)

        print("Collecting test set samples...")
        for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader)):
            for img_idx in range(len(images)):
                image = images[img_idx]
                label = labels[img_idx].item()

                # Get the actual image path from the dataset
                dataset = self.test_loader.dataset
                sample_idx = batch_idx * self.test_loader.batch_size + img_idx
                if sample_idx < len(dataset.samples):
                    img_path = dataset.samples[sample_idx][0]
                    species_samples[label].append((image, label, img_path))

        # Calculate proportional allocation
        total_available = sum(len(samples) for samples in species_samples.values())
        species_counts = {species: len(samples) for species, samples in species_samples.items()}

        print(f"\nTotal available test samples: {total_available}")
        print(f"Number of species in test set: {len(species_samples)}")

        # Allocate samples proportionally
        sampled = []
        remaining = self.sample_size

        # Sort species by count to handle rounding better
        sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)

        for species, count in sorted_species:
            if remaining <= 0:
                break

            # Calculate proportional allocation
            proportion = count / total_available
            allocated = max(1, int(proportion * self.sample_size))
            allocated = min(allocated, remaining, count)

            # Randomly sample from this species
            species_pool = species_samples[species]
            selected = random.sample(species_pool, allocated)
            sampled.extend(selected)
            remaining -= allocated

        # If we need more samples due to rounding, randomly add from any species
        while len(sampled) < self.sample_size:
            species = random.choice(list(species_samples.keys()))
            pool = species_samples[species]
            # Find samples not already selected
            existing_paths = {s[2] for s in sampled}
            available = [s for s in pool if s[2] not in existing_paths]
            if available:
                sampled.append(random.choice(available))
            else:
                break  # No more unique samples available

        # Shuffle the final sample
        random.shuffle(sampled)

        print(f"\nSampled {len(sampled)} images for comparison")

        # Print distribution
        sampled_species = defaultdict(int)
        for _, label, _ in sampled:
            sampled_species[label] += 1

        print(f"Distribution across {len(sampled_species)} species:")
        print(f"  Min samples per species: {min(sampled_species.values())}")
        print(f"  Max samples per species: {max(sampled_species.values())}")
        print(f"  Avg samples per species: {np.mean(list(sampled_species.values())):.1f}")

        return sampled


# ============================================================================
# Gemini Evaluator
# ============================================================================

class GeminiEvaluator:
    """
    Evaluate images using Google's Gemini models.
    """

    def __init__(self, api_key, model_name, species_list, api_delay=0.1, model_label='gemini'):
        """
        Args:
            api_key: Google AI API key
            model_name: Gemini model name (e.g., 'gemini-2.0-flash-exp', 'gemini-2.0-pro-exp')
            species_list: List of valid species names
            api_delay: Delay between API calls (seconds)
            model_label: Label for this model (e.g., 'flash', 'pro')
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        if not api_key:
            raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.model_label = model_label
        self.species_list = sorted(species_list)
        self.api_delay = api_delay

        # Create the structured schema
        self.create_schema()

    def create_schema(self):
        """Create JSON schema for structured output."""
        # Format species list for prompt
        self.species_options = '\n'.join([f"  - {species}" for species in self.species_list])

    def create_prompt(self):
        """
        Create the prompt for Gemini with structured JSON output.
        """
        prompt = f"""You are a plant identification expert. Identify the plant species in this image.

Observe that the plant is known to be one of the following species. You MUST choose from ONLY these {len(self.species_list)} valid species names:

{self.species_options}

IMPORTANT:
- Return ONLY valid JSON in the exact format specified
- The species name must EXACTLY match one of the species names listed above
- Use underscores, not spaces (e.g., "Acer_rubrum" not "Acer rubrum")
- Return your answer as JSON with this structure:

{{
  "species": "Species_name_from_list_above",
  "confidence": "high|medium|low"
}}

Respond with ONLY the JSON object, no other text."""

        return prompt

    def predict(self, image_path):
        """
        Get prediction from Gemini for a single image.

        Args:
            image_path: Path to image file

        Returns:
            tuple: (predicted_species, raw_response) or (None, error_message)
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')

            # Create prompt
            prompt = self.create_prompt()

            # Call Gemini API
            response = self.model.generate_content([prompt, img])

            # Add delay to avoid rate limits
            time.sleep(self.api_delay)

            # Parse response
            response_text = response.text.strip()

            # Try to extract JSON from response
            # Sometimes the model includes markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            # Parse JSON
            result = json.loads(response_text)
            species = result.get('species', None)

            # Validate species
            if species not in self.species_list:
                return None, f"Invalid species returned: {species}"

            return species, response_text

        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}. Response: {response_text[:200]}"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def evaluate_batch(self, samples, idx_to_species):
        """
        Evaluate a batch of images.

        Args:
            samples: List of (image_tensor, label, image_path) tuples
            idx_to_species: Mapping from label index to species name

        Returns:
            List of prediction results with keys prefixed by model_label
        """
        results = []

        print(f"\nQuerying Gemini {self.model_label.upper()} ({self.model_name}) for {len(samples)} images...")
        print(f"Estimated cost: ${len(samples) * 0.00001875:.4f}")
        print("This may take a few minutes...\n")

        for i, (image_tensor, true_label, img_path) in enumerate(tqdm(samples, desc=f'Gemini {self.model_label.upper()}')):
            true_species = idx_to_species[str(true_label)]

            # Get Gemini prediction
            pred_species, raw_response = self.predict(img_path)

            result = {
                'sample_index': i,
                'image_path': str(img_path),
                'true_label': true_label,
                'true_species': true_species,
                f'{self.model_label}_prediction': pred_species,
                f'{self.model_label}_raw_response': raw_response,
                f'{self.model_label}_correct': pred_species == true_species if pred_species else False,
                f'{self.model_label}_error': pred_species is None
            }

            results.append(result)

        return results


# ============================================================================
# Vision Transformer Evaluator
# ============================================================================

class VitEvaluator:
    """
    Evaluate images using trained Vision Transformer.
    """

    def __init__(self, checkpoint_path, device, num_classes):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: torch device
            num_classes: Number of output classes (from filtered dataset)
        """
        self.device = device

        # Load checkpoint first to check actual number of classes
        print("Loading Vision Transformer model...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Infer actual number of classes from checkpoint
        actual_num_classes = None
        for key in checkpoint['model_state_dict'].keys():
            if 'head.weight' in key or 'fc.weight' in key:
                actual_num_classes = checkpoint['model_state_dict'][key].shape[0]
                break

        if actual_num_classes is None:
            actual_num_classes = num_classes

        if actual_num_classes != num_classes:
            print(f"\n⚠️  WARNING: Model/Dataset class count mismatch!")
            print(f"   Checkpoint has {actual_num_classes} classes")
            print(f"   Current dataset has {num_classes} classes")
            print(f"   This likely means the checkpoint was trained with different filtering.")
            print(f"   Using checkpoint's {actual_num_classes} classes for model.\n")

        # Create model with actual number of classes from checkpoint
        try:
            self.model = create_vit_pretrained(
                model_name='vit_small_patch16_224',
                num_classes=actual_num_classes,
                pretrained=False
            )
            print(f"Created pretrained model with {actual_num_classes} classes")
        except:
            self.model = create_vit_small(num_classes=actual_num_classes, img_size=224)
            print(f"Created custom model with {actual_num_classes} classes")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        self.actual_num_classes = actual_num_classes

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    @torch.no_grad()
    def evaluate_batch(self, samples, idx_to_species):
        """
        Evaluate a batch of images.

        Args:
            samples: List of (image_tensor, label, image_path) tuples
            idx_to_species: Mapping from label index to species name

        Returns:
            List of prediction results
        """
        results = []

        print(f"\nEvaluating with Vision Transformer on {len(samples)} images...")

        for i, (image_tensor, true_label, img_path) in enumerate(tqdm(samples)):
            # Move to device
            image_batch = image_tensor.unsqueeze(0).to(self.device)

            # Get prediction
            output = self.model(image_batch)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

            pred_label = predicted.item()

            # Handle case where model predicts a class not in the filtered dataset
            # This happens when checkpoint was trained on different filtering
            if str(pred_label) not in idx_to_species:
                # Model predicted a class that was filtered out
                # Mark as incorrect and use placeholder
                pred_species = f"FILTERED_CLASS_{pred_label}"
                vit_correct = False
            else:
                pred_species = idx_to_species[str(pred_label)]
                vit_correct = pred_label == true_label

            true_species = idx_to_species.get(str(true_label), f"UNKNOWN_{true_label}")

            result = {
                'sample_index': i,
                'image_path': str(img_path),
                'true_label': true_label,
                'true_species': true_species,
                'vit_prediction_label': pred_label,
                'vit_prediction_species': pred_species,
                'vit_confidence': confidence.item(),
                'vit_correct': vit_correct,
                'vit_prediction_filtered': str(pred_label) not in idx_to_species
            }

            results.append(result)

        return results


# ============================================================================
# Main Comparison Function
# ============================================================================

def compare_models():
    """
    Compare Vision Transformer and Gemini on stratified sample.
    """
    print("=" * 80)
    print("Vision Transformer vs Gemini 2.5 Flash Comparison")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load configuration from checkpoint
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
    min_images_per_species = 50
    if 'config' in checkpoint and checkpoint['config']:
        min_images_per_species = checkpoint['config'].get('min_images_per_species', 0)

    print(f"Min images per species filter from checkpoint: {min_images_per_species}")

    # Load test dataset FIRST to get the correct label mapping
    print("\nLoading test dataset...")
    _, _, test_loader, species_to_idx = create_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=32,
        num_workers=4,
        img_size=224,
        classification_mode='species',
        min_images_per_species=min_images_per_species
    )

    # Get label mapping from the dataset (this will match the filtering)
    # This ensures we have the same species list that was used during training
    idx_to_species = {str(v): k for k, v in species_to_idx.items()}
    species_list = list(species_to_idx.keys())
    num_classes = len(species_list)

    print(f"Number of species after filtering: {num_classes}")
    print(f"Sample size: {CONFIG['sample_size']}")

    # Stratified sampling
    sampler = StratifiedSampler(test_loader, CONFIG['sample_size'], CONFIG['random_seed'])
    samples = sampler.sample()

    # Evaluate with Vision Transformer
    vit_evaluator = VitEvaluator(CONFIG['checkpoint_path'], device, num_classes)
    vit_results = vit_evaluator.evaluate_batch(samples, idx_to_species)

    # Evaluate with Gemini Flash
    gemini_flash_evaluator = GeminiEvaluator(
        api_key=CONFIG['api_key'],
        model_name=CONFIG['gemini_models']['flash'],
        species_list=species_list,
        api_delay=CONFIG['api_delay'],
        model_label='flash'
    )
    gemini_flash_results = gemini_flash_evaluator.evaluate_batch(samples, idx_to_species)

    # Evaluate with Gemini Pro
    gemini_pro_evaluator = GeminiEvaluator(
        api_key=CONFIG['api_key'],
        model_name=CONFIG['gemini_models']['pro'],
        species_list=species_list,
        api_delay=CONFIG['api_delay'],
        model_label='pro'
    )
    gemini_pro_results = gemini_pro_evaluator.evaluate_batch(samples, idx_to_species)

    # Merge results
    print("\nMerging results...")
    merged_results = []
    for vit_res, flash_res, pro_res in zip(vit_results, gemini_flash_results, gemini_pro_results):
        merged = {**vit_res}
        # Add flash results
        for k, v in flash_res.items():
            if k not in ['sample_index', 'image_path', 'true_label', 'true_species']:
                merged[k] = v
        # Add pro results
        for k, v in pro_res.items():
            if k not in ['sample_index', 'image_path', 'true_label', 'true_species']:
                merged[k] = v
        merged_results.append(merged)

    # Calculate accuracies
    vit_correct = sum(1 for r in merged_results if r['vit_correct'])
    flash_correct = sum(1 for r in merged_results if r['flash_correct'])
    pro_correct = sum(1 for r in merged_results if r['pro_correct'])
    flash_errors = sum(1 for r in merged_results if r['flash_error'])
    pro_errors = sum(1 for r in merged_results if r['pro_error'])
    vit_filtered_predictions = sum(1 for r in merged_results if r.get('vit_prediction_filtered', False))

    total = len(merged_results)
    vit_accuracy = 100.0 * vit_correct / total
    flash_accuracy = 100.0 * flash_correct / total
    pro_accuracy = 100.0 * pro_correct / total

    # Print results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nTested on {total} stratified samples from the test set\n")
    print(f"Vision Transformer:  {vit_accuracy:.2f}% accuracy ({vit_correct}/{total} correct)")
    print(f"Gemini 2.0 Flash:    {flash_accuracy:.2f}% accuracy ({flash_correct}/{total} correct)")
    print(f"Gemini 2.0 Pro:      {pro_accuracy:.2f}% accuracy ({pro_correct}/{total} correct)")

    if vit_filtered_predictions > 0:
        print(f"\n⚠️  Note: ViT made {vit_filtered_predictions} predictions for classes not in the filtered dataset")
        print(f"   This indicates the checkpoint was trained on a different set of species.")
        print(f"   These predictions were counted as incorrect.")

    if flash_errors > 0:
        print(f"\n⚠️  Note: {flash_errors} Gemini Flash predictions failed (errors/invalid responses)")

    if pro_errors > 0:
        print(f"\n⚠️  Note: {pro_errors} Gemini Pro predictions failed (errors/invalid responses)")

    # Determine winner
    accuracies = {
        'Vision Transformer': vit_accuracy,
        'Gemini 2.0 Flash': flash_accuracy,
        'Gemini 2.0 Pro': pro_accuracy
    }
    winner = max(accuracies, key=accuracies.get)
    winner_accuracy = accuracies[winner]

    print(f"\n{'='*80}")
    print(f"🏆 WINNER: {winner} with {winner_accuracy:.2f}% accuracy!")
    print(f"{'='*80}")

    # Show margins
    print(f"\nMargins:")
    for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        if model != winner:
            diff = winner_accuracy - acc
            print(f"  {winner} beats {model} by {diff:.2f} percentage points")

    # Save results
    config_without_api_key = CONFIG.copy()
    del config_without_api_key['api_key']

    output_data = {
        'config': config_without_api_key,
        'summary': {
            'total_samples': total,
            'vit_accuracy': vit_accuracy,
            'vit_correct': vit_correct,
            'flash_accuracy': flash_accuracy,
            'flash_correct': flash_correct,
            'flash_errors': flash_errors,
            'pro_accuracy': pro_accuracy,
            'pro_correct': pro_correct,
            'pro_errors': pro_errors,
            'winner': winner,
            'winner_accuracy': winner_accuracy
        },
        'detailed_results': merged_results
    }

    with open(CONFIG['output_file'], 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {CONFIG['output_file']}")
    print("=" * 80)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Check dependencies
    if not DOTENV_AVAILABLE:
        print("WARNING: python-dotenv not installed")
        print("Install with: pip install python-dotenv")
        print("Continuing with system environment variables...\n")

    if not GENAI_AVAILABLE:
        print("ERROR: google-generativeai not installed")
        print("Install with: pip install google-generativeai")
        exit(1)

    # Check API key
    if not CONFIG['api_key']:
        print("ERROR: API key not found!")
        print("\nPlease create a .env file in the project directory with:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("\nOr set the GOOGLE_API_KEY environment variable in your system.")
        exit(1)

    # Run comparison
    compare_models()
