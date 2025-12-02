"""
Compare Vision Transformer with Gemini 2.5 Flash Lite on Plant Genus Identification

This script tests Gemini 2.5 Flash Lite's ability to identify plant genera
by randomly selecting 100 images from the broader_transformer test set.

Features:
- Random sampling from test set (stratified across genera)
- Structured JSON output from Gemini
- Side-by-side accuracy comparison with trained ViT model
- Cost tracking for API usage

Requirements:
    pip install google-generativeai python-dotenv

Setup:
    Create a .env file in the project directory with:
    GOOGLE_API_KEY=your_api_key_here
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import random
import time
from PIL import Image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")

# Google AI imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_genus_vit_pretrained
from dataset import get_val_transforms


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Sample configuration
    'sample_size': 100,  # Number of images to test
    'random_seed': 42,

    # Paths
    'checkpoint_path': 'checkpoints_v2/best_checkpoint.pth',
    'manifest_file': None,  # Will auto-detect from test/val/train manifests
    'label_mapping': 'processed_data/metadata/label_mapping.json',
    'image_base_dir': 'processed_data/images',
    'metadata_dir': 'processed_data/metadata',

    # Gemini configuration
    'gemini_model': 'gemini-2.5-flash-lite',  # Gemini 2.0 Flash Experimental
    'api_delay': 0.5,  # Delay between API calls (seconds)

    # Output
    'output_file': 'gemini_genus_comparison.json',

    # API Key (loaded from .env file)
    'api_key': os.getenv('GOOGLE_API_KEY', None),
}


# ============================================================================
# Data Loading
# ============================================================================

def find_manifest(metadata_dir):
    """
    Find available manifest file (prefer test, then val, then train).

    Returns:
        Path to manifest file
    """
    metadata_path = Path(metadata_dir)

    # Try test manifest first
    test_manifest = metadata_path / 'test_manifest.csv'
    if test_manifest.exists():
        print(f"Using test manifest: {test_manifest}")
        return test_manifest

    # Try val manifest
    val_manifest = metadata_path / 'val_manifest.csv'
    if val_manifest.exists():
        print(f"Using validation manifest: {val_manifest}")
        return val_manifest

    # Try train manifest
    train_manifest = metadata_path / 'train_manifest.csv'
    if train_manifest.exists():
        print(f"Using train manifest: {train_manifest}")
        return train_manifest

    raise FileNotFoundError(f"No manifest files found in {metadata_dir}")


def load_test_data(manifest_path, label_mapping, image_base_dir):
    """
    Load data from manifest and label mapping.

    Returns:
        Tuple of (samples_list, idx_to_genus, genus_to_idx)
    """
    import pandas as pd

    print(f"\nLoading data from manifest...")

    # Load label mapping
    with open(label_mapping, 'r') as f:
        label_data = json.load(f)

    genus_to_idx = label_data['genus_to_id']
    idx_to_genus = label_data['id_to_genus']

    # Load manifest
    df = pd.read_csv(manifest_path)

    # Create samples list: (image_path, genus_id, genus_name)
    samples = []
    for _, row in df.iterrows():
        img_path = Path(image_base_dir) / row['file_path']
        genus_id = row['genus_id']
        genus_name = row['genus']

        if img_path.exists():
            samples.append((str(img_path), genus_id, genus_name))

    print(f"  Loaded {len(samples)} images")
    print(f"  Covering {len(set(s[2] for s in samples))} genera")

    return samples, idx_to_genus, genus_to_idx


def stratified_sample(samples, sample_size, seed=42):
    """
    Stratified random sampling across genera.

    Args:
        samples: List of (image_path, genus_id, genus_name) tuples
        sample_size: Number of images to sample
        seed: Random seed

    Returns:
        List of sampled (image_path, genus_id, genus_name) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"\nPerforming stratified sampling...")

    # Group by genus
    genus_samples = defaultdict(list)
    for sample in samples:
        genus_name = sample[2]
        genus_samples[genus_name].append(sample)

    total_available = len(samples)
    num_genera = len(genus_samples)

    print(f"  Total available: {total_available} images")
    print(f"  Number of genera: {num_genera}")

    # Calculate proportional allocation
    sampled = []
    remaining = sample_size

    # Sort genera by count for better allocation
    sorted_genera = sorted(genus_samples.items(), key=lambda x: len(x[1]), reverse=True)

    for genus_name, genus_pool in sorted_genera:
        if remaining <= 0:
            break

        # Proportional allocation
        proportion = len(genus_pool) / total_available
        allocated = max(1, int(proportion * sample_size))
        allocated = min(allocated, remaining, len(genus_pool))

        # Sample from this genus
        selected = random.sample(genus_pool, allocated)
        sampled.extend(selected)
        remaining -= allocated

    # Shuffle final sample
    random.shuffle(sampled)

    print(f"  Sampled {len(sampled)} images")

    # Print distribution
    sampled_genera = defaultdict(int)
    for _, _, genus_name in sampled:
        sampled_genera[genus_name] += 1

    print(f"  Distribution: {len(sampled_genera)} genera")
    print(f"    Min per genus: {min(sampled_genera.values())}")
    print(f"    Max per genus: {max(sampled_genera.values())}")
    print(f"    Avg per genus: {np.mean(list(sampled_genera.values())):.1f}")

    return sampled


# ============================================================================
# Gemini Evaluator
# ============================================================================

class GeminiGenusEvaluator:
    """
    Evaluate plant genus identification using Gemini.
    """

    def __init__(self, api_key, model_name, genus_list, api_delay=0.1):
        """
        Args:
            api_key: Google AI API key
            model_name: Gemini model name
            genus_list: List of valid genus names
            api_delay: Delay between API calls (seconds)
        """
        if not api_key:
            raise ValueError("API key required. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.genus_list = sorted(genus_list)
        self.api_delay = api_delay

    def create_prompt(self):
        """Create the prompt for Gemini with structured JSON output."""
        genus_options = '\n'.join([f"  - {genus}" for genus in self.genus_list])

        prompt = f"""You are a plant identification expert. Identify the plant genus in this image.

The plant is known to be one of the following {len(self.genus_list)} genera. You MUST choose from ONLY these valid genus names:

{genus_options}

IMPORTANT:
- Return ONLY valid JSON in the exact format specified
- The genus name must EXACTLY match one of the genus names listed above
- Return your answer as JSON with this structure:

{{
  "genus": "Genus_name_from_list_above",
  "confidence": "high|medium|low"
}}

Respond with ONLY the JSON object, no other text."""

        return prompt

    def predict(self, image_path):
        """
        Get genus prediction from Gemini for a single image.

        Args:
            image_path: Path to image file

        Returns:
            tuple: (predicted_genus, confidence, raw_response) or (None, None, error_message)
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
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            # Parse JSON
            result = json.loads(response_text)
            genus = result.get('genus', None)
            confidence = result.get('confidence', 'unknown')

            # Validate genus
            if genus not in self.genus_list:
                return None, None, f"Invalid genus returned: {genus}"

            return genus, confidence, response_text

        except json.JSONDecodeError as e:
            return None, None, f"JSON decode error: {e}. Response: {response_text[:200]}"
        except Exception as e:
            return None, None, f"Error: {str(e)}"

    def evaluate_batch(self, samples):
        """
        Evaluate a batch of images.

        Args:
            samples: List of (image_path, genus_id, genus_name) tuples

        Returns:
            List of prediction results
        """
        results = []

        # Estimate cost (Gemini 2.0 Flash: $0.00001875 per image for input <= 128k tokens)
        estimated_cost = len(samples) * 0.00001875

        print(f"\n{'='*80}")
        print(f"Querying Gemini ({self.model_name}) for {len(samples)} images...")
        print(f"Estimated cost: ${estimated_cost:.4f}")
        print(f"{'='*80}\n")

        for i, (img_path, true_genus_id, true_genus_name) in enumerate(tqdm(samples, desc='Gemini')):
            # Get Gemini prediction
            pred_genus, confidence, raw_response = self.predict(img_path)

            result = {
                'sample_index': i,
                'image_path': img_path,
                'true_genus_id': true_genus_id,
                'true_genus': true_genus_name,
                'gemini_prediction': pred_genus,
                'gemini_confidence': confidence,
                'gemini_raw_response': raw_response,
                'gemini_correct': pred_genus == true_genus_name if pred_genus else False,
                'gemini_error': pred_genus is None
            }

            results.append(result)

        return results


# ============================================================================
# Vision Transformer Evaluator
# ============================================================================

class VitGenusEvaluator:
    """
    Evaluate plant genus identification using trained Vision Transformer.
    """

    def __init__(self, checkpoint_path, device):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: torch device
        """
        self.device = device

        print("\nLoading Vision Transformer model...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        self.model = create_genus_vit_pretrained(
            model_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
            num_classes=500,
            pretrained=False
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        print(f"  Model loaded (epoch {checkpoint['epoch']})")
        try:
            print(f"  Best val accuracy: {checkpoint['best_val_accuracy']:.2f}%")
        except:
            pass

        # Load transforms
        self.transform = get_val_transforms(img_size=224)

    @torch.no_grad()
    def evaluate_batch(self, samples, idx_to_genus):
        """
        Evaluate a batch of images.

        Args:
            samples: List of (image_path, genus_id, genus_name) tuples
            idx_to_genus: Mapping from genus_id to genus name

        Returns:
            List of prediction results
        """
        results = []

        print(f"\n{'='*80}")
        print(f"Evaluating with Vision Transformer on {len(samples)} images...")
        print(f"{'='*80}\n")

        for i, (img_path, true_genus_id, true_genus_name) in enumerate(tqdm(samples, desc='ViT')):
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Get prediction
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

            pred_genus_id = predicted.item()
            pred_genus_name = idx_to_genus[str(pred_genus_id)]

            result = {
                'sample_index': i,
                'image_path': img_path,
                'true_genus_id': true_genus_id,
                'true_genus': true_genus_name,
                'vit_prediction_id': pred_genus_id,
                'vit_prediction': pred_genus_name,
                'vit_confidence': confidence.item(),
                'vit_correct': pred_genus_id == true_genus_id
            }

            results.append(result)

        return results


# ============================================================================
# Main Comparison Function
# ============================================================================

def main():
    """
    Compare Vision Transformer and Gemini on genus identification.
    """
    print("=" * 80)
    print("Vision Transformer vs Gemini 2.0 Flash - Genus Identification")
    print("=" * 80)

    # Check API key
    if not CONFIG['api_key']:
        print("\nERROR: API key not found!")
        print("Please create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Find available manifest file
    manifest_path = find_manifest(CONFIG['metadata_dir'])

    # Load test data
    samples, idx_to_genus, genus_to_idx = load_test_data(
        manifest_path,
        CONFIG['label_mapping'],
        CONFIG['image_base_dir']
    )

    genus_list = list(genus_to_idx.keys())
    print(f"Number of genera: {len(genus_list)}")

    # Stratified sampling
    sampled = stratified_sample(samples, CONFIG['sample_size'], CONFIG['random_seed'])

    # Evaluate with Vision Transformer
    vit_evaluator = VitGenusEvaluator(CONFIG['checkpoint_path'], device)
    vit_results = vit_evaluator.evaluate_batch(sampled, idx_to_genus)

    # Evaluate with Gemini
    gemini_evaluator = GeminiGenusEvaluator(
        api_key=CONFIG['api_key'],
        model_name=CONFIG['gemini_model'],
        genus_list=genus_list,
        api_delay=CONFIG['api_delay']
    )
    gemini_results = gemini_evaluator.evaluate_batch(sampled)

    # Merge results
    print("\nMerging results...")
    merged_results = []
    for vit_res, gemini_res in zip(vit_results, gemini_results):
        merged = {**vit_res}
        # Add gemini results (skip duplicate fields)
        for k, v in gemini_res.items():
            if k not in ['sample_index', 'image_path', 'true_genus_id', 'true_genus']:
                merged[k] = v
        merged_results.append(merged)

    # Calculate accuracies
    vit_correct = sum(1 for r in merged_results if r['vit_correct'])
    gemini_correct = sum(1 for r in merged_results if r['gemini_correct'])
    gemini_errors = sum(1 for r in merged_results if r['gemini_error'])

    total = len(merged_results)
    vit_accuracy = 100.0 * vit_correct / total
    gemini_accuracy = 100.0 * gemini_correct / total

    # Print results
    print("\n" + "=" * 80)
    print("GENUS IDENTIFICATION COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nTested on {total} stratified samples from test set")
    print(f"Number of genera: {len(genus_list)}\n")
    print(f"Vision Transformer:  {vit_accuracy:.2f}% accuracy ({vit_correct}/{total} correct)")
    print(f"Gemini 2.0 Flash:    {gemini_accuracy:.2f}% accuracy ({gemini_correct}/{total} correct)")

    if gemini_errors > 0:
        print(f"\n⚠️  Note: {gemini_errors} Gemini predictions failed (errors/invalid responses)")

    # Determine winner
    if vit_accuracy > gemini_accuracy:
        diff = vit_accuracy - gemini_accuracy
        print(f"\n{'='*80}")
        print(f"🏆 WINNER: Vision Transformer by {diff:.2f} percentage points!")
        print(f"{'='*80}")
    elif gemini_accuracy > vit_accuracy:
        diff = gemini_accuracy - vit_accuracy
        print(f"\n{'='*80}")
        print(f"🏆 WINNER: Gemini 2.0 Flash by {diff:.2f} percentage points!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"🤝 TIE: Both models achieved {vit_accuracy:.2f}% accuracy!")
        print(f"{'='*80}")

    # Confusion analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Both correct
    both_correct = sum(1 for r in merged_results if r['vit_correct'] and r['gemini_correct'])
    print(f"\nBoth correct: {both_correct}/{total} ({100.0*both_correct/total:.1f}%)")

    # Only ViT correct
    only_vit = sum(1 for r in merged_results if r['vit_correct'] and not r['gemini_correct'])
    print(f"Only ViT correct: {only_vit}/{total} ({100.0*only_vit/total:.1f}%)")

    # Only Gemini correct
    only_gemini = sum(1 for r in merged_results if not r['vit_correct'] and r['gemini_correct'])
    print(f"Only Gemini correct: {only_gemini}/{total} ({100.0*only_gemini/total:.1f}%)")

    # Both wrong
    both_wrong = sum(1 for r in merged_results if not r['vit_correct'] and not r['gemini_correct'])
    print(f"Both wrong: {both_wrong}/{total} ({100.0*both_wrong/total:.1f}%)")

    # Save results
    config_without_api_key = {k: v for k, v in CONFIG.items() if k != 'api_key'}

    output_data = {
        'config': config_without_api_key,
        'summary': {
            'total_samples': total,
            'num_genera': len(genus_list),
            'vit_accuracy': vit_accuracy,
            'vit_correct': vit_correct,
            'gemini_accuracy': gemini_accuracy,
            'gemini_correct': gemini_correct,
            'gemini_errors': gemini_errors,
            'both_correct': both_correct,
            'only_vit_correct': only_vit,
            'only_gemini_correct': only_gemini,
            'both_wrong': both_wrong
        },
        'detailed_results': merged_results
    }

    with open(CONFIG['output_file'], 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {CONFIG['output_file']}")
    print("=" * 80)


if __name__ == "__main__":
    if not GENAI_AVAILABLE:
        print("ERROR: google-generativeai not installed")
        print("Install with: pip install google-generativeai")
        sys.exit(1)

    main()
