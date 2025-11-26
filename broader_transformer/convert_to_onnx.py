"""
Convert trained Vision Transformer model to ONNX format with FP16 quantization.

This script takes a trained model checkpoint and converts it to ONNX format
optimized for mobile deployment with FP16 quantization for reduced model size.

Usage:
    python convert_to_onnx.py --checkpoint_dir checkpoints
    python convert_to_onnx.py --checkpoint_dir checkpoints_v2 --opset_version 14
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import onnx
import numpy as np

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from model import create_genus_vit_pretrained

# Try to import FP16 conversion tools
try:
    from onnxconverter_common import float16
    FP16_AVAILABLE = True
except ImportError:
    FP16_AVAILABLE = False


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in eval mode
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
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, config


def convert_to_onnx(model, output_path, img_size=224, opset_version=14):
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode
        output_path: Path to save ONNX model
        img_size: Input image size (default: 224)
        opset_version: ONNX opset version (default: 13)

    Returns:
        Path to saved ONNX model
    """
    print(f"\nConverting to ONNX format...")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Opset version: {opset_version}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ ONNX model saved to: {output_path}")

    # Get model size
    model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Model size (FP32): {model_size_mb:.2f} MB")

    return output_path


def quantize_to_fp16(onnx_model_path, output_path):
    """
    Quantize ONNX model to FP16.

    Args:
        onnx_model_path: Path to FP32 ONNX model
        output_path: Path to save FP16 ONNX model

    Returns:
        Path to quantized model
    """
    if not FP16_AVAILABLE:
        raise ImportError(
            "onnxconverter-common is required for FP16 quantization. "
            "Install with: pip install onnxconverter-common"
        )

    print(f"\nQuantizing to FP16...")

    # Load ONNX model
    model = onnx.load(onnx_model_path)

    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(model)

    # Save FP16 model
    onnx.save(model_fp16, output_path)

    print(f"✓ FP16 model saved to: {output_path}")

    # Get model sizes
    fp32_size_mb = Path(onnx_model_path).stat().st_size / (1024 * 1024)
    fp16_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"  Model size (FP32): {fp32_size_mb:.2f} MB")
    print(f"  Model size (FP16): {fp16_size_mb:.2f} MB")
    print(f"  Size reduction: {(1 - fp16_size_mb/fp32_size_mb) * 100:.1f}%")

    return output_path


def verify_onnx_model(onnx_model_path, pytorch_model, img_size=224):
    """
    Verify ONNX model produces similar outputs to PyTorch model.

    Args:
        onnx_model_path: Path to ONNX model
        pytorch_model: PyTorch model
        img_size: Input image size

    Returns:
        True if models match within tolerance
    """
    print(f"\nVerifying ONNX model...")

    import onnxruntime as ort

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(str(onnx_model_path))

    # Create test input
    test_input = torch.randn(1, 3, img_size, img_size)

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # Get ONNX output
    onnx_output = ort_session.run(
        None,
        {'input': test_input.numpy()}
    )[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Check if within tolerance
    tolerance = 1e-3 if 'fp16' in str(onnx_model_path) else 1e-5

    if max_diff < tolerance:
        print(f"✓ ONNX model verified (max diff < {tolerance})")
        return True
    else:
        print(f"⚠ Warning: ONNX model outputs differ from PyTorch (max diff {max_diff} >= {tolerance})")
        return False


def save_conversion_info(checkpoint_dir, info):
    """
    Save conversion information to JSON file.

    Args:
        checkpoint_dir: Checkpoint directory
        info: Conversion information dictionary
    """
    info_path = checkpoint_dir / 'onnx_conversion_info.json'

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nConversion info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX with FP16 quantization')
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
        help='Name of checkpoint file to convert (default: best_checkpoint.pth)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14, required for ViT models)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for conversion (default: cpu)'
    )
    parser.add_argument(
        '--skip_verification',
        action='store_true',
        help='Skip ONNX model verification'
    )

    args = parser.parse_args()

    # Convert paths
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("="*60)
    print("PyTorch to ONNX Conversion with FP16 Quantization")
    print("="*60)

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device=args.device)
    model = model.to('cpu')  # ONNX export works best on CPU

    # Define output paths
    onnx_fp32_path = checkpoint_dir / 'model_fp32.onnx'
    onnx_fp16_path = checkpoint_dir / 'model_fp16.onnx'

    # Convert to ONNX (FP32)
    convert_to_onnx(
        model,
        onnx_fp32_path,
        img_size=args.img_size,
        opset_version=args.opset_version
    )

    # Verify FP32 model
    if not args.skip_verification:
        try:
            verify_onnx_model(onnx_fp32_path, model, img_size=args.img_size)
        except Exception as e:
            print(f"⚠ Warning: Could not verify FP32 model: {e}")
            print("Install onnxruntime with: pip install onnxruntime")

    # Quantize to FP16
    try:
        quantize_to_fp16(onnx_fp32_path, onnx_fp16_path)

        # Verify FP16 model
        if not args.skip_verification:
            try:
                verify_onnx_model(onnx_fp16_path, model, img_size=args.img_size)
            except Exception as e:
                print(f"⚠ Warning: Could not verify FP16 model: {e}")

    except Exception as e:
        print(f"\n⚠ Warning: FP16 quantization failed: {e}")
        print("Install onnxconverter-common with: pip install onnxconverter-common")
        print("FP32 model is still available at:", onnx_fp32_path)

    # Save conversion info
    conversion_info = {
        'checkpoint_file': args.checkpoint_name,
        'img_size': args.img_size,
        'opset_version': args.opset_version,
        'fp32_model': str(onnx_fp32_path.name),
        'fp16_model': str(onnx_fp16_path.name) if onnx_fp16_path.exists() else None,
        'original_config': config
    }

    save_conversion_info(checkpoint_dir, conversion_info)

    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print("="*60)
    print(f"\nOutput files in {checkpoint_dir}:")
    print(f"  - {onnx_fp32_path.name} (FP32 ONNX model)")
    if onnx_fp16_path.exists():
        print(f"  - {onnx_fp16_path.name} (FP16 ONNX model, recommended for mobile)")
    print(f"  - onnx_conversion_info.json (conversion metadata)")

    if onnx_fp16_path.exists():
        print(f"\nFor mobile deployment, use: {onnx_fp16_path.name}")
    else:
        print(f"\nFor mobile deployment, use: {onnx_fp32_path.name}")


if __name__ == "__main__":
    main()
