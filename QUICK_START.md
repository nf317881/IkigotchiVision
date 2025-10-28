# Quick Start - Improved Vision Transformer

## TL;DR - Get Started in 2 Steps

### Step 1: Install Requirements
```bash
pip install timm
```

### Step 2: Train the Model
```bash
python train.py
```

That's it! The model now uses:
- ✅ Pretrained weights (transfer learning)
- ✅ 224x224 images (was 56x56)
- ✅ Better data augmentation
- ✅ Class weighting for imbalanced data
- ✅ Label smoothing
- ✅ Improved learning rate schedule

---

## What Changed?

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| **Image Size** | 56x56 | 224x224 | +15-30% |
| **Pretrained** | No | Yes (ImageNet) | +20-40% |
| **Learning Rate** | Linear decay | Cosine annealing | +5-15% |
| **Class Weights** | Disabled | Enabled | +5-10% |
| **Augmentation** | Basic | Advanced | +3-7% |
| **Label Smoothing** | None | 0.1 | +2-5% |
| **Expected Accuracy** | ~13% | **40-70%** | **+27-57%** |

---

## Training Time

- **Tiny model:** ~1-2 hours
- **Small model (recommended):** ~2-4 hours
- **Base model:** ~4-8 hours

---

## Customization

Edit these settings in `train.py` if needed:

```python
config = {
    'batch_size': 32,           # Lower if VRAM issues (try 16)
    'model_size': 'small',      # 'tiny', 'small', or 'base'
    'num_epochs': 500,          # Increase for better convergence
    'learning_rate': 3e-5,      # Fine-tuning LR (don't change unless needed)
}
```

---

## Troubleshooting

### CUDA Out of Memory
```python
config = {
    'batch_size': 16,          # Reduce from 32
    'model_size': 'tiny',      # Use smaller model
}
```

### Training Too Slow
```python
config = {
    'model_size': 'tiny',      # Faster model
    'num_epochs': 200,         # Fewer epochs
}
```

### Low Accuracy After Training
1. **Collect more images** (most important - aim for 50+ per species)
2. Try the best pretrained model:
   ```python
   config = {
       'pretrained_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
       'model_size': 'base',
   }
   ```
3. Increase epochs to 1000
4. Lower learning rate to 1e-5

---

## After Training

### Evaluate the Model
```bash
python evaluate.py
```

This generates:
- Confusion matrix
- Per-class accuracy
- Top-K accuracy curves
- Misclassification gallery
- Classification report

### Check Results
Look in `evaluation_results/` folder for visualizations and metrics.

---

## Need More Help?

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed explanation of all changes.
