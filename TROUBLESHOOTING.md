# Troubleshooting Guide

## Issue: 0% Accuracy in First Few Epochs

This is **NORMAL** and **EXPECTED** for your dataset! Here's why:

### Your Challenge:
- **161 classes** (species that have data)
- **213 total species** in label mapping
- **~4,500 training images** total
- **Average: <30 images per class**

### Why 0% Accuracy is Normal Initially:

1. **Random guessing accuracy**: 100 / 161 = **0.62%**
   - Even a random model would only get 0.62% correct
   - So 0% in early epochs just means "worse than random" initially

2. **The model is still learning!**
   - What matters is: **Is the LOSS decreasing?**
   - Loss should drop from ~5.0-5.5 → ~3.5-4.0 in first 20-30 epochs
   - Accuracy will jump suddenly once loss gets low enough

3. **With pretrained models:**
   - First few epochs: Model adjusts pretrained features
   - Epochs 10-30: Loss drops steadily
   - Epochs 30-50: Accuracy starts improving (1% → 5% → 10%)
   - Epochs 50-100: Faster improvement (10% → 20% → 30%+)

## What to Watch For

### ✅ Good Signs (Model is Learning):
```
Epoch 1:  Loss: 5.42, Acc: 0.00%    ← Normal!
Epoch 5:  Loss: 4.87, Acc: 0.00%    ← Loss dropping = good
Epoch 10: Loss: 4.23, Acc: 0.41%    ← First correct prediction!
Epoch 20: Loss: 3.85, Acc: 2.15%    ← Improving
Epoch 50: Loss: 3.12, Acc: 12.5%    ← Much better
Epoch 100: Loss: 2.65, Acc: 28.3%   ← Good progress
```

### ❌ Bad Signs (Model NOT Learning):
```
Epoch 1:  Loss: 5.42, Acc: 0.00%
Epoch 10: Loss: 5.41, Acc: 0.00%    ← Loss not decreasing!
Epoch 20: Loss: 5.40, Acc: 0.00%    ← This is bad
```

If loss is NOT decreasing after 20 epochs:
- Learning rate might be too low
- Or model might not be loading correctly

## Pydantic Warnings

The warnings you see:
```
UnsupportedFieldAttributeWarning: The 'frozen' attribute...
```

These are **HARMLESS** and come from the `timm` library. They don't affect training.

### To suppress them (already done in latest train.py):
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
```

## Checking If Model is Learning

### Method 1: Run Diagnostic Script
```bash
python diagnose.py
```

This will test:
- Data loading
- Model loading
- Forward/backward pass
- Learning rate settings

### Method 2: Check Loss in Training Output

Look at the console output during training:
```
Epoch 5/500
------------------------------------------------------------
Training: 100%|██████████| 141/141 [01:23<00:00, loss=4.87, acc=0.00]
```

**Key metric**: `loss=4.87`
- Should be decreasing each epoch
- If stuck at same value → problem

### Method 3: Check Training History

After training, check:
```python
import json
with open('checkpoints/training_history.json') as f:
    history = json.load(f)

print("Train loss:", history['train_loss'][:10])  # First 10 epochs
print("Val loss:", history['val_loss'][:10])
```

Loss should show downward trend.

## Common Issues & Solutions

### Issue 1: Loss Not Decreasing

**Cause**: Learning rate too low or model frozen

**Solution**:
```python
# In train.py, check the learning rate:
config = {
    'learning_rate': None,  # Auto-set based on pretrained
    # Or manually set:
    # 'learning_rate': 3e-4,  # For pretrained
    # 'learning_rate': 1e-3,  # For from scratch
}
```

### Issue 2: CUDA Out of Memory

**Solution**: Reduce batch size
```python
config = {
    'batch_size': 16,  # Or even 8
}
```

### Issue 3: Training Very Slow

**Solution**: Use smaller model
```python
config = {
    'model_size': 'tiny',  # Instead of 'small'
}
```

### Issue 4: Loss Exploding (becoming NaN)

**Cause**: Learning rate too high

**Solution**:
```python
config = {
    'learning_rate': 1e-4,  # Lower from 3e-4
}
```

## Expected Training Timeline

With pretrained `vit_small_patch16_224`:

| Epoch Range | Expected Loss | Expected Val Acc | What's Happening |
|-------------|---------------|------------------|------------------|
| 1-10        | 5.4 → 4.5     | 0-1%             | Feature adjustment |
| 10-30       | 4.5 → 3.8     | 1-5%             | Learning patterns |
| 30-60       | 3.8 → 3.2     | 5-15%            | Improving |
| 60-100      | 3.2 → 2.8     | 15-30%           | Good progress |
| 100-200     | 2.8 → 2.4     | 30-45%           | Fine-tuning |
| 200-500     | 2.4 → 2.0     | 45-60%+          | Best performance |

**Note**: These are estimates. Your results may vary based on dataset quality.

## When to Stop Training Early

Stop if you see:
1. **Overfitting**: Val loss increasing while train loss decreasing
2. **No improvement**: Val acc not improving for 50+ epochs (early stopping will handle this)
3. **Good enough**: You're happy with current accuracy

## Dataset Recommendations

Your current dataset is small for 161 classes. For better results:

### Current Dataset Stats:
- 161 classes with data
- ~4,500 images total
- Average: ~28 images/class
- Some classes: 1-3 images only

### Recommendations:
1. **Collect more images**: Aim for 50-100 per species
2. **Remove rare classes**: Remove species with <5 images
3. **Use data sources**: PlantNet, iNaturalist, Pl@ntNet API

## Still Having Issues?

1. **Run diagnostic**: `python diagnose.py`
2. **Check this file**: Look at training output, is loss decreasing?
3. **Wait longer**: Accuracy might jump after epoch 20-30
4. **Share info**:
   - What's the loss at epoch 10? Epoch 20?
   - Is it decreasing?
   - What learning rate is being used?

## Quick Checklist

Before reporting an issue, verify:

- [ ] `timm` is installed: `pip show timm`
- [ ] Model loaded successfully (check console for "✓ Successfully loaded")
- [ ] Loss is decreasing (even if accuracy is 0%)
- [ ] Waited at least 20 epochs
- [ ] Learning rate is appropriate (3e-4 for pretrained, 1e-3 for scratch)
- [ ] No CUDA errors
- [ ] Dataset loads correctly

If all checked and still issues, there might be a real problem!
