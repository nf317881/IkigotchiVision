# House Plants Vision Transformer

A Vision Transformer implementation for classifying 47 house plant species.

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision timm pillow tqdm matplotlib einops
```

### 2. Resize Images

First, resize your images to 224x224:

```bash
cd house_plants_transformer
python resize_images.py
```

This will:
- Read images from `house_plant_species/`
- Resize them to 224x224
- Save to `house_plant_species_224/`

**Optional arguments:**
```bash
python resize_images.py --input_dir house_plant_species --output_dir house_plant_species_224 --size 224 --quality 95
```

To just count images without resizing:
```bash
python resize_images.py --count_only
```

### 3. Test the Dataset

Verify your dataset is loaded correctly:

```bash
python dataset.py
```

### 4. Train the Model

Start training:

```bash
python train.py
```

## Model Architecture

The model uses a Vision Transformer architecture with:
- **Input**: 224x224 RGB images
- **Patch size**: 16x16 (196 patches total)
- **Classes**: 47 house plant species
- **Variants**:
  - `tiny`: ~5M parameters, fast training
  - `small`: ~22M parameters, good balance (default)
  - `base`: ~86M parameters, best accuracy

## Training Features

- **Transfer Learning**: Uses pretrained ImageNet weights (via timm library)
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training
- **Data Augmentation**: Random crops, flips, rotations, color jitter
- **Class Balancing**: Weighted loss to handle class imbalance
- **Learning Rate Schedule**: Cosine annealing with warmup
- **Early Stopping**: Stops training when validation accuracy plateaus
- **Checkpointing**: Saves best model based on validation accuracy

## Configuration

Edit the `config` dictionary in [train.py](train.py:424) to customize:

```python
config = {
    'data_dir': 'house_plant_species_224',
    'batch_size': 32,
    'num_epochs': 100,
    'model_size': 'small',  # 'tiny', 'small', or 'base'
    'use_pretrained': True,  # Highly recommended!
    'min_images_per_species': 20,  # Filter species with few images
    ...
}
```

## Files

- **[model.py](model.py)**: Vision Transformer architecture
- **[dataset.py](dataset.py)**: Data loading and augmentation
- **[train.py](train.py)**: Training script with all the bells and whistles
- **[resize_images.py](resize_images.py)**: Image preprocessing script

## Output

Training produces:
- `checkpoints_houseplants/best_checkpoint.pth`: Best model weights
- `checkpoints_houseplants/training_history.json`: Training metrics
- `checkpoints_houseplants/training_curves.png`: Loss/accuracy plots
- `house_plants_label_mapping.json`: Species label mappings

## Dataset Structure

Expected structure:
```
house_plant_species_224/
    African Violet (Saintpaulia ionantha)/
        1.jpg
        2.jpg
        ...
    Aloe Vera/
        1.jpg
        2.jpg
        ...
    [... 47 species total ...]
```

## Tips

1. **Use pretrained weights**: Set `use_pretrained=True` for much better results
2. **Adjust batch size**: Reduce if you run out of VRAM
3. **Filter small classes**: Set `min_images_per_species=20` to remove species with too few examples
4. **Monitor training**: Watch the loss curve - it should decrease steadily
5. **Early stopping**: Default patience is 15 epochs - increase if training is still improving

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` (try 16 or 8)
- Use `model_size='tiny'`
- Set `use_checkpoint=True` for base model

**Timm not installed?**
```bash
pip install timm
```

**Low accuracy?**
- Make sure `use_pretrained=True`
- Increase `num_epochs`
- Check that you have enough training data per species
- Verify images resized correctly

## Similar to Broad Transformer

This implementation follows the same patterns as the broad transformer:
- Same training loop and optimization
- Same data augmentation strategies
- Same checkpoint and early stopping logic
- Adapted for house plants (species-only, no organ types)
