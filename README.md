# Plant Vision Transformer

A Vision Transformer (ViT) implementation for plant species classification, optimized for 8GB VRAM GPUs.

## Features

- **Memory-efficient Vision Transformer** with three model sizes (Tiny, Small, Base)
- **Automated web scraping** for plant images from Harvard Arboretum and PlantNet
- **Flexible classification modes**: Species-only or Species+Organ joint classification
- **Comprehensive training pipeline** with mixed precision, early stopping, and checkpointing
- **Evaluation tools**: Confusion matrices, attention visualization, per-class accuracy, and more

## Project Structure

```
vision_transformer/
├── single_plant_test.ipynb    # Web scraping notebook
├── plant_list.txt              # List of plant species to scrape
├── vision_transformer.py       # Vision Transformer architecture
├── dataset.py                  # Dataset loader with train/val/test splits
├── train.py                    # Training script
├── evaluate.py                 # Evaluation and visualization tools
├── requirements.txt            # Python dependencies
├── plant_data/                 # Downloaded plant images (auto-created)
│   ├── Species_name_1/
│   │   ├── flower/
│   │   ├── leaf/
│   │   ├── bark/
│   │   ├── fruit/
│   │   └── whole_plant/
│   └── Species_name_2/
│       └── ...
├── checkpoints/                # Model checkpoints (auto-created)
└── evaluation_results/         # Evaluation outputs (auto-created)
```

## Installation

### 1. Install PyTorch

For CUDA 11.8 (adjust for your CUDA version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ChromeDriver (for web scraping)

Download ChromeDriver from https://chromedriver.chromium.org/ and add to PATH.

## Usage

### Step 1: Scrape Plant Images

1. Open `single_plant_test.ipynb` in Jupyter or VS Code
2. Ensure `plant_list.txt` contains your target species
3. Run all cells to scrape images

**Note**: Scraping 221 species can take several hours. Consider testing with a subset first:
```python
START_INDEX = 0
END_INDEX = 10  # Only scrape first 10 species
```

The scraper will create organized folders:
```
plant_data/
  Populus_tremuloides/
    flower/
      plantnet_0.jpg (56x56)
      plantnet_1.jpg (56x56)
      ...
    leaf/
      plantnet_0.jpg
      ...
    whole_plant/
      harvard_0.jpg
      ...
```

### Step 2: Train the Model

```bash
python train.py
```

**Configuration** (edit in `train.py`):
```python
config = {
    'data_dir': 'plant_data',
    'batch_size': 64,              # Adjust based on VRAM
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'model_size': 'small',         # 'tiny', 'small', or 'base'
    'classification_mode': 'species',  # 'species' or 'joint'
    'use_amp': True,               # Mixed precision training
    'use_checkpoint': False,       # Gradient checkpointing (saves VRAM)
}
```

**Model Sizes**:
- **Tiny**: ~3M parameters, ~150 MB VRAM (batch 64)
- **Small**: ~11M parameters, ~300 MB VRAM (batch 64) - **Recommended**
- **Base**: ~43M parameters, ~600 MB VRAM (batch 64, with checkpointing)

**Training Output**:
- `checkpoints/best_checkpoint.pth` - Best model based on validation accuracy
- `checkpoints/latest_checkpoint.pth` - Latest model state
- `checkpoints/training_curves.png` - Loss and accuracy plots
- `checkpoints/training_history.json` - Training metrics
- `label_mapping.json` - Species and organ type mappings

### Step 3: Evaluate the Model

```bash
python evaluate.py
```

**Generates**:
1. **Confusion Matrix** - Shows which species are confused with each other
2. **Per-Class Accuracy** - Bar chart of accuracy for each species
3. **Top-K Accuracy** - Accuracy curve for top-1, top-3, top-5 predictions
4. **Misclassifications Gallery** - Visual examples of incorrect predictions
5. **Classification Report** - Detailed precision, recall, F1-score per class
6. **Attention Maps** - Visualize what the model focuses on (optional)

All outputs saved to `evaluation_results/`

### Step 4: Use Trained Model for Inference

```python
import torch
from PIL import Image
from vision_transformer import create_vit_small
from dataset import get_val_transforms
import json

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_vit_small(num_classes=221, img_size=56)
checkpoint = torch.load('checkpoints/best_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load label mapping
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
idx_to_species = label_mapping['idx_to_species']

# Prepare image
transform = get_val_transforms()
image = Image.open('test_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = probs.topk(5, dim=1)

# Print results
print("Top 5 Predictions:")
for prob, idx in zip(top5_probs[0], top5_indices[0]):
    species = idx_to_species[str(idx.item())]
    print(f"  {species}: {prob.item()*100:.2f}%")
```

## Architecture Details

### Vision Transformer (ViT-Small)

- **Input**: 56×56 RGB images
- **Patch Size**: 7×7 (produces 8×8 = 64 patches)
- **Embedding Dimension**: 192
- **Layers**: 12 transformer blocks
- **Attention Heads**: 6
- **MLP Ratio**: 4.0
- **Parameters**: ~11M
- **FLOPs**: ~0.6 GFLOPs per image

### Key Components

1. **Patch Embedding**: Splits image into 7×7 patches and projects to 192-dim embeddings
2. **Positional Encoding**: Learnable positional embeddings for spatial information
3. **Class Token**: Special learnable token for classification
4. **Transformer Blocks**: Multi-head self-attention + MLP with pre-norm architecture
5. **Classification Head**: Linear layer mapping class token to species logits

### Training Features

- **Mixed Precision Training (AMP)**: 2x faster training with minimal accuracy loss
- **Gradient Checkpointing**: Trade computation for memory (enables larger models)
- **Learning Rate Warmup**: 5 epochs warmup for stable training
- **Cosine Annealing**: Gradual learning rate decay
- **Early Stopping**: Stop training when validation stops improving
- **Data Augmentation**: Random flips, rotations, color jitter, affine transforms

## Performance Tips

### For 8GB VRAM GPU:

1. **Start with VIT-Small** (recommended)
2. **Batch size**:
   - 64-128 for VIT-Tiny
   - 32-64 for VIT-Small
   - 16-32 for VIT-Base (with checkpointing)
3. **Enable mixed precision** (`use_amp=True`)
4. **For larger models**, enable gradient checkpointing (`use_checkpoint=True`)

### For CPU Training:

1. Use **VIT-Tiny** model
2. Lower batch size (8-16)
3. Disable mixed precision (`use_amp=False`)
4. Expect 10-20x slower training

### Data Collection Tips:

1. **Test with small subset first** (10 species)
2. **Run overnight** for full 221 species
3. **Check scraper output** - some species may have fewer images
4. **Harvard images**: Stored in `whole_plant/` (not organ-labeled)

## Classification Modes

### 1. Species Classification (`classification_mode='species'`)

- Predicts plant species only (ignores organ type)
- 221 output classes (one per species)
- **Use case**: General plant identification

### 2. Joint Classification (`classification_mode='joint'`)

- Predicts species AND organ type
- 221 × 5 = 1,105 output classes
- **Use case**: Fine-grained plant part recognition

## Common Issues

### ChromeDriver Version Mismatch
```
Error: Chrome version mismatch
```
**Solution**: Download ChromeDriver matching your Chrome version from https://chromedriver.chromium.org/

### Out of Memory (OOM)
```
CUDA out of memory
```
**Solution**:
- Reduce batch size
- Use smaller model (Tiny instead of Small)
- Enable gradient checkpointing
- Close other GPU applications

### ImportError: einops
```
ModuleNotFoundError: No module named 'einops'
```
**Solution**: `pip install einops`

### Empty Dataset
```
Total samples: 0
```
**Solution**: Run the scraper first to collect plant images

## Future Improvements

- [ ] Add transfer learning from ImageNet pretrained weights
- [ ] Implement contrastive learning for better feature representations
- [ ] Add ensemble methods (multiple models voting)
- [ ] Support for larger image sizes (224×224)
- [ ] Multi-GPU training with DistributedDataParallel
- [ ] Integration with Weights & Biases for experiment tracking
- [ ] Mobile-optimized models (quantization, pruning)

## Citation

If you use this code, please cite:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```

## License

MIT License - Feel free to use for research and educational purposes.

## Acknowledgments

- Plant images from Harvard Arboretum and PlantNet
- Vision Transformer architecture based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
