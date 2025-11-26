# Quick Start Guide

This guide provides quick instructions for getting started with each of the three Vision Transformer models in this repository.

---

## Model 1: Broader Transformer (500 Genera) - **RECOMMENDED**

**Best performance for plant genus classification**

### Prerequisites
1. Download GBIF data from [https://www.gbif.org/dataset/7a3679ef-5582-4aaa-81f0-8c2545cafc81/project](https://www.gbif.org/dataset/7a3679ef-5582-4aaa-81f0-8c2545cafc81/project)
2. Place downloaded data in `broader_transformer/data/` folder

### Setup & Training
```bash
cd broader_transformer

# Install dependencies
pip install -r requirements.txt

# Process GBIF data (may need complete_metadata.py first)
python process_gbif_data.py

# Train the model
python train.py

# Evaluate the model
python evaluate.py
```

### Optional: Model Export & Analysis
```bash
# Convert to ONNX for mobile deployment (FP16 quantized)
python convert_to_onnx.py --checkpoint_dir checkpoints_v2

# Analyze confidence thresholds
python evaluate_confidence_thresholds.py --checkpoint_dir checkpoints_v2 --dataset test
```

**Note**: Our model achieved ~92% validation accuracy with stronger regularization.

---

## Model 2: House Plants Transformer

**For house plant species classification**

### Prerequisites
1. Download data from [Kaggle: House Plant Species Dataset](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species/data)
2. Place downloaded data in `house_plant_species/` folder

### Setup & Training
```bash
cd house_plants_transformer

# Install dependencies
pip install -r requirements.txt

# Resize images to 224x224
python resize_images.py

# Train the model
python train.py
```

**Note**: Optimized for common house plant identification.

---

## Model 3: Broad Transformer (PlantNet) - **DO NOT USE**

⚠️ **Warning**: Gathering the data for this model violates PlantNet's Terms of Service (and probably Copyright Law). This information is provided for educational purposes only.

### If You Had Permission (Hypothetical)
```bash
cd broad_transformer

# Install dependencies
pip install -r requirements.txt

# Collect data (issue)
# Run single_plant_test.ipynb

# Train the model
python train.py

# Evaluate the model
python evaluate.py
```

---

## Choosing the Right Model

| Model | Dataset | Classes | Best For | Status |
|-------|---------|---------|----------|--------|
| **Broader Transformer** | GBIF (Public) | 500 genera | General plant genus ID | ✅ **Recommended** |
| **House Plants Transformer** | Kaggle | ~50 species | House plant ID | ✅ Available |
| **Broad Transformer** | PlantNet | 221 species | - | ❌ TOS Violation |

**Start with Broader Transformer**

---

## General Tips

### First Time Setup
1. **Check Python version**: Requires Python 3.8+
2. **GPU recommended**: Training on CPU is 10-20x slower
3. **Storage**: Ensure 10-50GB free space depending on model

### Training Considerations
- **Monitor GPU usage**: Use `nvidia-smi` to check VRAM
- **Save checkpoints**: Training can take hours, enable auto-resume
- **Adjust batch size**: If out of memory, reduce batch_size in config

### Getting Help
- Check `README.md` for detailed documentation
- Review requirements.txt for dependency issues
- Ensure data is in the correct directory structure
- Verify GPU drivers and CUDA installation (for GPU training)
