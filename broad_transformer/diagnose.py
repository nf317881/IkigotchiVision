"""
Diagnostic script to check what's happening with the model
"""
import torch
import torch.nn as nn
from broad_transformer.dataset import create_dataloaders
from vision_transformer import create_vit_pretrained, create_vit_small

print("="*60)
print("DIAGNOSTIC SCRIPT")
print("="*60)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n1. Device: {device}")

# Load data
print("\n2. Loading dataset...")
try:
    train_loader, val_loader, test_loader, species_to_idx = create_dataloaders(
        data_dir='plant_data',
        batch_size=4,
        num_workers=0,
        img_size=224,
        classification_mode='species'
    )
    print(f"   ✓ Dataset loaded successfully")
    print(f"   Number of classes: {len(species_to_idx)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
except Exception as e:
    print(f"   ✗ Error loading dataset: {e}")
    exit(1)

# Get a batch
print("\n3. Testing data loading...")
try:
    images, labels = next(iter(train_loader))
    print(f"   ✓ Batch shape: {images.shape}")
    print(f"   ✓ Labels shape: {labels.shape}")
    print(f"   ✓ Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   ✓ Label range: [{labels.min()}, {labels.max()}]")
    print(f"   Number of unique labels in batch: {len(torch.unique(labels))}")
except Exception as e:
    print(f"   ✗ Error getting batch: {e}")
    exit(1)

# Try loading pretrained model
print("\n4. Testing pretrained model loading...")
try:
    model = create_vit_pretrained(
        model_name='vit_small_patch16_224',
        num_classes=len(species_to_idx),
        pretrained=True
    )
    print(f"   ✓ Pretrained model loaded successfully")
    print(f"   Model type: {type(model).__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    pretrained_ok = True
except Exception as e:
    print(f"   ✗ Error loading pretrained model: {e}")
    print(f"   Trying custom model instead...")
    pretrained_ok = False

    try:
        model = create_vit_small(
            num_classes=len(species_to_idx),
            img_size=224,
            use_checkpoint=False
        )
        print(f"   ✓ Custom model loaded successfully")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"   ✗ Error loading custom model: {e}")
        exit(1)

# Test forward pass
print("\n5. Testing forward pass...")
try:
    model = model.to(device)
    images_device = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images_device)

    print(f"   ✓ Forward pass successful")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")

    # Check predictions
    probs = torch.softmax(outputs, dim=1)
    preds = outputs.argmax(dim=1)
    print(f"   Predictions: {preds.tolist()}")
    print(f"   Max probabilities: {[f'{p:.3f}' for p in probs.max(dim=1)[0].tolist()]}")

except Exception as e:
    print(f"   ✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test backward pass
print("\n6. Testing backward pass...")
try:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward
    outputs = model(images_device)
    loss = criterion(outputs, labels.to(device))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"   ✓ Backward pass successful")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Expected initial loss (random): ~{torch.log(torch.tensor(len(species_to_idx))).item():.4f}")

except Exception as e:
    print(f"   ✗ Error in backward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test learning rate
print("\n7. Checking learning rates...")
if pretrained_ok:
    print(f"   Recommended LR (fine-tuning): 1e-4 to 3e-4")
    print(f"   ⚠ If using 3e-5, might be too low and train very slowly")
else:
    print(f"   Recommended LR (from scratch): 1e-3 to 3e-3")
    print(f"   ⚠ If using 3e-5, definitely too low!")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("\nIf all tests passed, the model should work!")
print("If you're getting 0% accuracy:")
print("  1. Check learning rate (should be 1e-4 for pretrained, 1e-3 for scratch)")
print("  2. Wait longer (accuracy might start improving after epoch 10-20)")
print("  3. Check if model is actually learning (loss should decrease)")
