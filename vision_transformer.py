"""
Vision Transformer for Plant Classification

A memory-efficient Vision Transformer implementation optimized for 8GB VRAM.
Designed for 56x56 RGB images with organ-type classification.

Architecture features:
- Small patch size (7x7) suitable for 56x56 images
- Efficient embedding dimensions
- Gradient checkpointing support for memory savings
- Multi-head self-attention
- MLP with GELU activation
- Layer normalization
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    For 56x56 images with 7x7 patches, we get 8x8 = 64 patches.
    """
    def __init__(self, img_size=56, patch_size=7, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Sequential(
            # Rearrange image into patches: (B, C, H, W) -> (B, N, P*P*C)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_size * patch_size * in_channels),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            (batch_size, n_patches, embed_dim)
        """
        return self.projection(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim=192, num_heads=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches + 1, embed_dim)
        Returns:
            (batch_size, n_patches + 1, embed_dim)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation.
    """
    def __init__(self, embed_dim=192, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with pre-norm architecture.
    """
    def __init__(self, embed_dim=192, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        # Pre-norm: normalize before attention
        x = x + self.attn(self.norm1(x))

        # Pre-norm: normalize before MLP
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for plant organ classification.

    Optimized for:
    - 56x56 RGB images
    - 8GB VRAM constraint
    - Multi-class classification (species + organ type)
    """
    def __init__(
        self,
        img_size=56,
        patch_size=7,
        in_channels=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_checkpoint=False
    ):
        """
        Args:
            img_size: Input image size (default: 56)
            patch_size: Size of image patches (default: 7, gives 8x8=64 patches)
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
            embed_dim: Embedding dimension (default: 192, memory efficient)
            depth: Number of transformer blocks (default: 12)
            num_heads: Number of attention heads (default: 6)
            mlp_ratio: Ratio of MLP hidden dim to embed dim (default: 4.0)
            dropout: Dropout rate (default: 0.1)
            use_checkpoint: Use gradient checkpointing to save memory (default: False)
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            (batch_size, num_classes)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory during training
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Classification: use only the class token
        cls_token_final = x[:, 0]  # (B, embed_dim)

        # Classification head
        logits = self.head(cls_token_final)  # (B, num_classes)

        return logits

    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization.

        Args:
            x: (batch_size, channels, height, width)
        Returns:
            List of attention maps from each layer
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Collect attention maps
        attention_maps = []
        for block in self.blocks:
            # Get attention from each block
            attn = block.attn
            qkv = attn.qkv(block.norm1(x))
            B_temp, N, C = x.shape
            qkv = qkv.reshape(B_temp, N, 3, attn.num_heads, attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_map = (q @ k.transpose(-2, -1)) * attn.scale
            attn_map = attn_map.softmax(dim=-1)
            attention_maps.append(attn_map.detach())

            # Forward pass
            x = block(x)

        return attention_maps


def create_vit_small(num_classes=1000, img_size=56, use_checkpoint=False):
    """
    Create a small Vision Transformer optimized for 8GB VRAM.

    Configuration:
    - 12 layers
    - 192 embedding dimensions
    - 6 attention heads
    - ~11M parameters
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=7,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_checkpoint=use_checkpoint
    )


def create_vit_tiny(num_classes=1000, img_size=56, use_checkpoint=False):
    """
    Create a tiny Vision Transformer for very limited VRAM or fast training.

    Configuration:
    - 8 layers
    - 128 embedding dimensions
    - 4 attention heads
    - ~3M parameters
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=7,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=128,
        depth=8,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        use_checkpoint=use_checkpoint
    )


def create_vit_base(num_classes=1000, img_size=56, use_checkpoint=True):
    """
    Create a base Vision Transformer (larger model, requires gradient checkpointing).

    Configuration:
    - 12 layers
    - 384 embedding dimensions
    - 8 attention heads
    - ~43M parameters

    Note: Requires use_checkpoint=True for 8GB VRAM
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=7,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        use_checkpoint=use_checkpoint
    )


if __name__ == "__main__":
    # Test the model
    print("Testing Vision Transformer...")

    # Create model
    model = create_vit_small(num_classes=221, img_size=56)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: VisionTransformer-Small")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 3, 56, 56)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test with different batch sizes to estimate VRAM usage
    print("\n" + "="*60)
    print("Estimated VRAM usage (forward + backward pass):")
    print("="*60)

    for bs in [16, 32, 64, 128]:
        params_size = total_params * 4 / 1024 / 1024  # MB
        activation_size = bs * 64 * 192 * 4 / 1024 / 1024 * 12  # Rough estimate
        gradient_size = params_size  # Same as params
        total_vram = params_size + activation_size + gradient_size

        print(f"Batch size {bs:3d}: ~{total_vram:.0f} MB")

    print("\n Model test completed successfully!")
