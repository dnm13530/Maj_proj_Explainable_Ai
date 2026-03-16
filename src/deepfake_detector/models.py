import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DetectorOutput, ModelConfig


# ---------------------------------------------------------------------------
# CNN Backbone
# ---------------------------------------------------------------------------

class CNNBackbone(nn.Module):
    """Lightweight 4-layer CNN for local forensic artifact extraction."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # [B, C, H', W']


# ---------------------------------------------------------------------------
# ViT Module
# ---------------------------------------------------------------------------

class ViTModule(nn.Module):
    """
    Vision Transformer operating on CNN feature map patches.
    Captures self-attention weights from the final layer via forward hook.
    """

    def __init__(self, config: ModelConfig, feature_hw: int = 28):
        super().__init__()
        self.config = config
        # Each spatial position of the CNN feature map becomes one token
        self.num_patches = feature_hw * feature_hw
        seq_len = self.num_patches + 1  # +1 for CLS token

        # Input to patch_embed is cnn_out_channels (one vector per spatial position)
        patch_dim = config.cnn_out_channels
        self.patch_embed = nn.Linear(patch_dim, config.vit_embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vit_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, config.vit_embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vit_embed_dim,
            nhead=config.vit_num_heads,
            dim_feedforward=config.vit_embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.vit_num_layers)

        # Dedicated final attention layer for weight capture
        self.final_attn = nn.MultiheadAttention(
            embed_dim=config.vit_embed_dim,
            num_heads=config.vit_num_heads,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(config.vit_embed_dim)

        self._attention_weights: Optional[torch.Tensor] = None
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feature_map.shape

        # Flatten spatial dims to patch sequence
        x = feature_map.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.patch_embed(x)                      # [B, N, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)               # [B, N+1, D]
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer layers (all but final)
        x = self.transformer(x)

        # Final attention layer — capture weights
        x_norm = self.final_norm(x)
        attn_out, attn_weights = self.final_attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
        self._attention_weights = attn_weights  # [B, num_heads, seq_len, seq_len]
        x = x + attn_out

        return x, attn_weights  # x: [B, N+1, D]


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls_token))  # [B, num_classes]


# ---------------------------------------------------------------------------
# Hybrid CNN-ViT
# ---------------------------------------------------------------------------

class HybridCNNViT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cnn = CNNBackbone(out_channels=config.cnn_out_channels)
        # After 3x MaxPool2d on 224 input: 224/8 = 28
        self.vit = ViTModule(config, feature_hw=28)
        self.head = ClassificationHead(config.vit_embed_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> DetectorOutput:
        B = x.shape[0]
        input_hw = (x.shape[2], x.shape[3])

        feature_map = self.cnn(x)                        # [B, C, H', W']
        tokens, attn_weights = self.vit(feature_map)     # [B, N+1, D], [B, heads, N+1, N+1]
        cls_token = tokens[:, 0, :]                      # [B, D]
        logits = self.head(cls_token)                    # [B, 2]

        probs = F.softmax(logits, dim=-1)
        confidence = probs[:, 1]                         # deepfake probability

        attn_map = self.generate_artifact_attention_map(attn_weights, input_hw)

        # Return single-image output for batch size 1, else return batch
        label = int(torch.argmax(logits, dim=-1)[0].item())
        conf_val = float(confidence[0].item())

        return DetectorOutput(
            label=label,
            confidence=conf_val,
            attention_map=attn_map[0],  # [H, W] for first image
        )

    def forward_batch(self, x: torch.Tensor):
        """Returns logits, confidence scores, and attention maps for a full batch."""
        feature_map = self.cnn(x)
        tokens, attn_weights = self.vit(feature_map)
        cls_token = tokens[:, 0, :]
        logits = self.head(cls_token)
        probs = F.softmax(logits, dim=-1)
        confidence = probs[:, 1]
        attn_maps = self.generate_artifact_attention_map(attn_weights, (x.shape[2], x.shape[3]))
        return logits, confidence, attn_maps

    def generate_artifact_attention_map(
        self,
        attn_weights: torch.Tensor,
        input_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Extract CLS attention, average over heads, reshape, upsample, normalize.
        Returns: [B, H, W]
        """
        # attn_weights: [B, num_heads, seq_len, seq_len]
        # CLS token is index 0; its attention to all patch tokens is index 1:
        cls_attn = attn_weights[:, :, 0, 1:]  # [B, heads, N]
        avg_attn = cls_attn.mean(dim=1)        # [B, N]

        B, N = avg_attn.shape
        grid_size = int(math.sqrt(N))

        # Reshape to spatial grid
        spatial = avg_attn.reshape(B, 1, grid_size, grid_size)  # [B, 1, g, g]

        # Upsample to input resolution
        upsampled = F.interpolate(spatial, size=input_hw, mode='bilinear', align_corners=False)
        upsampled = upsampled.squeeze(1)  # [B, H, W]

        # Normalize to [0, 1]
        B_size = upsampled.shape[0]
        maps = []
        for i in range(B_size):
            m = upsampled[i]
            max_val = m.max()
            if max_val > 0:
                m = m / max_val
            maps.append(m)

        return torch.stack(maps, dim=0)  # [B, H, W]

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> 'HybridCNNViT':
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(ckpt['config'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model
