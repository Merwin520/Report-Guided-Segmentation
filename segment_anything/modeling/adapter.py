"""
MedSAM ViT-B Adapter Implementation
Inspired by DETRIS DenseAligner but optimized for MedSAM image encoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from torch import Tensor

# Try to import FDConv, fallback to standard Conv2d if not available
try:
    from .FDConv import FDConv
    HAS_FDCONV = True
except ImportError:
    HAS_FDCONV = False


class BasicConv2d(nn.Module):
    """
    Basic 2D convolution block optimized for MedSAM ViT-B
    Supports both FDConv and standard Conv2d
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_fdconv: bool = True, **kwargs) -> None:
        super().__init__()
        
        if use_fdconv and HAS_FDCONV:
            # FDConv configuration optimized for medical images
            self.conv = FDConv(
                in_channels, out_channels, bias=True,
                # Medical image specific settings
                kernel_num=3,
                use_fdconv_if_c_gt=4,
                use_fdconv_if_k_in=[1, 3, 5],
                use_fbm_if_k_in=[3, 5],
                temp=1.2,
                kernel_temp=1.0,
                param_ratio=2,
                param_reduction=1.0,
                ksm_local_act='gelu',
                ksm_global_act='sigmoid',
                fbm_cfg={
                    'k_list': [2, 4, 6],
                    'lowfreq_att': True,   # Important for medical images
                    'spatial_group': min(16, in_channels),
                    'spatial_kernel': 5,
                    'init': 'zero',
                },
                **kwargs
            )
        else:
            # Standard Conv2d fallback
            self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
            
        # Use SyncBatchNorm to match DETRIS implementation
        self.bn = nn.SyncBatchNorm(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MedSAMAdapter(nn.Module):
    """
    Medical SAM Adapter for ViT-B Image Encoder
    
    Based on DenseAligner architecture but optimized for MedSAM:
    - Input: [B, 4097, 768] (1 CLS + 64*64 patches, embed_dim=768)
    - Output: [B, 4097, 768] (same shape)
    - Multi-scale dense connections with medical image optimizations
    """
    
    def __init__(
        self,
        embed_dim: int = 768,           # MedSAM ViT-B embed dimension
        adapter_dim: int = 128,         # Internal adapter dimension
        ch1x1: int = 64,               # 1x1 conv output channels
        ch3x3red: int = 8,             # 3x3 conv reduction channels
        ch3x3: int = 32,               # 3x3 conv output channels
        ch5x5red: int = 8,             # 5x5 conv reduction channels
        ch5x5: int = 32,               # 5x5 conv output channels
        skip_connect: bool = True,      # Global skip connection
        use_fdconv: bool = True,        # Use FDConv if available
        dropout_rate: float = 0.1,      # Dropout rate
    ) -> None:
        super().__init__()
        
        # Validate dimensions for MedSAM ViT-B
        assert embed_dim == 768, f"MedSAM ViT-B embed_dim must be 768, got {embed_dim}"
        assert ch1x1 + ch3x3 + ch5x5 == adapter_dim, \
            f"Branch channels sum ({ch1x1 + ch3x3 + ch5x5}) must equal adapter_dim ({adapter_dim})"
        
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.skip_connect = skip_connect
        
        # Down-projection: 768 -> adapter_dim (following DenseAligner D_fc1)
        self.D_fc1 = nn.Linear(embed_dim, adapter_dim)
        
        # Multi-branch dense convolution structure (exactly like DenseAligner)
        conv_block = BasicConv2d
        
        # Branch 1: 1x1 convolution
        self.dense_branch1 = conv_block(
            adapter_dim, ch1x1, 
            kernel_size=1, 
            use_fdconv=use_fdconv
        )
        
        # Branch 2: 1x1 reduction + 3x3 convolution
        self.dense_branch2 = nn.Sequential(
            conv_block(
                adapter_dim + ch1x1, ch3x3red, 
                kernel_size=1, 
                use_fdconv=use_fdconv
            ),
            conv_block(
                ch3x3red, ch3x3, 
                kernel_size=3, padding=1, 
                use_fdconv=use_fdconv
            )
        )
        
        # Branch 3: 1x1 reduction + 5x5 convolution
        self.dense_branch3 = nn.Sequential(
            conv_block(
                adapter_dim + ch1x1 + ch3x3, ch5x5red, 
                kernel_size=1, 
                use_fdconv=use_fdconv
            ),
            conv_block(
                ch5x5red, ch5x5, 
                kernel_size=5, padding=2, 
                use_fdconv=use_fdconv
            )
        )
        
        # Up-projection: adapter_dim -> 768 (following DenseAligner D_fc2)
        self.D_fc2 = nn.Linear(adapter_dim, embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()

    # 在 segment_anything/modeling/adapter.py 中，替换 _initialize_weights 方法：

    def _initialize_weights(self):
        """Initialize weights, completely skip FDConv modules"""
        for name, module in self.named_modules():
            # 完全跳过FDConv相关的模块
            if 'FDConv' in module.__class__.__name__:
                print(f"Skipping FDConv initialization: {name}")
                continue
            
            # 跳过包含FDConv的BasicConv2d
            if hasattr(module, 'conv') and 'FDConv' in module.conv.__class__.__name__:
                print(f"Skipping BasicConv2d with FDConv: {name}")
                continue
            
            # 对其他模块进行正常初始化
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # 再次确认这不是FDConv
                if 'FDConv' not in module.__class__.__name__:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass following DenseAligner structure exactly
        
        Args:
            x: Input features from MedSAM ViT-B encoder [B, 4097, 768]
               4097 = 1 CLS token + 4096 spatial tokens (64x64 patches)
        
        Returns:
            Adapted features [B, 4097, 768]
        """
        B, N, C = x.shape
        
        # Validate MedSAM ViT-B specific dimensions
        assert N == 4097, f"Expected 4097 tokens (1 CLS + 64*64 patches), got {N}"
        assert C == 768, f"Expected 768 channels, got {C}"
        
        # Save for global skip connection
        identity = x
        
        # Down-projection (following DenseAligner)
        x0 = self.D_fc1(x)  # [B, 4097, adapter_dim]
        x0 = F.relu(x0, inplace=True)
        
        # Split CLS token and spatial tokens (following DenseAligner split_token logic)
        cls_token = x0[:, 0:1, :]       # [B, 1, adapter_dim]
        spatial_tokens = x0[:, 1:, :]   # [B, 4096, adapter_dim]
        
        # Reshape to 2D spatial format: [B, 4096, adapter_dim] -> [B, adapter_dim, 64, 64]
        H = W = 64  # MedSAM ViT-B: 1024/16 = 64
        xs = spatial_tokens.reshape(B, H, W, self.adapter_dim).permute(0, 3, 1, 2)
        
        # Multi-branch dense connections (exactly following DenseAligner)
        dense_branch1 = self.dense_branch1(xs)  # [B, ch1x1, 64, 64]
        
        dense_branch2 = self.dense_branch2(
            torch.cat([xs, dense_branch1], dim=1)
        )  # [B, ch3x3, 64, 64]
        
        dense_branch3 = self.dense_branch3(
            torch.cat([xs, dense_branch1, dense_branch2], dim=1)
        )  # [B, ch5x5, 64, 64]
        
        # Concatenate all branch outputs
        outputs = torch.cat([dense_branch1, dense_branch2, dense_branch3], dim=1)
        # [B, adapter_dim, 64, 64] where adapter_dim = ch1x1 + ch3x3 + ch5x5
        
        # Add residual connection in spatial domain
        outputs = outputs + xs
        
        # Reshape back to sequence format: [B, adapter_dim, 64, 64] -> [B, 4096, adapter_dim]
        outputs = outputs.reshape(B, self.adapter_dim, H * W).permute(0, 2, 1)
        
        # Recombine with CLS token
        outputs = torch.cat([cls_token, outputs], dim=1)  # [B, 4097, adapter_dim]
        
        # Add residual connection in feature space (following DenseAligner)
        outputs = outputs + x0
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        # Up-projection back to original dimension
        outputs = self.D_fc2(outputs)  # [B, 4097, 768]
        
        # Global skip connection
        if self.skip_connect:
            outputs = outputs + identity
            
        return outputs


class MedSAMLightAdapter(nn.Module):
    """
    Lightweight version of MedSAM Adapter
    Fewer parameters, suitable for resource-constrained scenarios
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        adapter_dim: int = 64,          # Smaller internal dimension
        expansion_ratio: int = 4,        # Expansion ratio for MLP
        skip_connect: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        assert embed_dim == 768, f"MedSAM ViT-B embed_dim must be 768, got {embed_dim}"
        
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.skip_connect = skip_connect
        
        # Simplified adapter structure
        self.down_proj = nn.Linear(embed_dim, adapter_dim)
        
        # Depthwise separable convolution for multi-scale features
        self.spatial_conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(adapter_dim, adapter_dim * expansion_ratio, 3, padding=1, groups=adapter_dim),
            nn.SyncBatchNorm(adapter_dim * expansion_ratio, eps=0.001),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(adapter_dim * expansion_ratio, adapter_dim, 1),
            nn.SyncBatchNorm(adapter_dim, eps=0.001),
        )
        
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features [B, 4097, 768]
        Returns:
            Adapted features [B, 4097, 768]
        """
        B, N, C = x.shape
        assert N == 4097 and C == 768, f"Expected [B, 4097, 768], got [B, {N}, {C}]"
        
        identity = x
        
        # Down-projection
        x_down = self.down_proj(x)  # [B, 4097, adapter_dim]
        x_down = F.relu(x_down, inplace=True)
        
        # Separate tokens
        cls_token = x_down[:, 0:1, :]
        spatial_tokens = x_down[:, 1:, :]
        
        # 2D processing
        H = W = 64
        spatial_features = spatial_tokens.reshape(B, H, W, self.adapter_dim).permute(0, 3, 1, 2)
        
        # Spatial convolution with residual
        adapted_features = self.spatial_conv(spatial_features) + spatial_features
        
        # Back to sequence format
        adapted_spatial = adapted_features.permute(0, 2, 3, 1).reshape(B, H*W, self.adapter_dim)
        adapted_tokens = torch.cat([cls_token, adapted_spatial], dim=1)
        adapted_tokens = adapted_tokens + x_down
        
        # Up-projection with dropout
        adapted_tokens = self.dropout(adapted_tokens)
        output = self.up_proj(adapted_tokens)
        
        if self.skip_connect:
            output = output + identity
            
        return output


def create_medsam_adapter(
    variant: str = 'full',
    use_fdconv: bool = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create MedSAM adapters
    
    Args:
        variant: 'full' or 'light'
        use_fdconv: Whether to use FDConv (auto-detect if None)
        **kwargs: Additional arguments for adapter
    
    Returns:
        MedSAM adapter instance
    """
    if use_fdconv is None:
        use_fdconv = HAS_FDCONV
    
    if variant == 'full':
        return MedSAMAdapter(use_fdconv=use_fdconv, **kwargs)
    elif variant == 'light':
        return MedSAMLightAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'light'")