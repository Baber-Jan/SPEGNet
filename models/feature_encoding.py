"""
Feature encoding module for SPEGNet using SAM2's Hiera backbone specialized for visual tasks.
This implementation details the hierarchical architecture that enables efficient
multi-scale feature extraction through progressive windowed attention mechanisms.

Detailed Architecture:
---------------------
1. Patch Embedding Layer:
   - Input: x ∈ R^(H×W×3)
   - Output: z_0 ∈ R^(H/4×W/4×C_0)
   - Operation: Conv7x7(stride=4) followed by LayerNorm
   - Mathematical form: z_0 = LN(Conv(x) * W_e + b_e)
   - For 512x512 input: (512×512×3) -> (128×128×C_0)

2. Hierarchical Transformer Stages:
   Each stage i consists of multiple blocks with:
   
   a) Hierarchical Attention:
      Stage 1: Local attention within 8×8 regions
              A(z) = {Attention(z_k) | k ∈ regions(H/4 × W/4)}
      Stage 2: Local attention within 4×4 regions
              A(z) = {Attention(z_k) | k ∈ regions(H/8 × W/8)}
      Stage 3,4: Global attention
              A(z) = Attention(z)
   
   b) MLP Block:
      - Two-layer feed-forward: FFN(x) = W_2σ(W_1x + b_1) + b_2
      - GELU activation: σ(x) = x * Φ(x)
   
   c) Skip Connections:
      - z'_i = MSA(LN(z_i)) + z_i
      - z_(i+1) = FFN(LN(z'_i)) + z'_i

   d) Stage Transition:
      - Spatial pooling: 2×2 maxpool
      - Channel expansion: 2× via linear projection

Model Variants:
-----------------------------------------
Variant  Channels (C_i)               Blocks/Stage         
----------------------------------------------------------------
Tiny     [96-192-384-768]            [1-2-7-2]
Small    [96-192-384-768]            [1-2-11-2]
Base     [96-192-384-768]            [2-3-16-3]
Base+    [112-224-448-896]           [2-3-16-3]
Large    [144-288-576-1152]          [2-6-36-4]
Huge     [256-512-1024-2048]         [2-6-36-4]

Feature Map Dimensions (Large Variant):
------------------------------------
Stage    Output Size    Channels    RF Size    Parameters    Primary Role
-------------------------------------------------------------------------
Stem     H/4×W/4       144         4×4        7K            Basic patterns
Stage 1  H/4×W/4       144         32×32      152K          Local features
Stage 2  H/8×W/8       288         64×64      2.1M          Edge detection
Stage 3  H/16×W/16     576         128×128    45M           Object parts
Stage 4  H/32×W/32     1152        Global     39M           Scene context

Window Attention Details:
-----------------------
1. Local Window Creation:
   - Window size M×M (typically 7×7)
   - Non-overlapping partitioning
   - Relative position encoding within windows

2. Attention Computation (per window):
   - Q,K,V ∈ R^(M²×C)
   - Attention map: A ∈ R^(M²×M²)
   - Memory complexity: O(M²) vs O(N²) for global
   
3. Cross-window Connection:
   - Shifted window partitioning
   - Alternate between regular and shifted
   - Enables information flow between windows

Key Differences from Vanilla Hiera:
--------------------------------
1. Architecture:
   - SAM2 uses window attention vs global attention
   - Additional skip connections for stability

2. Training:
   - Pretrained on SA-1B + video data
   - Mask-based pretraining objectives
   - Multi-scale supervision

3. Memory Efficiency:
   - Streaming architecture design
   - Efficient attention mechanisms
   - Optional memory optimization

References:
----------
1. SAM2: Segment Anything Model in Images and Videos (Meta AI, 2024)
2. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles (ICML 2023)
3. Window Attention Is All You Need (ICCV 2023)
4. Feature Pyramid Networks for Object Detection (CVPR 2017)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Tuple
import gc
import logging
import numpy as np

from sam2.build_sam import build_sam2

logger = logging.getLogger(__name__)

class HieraSAM2FeatureEncoder(nn.Module):
    """
    SAM2's Hiera backbone for hierarchical feature extraction.
    See module docstring for detailed architecture description.
    """
    
    def __init__(
        self,
        model_cfg: str,
        checkpoint_path: str,
        variant: str = 'large',
        device : str = 'cuda'
    ):
        """
        Initialize Hiera encoder from SAM2.
        
        Args:
            model_cfg: Path to SAM2 config
            checkpoint_path: Path to SAM2 weights
            variant: Model size ['tiny', 'small', 'base', 'base_plus', 'large', 'huge']
            device: Device to load model on
            
        Memory Management:
        - Loads only necessary weights
        - Clears unused components immediately
        - Forces garbage collection
        """
        super().__init__()

        # Register channel dimensions
        self.channels_dict = {
            'tiny': [96, 192, 384, 768],
            'small': [96, 192, 384, 768],
            'base': [96, 192, 384, 768],
            'base_plus': [112, 224, 448, 896],
            'large': [144, 288, 576, 1152],
            'huge': [256, 512, 1024, 2048]
        }
        
        if variant not in self.channels_dict:
            raise ValueError(f"Invalid variant. Choose from: {list(self.channels_dict.keys())}")
            
        try:
            # Load SAM2 model
            logger.info(f"Loading SAM2 model with {variant} Hiera backbone...")
            sam2 = build_sam2(model_cfg, checkpoint_path, apply_postprocessing=False, device=device)
            
            # Extract and store only needed components
            self.encoder = sam2.image_encoder.trunk
            logger.info("Using encoder backbone only (hierarchical channels)")
                
            # Clear unused components
            self._clear_unused_components(sam2)
            
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {str(e)}")
            raise
            
        self.variant = variant
                
        # Log initialization success
        logger.info(f"Initialized {variant} encoder with {self.param_count:,} parameters")
        
    def _clear_unused_components(self, model: nn.Module) -> None:
        """
        Thoroughly clean up unused model components and clear memory.
        
        Args:
            model: Full SAM2 model to clean
            
        Steps:
        1. Delete unused attributes
        2. Clear CUDA cache
        3. Force garbage collection
        """
        # Get list of components to remove first
        to_remove = []
        for name, _ in model.named_children():
            if name not in ['image_encoder']:
                to_remove.append(name)
                
        # Then delete them
        for name in to_remove:
            delattr(model, name)
            
        del model
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()  # Reset peak stats
            except:
                pass  # Ignore if fails
            
        # Force garbage collection
        gc.collect()
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract hierarchical features through progressive processing.
        
        Args:
            x: Input tensor (B, 3, H, W)
               H, W must be divisible by 32
               For 512x512: x.shape = (B, 3, 512, 512)
               
        Returns:
            List of feature tensors at different scales:
            [
                stage1: (B, C1, H/4, W/4),    # Local patterns
                stage2: (B, C2, H/8, W/8),    # Edge features
                stage3: (B, C3, H/16, W/16),  # Object parts
                stage4: (B, C4, H/32, W/32)   # Scene context
            ]
            where Ci depends on variant
        
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.dim()}D")
        if any(s % 32 != 0 for s in x.shape[-2:]):
            raise ValueError("Input spatial dims must be divisible by 32")
            
        # Direct hierarchical features
        return self.encoder(x)
            
                
    def get_output_shapes(self, height: int, width: int) -> List[Tuple[int, int, int]]:
        """
        Calculate output shapes for given input dimensions.
        
        Args:
            height: Input height (must be divisible by 32)
            width: Input width (must be divisible by 32)
            
        Returns:
            List of (channels, height, width) tuples for each stage
            
        Example for 512x512 input with Large variant:
            [
                (144, 128, 128),  # Stage 1
                (288, 64, 64),    # Stage 2  
                (576, 32, 32),    # Stage 3
                (1152, 16, 16)    # Stage 4
            ]
        """
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError("Input dimensions must be divisible by 32")
            
        channels = self.channels

        shapes = []
        for i, c in enumerate(channels):
            if i == 0:
                h, w = height // 4, width // 4
            else:
                h, w = height // (2 ** (i+2)), width // (2 ** (i+2))
            shapes.append((c, h, w))
            
        return shapes

    @property
    def channels(self) -> List[int]:
        """Return channel dimensions for current variant."""
        return self.channels_dict[self.variant]
        
    @property
    def param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
        
    def __repr__(self) -> str:
        """Detailed string representation with architecture info."""
        return (
            f"HieraSAM2Encoder(\n"
            f"  variant={self.variant}\n"
            f"  channels={self.channels}\n"
            f"  params={self.param_count:,}\n"
            f")"
        )