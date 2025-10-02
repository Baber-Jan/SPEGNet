"""
Edge Feature Extraction (EFE) and Progressive Edge-guided Decoder (PED) for SPEGNet

This module implements the final stages of camouflaged object detection through
edge-guided progressive refinement. It processes context-enhanced features from
CFI to generate high-resolution segmentation masks with precise boundaries.

Architecture Components:

1. Edge Feature Extraction (EFE) - Section 3.3:
   Purpose: Maintain semantic-spatial alignment in boundary features
   Innovation: Direct extraction from context-rich representations
   Unlike traditional edge detectors, EFE preserves object understanding

   Input: Context features from CFI [B, 256, 64, 64]
   Processing:
   - Feature extraction: 3×3 conv → [B, 64, 64, 64]
   - Edge map generation: 1×1 conv → [B, 1, 64, 64]
   Output: Semantically-aware boundary features

2. Progressive Edge-guided Decoder (PED) - Section 3.4:
   Three-stage refinement with scale-adaptive edge modulation (peak-and-fade strategy).

   Edge Influence Pattern (NON-MONOTONIC):
   - Stage 1 (H/4): 20% edge channels (64/320) - Initial boundary hypothesis
   - Stage 2 (H/2): 33% edge channels (64/192) - PEAK refinement
   - Stage 3 (H): 0% edge channels (0/64) - Pure region consistency

   Justification: Peak at intermediate scale where edges most informative,
   prevents texture-induced over-segmentation at fine resolution.
   
   Stage-wise Processing:
   a) Stage 1: (2× upsampling + edge guidance)
      - Input:  [B, 256, 64, 64]
      - Edge:   [B, 64, 64, 64]
      - Output: [B, 256, 128, 128] → pred1 [B, 1, 128, 128]
      
   b) Stage 2: (2× upsampling + edge guidance)
      - Input:  [B, 256, 128, 128]
      - Edge:   [B, 64, 128, 128]
      - Output: [B, 128, 256, 256] → pred2 [B, 1, 256, 256]
      
   c) Stage 3: (2× upsampling, no edge)
      - Input:  [B, 128, 256, 256]
      - Output: [B, 64, 512, 512] → pred3 [B, 1, 512, 512]

Features:
1. Edge-Guided Refinement:
   - Early stages use edge features for boundary precision
   - Progressive reduction in edge influence for stability
   - Dual convolution refinement at each stage

2. Deep Supervision:
   - Multi-scale predictions for better gradient flow
   - Stage-specific loss computation
   - Balanced feature hierarchy learning

3. Channel Progression:
   - Progressive channel reduction: 256 → 128 → 64
   - Maintains balance between detail and efficiency
   - Final stage focuses on fine spatial details

Processing Pipeline:
1. Edge Detection:
   Context Features → Edge Features + Edge Map
   [256, 64, 64] → [64, 64, 64] + [1, 64, 64]

2. Progressive Refinement:
   Each stage performs:
   a) Feature upsampling (2×)
   b) Edge feature integration (if available)
   c) Dual convolution refinement
   d) Prediction generation

Note:
All spatial dimensions assume input images are 512×512.
For different input sizes, the spatial dimensions will scale accordingly
while maintaining the same processing pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class EdgeDetectionModule(nn.Module):
    """
    Edge detection module for boundary feature extraction and map generation.
    
    Architecture:
    1. Feature Extraction:
       [B, 256, 64, 64] → Conv3x3 → BN → ReLU → [B, 64, 64, 64]
       - Captures local gradient patterns
       - Maintains spatial resolution
       - Channel reduction: 256 → 64
       
    2. Edge Map Generation:
       [B, 64, 64, 64] → Conv1x1 → [B, 1, 64, 64]
       - Projects to binary-like edge predictions
       - Supervised with edge ground truth
    
    Args:
        in_channels (int): Input feature channels (256 from e-ASPP)
        out_channels (int): Edge feature channels (64 for decoder guidance)
        
    Shape:
        - Input: (B, 256, 64, 64)
        - Outputs: 
            - edge_map: (B, 1, 64, 64)
            - edge_features: (B, 64, 64, 64)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Edge feature extraction (256 → 64 channels)
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Edge map generation (64 → 1 channel)
        self.edge_conv = nn.Conv2d(
            out_channels,
            1,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Edge detection forward pass.

        Args:
            x: Context features [B, 256, 64, 64]
               From e-ASPP module

        Returns:
            Tuple containing:
                - edge_map: Binary-like edge predictions [B, 1, 64, 64]
                - edge_features: Edge guidance features [B, 64, 64, 64]
                
        Steps:
            1. Feature extraction with channel reduction
            2. Edge map generation through projection
        """
        # Extract edge features
        edge_features = self.conv1(x)
        edge_features = self.bn1(edge_features)
        edge_features = self.relu(edge_features)
        
        # Generate edge map
        edge_map = self.edge_conv(edge_features)
        
        return edge_map, edge_features

class DecoderBlock(nn.Module):
    """
    Single decoder block for progressive upsampling with edge guidance.
    
    Architecture:
    1. Feature Upsampling:
       [B, C_in, H, W] → [B, C_in, 2H, 2W]
       
    2. Edge Feature Integration (if provided):
       - Edge upsampling: [B, 64, H, W] → [B, 64, 2H, 2W]
       - Concatenation: [B, C_in+64, 2H, 2W]
       
    3. Dual Refinement:
       - Conv3x3 → BN → ReLU → [B, C_out, 2H, 2W]
       - Conv3x3 → BN → ReLU → [B, C_out, 2H, 2W]

    Args:
        in_channels (int): Input feature channels
        out_channels (int): Output feature channels
        edge_channels (Optional[int]): Edge feature channels (64 if provided)
        
    Shape:
        - Input: (B, C_in, H, W)
        - Edge (optional): (B, 64, H, W)
        - Output: (B, C_out, 2H, 2W)
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 edge_channels: Optional[int] = None):
        super().__init__()
        total_channels = in_channels + (edge_channels or 0)
        
        # First refinement
        self.conv1 = nn.Conv2d(total_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second refinement
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, 
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for progressive refinement.

        Args:
            x: Input features [B, C_in, H, W]
            edge_features: Optional edge guidance [B, 64, H, W]

        Returns:
            Refined features [B, C_out, 2H, 2W]
            
        Steps:
            1. Bilinear upsampling of input features
            2. Edge feature integration if provided
            3. Dual convolution refinement
        """
        # 1. Spatial upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 2. Edge feature integration
        if edge_features is not None:
            edge_features = F.interpolate(edge_features, 
                                        size=x.shape[2:],
                                        mode='bilinear', 
                                        align_corners=False)
            x = torch.cat([x, edge_features], dim=1)
        
        # 3. Feature refinement
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class BoundaryAwareDecoder(nn.Module):
    """
    Progressive decoder with edge-guided refinement and deep supervision.
    
    Architecture:
    Three-stage refinement with progressive resolution increase:
    
    Stage 1: Edge-guided (64→128)
        Input:  [B, 256, 64, 64]
        Edge:   [B, 64, 64, 64]
        Output: [B, 256, 128, 128] → pred1 [B, 1, 128, 128]
        
    Stage 2: Edge-guided (128→256)
        Input:  [B, 256, 128, 128]
        Edge:   [B, 64, 128, 128]
        Output: [B, 128, 256, 256] → pred2 [B, 1, 256, 256]
        
    Stage 3: Detail refinement (256→512)
        Input:  [B, 128, 256, 256]
        Output: [B, 64, 512, 512] → pred3 [B, 1, 512, 512]

    Args:
        in_channels (int): Input channels from e-ASPP (256)
        decoder_channels (List[int]): Channel progression [256, 128, 64]
        n_classes (int): Number of output classes (1 for binary)
        edge_channels_list (List[Optional[int]]): Edge channels per stage [64, 64, None]
        
    Shape:
        - Input: (B, 256, 64, 64)
        - Edge features list: 
            [
                (B, 64, 64, 64),    # Stage 1
                (B, 64, 128, 128),  # Stage 2
                None                # Stage 3
            ]
        - Outputs: List of predictions
            [
                (B, 1, 128, 128),   # Stage 1
                (B, 1, 256, 256),   # Stage 2
                (B, 1, 512, 512)    # Stage 3
            ]
    """
    def __init__(self,
                 in_channels: int,
                 decoder_channels: List[int],
                 n_classes: int = 1,
                 edge_channels_list: Optional[List[int]] = None):
        super().__init__()
        
        if edge_channels_list is None:
            edge_channels_list = [None] * len(decoder_channels)
            
        assert len(decoder_channels) == len(edge_channels_list), \
            "decoder_channels and edge_channels_list must have same length"
        
        # Create progressive decoder stages
        self.decoder_blocks = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        
        prev_channels = in_channels
        for out_channels, edge_channels in zip(decoder_channels, edge_channels_list):
            # Decoder block
            self.decoder_blocks.append(
                DecoderBlock(prev_channels, out_channels, edge_channels)
            )
            # Prediction head
            self.pred_heads.append(nn.Conv2d(out_channels, n_classes, kernel_size=1))
            prev_channels = out_channels

    def forward(self, x: torch.Tensor, 
                edge_features_list: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        Progressive upsampling with deep supervision.

        Args:
            x: Context features [B, 256, 64, 64]
            edge_features_list: Edge features for each stage
                [
                    [B, 64, 64, 64],     # Stage 1
                    [B, 64, 128, 128],   # Stage 2
                    None                 # Stage 3
                ]

        Returns:
            List[torch.Tensor]: Multi-scale predictions
                [
                    pred1: [B, 1, 128, 128],
                    pred2: [B, 1, 256, 256],
                    pred3: [B, 1, 512, 512]
                ]
        """
        if edge_features_list is None:
            edge_features_list = [None] * len(self.decoder_blocks)
            
        predictions = []
        for i, (decoder, edge_feat) in enumerate(zip(self.decoder_blocks, edge_features_list)):
            # Process features
            x = decoder(x, edge_feat)
            # Generate prediction
            pred = self.pred_heads[i](x)
            predictions.append(pred)
            
        return predictions