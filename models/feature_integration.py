"""
Contextual Feature Integration (CFI) Module for SPEGNet

CFI addresses intrinsic similarity through synergistic channel-spatial processing.
Unlike accumulated attention mechanisms, CFI integrates channel recalibration with
multi-scale spatial enhancement in a unified module.

Key Innovation: Complementary processing where channel attention identifies
camouflage-discriminative features while e-ASPP captures scale-dependent patterns.

Reference: IEEE TIP Paper Section 3.2 - Contextual Feature Integration

This module implements two main components:

1. Adaptive Attention Fusion:
   Combines multi-scale features through channel attention and spatial alignment.
   
   Input Features (Hiera-Large):
   - Stage 2: [B, 288, 64, 64]  (mid-level patterns)  
   - Stage 3: [B, 576, 32, 32]  (semantic features)
   - Stage 4: [B, 1152, 16, 16] (global context)
   
   Processing Flow:
   a) Spatial Alignment: Upsample all features to Stage 2 resolution (64×64)
   b) Channel Concatenation: [B, 2016, 64, 64]
   c) Channel Reduction: 1×1 conv reduces to [B, 512, 64, 64]
   d) Attention Weighting: SE block for adaptive channel recalibration
   
   Output: [B, 512, 64, 64] fused features

2. Efficient ASPP (e-ASPP):
   Memory-efficient multi-scale context modeling through depth-wise operations.
   
   Input: [B, 512, 64, 64] from fusion stage
   
   Processing Branches:
   - Rate 1:  3×3 DW conv → Local patterns (3×3 RF)
   - Rate 6:  3×3 DW conv → Object parts (13×13 RF)
   - Rate 12: 3×3 DW conv → Small objects (25×25 RF)
   - Rate 18: 3×3 DW conv → Large objects (37×37 RF)
   - Global:  Avg pool → Scene context (64×64 RF)
   
   Memory Optimization:
   - Channel Reduction: 512 -> 128 channels
   - Depth-wise Convolutions: O(C) vs O(C²)
   - Independent Scale Processing
   
   Output: [B, 256, 64, 64] context-enhanced features

Key Features:
1. Memory Efficiency:
   - Depth-wise operations for multi-scale processing
   - Progressive channel reduction/expansion
   - Efficient feature alignment

2. Adaptive Processing:
   - Channel attention for feature importance
   - Multi-scale context modeling
   - Balanced receptive field coverage

3. Architectural Benefits:
   - Maintains spatial resolution for boundary detail
   - Integrates global context with local patterns
   - Memory-efficient parallel processing

Processing Pipeline:
Input Features → Spatial Alignment → Feature Fusion → 
Channel Attention → Context Enhancement → Output Features

References:
1. "Squeeze-and-Excitation Networks" (CVPR 2018)
   Channel attention mechanism for adaptive feature recalibration
   
2. "Rethinking Atrous Convolution" (arXiv:1706.05587)
   Base ASPP module for multi-scale context modeling

Note:
This module expects features from a Hiera-Large backbone. For other variants,
channel dimensions should be adjusted accordingly while maintaining the same
processing flow.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SqueezeExcitationBlock(nn.Module):
    """
    Channel attention through Squeeze-and-Excitation.
    
    Theoretical Basis:
    SE Formula: F_SE(x) = x ⊗ σ(W_2(δ(W_1(GAP(x)))))
    where: ⊗ is channel-wise multiplication
           σ is sigmoid
           δ is ReLU
           GAP is global average pooling
    
    For Hiera-Large with 512 channels:
    1. GAP: [B, 512, 64, 64] -> [B, 512, 1, 1]
    2. FC1: 512 -> 32 (reduction=16)
    3. FC2: 32 -> 512
    4. Scale: [B, 512, 64, 64] * [B, 512, 1, 1]

    Args:
        channels (int): Number of input channels
        reduction (int, optional): Channel reduction factor. Default: 16
            For 512 channels, reduces to 32 channels internally
            
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 32)  # Minimum 32 channels
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE block.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W)
                For feature fusion stage: (B, 512, 64, 64)

        Returns:
            torch.Tensor: Recalibrated feature map, same shape as input
                For feature fusion stage: (B, 512, 64, 64)
                
        Steps:
            1. Global average pooling for channel statistics
            2. Two FC layers for channel relationship modeling
            3. Channel-wise multiplication with input features
        """
        b, c, _, _ = x.size()
        # Global information aggregation
        y = self.avg_pool(x).view(b, c)  # Squeeze: [B, C]
        # Channel-wise dependencies
        y = self.fc(y).view(b, c, 1, 1)  # Excitation: [B, C, 1, 1]
        # Feature recalibration
        return x * y.expand_as(x)  # Scale: [B, C, H, W]

class AdaptiveAttentionFusion(nn.Module):
    """
    Fuses multi-scale features through concatenation and channel attention.
    
    Theoretical Basis:
    The fusion process follows: F_out = SE(Conv1x1(Concat([F_2, F_3, F_4])))
    where F_i are features from stages 2,3,4 respectively.
    
    For Hiera-Large dimensions (B=batch_size):
    Input Features:
    - stage2: [B, 288, 64, 64]   (mid-level features)
    - stage3: [B, 576, 32, 32]   (semantic features) 
    - stage4: [B, 1152, 16, 16]  (global context)
    
    Processing steps:
    1. Spatial Alignment (to stage2 resolution):
       F_align(x_i) = Upsample(x_i) to 64×64
    
    2. Feature Concatenation:
       F_cat = Concat([x_2, x_3, x_4]) -> [B, 2016, 64, 64]
    
    3. Channel Reduction:
       F_red = Conv1x1(F_cat) -> [B, 512, 64, 64]
    
    4. Attention Weighting:
       F_out = SE(F_red) -> [B, 512, 64, 64]

    Args:
        in_channels_list (List[int]): Channel dimensions from stages 2,3,4
            For Hiera-Large: [288, 576, 1152]
        out_channels (int): Output channel dimension after fusion
            Default: 512 for compatibility with subsequent modules
            
    Shape:
        - Input: List of 3 tensors:
            - stage2: (B, 288, 64, 64)
            - stage3: (B, 576, 32, 32)
            - stage4: (B, 1152, 16, 16)
        - Output: (B, 512, 64, 64)
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 512):
        super().__init__()
        total_channels = sum(in_channels_list)  # 3584 for Hiera-Huge
        
        # Channel reduction through 1x1 convolution
        self.conv1x1 = nn.Conv2d(total_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel attention module
        self.se_block = SqueezeExcitationBlock(out_channels)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for feature fusion.

        Args:
            features (List[torch.Tensor]): Multi-scale features [stage2, stage3, stage4]           
                For Hiera-Large:
                    - features[0]: stage2 (B, 288, 64, 64)
                    - features[1]: stage3 (B, 576, 32, 32)
                    - features[2]: stage4 (B, 1152, 16, 16)

        Returns:
            torch.Tensor: Fused features (B, 512, 64, 64)
            
        Steps:
            1. Upsamples stage3 and stage4 to match stage2's resolution
            2. Concatenates all features along channel dimension
            3. Reduces channels through 1x1 convolution
            4. Applies channel attention through SE block
        """
        # Get target spatial dimensions from stage2
        target_size = features[0].shape[2:]  # 64×64
        
        # 1. Spatial alignment through bilinear upsampling
        aligned_features = [
            F.interpolate(f, size=target_size, mode='bilinear', 
                         align_corners=False) if f.shape[2:] != target_size else f
            for f in features
        ]
        
        # 2. Feature concatenation
        x = torch.cat(aligned_features, dim=1)  # [B, 2016, 64, 64]
        
        # 3. Channel reduction
        x = self.conv1x1(x)  # [B, 512, 64, 64]
        x = self.bn(x)
        x = self.relu(x)
        
        # 4. Attention weighting
        x = self.se_block(x)  # [B, 512, 64, 64]
        
        return x

class EfficientASPP(nn.Module):
    """
    Memory-efficient variant of Atrous Spatial Pyramid Pooling.
    
    Theoretical Basis:
    e-ASPP optimizes ASPP through four key stages:
    1. Channel Reduction: Reduce input channels by factor of 4
    2. Multi-Scale Context: Parallel depth-wise dilated convolutions
    3. Feature Aggregation: Multi-branch feature fusion
    4. Channel Restoration: Restore channel capacity
    
    For Hiera-Large dimensions (B=batch_size):
    Input: [B, 512, 64, 64] (from feature fusion)
    
    Processing Flow:
    1. Channel Reduction: 
       [B, 512, 64, 64] -> [B, 128, 64, 64]
       Reduces computation by 75%
       
    2. Multi-Scale Feature Extraction:
       Parallel depth-wise branches (each processes 128 channels independently):
       - Rate 1:  RF: 3×3    Local patterns
       - Rate 6:  RF: 13×13  Object parts
       - Rate 12: RF: 25×25  Small objects
       - Rate 18: RF: 37×37  Large objects
       - Global:  RF: 64×64  Scene context
       Memory efficient: O(C) vs O(C²) for standard conv

    3. Multi-Scale Feature Fusion:
       Combines features while maintaining channel separation
       [B, 640, 64, 64] -> [B, 128, 64, 64]
       
    4. Channel Expansion:
       [B, 128, 64, 64] -> [B, 256, 64, 64]
       Restores representation capacity

    Args:
        in_channels (int): Input channels (512 from fusion)
        out_channels (int): Output channels (256 for decoder)
        reduction_factor (int): Channel reduction factor (4)
        dilation_rates (List[int]): Dilation rates for different scales
            Default: [1,6,12,18] for balanced receptive fields
            
    Shape:
        - Input: (B, 512, 64, 64)
        - Output: (B, 256, 64, 64)
        
    Memory Efficiency:
        - Standard ASPP: O(C²HW) memory
        - e-ASPP: O(CHW) memory
        - 75% reduction in parameters
    """
    def __init__(self,
                 in_channels: int = 512,
                 out_channels: int = 256,
                 reduction_factor: int = 4,
                 dilation_rates: List[int] = [1, 6, 12, 18]):
        super().__init__()
        
        self.reduced_channels = in_channels // reduction_factor
        
        # 1. Channel Reduction (512 -> 128)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Multi-Scale Feature Extraction (depth-wise)
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(nn.Sequential(
                # Depth-wise dilated convolution
                nn.Conv2d(
                    self.reduced_channels, 
                    self.reduced_channels, 
                    3,
                    padding=rate, 
                    dilation=rate,
                    groups=self.reduced_channels,  # Depth-wise
                    bias=False
                ),
                nn.BatchNorm2d(self.reduced_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global context branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                self.reduced_channels, 
                self.reduced_channels, 
                1, 
                bias=False
            ),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. Multi-Scale Feature Fusion (depth-wise)
        total_branches = len(dilation_rates) + 1  # + global branch
        self.fusion = nn.Sequential(
            # Fuse multiple scales while maintaining channel separation
            nn.Conv2d(
                self.reduced_channels * total_branches,
                self.reduced_channels,
                1,
                groups=self.reduced_channels,  # Maintain channel separation
                bias=False
            ),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4. Channel Expansion {Inter-Channel Feature Fusion} (128 -> 256)
        self.expand = nn.Sequential(
            nn.Conv2d(self.reduced_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of e-ASPP.

        Args:
            x (torch.Tensor): Input features [B, 512, 64, 64]
                Features from previous fusion stage

        Returns:
            torch.Tensor: Context-enhanced features [B, 256, 64, 64]
            Combines local detail and global context information
            
        Processing Steps:
        2. Reduce channels for efficiency (512->128)
        3. Extract multi-scale features through parallel branches
        4. Combine features while maintaining channel separation
        5. Expand channels for decoder (128->256)
        """
        # Save spatial dimensions
        size = x.shape[2:]  # 64×64
        
        # 1. Channel reduction
        x = self.reduce(x)  # [B, 128, 64, 64]
        
        # 2. Multi-scale feature extraction
        branch_outputs = []
        
        # Process through dilated branches
        for branch in self.branches:
            branch_outputs.append(branch(x))
            
        # Global context
        global_features = self.global_branch(x)
        global_features = F.interpolate(
            global_features, 
            size=size,
            mode='bilinear', 
            align_corners=False
        )
        branch_outputs.append(global_features)
        
        # 3. Multi-scale feature fusion
        x = torch.cat(branch_outputs, dim=1)  # [B, 640, 64, 64]
        x = self.fusion(x)  # [B, 128, 64, 64]
        
        # 4. Channel expansion
        x = self.expand(x)  # [B, 256, 64, 64]
        
        return x