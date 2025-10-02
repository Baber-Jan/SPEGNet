"""
SPEGNet: Synergistic Perception-Guided Network for Camouflaged Object Detection
---------------------
A synergistic framework with complementary modules working in concert to address
camouflaged object detection through integrated perception rather than component accumulation.

Architecture Components:
1. Feature Encoder (Hiera-Large):
   - Input:  [B, 3, 512, 512]
   - Stages: Progressive feature hierarchy
     * Stage 1: [B, 144, 128, 128] Local patterns
     * Stage 2: [B, 288, 64, 64]   Edge features
     * Stage 3: [B, 576, 32, 32]   Object parts
     * Stage 4: [B, 1152, 16, 16]  Scene context

2. Feature Integration:
   a) Multi-scale Fusion:
      - Uses stages 2,3,4
      - Upsamples to 64×64
      - Concatenates: [B, 2016, 64, 64]
      - Reduces: [B, 512, 64, 64]
      
   b) Context Enhancement (e-ASPP):
      - Input:  [B, 512, 64, 64]
      - Output: [B, 256, 64, 64]
      - Multi-scale context through depth-wise ops

3. Edge-Guided Detection:
   a) Edge Detection:
      - Features: [B, 64, 64, 64]
      - Edge Map: [B, 1, 64, 64]
      
   b) Progressive Decoder:
      Stage 1: 64→128   + edge → [B, 1, 128, 128]
      Stage 2: 128→256  + edge → [B, 1, 256, 256]
      Stage 3: 256→512        → [B, 1, 512, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_encoding import HieraSAM2FeatureEncoder
from models.feature_integration import AdaptiveAttentionFusion, EfficientASPP
from models.object_detection import EdgeDetectionModule, BoundaryAwareDecoder
from typing import Dict, Optional, Tuple, Union

class SPEGNet(nn.Module):
    """
    Synergistic Perception-Guided Network for Camouflaged Object Detection.

    SPEGNet addresses fragmentation in current COD methods through synergistic design.
    Three complementary modules work in concert rather than accumulation:

    1. CFI (Contextual Feature Integration):
       - Channel recalibration identifies camouflage-discriminative features
       - Multi-scale spatial enhancement captures pattern-dependent properties
       - Synergistic processing addresses intrinsic similarity

    2. EFE (Edge Feature Extraction):
       - Direct boundary extraction from context-rich representations
       - Maintains semantic-spatial alignment throughout
       - Prevents texture-induced false boundaries

    3. PED (Progressive Edge-guided Decoder):
       - Scale-adaptive edge modulation with peak-and-fade strategy
       - Stage 1 (20%): Initial boundary hypothesis
       - Stage 2 (33%): Peak refinement at intermediate scale
       - Stage 3 (0%): Pure region consistency

    Reference: IEEE Transactions on Image Processing, 2024 (Under Review)

    Args:
        config: Model configuration dictionary
            encoder:
                config_path: Path to SAM2 config
                checkpoint_path: Path to SAM2 weights
                variant: Model size (default: 'large')                
    Shape:
        - Input: (B, 3, 512, 512)
        - Output: Dictionary containing
            - predictions: List of [
                (B, 1, 128, 128),
                (B, 1, 256, 256),
                (B, 1, 512, 512)
            ]
            - edge_map: (B, 1, 64, 64)
            - features: Dict of intermediate features
    """
    
    def __init__(self, config: Dict):
        super(SPEGNet, self).__init__()

        # 1. Feature Encoder
        self.encoder = HieraSAM2FeatureEncoder(
            model_cfg=config['encoder']['config_path'],
            checkpoint_path=config['encoder']['checkpoint_path'],
            variant=config['encoder'].get('variant', 'large')
        )
        
        # Get encoder channels
        encoder_channels = self.encoder.channels  # [144, 288, 576, 1152]
        
        # 2. Feature Integration: Fusion and Context enhancement
        # Channel dimensions for each stage 
        self.in_channels_list = encoder_channels[1:4]  # [C2, C3, C4] = [288, 576, 1152]
        
        # a) Multi-scale fusion with attention (output: B x 512 x H/16 x W/16)
        self.fusion = AdaptiveAttentionFusion(
            in_channels_list=self.in_channels_list,
            out_channels=512
        )
        
        # b) Context enhancement through e-ASPP (output: B x 256 x H/16 x W/16)
        # Input: [B,512,64,64] -> Output: [B,256,64,64]
        self.context = EfficientASPP(
            in_channels=512,
            out_channels=256,
            reduction_factor=4,
            dilation_rates=[1, 6, 12, 18]
        )
        
        # 3. Edge Detection & Refinement
        # a) Edge detection branch (outputs: B x 1 x H/16 x W/16, B x 64 x H/16 x W/16)
        self.edge_detector = EdgeDetectionModule(
            in_channels=256,  # From context module
            out_channels=64   # Edge features
        )
        
        # b) Progressive decoder with edge guidance: H/16 -> H/8 -> H/4 -> H/2
        self.decoder = BoundaryAwareDecoder(
            in_channels=256,                  # From context module
            decoder_channels=[256, 128, 64],  # Channel reduction at each stage
            n_classes=1,
            edge_channels_list=[64, 64, None]  # Peak-and-fade: 20%→33%→0% edge influence
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SPEGNet.

        Args:
            x: Input image [B, 3, 512, 512]
               Expected range: [0, 1]

        Returns:
            Dictionary containing:
                predictions: List of segmentation maps
                    [
                        pred1: (B, 1, 128, 128),
                        pred2: (B, 1, 256, 256),
                        pred3: (B, 1, 512, 512)
                    ]
                edge: Edge map (B, 1, 64, 64)
                features: Dict of intermediate features
                    context: (B, 256, 64, 64)
                    fused: (B, 512, 64, 64)
                    edge_features: (B, 64, 64, 64)
        
        Processing Steps:
        1. Hierarchical feature extraction
        2. Multi-scale feature integration
        3. Edge-guided progressive refinement
        """

        # 1. Multi-scale feature extraction
        features = self.encoder(x) # List of 4 feature maps
        
        # 2. Feature integration
        stage2 = features[1]  # [B, C2, H/8, W/8]              = [B, 288, 64, 64]
        stage3 = features[2]  # [B, C3, H/16, W/16]            = [B, 576, 32, 32]
        stage4 = features[3]  # [B, C4, H/32, W/32]            = [B, 1152, 16, 16]
        
        # a) Feature fusion at stage2's resolution
        fused = self.fusion([stage2, stage3, stage4])          # [B, 512, 64, 64]
        
        # b) Context enhancement through ASPP
        context = self.context(fused)                          # [B, 256, 64, 64]

        # 3. Edge detection & refinement
        # a) Edge detection branch
        edge_map, edge_features = self.edge_detector(context)
        # edge_map                                             : [B, 1, 64, 64]
        # edge_features                                        : [B, 64, 64, 64]

        # b) Progressive refinement with edge guidance
        # Prepare edge features for each decoder stage
        edge_features_list = [
            edge_features,  # Stage 1: [B, 64, 64, 64]
            edge_features,  # Stage 2: Will be upsampled in decoder
            None           # Stage 3: No edge guidance
        ]
        # Generate multi-scale predictions
        predictions = self.decoder(context, edge_features_list=edge_features_list) # predictions: [pred1, pred2, pred3]
        # pred1                                               : (B, 1, 128, 128)
        # pred2                                               : (B, 1, 256, 256)
        # pred3                                               : (B, 1, 512, 512)
        
        return {
            'predictions': predictions,  # List of multi-scale predictions
            'edge': edge_map,
            'features': {
                'context': context,
                'fused': fused,
                'edge_features': edge_features
            }
        }