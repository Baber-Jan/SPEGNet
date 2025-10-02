"""
Camouflaged Object Detection Loss Functions.

This module implements the multi-component loss function described in Section 3.4
of the SPEGNet paper. The loss combines three key components for effective
camouflage detection:

1. Structure Loss (Segmentation):
   Ls = λbce * Lbce + λiou * Liou
   - Boundary-aware weighting with Laplacian edge detection
   - Class-balanced BCE for handling foreground-background imbalance
   - Weighted IOU loss for structural consistency

2. Edge Detection Loss:
   Le = Lfocal + Ldice
   - Focal loss for hard example mining
   - Dice loss for boundary consistency
   - Automatic class balancing for sparse edges

3. Multi-scale Supervision:
   L = Σ(wi * Ls_i) + λe * Le
   - Progressive scale weights [0.2, 0.3, 0.5]
   - Edge guidance weight λe = 0.75

Implementation focuses on:
- Numerical stability with proper epsilon values
- Memory efficiency through in-place operations
- Batch processing optimization
- Proper gradient flow with autograd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class CODLoss(nn.Module):
    """
    SPEGNet's multi-component loss function combining structure and edge detection.
    
    The loss implements the formulation from Section 3.4:
    L = Σ(wi * Ls_i) + λe * Le

    Where:
    - Ls_i: Structure loss at scale i
    - wi: Scale weights [0.2, 0.3, 0.5]
    - Le: Edge detection loss
    - λe: Edge guidance weight (0.75)
    
    Key Components:
    1. Structure Loss:
       - Boundary-enhanced weighting
       - Class-balanced BCE
       - Weighted IOU
       
    2. Edge Loss:
       - Focal loss for hard examples
       - Dice loss for consistency
       
    Args:
        scale_weights: Progressive supervision weights [w1, w2, w3]
        boundary_weight: Weight λb for boundary regions
        bce_weight: Weight λbce for BCE term
        iou_weight: Weight λiou for IOU term
        edge_weight: Weight λe for edge loss
        edge_focal_alpha: Focal loss α modulating factor
        edge_focal_gamma: Focal loss γ focusing parameter
    """

    def __init__(self,
                 scale_weights: Optional[List[float]] = None,
                 boundary_weight: float = 5.0,
                 bce_weight: float = 0.4,
                 iou_weight: float = 0.6,
                 edge_weight: float = 0.75,
                 edge_focal_alpha: float = 0.75,
                 edge_focal_gamma: float = 2.0):
        """
            Args:
            scale_weights: List of weights for multi-scale predictions, default [0.2, 0.3, 0.5]
            boundary_weight: Multiplication factor for boundary regions, default 5.0
            bce_weight: Weight for BCE in structure loss, default 0.4
            iou_weight: Weight for IOU in structure loss, default 0.6
            edge_weight: Weight for edge detection loss, default 0.75
            edge_focal_alpha: Alpha modulating factor in focal loss, default 0.75
            edge_focal_gamma: Focusing parameter gamma in focal loss, default 2.0
            
        Shape:
            - Predictions:
                predictions: List[Tensor], each [B, 1, H_i, W_i]
                edge: Tensor [B, 1, H_e, W_e]
            - Targets:
                mask: Tensor [B, 1, H, W]
                edge: Tensor [B, 1, H, W]
        """
        
        super().__init__()
        
        # Initialize loss weights and parameters
        self.scale_weights = scale_weights or [0.2, 0.3, 0.5]
        self.boundary_weight = boundary_weight
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.edge_weight = edge_weight
        self.edge_focal_alpha = edge_focal_alpha
        self.edge_focal_gamma = edge_focal_gamma
        
        # Register Laplacian kernel for boundary detection
        kernel = torch.tensor([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer('boundary_kernel', kernel.view(1, 1, 3, 3))

    def compute_boundary_weights(self, masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes boundary-aware weight maps for each mask in the batch.
        
        Uses Laplacian kernel for edge detection and distance transform approximation
        to generate weights that emphasize boundary regions.
        
        Args:
            masks: List of binary segmentation masks [1, H, W]
            
        Returns:
            List of weight maps emphasizing boundary regions [1, H, W]
            
        Note:
            Weight maps combine local boundary detection with distance transform
            to emphasize both edge regions and transitional areas.
        """
        weight_maps = []
        for mask in masks:
            # Add batch dimension for conv2d
            mask = mask.unsqueeze(0)  # [1, 1, H, W]
            
            # Local boundary detection using Laplacian
            boundary = F.conv2d(mask, self.boundary_kernel, padding=1).abs()
            
            # Distance transform approximation via pooling
            pooled = F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)
            distance = (pooled - mask).abs()
            
            # Combine boundary and distance weights
            weight_map = 1.0 + self.boundary_weight * (boundary + distance)
            
            weight_maps.append(weight_map.squeeze(0))  # Remove batch dim
            
        return weight_maps

    def structure_loss(self, pred: torch.Tensor, masks: List[torch.Tensor], 
                      weight_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes weighted structure loss combining BCE and IOU.
        
        Features:
        - Class-balanced BCE with positive/negative ratio weighting
        - Boundary-aware weighting using weight maps
        - Weighted IOU loss for structural consistency
        
        Args:
            pred: Prediction logits [B, 1, H, W]
            masks: Ground truth masks [1, H, W]
            weight_maps: Boundary weight maps [1, H, W]
            
        Returns:
            Combined structure loss averaged over batch
        """
        batch_losses = []
        
        for idx, (mask, weight_map) in enumerate(zip(masks, weight_maps)):
            # Get prediction for this sample
            pred_sample = pred[idx:idx+1]  # Keep batch dim: [1, 1, H, W]
            
            # Add batch dimension to mask and weight map
            mask = mask.unsqueeze(0)      # [1, 1, H, W]
            weight_map = weight_map.unsqueeze(0)  # [1, 1, H, W]
            
            # Compute class-balanced BCE
            num_pos = mask.sum((2,3), keepdim=True)
            num_neg = (1 - mask).sum((2,3), keepdim=True)
            pos_weight = (num_neg / (num_pos + 1e-7)).clamp(0.1, 10.0)
            bce = F.binary_cross_entropy_with_logits(
                pred_sample, mask,
                pos_weight=pos_weight,
                reduction='none'
            )
            weighted_bce = (weight_map * bce).sum((2, 3)) / weight_map.sum((2, 3))
            
            # Compute weighted IOU loss
            pred_sigmoid = torch.sigmoid(pred_sample)
            inter = (pred_sigmoid * mask * weight_map).sum((2, 3))
            union = ((pred_sigmoid + mask) * weight_map).sum((2, 3))
            weighted_iou = 1 - (inter + 1) / (union - inter + 1)
            
            # Combine losses with weights
            loss = self.bce_weight * weighted_bce + self.iou_weight * weighted_iou
            batch_losses.append(loss)
            
        return torch.stack(batch_losses).mean()

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes edge detection loss combining focal and dice losses.
        
        Features:
        - Class-balanced focal loss for hard example mining
        - Dice loss for better boundary consistency
        - Automatic class weight computation based on edge sparsity
        
        Args:
            pred: Edge prediction logits [B, 1, H, W]
            target: Ground truth edges [B, 1, H, W]
            
        Returns:
            Combined edge loss averaged over batch
            
        Note:
            Combines focal loss for hard example mining with dice loss for
            better boundary consistency. Includes automatic class balancing
            to handle sparse edge maps.
        """
        # Compute sigmoid once for efficiency
        pred_sigmoid = torch.sigmoid(pred)
        
        # Compute class balancing weights
        num_pos = target.sum((2, 3), keepdim=True)
        num_neg = (1 - target).sum((2, 3), keepdim=True)
        pos_weight = (num_neg / (num_pos + 1e-7)).clamp(0.1, 10.0)
        
        # Focal loss computation
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_weight = (1 - pt).pow(self.edge_focal_gamma)
        focal = -pos_weight * self.edge_focal_alpha * focal_weight * torch.log(pt.clamp(min=1e-7))
        
        # Dice loss computation
        inter = (pred_sigmoid * target).sum((2, 3))
        union = pred_sigmoid.sum((2, 3)) + target.sum((2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        
        return focal.mean() + dice.mean()

    def forward(self, predictions: List[List[torch.Tensor]], 
            edge_pred: List[torch.Tensor],
            masks: List[torch.Tensor], 
            edges: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all losses.
        
        Args:
            predictions: List[List[Tensor]] organized as [batch_size][num_scales]
                        where each prediction is [1, 1, H, W]
            edge_pred: List[Tensor] of edge predictions [1, 1, H, W]
            masks: List[Tensor] of ground truth masks [1, H, W]
            edges: List[Tensor] of ground truth edges [1, H, W]
        """
        batch_size = len(masks)
        total_seg_loss = 0
        total_edge_loss = 0

        # Process each sample in batch
        for i in range(batch_size):
            # Get all predictions for this sample
            sample_preds = predictions[i]  # List of predictions at different scales
            
            # Compute boundary weights once per sample
            weight_map = self.compute_boundary_weights(masks[i].unsqueeze(0))
            
            # Multi-scale segmentation loss
            seg_loss = 0
            for pred, scale_weight in zip(sample_preds, self.scale_weights):
                seg_loss += scale_weight * self.structure_loss(
                    pred,
                    masks[i].unsqueeze(0),
                    weight_map
                )
            
            # Edge loss
            edge_loss = self.edge_loss(
                edge_pred[i],
                edges[i].unsqueeze(0)
            )
            
            total_seg_loss += seg_loss
            total_edge_loss += edge_loss

        # Average losses
        avg_seg_loss = total_seg_loss / batch_size
        avg_edge_loss = total_edge_loss / batch_size
        total_loss = avg_seg_loss + self.edge_weight * avg_edge_loss

        return {
            'loss': total_loss,
            'seg_loss': avg_seg_loss,
            'edge_loss': avg_edge_loss
        }