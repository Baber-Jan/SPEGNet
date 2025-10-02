"""
Camouflaged Object Detection (COD) Image Processor

This module implements the image processing pipeline for SPEGNet's camouflaged object
detection tasks. It handles RGB image processing, binary mask conversion, and edge map
preparation as described in Section 3.1 of the paper.

The processor  uses ImageNet normalization for consistent model input preparation. 
All operations are optimized for memory efficiency using torch.no_grad() contexts.

Key Operations:
    Images: RGB → Tensor → Resize → Normalize
    Masks/Edges: Grayscale → Binary Tensor [0,1]

Memory Efficiency:
    - No-grad contexts for all operations
    - Efficient tensor broadcasting
    - In-place operations where possible
    - Single-pass processing pipeline
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProcessedCOD:
    """
    Container for processed COD image data.
    
    All tensors are on CPU and in torch.float32 format.
    
    Attributes:
        image: Normalized tensor [C, H, W], range [-1,1] after normalization
        mask: Binary tensor [1, H, W], range [0,1], original resolution
        edge: Binary tensor [1, H, W], range [0,1], original resolution
    """
    image: torch.Tensor  
    mask: Optional[torch.Tensor] = None
    edge: Optional[torch.Tensor] = None


class CODImageProcessor:
    """
    Optimized image processor for Camouflaged Object Detection tasks.
    
    The processor handles:
    1. Images: RGB → Tensor → [0,1] → Resize → Normalize
    2. Masks/Edges: Grayscale → Binary Tensor [0,1]
    
    All operations are optimized for memory efficiency and proper tensor handling.
    
    Attributes:
        target_size: Final size for processed images (H, W)
        norm_mean: ImageNet normalization mean
        norm_std: ImageNet normalization std
    """
    
    def __init__(
        self,
        target_size: int = 512,  
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet stats
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),   # ImageNet stats
    ):
        """
        Initialize the COD image processor.
        
        Args:
            target_size: Target size for processed images (default: 352)
            normalize_mean: RGB normalization means (default: ImageNet)
            normalize_std: RGB normalization stds (default: ImageNet)
        """
        self.target_size = (target_size, target_size)
        self.register_norm_stats(normalize_mean, normalize_std)

    def register_norm_stats(self, mean: Tuple[float, ...], std: Tuple[float, ...]) -> None:
        """
        Register normalization statistics as tensors.
        
        Args:
            mean: RGB means for normalization
            std: RGB standard deviations for normalization
        """
        # Store as broadcasted tensors for efficient normalization
        self.norm_mean = torch.tensor(mean).view(-1, 1, 1)
        self.norm_std = torch.tensor(std).view(-1, 1, 1)

    @torch.no_grad()  # Memory optimization
    def process_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Process RGB images following SPEGNet preprocessing protocol.
        
        Processing Pipeline:
        1. Load and convert to RGB (ensuring 3 channels)
        2. Scale to [0,1] range and convert to tensor
        3. Resize maintaining aspect ratio with bilinear interpolation
        4. Apply ImageNet normalization
        
        Args:
            image_path: Path to input image file (jpg/png)
            
        Returns:
            Processed tensor [3, H, W] in normalized range
            
        Raises:
            RuntimeError: If image loading or processing fails
            ValueError: If image format is invalid
        """
        try:
            # Load and convert to RGB (ensures 3 channels)
            img = Image.open(str(image_path)).convert('RGB')
            
            # Convert to tensor and scale to [0,1]
            # permute: HWC → CHW format
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            
            # Resize with bilinear interpolation (best practice for images)
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),  # Add batch dimension
                size=self.target_size,
                mode='bilinear',
                align_corners=False,  # Best practice for feature maps
                antialias=True  # Prevent aliasing artifacts
            ).squeeze(0)  # Remove batch dimension
            
            # Normalize using ImageNet statistics
            img_tensor = (img_tensor - self.norm_mean) / self.norm_std
            
            return img_tensor
            
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {str(e)}")

    @torch.no_grad()  # Memory optimization
    def process_mask(self, mask_path: Union[str, Path]) -> torch.Tensor:
        """
        Process binary masks/edges.
        
        Processing steps:
        1. Load as grayscale (single channel)
        2. Convert to binary tensor (threshold at 127.5)
        3. Keep original resolution
        
        Args:
            mask_path: Path to the mask/edge image
            
        Returns:
            Binary tensor of shape [1, H, W]
            
        Raises:
            RuntimeError: If mask processing fails
        """
        try:
            # Load as grayscale (L mode ensures single channel)
            mask = Image.open(str(mask_path)).convert('L')
            
            # Convert to binary tensor [0, 1]
            mask_tensor = torch.from_numpy(np.array(mask)).float()
            mask_tensor = (mask_tensor > 127.5).float()  # Binary threshold
            
            # Add channel dimension [H, W] → [1, H, W]
            mask_tensor = mask_tensor.unsqueeze(0)
            
            return mask_tensor
            
        except Exception as e:
            raise RuntimeError(f"Error processing mask {mask_path}: {str(e)}")

    @torch.no_grad()  # Memory optimization
    def __call__(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        edge_path: Optional[Union[str, Path]] = None
    ) -> ProcessedCOD:
        """
        Process all inputs for COD task.
        
        Args:
            image_path: Path to input image
            mask_path: Optional path to mask (for training)
            edge_path: Optional path to edge mask (for training)
            
        Returns:
            ProcessedCOD object containing all processed tensors
            
        Example:
            processor = CODImageProcessor()
            processed = processor(
                'image.jpg',
                'mask.png',
                'edge.png'
            )
            model_input = processed.image
        """
        # Process main image (always required)
        processed_image = self.process_image(image_path)
        
        # Process mask and edge if provided (training mode)
        processed_mask = self.process_mask(mask_path) if mask_path is not None else None
        processed_edge = self.process_mask(edge_path) if edge_path is not None else None
            
        return ProcessedCOD(
            image=processed_image,
            mask=processed_mask,
            edge=processed_edge
        )