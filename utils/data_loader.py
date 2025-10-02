"""
Camouflaged Object Detection (COD) Dataset and DataLoader Implementation

This module provides dataset handling and loading functionalities for training and evaluating 
camouflaged object detection models. It manages image processing, data organization, and batching
with proper memory handling for variable-sized ground truths.

Main Components:
    - CODDataset: Custom dataset class handling both training and test data
    - Training mode: Processes images, masks, and edge maps
    - Testing mode: Processes images and masks, tracks filenames
    - Custom collation: Handles variable-sized masks/edges efficiently

Features:
    - Multiple dataset support with unified loading
    - Optional validation split for training data
    - Maintenance of original ground truth sizes
    - Memory efficient batch processing
    - Configurable image processing via model config
    - Device-aware processing (CPU/CUDA)

Typical Usage:
    # Create training loaders
    train_loader, val_loader = get_training_loaders(
        dataset_dirs=['path/to/dataset1', 'path/to/dataset2'],
        model_config=config,
        batch_size=16,
        val_ratio=0.1
    )

    # Create test loaders
    test_loaders = get_test_loaders(
        dataset_dirs=['path/to/dataset1', 'path/to/dataset2'],
        model_config=config
    )

Data Organization:
    Each dataset directory should have:
    ├── train/
    │   ├── Imgs/      # Training images (.jpg or .png)
    │   ├── GT/        # Ground truth masks (.png)
    │   └── Edges/     # Edge maps (.png)
    └── test/
        ├── Imgs/      # Test images (.jpg or .png)
        └── GT/        # Ground truth masks (.png)

Returns:
    Training mode batches contain:
        - images: Tensor[B, C, H, W] - Normalized and resized images
        - masks: List[B] of Tensor[1, H, W] - Original size masks
        - edges: List[B] of Tensor[1, H, W] - Original size edges

    Testing mode batches contain:
        - images: Tensor[B, C, H, W] - Normalized and resized images
        - masks: List[B] of Tensor[1, H, W] - Original size masks
        - names: List[B] of str - Original filenames
"""

import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.image_processor import CODImageProcessor

class CODDataset(Dataset):
    """
    Dataset class for Camouflaged Object Detection (COD) handling training and testing modes.

    Training Mode:
        - Loads images, masks and edge maps
        - Images are resized and normalized
        - Masks and edges maintain original dimensions

    Testing Mode:
        - Loads only images and masks
        - Images are resized and normalized
        - Masks maintain original dimensions
        - Returns filenames for evaluation
    """
    
    def __init__(
        self, 
        root_dir: str,
        processor_params: Dict,
        is_train: bool = True
    ):
        """
        Initialize COD dataset.
        
        Args:
            root_dir: Path containing 'Imgs', 'GT', and optionally 'Edges' subdirectories
            processor_params: Parameters for processor (target_size, normalize_mean, normalize_std, device)
            is_train: If True, loads edge maps and enables training mode
        """
        # Initialize paths and processor
        self.is_train = is_train
        self.image_dir = os.path.join(root_dir, 'Imgs')
        self.mask_dir = os.path.join(root_dir, 'GT')
        self.edge_dir = os.path.join(root_dir, 'Edges') if is_train else None
        
        # Validate directories exist
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Required directories not found in {root_dir}")
        if is_train and not os.path.exists(self.edge_dir):
            raise FileNotFoundError(f"Edge directory not found for training in {root_dir}")
            
        # Initialize image processor
        self.processor = CODImageProcessor(**processor_params)
        
        # Get sorted filenames (only images with corresponding masks)
        self.filenames = self._get_valid_files()
        
    def _get_valid_files(self) -> List[str]:
        """Get list of image files that have corresponding masks/edges."""
        # Get all image files
        image_files = {f.split('.')[0] for f in os.listdir(self.image_dir) 
                      if f.endswith(('.jpg', '.png'))}
        # Get all mask files
        mask_files = {f.split('.')[0] for f in os.listdir(self.mask_dir)
                     if f.endswith('.png')}
        
        # For training, also check edge files
        if self.is_train:
            edge_files = {f.split('.')[0] for f in os.listdir(self.edge_dir)
                         if f.endswith('.png')}
            valid_files = sorted(list(image_files & mask_files & edge_files))
        else:
            valid_files = sorted(list(image_files & mask_files))
        
        if not valid_files:
            raise ValueError(f"No valid samples found in {self.image_dir}")
        
        return valid_files
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a processed sample.
        
        Returns dict with:
            - 'image': Processed and normalized image tensor
            - 'mask': Binary mask tensor at original size
            - 'edge': Binary edge tensor at original size (training only)
            - 'name': Filename (testing only)
        """
        filename = self.filenames[idx]
        
        # Construct full paths
        img_path = os.path.join(self.image_dir, f"{filename}.jpg")
        if not os.path.exists(img_path):  # Try png if jpg doesn't exist
            img_path = os.path.join(self.image_dir, f"{filename}.png")
        mask_path = os.path.join(self.mask_dir, f"{filename}.png")
        edge_path = os.path.join(self.edge_dir, f"{filename}.png") if self.is_train else None
        
        # Process using image processor
        processed = self.processor(
            image_path=img_path,
            mask_path=mask_path,
            edge_path=edge_path
        )
        
        # Construct return dictionary
        sample = {
            'image': processed.image,
            'mask': processed.mask
        }
        
        if self.is_train:
            sample['edge'] = processed.edge
        else:
            sample['name'] = filename
            
        return sample

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collation function for variable size masks/edges.
    
    Args:
        batch: List of dictionaries containing processed samples
        
    Returns:
        Dictionary with:
        - 'images': Stacked image tensors [B, C, H, W]
        - 'masks': List of mask tensors [1, H, W]
        - 'edges': List of edge tensors [1, H, W] (training only)
        - 'names': List of filenames (testing only)
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    # Stack images (all same size)
    images = torch.stack([item['image'] for item in batch])
    
    # Collect masks as list (variable sizes)
    masks = [item['mask'] for item in batch]
    
    # Construct output dictionary
    output = {
        'images': images,
        'masks': masks
    }
    
    # Add edges for training or names for testing
    if 'edge' in batch[0]:
        output['edges'] = [item['edge'] for item in batch]
    else:
        output['names'] = [item['name'] for item in batch]
        
    return output

def get_training_loaders(
    dataset_dirs: List[str],
    model_config: Dict,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    device: str = 'cuda'
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders.
    
    Args:
        dataset_dirs: List of dataset root directories
        model_config: Model configuration containing:
            - image_processing.target_size: Size to resize images to
            - image_processing.normalize_mean: RGB normalization means
            - image_processing.normalize_std: RGB normalization stds
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        val_ratio: Ratio of training data for validation (0 for no validation)
        device: Device for image processor operations ('cuda' or 'cpu')
        
    Returns:
        Tuple containing:
        - Training DataLoader producing batches with:
            - images: Tensor[B, C, H, W] - Normalized and resized images
            - masks: List[B] of Tensor[1, H, W] - Original size masks
            - edges: List[B] of Tensor[1, H, W] - Original size edges
        - Validation DataLoader (None if val_ratio=0) with same format
    
    Raises:
        ValueError: If no valid training datasets found
    """
    # Extract processor parameters from config
    processor_params = {
        'target_size': model_config['image_processing']['target_size'],
        'normalize_mean': tuple(model_config['image_processing']['normalize_mean']),
        'normalize_std': tuple(model_config['image_processing']['normalize_std'])
    }
    
    # Create datasets from all training directories
    train_datasets = []
    for train_dir in dataset_dirs:
        train_path = os.path.join(train_dir, 'train')
        if not os.path.exists(train_path):
            continue
            
        train_datasets.append(
            CODDataset(
                root_dir=train_path,
                processor_params=processor_params,
                is_train=True
            )
        )
    
    if not train_datasets:
        raise ValueError("No valid training datasets found")
        
    # Combine all training datasets
    combined_dataset = ConcatDataset(train_datasets)
    
    if val_ratio > 0:
        # Split combined dataset
        train_size = int((1 - val_ratio) * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            combined_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
        
    else:
        # No validation split
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        return train_loader, None

def get_test_loaders(
    dataset_dirs: List[str],
    model_config: Dict,
    batch_size: int = 16,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create test data loaders for each dataset.
    
    Args:
        dataset_dirs: List of dataset root directories
        model_config: Model configuration containing:
            - image_processing.target_size: Size to resize images to
            - image_processing.normalize_mean: RGB normalization means
            - image_processing.normalize_std: RGB normalization stds
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        device: Device for image processor operations ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping dataset names to DataLoaders producing batches with:
            - images: Tensor[B, C, H, W] - Normalized and resized images
            - masks: List[B] of Tensor[1, H, W] - Original size masks
            - names: List[B] of str - Original filenames for evaluation
    
    Raises:
        ValueError: If no valid test datasets found
    """
    # Extract processor parameters from config
    processor_params = {
        'target_size': model_config['image_processing']['target_size'],
        'normalize_mean': tuple(model_config['image_processing']['normalize_mean']),
        'normalize_std': tuple(model_config['image_processing']['normalize_std'])
    }
    
    test_loaders = {}
    for test_dir in dataset_dirs:
        test_path = os.path.join(test_dir, 'test')
        if not os.path.exists(test_path):
            continue
            
        dataset_name = os.path.basename(test_dir)
        test_dataset = CODDataset(
            root_dir=test_path,
            processor_params=processor_params,
            is_train=False
        )
        
        test_loaders[dataset_name] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    if not test_loaders:
        raise ValueError("No valid test datasets found")
        
    return test_loaders