"""
Training engine for Camouflaged Object Detection models.

This module provides a comprehensive training implementation with:
1. Automatic mixed precision training
2. Multi-scale segmentation and edge supervision
3. Resource-optimized batch processing
4. Metric tracking and model checkpointing

Core Components:
- TrainingMonitor: Handles metric tracking and checkpoint management
- Trainer: Manages training loops and optimization
- Batch Processing: Efficient multi-scale prediction and loss computation

Processing Pipeline:
1. Data Loading: Non-blocking tensor transfers
2. Forward Pass: Mixed precision inference
3. Multi-scale Loss: Segmentation and edge predictions
4. Optimization: Gradient scaling and clipping
5. Metric Tracking: Comprehensive performance monitoring
"""

import logging
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from utils.metrics import MetricsProcessor
from utils.loss_functions import CODLoss
from utils.run_manager import DirectoryManager
from models.spegnet import SPEGNet
from utils.data_loader import get_training_loaders

class TrainingMonitor:
    """
    Tracks and manages training metrics and model checkpoints.
    
    This class handles:
    1. Per-batch statistic aggregation
    2. Best model selection via F-measure
    3. Metric history management
    4. Atomic checkpoint saving
    
    Attributes:
        metrics_file (Path): JSON file for metric storage
        checkpoint_dir (Path): Directory for model checkpoints
        batch_stats (defaultdict): Running statistics tracker
        epoch_start (float): Epoch timing reference
        history (dict): Complete training history
            epochs: List of epoch statistics
            best_metrics: Best performance records
    """
    
    def __init__(self, dir_manager: DirectoryManager):
        """Initialize training monitor with directory structure."""
        # Output paths
        self.metrics_file = dir_manager.run_dirs.metrics_file
        self.checkpoint_dir = dir_manager.run_dirs.checkpoints
        
        # Statistics tracking
        self.batch_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        
        # Epoch tracking
        self.epoch_start = None
        
        # Initialize or load history
        self.history = {
            "epochs": [],
            "best_metrics": {
                "weighted_f": 0.0,
                "s_alpha": 0.0,
                "mae": float("inf")
            }
        }
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.history = json.load(f)

    def start_epoch(self):
        """Reset statistics and start timing for new epoch."""
        self.batch_stats.clear()
        self.epoch_start = time.time()

    def update_batch(self, metrics: Dict[str, float], timing: Dict[str, float], batch_size: int):
        """
        Updates running statistics with batch results.
        
        Accumulates weighted averages of metrics and timing statistics,
        accounting for variable batch sizes.
        
        Args:
            metrics: Current batch metric values
            timing: Timing measurements (data_time, forward_time, etc.)
            batch_size: Number of samples in batch
            
        Updates:
            - Running sums and counts for each statistic
            - Enables proper weighted averaging
        """
        # Update running statistics
        for key, value in {**metrics, **timing}.items():
            value = float(value) if torch.is_tensor(value) else value
            self.batch_stats[key]['sum'] += value * batch_size
            self.batch_stats[key]['count'] += batch_size

    def get_current_stats(self) -> Dict[str, float]:
        """Get current averaged statistics."""
        return {
            key: stats['sum'] / stats['count']
            for key, stats in self.batch_stats.items()
            if stats['count'] > 0
        }

    def check_best_model(self, current_metrics: Dict[str, float]) -> bool:
        """
        Determines if current model achieves best performance.
        
        Compares current F-measure against historical best and
        updates records if improved.
        
        Args:
            current_metrics: Current epoch evaluation metrics
        
        Returns:
            bool: True if new best model achieved
            
        Updates:
            - Best metrics history if improved
            - Saves updated history to disk
        """
        if current_metrics['weighted_f'] > self.history['best_metrics']['weighted_f']:
            self.history['best_metrics'] = current_metrics.copy()
            self.save_history()
            logging.info(
                f"New best model -> F-Measure: {current_metrics['weighted_f']:.4f}"
            )
            return True
        return False
    
    def save_history(self):
        """Atomic save of complete history structure."""
        tmp_path = self.metrics_file.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        tmp_path.rename(self.metrics_file)

    def save_epoch(self, epoch: int, phase: str):
        """
        Save epoch results to metrics file.
        
        Args:
            epoch: Current epoch number
            phase: Training phase ('train' or 'val')
        """
        stats = self.get_current_stats()
        epoch_time = time.time() - self.epoch_start

        # Split into metrics and timing
        metrics = {k: v for k, v in stats.items() if not k.endswith('_time')}
        timing = {k: v for k, v in stats.items() if k.endswith('_time')}
        timing['epoch_time'] = epoch_time

        # Ensure epoch entry exists
        while len(self.history['epochs']) <= epoch:
            self.history['epochs'].append({"epoch": len(self.history['epochs'])})
        
        # Update epoch data
        self.history['epochs'][epoch][phase] = {
            "metrics": metrics,
            "timing": timing
        }

        # Save complete history structure
        self.save_history()
        if phase == 'val':
            # Log epoch summary
            logging.info(
                f"Epoch {epoch} ({phase}) - "
                f"F-measure: {stats['weighted_f']:.4f}, "
                f"S-alpha: {stats['s_alpha']:.4f}, "
                f"MAE: {stats['mae']:.4f}, "
                f"Loss: {stats['loss']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        else:
            logging.info(
                f"Epoch {epoch} ({phase}) - "
                f"Loss: {stats['loss']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

class Trainer:
    """
    Manages complete training pipeline for Camouflaged Object Detection.
    
    Implements:
    1. Multi-scale training with automatic mixed precision
    2. Joint segmentation and edge supervision
    3. Layer-wise learning rate optimization
    4. Comprehensive validation and checkpointing
    
    Processing Flow:
    1. Batch Loading: Images, masks, and edge maps
    2. Forward Pass: Generate multi-scale predictions
    3. Loss Computation: Combined segmentation and edge losses
    4. Optimization: Scaled gradients with clipping
    5. Validation: Multi-metric performance tracking
    
    Args:
        config (Dict): Model and training configuration
        dir_manager (DirectoryManager): Output directory handler
        device (torch.device): Computing device
    """
    
    def __init__(
        self,
        config: Dict,
        dir_manager: DirectoryManager,
        device: torch.device,
    ):
        # Basic setup
        self.config = config['training']
        self.model_config = config['model']
        self.device = device
        
        # Initialize model
        self.model = SPEGNet(config['model']).to(device)
        
        # Training parameters
        self.batch_size = self.config['batch_size']
        self.num_epochs = self.config['num_epochs']
        self.grad_clip = self.config.get('gradient_clip', 1.0)
        self.early_stop_patience = self.config.get('early_stop_patience', 15)
        self.save_freq = self.config.get('save_freq', 1)
        
        # Initialize components
        self._setup_optimization()
        self.criterion = CODLoss(**self.config['loss']).to(device)
        self.monitor = TrainingMonitor(dir_manager)
        self.metrics_processor = MetricsProcessor()
        
        # Mixed precision setup
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        
    def _setup_optimization(self):
        """Configure optimization with layerwise learning rates."""
        param_groups = self._get_param_groups()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['optimizer'].get('weight_decay', 0.01)
        )
        
        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.config['scheduler'].get('factor', 0.5),
            patience=self.config['scheduler'].get('patience', 5),
            min_lr=self.config['scheduler'].get('min_lr', 1e-6)
        )

    def _get_param_groups(self) -> List[Dict]:
        """Create parameter groups for different learning rates."""
        # Parameter groups
        encoder_params = {'params': [], 'weight_decay': 0.0}
        encoder_norm_params = {'params': [], 'weight_decay': 0.0}
        decoder_params = {'params': [], 
                         'weight_decay': self.config['optimizer'].get('weight_decay', 0.01)}
        decoder_norm_params = {'params': [], 'weight_decay': 0.0}
        
        # Group parameters
        for name, p in self.model.named_parameters():
            if 'encoder' in name:
                if 'norm' in name or 'bn' in name:
                    encoder_norm_params['params'].append(p)
                else:
                    encoder_params['params'].append(p)
            else:
                if 'norm' in name or 'bn' in name:
                    decoder_norm_params['params'].append(p)
                else:
                    decoder_params['params'].append(p)
        
        # Set learning rates
        base_lr = self.config['optimizer']['learning_rate']
        encoder_lr_scale = self.config['optimizer'].get('encoder_lr_ratio', 0.1)
        
        encoder_params['lr'] = base_lr * encoder_lr_scale
        encoder_norm_params['lr'] = base_lr * encoder_lr_scale
        decoder_params['lr'] = base_lr
        decoder_norm_params['lr'] = base_lr
        
        return [encoder_params, encoder_norm_params, 
                decoder_params, decoder_norm_params]
    
    def _process_batch(self, batch: Dict[str, torch.Tensor], 
                  is_train: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Processes single batch through complete training pipeline.
        
        Steps:
        1. Data Transfer: Non-blocking device placement
        2. Forward Pass: Mixed precision inference
        3. Prediction Resizing: Multi-scale alignment
        4. Loss Computation: Segmentation and edge losses
        5. Optimization: Gradient scaling and updates (training only)
        6. Metric Computation: Performance measurements
        
        Args:
            batch: Input data containing:
                - images: Input images
                - masks: Ground truth segmentations
                - edges: Ground truth edge maps
            is_train: Whether in training mode
            
        Returns:
            metrics: Performance measurements
            timing: Processing time breakdown
        """

        timing = {}
        batch_start = time.time()
        
        # Data loading
        data_start = time.time()
        images = batch['images'].to(self.device, non_blocking=True)
        masks = [m.to(self.device, non_blocking=True) for m in batch['masks']]
        edges = [e.to(self.device, non_blocking=True) for e in batch['edges']]
        timing['data_time'] = time.time() - data_start
        
        # Forward pass
        forward_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
        timing['forward_time'] = time.time() - forward_start
        
        # Prediction resizing
        resize_start = time.time()
        batch_size = len(masks)
        # Reorganize by batch first, then scale
        batch_predictions = []  # [batch_size][num_scales]
        batch_edges = []
        
        # Single loop for both segmentation and edge predictions
        for i in range(batch_size):
            target_size = masks[i].shape[-2:]
            edge_size = edges[i].shape[-2:]
            
            # Get all scale predictions for this sample
            sample_preds = []
            for scale_preds in outputs['predictions']:
                sample_preds.append(
                    F.interpolate(
                        scale_preds[i:i+1],
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                )
            batch_predictions.append(sample_preds)

            # Resize edge prediction
            batch_edges.append(
                F.interpolate(
                    outputs['edge'][i:i+1],
                    size=edge_size,
                    mode='bilinear',
                    align_corners=False
                )
            )
        timing['resize_time'] = time.time() - resize_start
        
        # Loss computation
        loss_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_dict = self.criterion(
                predictions=batch_predictions,
                edge_pred=batch_edges,
                masks=masks,
                edges=edges
            )
        timing['loss_time'] = time.time() - loss_start
        
        
        # Optimization
        if is_train:
            backward_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss_dict['loss']).backward()
            
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            timing['backward_time'] = time.time() - backward_start
            metrics = loss_dict
        else:
            # Metrics computation
            metric_start = time.time()
            with torch.no_grad():
                metrics = self.metrics_processor.compute_metrics(
                    seg_pred=[b[-1] for b in batch_predictions],  # Last scale predictions
                    seg_gt=masks,
                    edge_pred=batch_edges,
                    edge_gt=edges
                )
            metrics.update(loss_dict)
            timing['metric_time'] = time.time() - metric_start
            
        timing['batch_time'] = time.time() - batch_start
        
        return metrics, timing

    def _resize_predictions(self, preds, targets):
        """Resize predictions to match target sizes."""
        if isinstance(preds, (list, tuple)):
            return [F.interpolate(p, size=t.shape[-2:], 
                                mode='bilinear', align_corners=False)
                    for p, t in zip(preds, targets)]
        return F.interpolate(preds, size=targets[0].shape[-2:], 
                            mode='bilinear', align_corners=False)

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Executes single training epoch.
        
        Processing:
        1. Sets model to training mode
        2. Iterates through batches with progress tracking
        3. Processes each batch with optimization
        4. Updates monitoring statistics
        
        Args:
            loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            dict: Averaged epoch metrics
        """

        self.model.train()
        self.monitor.start_epoch()
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Process batch
            metrics, timing = self._process_batch(batch, is_train=True)
            
            # Update monitoring
            self.monitor.update_batch(
                metrics=metrics,
                timing=timing,
                batch_size=len(batch['images'])
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'seg_loss': f"{metrics['seg_loss']:.4f}",
                'edge_loss': f"{metrics['edge_loss']:.4f}",
                'time': f"{timing['batch_time']:.3f}s"
            })
        
        return self.monitor.get_current_stats()

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Performs validation pass.
        
        Processing:
        1. Sets model to evaluation mode
        2. Processes batches without gradients
        3. Computes comprehensive metrics
        4. Tracks performance statistics
        
        Args:
            loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            dict: Complete validation metrics
        """
        self.model.eval()
        self.monitor.start_epoch()
        
        pbar = tqdm(loader, desc='Validation')
        for batch_idx, batch in enumerate(pbar):
            metrics, timing = self._process_batch(batch, is_train=False)
            
            self.monitor.update_batch(
                metrics=metrics,
                timing=timing,
                batch_size=len(batch['images'])
            )
            
            pbar.set_postfix({
                'f_measure': f"{metrics['weighted_f']:.4f}",
                's_alpha': f"{metrics['s_alpha']:.4f}",
                'mae': f"{metrics['mae']:.4f}",
                'loss': f"{metrics['loss']:.4f}",
                'time': f"{timing['batch_time']:.3f}s"
            })
            
        return self.monitor.get_current_stats()
        
    def train(self, dataset_dirs: List[str]):
        """Main training loop."""
        try:
            # Setup data loaders
            train_loader, val_loader = get_training_loaders(
                dataset_dirs=dataset_dirs,
                model_config=self.model_config,
                batch_size=self.batch_size,
                num_workers=self.config['num_workers'],
                val_ratio=self.config['val_ratio']
            )
            
            # Log setup
            logging.info(f"Training samples: {len(train_loader.dataset)}")
            if val_loader:
                logging.info(f"Validation samples: {len(val_loader.dataset)}")

            # Initialize tracking variables
            best_weighted_f = 0.0
            early_stop_counter = 0
            min_delta = self.config.get('min_delta', 1e-4)

            # Training loop
            for epoch in range(self.num_epochs):
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                self.monitor.save_epoch(epoch, 'train')
                
                # Validation phase
                if val_loader:
                    val_metrics = self.validate(val_loader, epoch)
                    self.monitor.save_epoch(epoch, 'val')
                    
                    # Learning rate adjustment
                    self.scheduler.step(val_metrics['weighted_f'])
                    
                    # Early stopping check using validation loss
                    if val_metrics['weighted_f'] - best_weighted_f > min_delta:
                        best_weighted_f = val_metrics['weighted_f']
                        early_stop_counter = 0
                        # Save model if structure measure also improved
                        if self.monitor.check_best_model(val_metrics):
                            self._save_checkpoint(epoch, val_metrics, is_best=True)
                    
                    else:
                        early_stop_counter += 1
                        
                    if early_stop_counter >= self.early_stop_patience:
                        logging.info("Early stopping triggered")
                        break
                
                # Regular checkpointing
                if (epoch + 1) % self.save_freq == 0:
                    self._save_checkpoint(
                        epoch,
                        val_metrics if val_loader else train_metrics,
                        is_best=False
                    )
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Training error: {str(e)}", exc_info=True)
            raise
            
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint."""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'metrics': metrics,
            'config': {
                'training': self.config,
                'model': self.model_config
            }
        }
        
        filename = 'model_best.pth' if is_best else f'checkpoint_{epoch:03d}.pth'
        save_path = self.monitor.checkpoint_dir / filename
        torch.save(save_dict, save_path)
        logging.info(f"Saved checkpoint: {save_path}")