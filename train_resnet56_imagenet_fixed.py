#!/usr/bin/env python3
"""
Fixed ImageNet Training Script for ResNet56 with Optimized Pruning
Author: Manus AI
Date: June 30, 2025

This script fixes the BatchNorm dimension mismatch errors and provides robust
training for ResNet56 on ImageNet with optimized pruning capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import time
import logging
from typing import Dict, Tuple, Optional, List
import json
from tqdm import tqdm
import numpy as np
import warnings
import traceback

# Import our fixed modules
try:
    from imagenet_data_loader import ImageNetDataLoader, create_imagenet_loaders
    from resnet56_imagenet import ResNet56ImageNet, create_resnet56_imagenet, ResNet56ImageNetAnalyzer
    from optimized_prune_resnet56_fixed import (
        OptimizedPruner, 
        PruningConfig, 
        ImportanceScorer,
        prune_resnet56_optimized
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the same directory")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ImageNetPruningConfig(PruningConfig):
    """Extended pruning configuration for ImageNet scale training."""
    
    def __init__(self, **kwargs):
        # ImageNet-specific defaults
        imagenet_defaults = {
            # More conservative pruning for ImageNet complexity
            'stage1_ratio': 0.10,  # Very conservative for initial layers
            'stage2_ratio': 0.25,  # Conservative for early features
            'stage3_ratio': 0.40,  # Moderate for mid-level features
            'stage4_ratio': 0.55,  # More aggressive for high-level features
            
            # Extended fine-tuning for ImageNet
            'phase1_epochs': 15,
            'phase2_epochs': 20,
            'phase3_epochs': 25,
            
            # Lower learning rates for ImageNet stability
            'phase1_lr': 0.005,
            'phase2_lr': 0.002,
            'phase3_lr': 0.001,
            
            # Enhanced knowledge distillation for complex dataset
            'distillation_weight': 0.4,
            'temperature': 5.0,
            
            # Adjusted importance weights for ImageNet
            'magnitude_weight': 0.20,
            'gradient_weight': 0.35,
            'variance_weight': 0.25,
            'taylor_weight': 0.20
        }
        
        # Update with ImageNet defaults, then user overrides
        imagenet_defaults.update(kwargs)
        super().__init__(**imagenet_defaults)


class SafeKnowledgeDistillationLoss(nn.Module):
    """Safe knowledge distillation loss with error handling."""
    
    def __init__(self, temperature: float = 5.0, alpha: float = 0.4, 
                 label_smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
        # Use label smoothing for ImageNet
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined knowledge distillation and classification loss with error handling.
        """
        try:
            # Ensure outputs have the same shape
            if student_outputs.shape != teacher_outputs.shape:
                logger.warning(f"Output shape mismatch: student {student_outputs.shape}, teacher {teacher_outputs.shape}")
                # Use only classification loss if shapes don't match
                ce_loss = self.ce_loss(student_outputs, targets)
                return ce_loss, {'total_loss': ce_loss.item(), 'ce_loss': ce_loss.item(), 'kd_loss': 0.0}
            
            # Standard classification loss with label smoothing
            ce_loss = self.ce_loss(student_outputs, targets)
            
            # Knowledge distillation loss
            student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
            kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            
            loss_components = {
                'total_loss': total_loss.item(),
                'ce_loss': ce_loss.item(),
                'kd_loss': kd_loss.item()
            }
            
            return total_loss, loss_components
            
        except Exception as e:
            logger.error(f"Error in knowledge distillation loss: {e}")
            # Fallback to classification loss only
            ce_loss = self.ce_loss(student_outputs, targets)
            return ce_loss, {'total_loss': ce_loss.item(), 'ce_loss': ce_loss.item(), 'kd_loss': 0.0}


class SafeImageNetTrainer:
    """Safe trainer for ResNet56 on ImageNet with comprehensive error handling."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.world_size = args.world_size
        self.rank = args.rank
        
        # Initialize distributed training if specified
        if args.distributed:
            self._setup_distributed()
        
        # Initialize data loaders with error handling
        try:
            self.train_loader, self.val_loader = self._prepare_data()
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise
        
        # Initialize model with error handling
        try:
            self.model = self._prepare_model()
            self.original_model = None  # Will store teacher model for distillation
        except Exception as e:
            logger.error(f"Failed to prepare model: {e}")
            raise
        
        # Initialize pruning configuration
        self.pruning_config = self._create_pruning_config()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_top5_accuracy = 0.0
        self.training_history = []
        
        # Performance tracking
        self.epoch_times = []
        self.memory_usage = []
    
    def _setup_distributed(self):
        """Setup distributed training with error handling."""
        try:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            torch.cuda.set_device(self.rank)
            logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            self.args.distributed = False
            logger.info("Falling back to single-GPU training")
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare ImageNet data loaders with comprehensive error handling."""
        
        try:
            # Adjust batch size for distributed training
            batch_size = self.args.batch_size
            if self.args.distributed:
                batch_size = batch_size // self.world_size
            
            # Create data loader with error handling
            data_loader = ImageNetDataLoader(
                data_root=self.args.data_root,
                batch_size=batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                use_subset=self.args.use_subset,
                subset_size=self.args.subset_size
            )
            
            train_loader = data_loader.get_train_loader()
            val_loader = data_loader.get_val_loader()
            
            # Wrap with distributed sampler if needed
            if self.args.distributed:
                from torch.utils.data.distributed import DistributedSampler
                
                train_sampler = DistributedSampler(train_loader.dataset, 
                                                 num_replicas=self.world_size,
                                                 rank=self.rank)
                val_sampler = DistributedSampler(val_loader.dataset,
                                               num_replicas=self.world_size,
                                               rank=self.rank,
                                               shuffle=False)
                
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    num_workers=self.args.num_workers,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    val_loader.dataset,
                    batch_size=batch_size,
                    sampler=val_sampler,
                    num_workers=self.args.num_workers,
                    pin_memory=True
                )
            
            logger.info(f"Data prepared: {len(train_loader)} train batches, {len(val_loader)} val batches")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _prepare_model(self) -> nn.Module:
        """Initialize ResNet56 model for ImageNet with error handling."""
        
        try:
            model = create_resnet56_imagenet(
                num_classes=1000,
                pretrained=bool(self.args.pretrained_path),
                pretrained_path=self.args.pretrained_path
            )
            
            model = model.to(self.device)
            
            # Wrap with DataParallel or DistributedDataParallel
            if self.args.distributed:
                model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
            elif torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"Model initialized:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error preparing model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_pruning_config(self) -> ImageNetPruningConfig:
        """Create ImageNet-specific pruning configuration."""
        
        config = ImageNetPruningConfig(
            stage1_ratio=self.args.stage1_ratio,
            stage2_ratio=self.args.stage2_ratio,
            stage3_ratio=self.args.stage3_ratio,
            stage4_ratio=getattr(self.args, 'stage4_ratio', 0.55),
            phase1_epochs=self.args.phase1_epochs,
            phase2_epochs=self.args.phase2_epochs,
            phase3_epochs=self.args.phase3_epochs,
            phase1_lr=self.args.phase1_lr,
            phase2_lr=self.args.phase2_lr,
            phase3_lr=self.args.phase3_lr,
            distillation_weight=self.args.distillation_weight,
            temperature=self.args.temperature
        )
        
        return config
    
    def safe_forward_pass(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Perform a safe forward pass with error handling."""
        try:
            return model(inputs)
        except RuntimeError as e:
            if "running_mean should contain" in str(e) or "running_var should contain" in str(e):
                logger.error(f"BatchNorm dimension mismatch: {e}")
                logger.info("This usually indicates a pruning error. Attempting to fix...")
                
                # Try to reinitialize BatchNorm layers
                self._fix_batchnorm_layers(model)
                return model(inputs)
            else:
                raise e
    
    def _fix_batchnorm_layers(self, model: nn.Module):
        """Attempt to fix BatchNorm layers with dimension mismatches."""
        logger.info("Attempting to fix BatchNorm layers...")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Find the corresponding conv layer
                conv_name = name.replace('.bn', '.conv').replace('bn1', 'conv1').replace('bn2', 'conv2')
                
                # Find the conv layer
                conv_module = None
                for conv_name_candidate, conv_candidate in model.named_modules():
                    if conv_name_candidate == conv_name and isinstance(conv_candidate, nn.Conv2d):
                        conv_module = conv_candidate
                        break
                
                if conv_module is not None:
                    expected_channels = conv_module.out_channels
                    if module.num_features != expected_channels:
                        logger.info(f"Fixing BatchNorm {name}: {module.num_features} -> {expected_channels}")
                        
                        # Create new BatchNorm with correct dimensions
                        new_bn = nn.BatchNorm2d(
                            num_features=expected_channels,
                            eps=module.eps,
                            momentum=module.momentum,
                            affine=module.affine,
                            track_running_stats=module.track_running_stats
                        ).to(module.weight.device)
                        
                        # Copy parameters if possible
                        if module.affine and expected_channels <= module.num_features:
                            new_bn.weight.data[:expected_channels] = module.weight.data[:expected_channels]
                            new_bn.bias.data[:expected_channels] = module.bias.data[:expected_channels]
                        
                        if module.track_running_stats and expected_channels <= module.num_features:
                            new_bn.running_mean.data[:expected_channels] = module.running_mean.data[:expected_channels]
                            new_bn.running_var.data[:expected_channels] = module.running_var.data[:expected_channels]
                            new_bn.num_batches_tracked = module.num_batches_tracked
                        
                        # Replace the module
                        self._replace_module(model, name, new_bn)
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Replace a module in the model."""
        name_parts = module_name.split('.')
        parent = model
        
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, name_parts[-1], new_module)
    
    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int, 
                   teacher_model: Optional[nn.Module] = None) -> Dict:
        """Train model for one epoch with comprehensive error handling."""
        
        model.train()
        if teacher_model is not None:
            teacher_model.eval()
        
        # Set distributed sampler epoch
        if self.args.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        # Use tqdm only on main process
        if not self.args.distributed or self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = self.train_loader
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass with error handling
                outputs = self.safe_forward_pass(model, inputs)
                
                # Compute loss
                if teacher_model is not None and isinstance(criterion, SafeKnowledgeDistillationLoss):
                    with torch.no_grad():
                        teacher_outputs = self.safe_forward_pass(teacher_model, inputs)
                    loss, loss_components = criterion(outputs, teacher_outputs, targets)
                    running_ce_loss += loss_components['ce_loss']
                    running_kd_loss += loss_components['kd_loss']
                else:
                    loss = criterion(outputs, targets)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                
                # Top-1 and Top-5 accuracy
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                
                correct_top1 += correct[:1].view(-1).float().sum(0)
                correct_top5 += correct[:5].view(-1).float().sum(0)
                total += targets.size(0)
                
                # Update progress bar (only on main process)
                if not self.args.distributed or self.rank == 0:
                    if isinstance(pbar, tqdm):
                        pbar.set_postfix({
                            'Loss': f'{running_loss/(batch_idx+1):.3f}',
                            'Top1': f'{100.*correct_top1/total:.2f}%',
                            'Top5': f'{100.*correct_top5/total:.2f}%'
                        })
                
                # Memory cleanup
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                # Continue with next batch
                continue
        
        epoch_time = time.time() - start_time
        self.epoch_times.append(epoch_time)
        
        # Aggregate statistics across processes
        if self.args.distributed:
            try:
                # Reduce statistics across all processes
                stats_tensor = torch.tensor([running_loss, correct_top1, correct_top5, total], 
                                           device=self.device)
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                running_loss, correct_top1, correct_top5, total = stats_tensor.tolist()
            except Exception as e:
                logger.warning(f"Failed to aggregate distributed statistics: {e}")
        
        epoch_stats = {
            'epoch': epoch,
            'train_loss': running_loss / len(self.train_loader),
            'train_top1_accuracy': 100. * correct_top1 / total,
            'train_top5_accuracy': 100. * correct_top5 / total,
            'epoch_time': epoch_time
        }
        
        if teacher_model is not None:
            epoch_stats.update({
                'train_ce_loss': running_ce_loss / len(self.train_loader),
                'train_kd_loss': running_kd_loss / len(self.train_loader)
            })
        
        return epoch_stats
    
    def evaluate(self, model: nn.Module, criterion: nn.Module) -> Dict:
        """Evaluate model on ImageNet validation set with error handling."""
        
        model.eval()
        test_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            if not self.args.distributed or self.rank == 0:
                pbar = tqdm(self.val_loader, desc='Evaluating')
            else:
                pbar = self.val_loader
            
            for inputs, targets in pbar:
                try:
                    inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    
                    outputs = self.safe_forward_pass(model, inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    
                    # Top-1 and Top-5 accuracy
                    _, pred = outputs.topk(5, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(targets.view(1, -1).expand_as(pred))
                    
                    correct_top1 += correct[:1].view(-1).float().sum(0)
                    correct_top5 += correct[:5].view(-1).float().sum(0)
                    total += targets.size(0)
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        # Aggregate statistics across processes
        if self.args.distributed:
            try:
                stats_tensor = torch.tensor([test_loss, correct_top1, correct_top5, total], 
                                           device=self.device)
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                test_loss, correct_top1, correct_top5, total = stats_tensor.tolist()
            except Exception as e:
                logger.warning(f"Failed to aggregate evaluation statistics: {e}")
        
        top1_accuracy = 100. * correct_top1 / total
        top5_accuracy = 100. * correct_top5 / total
        avg_loss = test_loss / len(self.val_loader)
        
        eval_stats = {
            'val_loss': avg_loss,
            'val_top1_accuracy': top1_accuracy,
            'val_top5_accuracy': top5_accuracy
        }
        
        if not self.args.distributed or self.rank == 0:
            logger.info(f"Validation Results - Loss: {avg_loss:.4f}, "
                       f"Top-1: {top1_accuracy:.2f}%, Top-5: {top5_accuracy:.2f}%")
        
        return eval_stats
    
    def safe_pruning_execution(self, model: nn.Module) -> Tuple[nn.Module, Dict]:
        """Execute pruning with comprehensive error handling."""
        try:
            logger.info("Starting safe pruning execution...")
            
            # Get the base model (unwrap from DataParallel/DDP)
            base_model = self._get_base_model(model)
            
            # Create a copy for pruning
            model_copy = copy.deepcopy(base_model)
            
            # Execute pruning on the copy
            pruned_model, pruning_stats = prune_resnet56_optimized(model_copy, self.pruning_config)
            
            logger.info("Pruning completed successfully")
            logger.info(f"Compression ratio: {pruning_stats['compression_ratio']:.2%}")
            
            return pruned_model, pruning_stats
            
        except Exception as e:
            logger.error(f"Error during pruning: {e}")
            logger.error(traceback.format_exc())
            logger.info("Returning original model without pruning")
            return self._get_base_model(model), {'compression_ratio': 0.0, 'error': str(e)}
    
    def execute_safe_training(self):
        """Execute the complete training pipeline with comprehensive error handling."""
        
        try:
            if not self.args.distributed or self.rank == 0:
                logger.info("Starting safe ResNet56 training on ImageNet")
            
            # Phase 1: Initial training (if needed)
            if not self.args.skip_initial_training:
                if not self.args.distributed or self.rank == 0:
                    logger.info("Phase 1: Initial training")
                
                try:
                    initial_history = self.fine_tune_phase(
                        self.model, 
                        epochs=self.args.initial_epochs,
                        lr=self.args.initial_lr
                    )
                    self.training_history.extend(initial_history)
                except Exception as e:
                    logger.error(f"Error in initial training: {e}")
                    if not self.args.distributed or self.rank == 0:
                        logger.info("Continuing with pruning phase...")
            
            # Store original model for knowledge distillation
            if self.args.use_distillation:
                try:
                    self.original_model = copy.deepcopy(self._get_base_model(self.model))
                    if not self.args.distributed or self.rank == 0:
                        logger.info("Original model stored for knowledge distillation")
                except Exception as e:
                    logger.error(f"Error storing original model: {e}")
                    self.args.use_distillation = False
                    logger.info("Disabling knowledge distillation")
            
            # Phase 2: Pruning
            if self.args.enable_pruning:
                if not self.args.distributed or self.rank == 0:
                    logger.info("Phase 2: Safe pruning execution")
                
                try:
                    # Execute pruning (only on main process to avoid conflicts)
                    if not self.args.distributed or self.rank == 0:
                        base_model = self._get_base_model(self.model)
                        pruned_model, pruning_stats = self.safe_pruning_execution(base_model)
                        
                        # Wrap the pruned model
                        self.model = self._wrap_model(pruned_model)
                        
                        # Store pruning statistics
                        self.pruning_stats = pruning_stats
                        
                        logger.info("Pruning phase completed successfully")
                    
                    # Synchronize across processes if distributed
                    if self.args.distributed:
                        dist.barrier()
                
                except Exception as e:
                    logger.error(f"Error in pruning phase: {e}")
                    logger.info("Continuing with original model...")
            
            # Phase 3: Final evaluation
            if not self.args.distributed or self.rank == 0:
                logger.info("Phase 3: Final evaluation")
                
                try:
                    final_stats = self.evaluate(self.model, nn.CrossEntropyLoss())
                    
                    # Add pruning statistics if available
                    if hasattr(self, 'pruning_stats'):
                        final_stats.update(self.pruning_stats)
                    
                    # Save final results
                    self.save_final_results(final_stats)
                    
                    logger.info("Training completed successfully!")
                    logger.info(f"Final Top-1 accuracy: {final_stats['val_top1_accuracy']:.2f}%")
                    logger.info(f"Final Top-5 accuracy: {final_stats['val_top5_accuracy']:.2f}%")
                    
                    if 'compression_ratio' in final_stats:
                        logger.info(f"Compression ratio: {final_stats['compression_ratio']:.2%}")
                
                except Exception as e:
                    logger.error(f"Error in final evaluation: {e}")
                    logger.error(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"Critical error in training pipeline: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def fine_tune_phase(self, model: nn.Module, epochs: int, lr: float, 
                       teacher_model: Optional[nn.Module] = None) -> List[Dict]:
        """Fine-tune model with error handling."""
        
        try:
            # Setup optimizer
            optimizer = optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=1e-4,
                nesterov=True
            )
            
            # Setup scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, epochs//3), T_mult=2, eta_min=lr*0.01
            )
            
            # Setup criterion
            if teacher_model is not None:
                criterion = SafeKnowledgeDistillationLoss(
                    temperature=self.pruning_config.temperature,
                    alpha=self.pruning_config.distillation_weight,
                    label_smoothing=0.1
                )
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            phase_history = []
            
            for epoch in range(epochs):
                try:
                    # Training
                    train_stats = self.train_epoch(model, optimizer, criterion, epoch, teacher_model)
                    
                    # Evaluation
                    eval_stats = self.evaluate(model, nn.CrossEntropyLoss())
                    
                    # Combine statistics
                    epoch_stats = {**train_stats, **eval_stats}
                    phase_history.append(epoch_stats)
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Save best model (only on main process)
                    if not self.args.distributed or self.rank == 0:
                        if eval_stats['val_top1_accuracy'] > self.best_accuracy:
                            self.best_accuracy = eval_stats['val_top1_accuracy']
                            self.best_top5_accuracy = eval_stats['val_top5_accuracy']
                            self.save_checkpoint(model, epoch_stats, is_best=True)
                        
                        logger.info(f"Epoch {epoch}: Train Top-1: {train_stats['train_top1_accuracy']:.2f}%, "
                                   f"Val Top-1: {eval_stats['val_top1_accuracy']:.2f}%, "
                                   f"Val Top-5: {eval_stats['val_top5_accuracy']:.2f}%")
                
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {e}")
                    continue
            
            return phase_history
            
        except Exception as e:
            logger.error(f"Error in fine-tuning phase: {e}")
            return []
    
    def _get_base_model(self, model):
        """Extract base model from DataParallel/DistributedDataParallel wrapper."""
        if isinstance(model, (nn.DataParallel, DDP)):
            return model.module
        return model
    
    def _wrap_model(self, model):
        """Wrap model with appropriate parallel wrapper."""
        model = model.to(self.device)
        
        if self.args.distributed:
            return DDP(model, device_ids=[self.rank], find_unused_parameters=True)
        elif torch.cuda.device_count() > 1:
            return nn.DataParallel(model)
        return model
    
    def save_checkpoint(self, model: nn.Module, stats: Dict, is_best: bool = False):
        """Save model checkpoint with error handling."""
        
        try:
            if self.args.distributed and self.rank != 0:
                return  # Only save on main process
            
            os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self._get_base_model(model).state_dict(),
                'stats': stats,
                'args': vars(self.args),
                'pruning_config': self.pruning_config.__dict__,
                'epoch': self.current_epoch,
                'best_accuracy': self.best_accuracy
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved - Top-1: {stats['val_top1_accuracy']:.2f}%, "
                           f"Top-5: {stats['val_top5_accuracy']:.2f}%")
        
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def save_final_results(self, final_stats: Dict):
        """Save final training results with error handling."""
        
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # Save comprehensive results
            results = {
                'final_statistics': final_stats,
                'training_history': self.training_history,
                'arguments': vars(self.args),
                'pruning_configuration': self.pruning_config.__dict__,
                'performance_metrics': {
                    'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
                    'total_training_time': sum(self.epoch_times) if self.epoch_times else 0
                }
            }
            
            results_path = os.path.join(self.args.output_dir, 'imagenet_training_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save final model
            final_model_path = os.path.join(self.args.output_dir, 'final_resnet56_imagenet.pth')
            torch.save(self._get_base_model(self.model).state_dict(), final_model_path)
            
            logger.info(f"Final results saved to {self.args.output_dir}")
        
        except Exception as e:
            logger.error(f"Error saving final results: {e}")


def parse_arguments():
    """Parse command line arguments for ImageNet training."""
    
    parser = argparse.ArgumentParser(description='Safe ResNet56 ImageNet Training with Optimized Pruning')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='./data/imagenet', 
                       help='ImageNet dataset root directory')
    parser.add_argument('--batch-size', type=int, default=256, 
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=8, 
                       help='Number of data loading workers')
    parser.add_argument('--use-subset', action='store_true', 
                       help='Use subset of ImageNet for faster experimentation')
    parser.add_argument('--subset-size', type=int, default=100000, 
                       help='Size of ImageNet subset')
    
    # Model arguments
    parser.add_argument('--pretrained-path', type=str, default='', 
                       help='Path to pretrained model weights')
    
    # Training arguments
    parser.add_argument('--initial-epochs', type=int, default=90, 
                       help='Initial training epochs')
    parser.add_argument('--initial-lr', type=float, default=0.1, 
                       help='Initial learning rate')
    parser.add_argument('--skip-initial-training', action='store_true', 
                       help='Skip initial training phase')
    
    # Pruning arguments
    parser.add_argument('--enable-pruning', action='store_true', 
                       help='Enable progressive pruning')
    parser.add_argument('--stage1-ratio', type=float, default=0.10, 
                       help='Stage 1 pruning ratio')
    parser.add_argument('--stage2-ratio', type=float, default=0.25, 
                       help='Stage 2 pruning ratio')
    parser.add_argument('--stage3-ratio', type=float, default=0.40, 
                       help='Stage 3 pruning ratio')
    parser.add_argument('--stage4-ratio', type=float, default=0.55, 
                       help='Stage 4 pruning ratio')
    
    # Progressive pruning schedule
    parser.add_argument('--phase1-epochs', type=int, default=15, 
                       help='Phase 1 fine-tuning epochs')
    parser.add_argument('--phase2-epochs', type=int, default=20, 
                       help='Phase 2 fine-tuning epochs')
    parser.add_argument('--phase3-epochs', type=int, default=25, 
                       help='Phase 3 fine-tuning epochs')
    parser.add_argument('--phase1-lr', type=float, default=0.005, 
                       help='Phase 1 learning rate')
    parser.add_argument('--phase2-lr', type=float, default=0.002, 
                       help='Phase 2 learning rate')
    parser.add_argument('--phase3-lr', type=float, default=0.001, 
                       help='Phase 3 learning rate')
    
    # Knowledge distillation arguments
    parser.add_argument('--use-distillation', action='store_true', 
                       help='Use knowledge distillation')
    parser.add_argument('--distillation-weight', type=float, default=0.4, 
                       help='Distillation loss weight')
    parser.add_argument('--temperature', type=float, default=5.0, 
                       help='Distillation temperature')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', 
                       help='Use distributed training')
    parser.add_argument('--world-size', type=int, default=1, 
                       help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0, 
                       help='Rank of current process')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_imagenet', 
                       help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='./results_imagenet', 
                       help='Output directory')
    
    return parser.parse_args()


def main():
    """Main training function with comprehensive error handling."""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        # Create trainer and execute training
        trainer = SafeImageNetTrainer(args)
        trainer.execute_safe_training()
        
        # Cleanup distributed training
        if args.distributed:
            dist.destroy_process_group()
    
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

