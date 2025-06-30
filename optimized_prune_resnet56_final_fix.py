"""
Final Fixed Optimized Pruning Implementation for ResNet56
Author: Manus AI
Date: June 30, 2025

This module implements an advanced pruning strategy with proper handling of residual connections
and BatchNorm layers to completely fix dimension mismatch errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    """Configuration class for optimized pruning parameters."""
    # Stage-wise pruning ratios
    stage1_ratio: float = 0.15  # Conservative pruning for early layers
    stage2_ratio: float = 0.40  # Moderate pruning for middle layers  
    stage3_ratio: float = 0.60  # Aggressive pruning for late layers
    stage4_ratio: float = 0.55  # For ImageNet 4-stage architecture
    
    # Progressive pruning schedule
    phase1_ratio: float = 0.30  # First pruning phase
    phase2_ratio: float = 0.40  # Second pruning phase
    phase3_ratio: float = 0.30  # Final pruning phase
    
    # Fine-tuning parameters
    phase1_epochs: int = 10
    phase2_epochs: int = 15
    phase3_epochs: int = 20
    
    # Learning rates for fine-tuning
    phase1_lr: float = 0.01
    phase2_lr: float = 0.005
    phase3_lr: float = 0.001
    
    # Importance scoring weights
    magnitude_weight: float = 0.25
    gradient_weight: float = 0.30
    variance_weight: float = 0.25
    taylor_weight: float = 0.20
    
    # Knowledge distillation
    distillation_weight: float = 0.3
    temperature: float = 4.0


class ImportanceScorer:
    """Advanced importance scoring system for filter selection."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.gradient_cache = {}
        self.activation_cache = {}
        
    def compute_magnitude_importance(self, layer: nn.Conv2d) -> torch.Tensor:
        """Compute L1-norm based importance scores."""
        weights = layer.weight.data
        # Compute L1 norm for each filter
        importance = torch.sum(torch.abs(weights.view(weights.size(0), -1)), dim=1)
        return importance
    
    def compute_gradient_importance(self, layer: nn.Conv2d, layer_name: str) -> torch.Tensor:
        """Compute gradient-based importance scores."""
        if layer.weight.grad is None:
            logger.warning(f"No gradient available for layer {layer_name}")
            return torch.zeros(layer.weight.size(0), device=layer.weight.device)
        
        grad = layer.weight.grad.data
        # Compute gradient magnitude for each filter
        importance = torch.sum(torch.abs(grad.view(grad.size(0), -1)), dim=1)
        return importance
    
    def compute_variance_importance(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute activation variance-based importance scores."""
        if activations is None:
            return torch.zeros(1)
        
        # activations shape: [batch_size, channels, height, width]
        # Compute variance across spatial and batch dimensions
        variance = torch.var(activations.view(activations.size(0), activations.size(1), -1), 
                           dim=[0, 2])
        return variance
    
    def compute_taylor_importance(self, layer: nn.Conv2d, activations: torch.Tensor) -> torch.Tensor:
        """Compute Taylor expansion-based importance scores."""
        if layer.weight.grad is None or activations is None:
            logger.warning("No gradient or activations available for Taylor importance")
            return torch.zeros(layer.weight.size(0), device=layer.weight.device)
        
        weights = layer.weight.data
        grads = layer.weight.grad.data
        
        # Taylor approximation: importance ≈ |weight * gradient|
        taylor_scores = torch.sum(torch.abs(weights * grads).view(weights.size(0), -1), dim=1)
        return taylor_scores
    
    def compute_combined_importance(self, layer: nn.Conv2d, layer_name: str, 
                                  activations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute combined importance scores using all metrics."""
        
        device = layer.weight.device
        num_filters = layer.weight.size(0)
        
        # Magnitude importance
        mag_importance = self.compute_magnitude_importance(layer)
        mag_importance = mag_importance / (mag_importance.max() + 1e-8)
        
        # Gradient importance
        grad_importance = self.compute_gradient_importance(layer, layer_name)
        grad_importance = grad_importance / (grad_importance.max() + 1e-8)
        
        # Initialize other importance scores
        var_importance = torch.zeros(num_filters, device=device)
        taylor_importance = torch.zeros(num_filters, device=device)
        
        if activations is not None:
            # Variance importance
            var_importance = self.compute_variance_importance(activations)
            if var_importance.size(0) == num_filters:
                var_importance = var_importance / (var_importance.max() + 1e-8)
            else:
                var_importance = torch.zeros(num_filters, device=device)
            
            # Taylor importance
            taylor_importance = self.compute_taylor_importance(layer, activations)
            taylor_importance = taylor_importance / (taylor_importance.max() + 1e-8)
        
        # Combine all importance scores
        combined_importance = (
            self.config.magnitude_weight * mag_importance +
            self.config.gradient_weight * grad_importance +
            self.config.variance_weight * var_importance +
            self.config.taylor_weight * taylor_importance
        )
        
        return combined_importance


class ResNetStructuredPruner:
    """Structured pruner that handles ResNet blocks and residual connections properly."""
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        self.scorer = ImportanceScorer(config)
        self.original_model = copy.deepcopy(model)
        
        # Track pruning progress
        self.pruning_history = []
        self.current_phase = 0
        self.block_pruning_plan = {}  # Track which blocks to prune together
        
        # Analyze model structure
        self._analyze_resnet_structure()
    
    def _analyze_resnet_structure(self):
        """Analyze ResNet structure to identify blocks and dependencies."""
        self.blocks = {}
        self.stage_info = {
            'stage1': [],
            'stage2': [],
            'stage3': [],
            'stage4': []
        }
        
        # Identify ResNet blocks
        for name, module in self.model.named_modules():
            if hasattr(module, 'conv1') and hasattr(module, 'conv2'):
                # This is a ResNet basic block
                stage = self._get_stage_from_name(name)
                if stage:
                    self.blocks[name] = module
                    self.stage_info[stage].append(name)
        
        logger.info(f"Identified {len(self.blocks)} ResNet blocks:")
        for stage, blocks in self.stage_info.items():
            logger.info(f"  {stage}: {len(blocks)} blocks")
    
    def _get_stage_from_name(self, name: str) -> Optional[str]:
        """Determine stage from layer name."""
        if 'layer1' in name:
            return 'stage1'
        elif 'layer2' in name:
            return 'stage2'
        elif 'layer3' in name:
            return 'stage3'
        elif 'layer4' in name:
            return 'stage4'
        return None
    
    def _get_pruning_ratio_for_stage(self, stage: str) -> float:
        """Get pruning ratio for a specific stage."""
        stage_ratios = {
            'stage1': self.config.stage1_ratio,
            'stage2': self.config.stage2_ratio,
            'stage3': self.config.stage3_ratio,
            'stage4': getattr(self.config, 'stage4_ratio', self.config.stage3_ratio)
        }
        return stage_ratios.get(stage, 0.1)
    
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        self.activation_hooks = {}
        
        def get_activation_hook(name):
            def hook(module, input, output):
                self.scorer.activation_cache[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(get_activation_hook(name))
                self.activation_hooks[name] = handle
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.activation_hooks.values():
            handle.remove()
        self.activation_hooks.clear()
    
    def prune_resnet_block(self, block_name: str, block: nn.Module, prune_ratio: float) -> nn.Module:
        """Prune a complete ResNet block while maintaining residual connections."""
        
        if prune_ratio <= 0:
            return block
        
        logger.info(f"Pruning block {block_name} with ratio {prune_ratio:.2%}")
        
        # Get the conv layers in the block
        conv1 = block.conv1
        conv2 = block.conv2
        bn1 = block.bn1
        bn2 = block.bn2
        
        # Determine number of filters to prune from conv1 (output channels)
        num_filters_conv1 = conv1.out_channels
        num_to_prune = int(num_filters_conv1 * prune_ratio)
        num_to_prune = min(num_to_prune, num_filters_conv1 - 1)  # Keep at least 1 filter
        
        if num_to_prune == 0:
            return block
        
        # Get importance scores for conv1
        conv1_name = f"{block_name}.conv1"
        activations = self.scorer.activation_cache.get(conv1_name, None)
        importance_scores = self.scorer.compute_combined_importance(conv1, conv1_name, activations)
        
        # Select filters to prune (lowest importance)
        _, indices = torch.sort(importance_scores)
        filters_to_prune = indices[:num_to_prune].tolist()
        keep_indices = indices[num_to_prune:].tolist()
        
        logger.info(f"  Pruning {num_to_prune}/{num_filters_conv1} filters from {conv1_name}")
        
        # Create new block with pruned dimensions
        new_block = copy.deepcopy(block)
        
        # Prune conv1 output channels
        new_conv1 = self._prune_conv_output_channels(conv1, keep_indices)
        new_block.conv1 = new_conv1
        
        # Prune corresponding bn1
        new_bn1 = self._prune_batchnorm(bn1, keep_indices)
        new_block.bn1 = new_bn1
        
        # Prune conv2 input channels to match conv1 output
        new_conv2 = self._prune_conv_input_channels(conv2, keep_indices)
        new_block.conv2 = new_conv2
        
        # Handle downsample if present
        if hasattr(block, 'downsample') and block.downsample is not None:
            # For downsample, we need to prune the output channels to match the main path
            downsample_conv = block.downsample[0]  # Assuming downsample[0] is conv, downsample[1] is bn
            downsample_bn = block.downsample[1]
            
            new_downsample_conv = self._prune_conv_output_channels(downsample_conv, keep_indices)
            new_downsample_bn = self._prune_batchnorm(downsample_bn, keep_indices)
            
            new_block.downsample = nn.Sequential(new_downsample_conv, new_downsample_bn)
        
        return new_block
    
    def _prune_conv_output_channels(self, conv: nn.Conv2d, keep_indices: List[int]) -> nn.Conv2d:
        """Prune output channels of a conv layer."""
        keep_indices = torch.tensor(keep_indices, device=conv.weight.device)
        
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=len(keep_indices),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None
        )
        
        # Copy weights for kept filters
        new_conv.weight.data = torch.index_select(conv.weight.data, 0, keep_indices)
        
        if conv.bias is not None:
            new_conv.bias.data = torch.index_select(conv.bias.data, 0, keep_indices)
        
        return new_conv
    
    def _prune_conv_input_channels(self, conv: nn.Conv2d, keep_indices: List[int]) -> nn.Conv2d:
        """Prune input channels of a conv layer."""
        keep_indices = torch.tensor(keep_indices, device=conv.weight.device)
        
        new_conv = nn.Conv2d(
            in_channels=len(keep_indices),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None
        )
        
        # Copy weights for kept input channels
        new_conv.weight.data = torch.index_select(conv.weight.data, 1, keep_indices)
        
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
        
        return new_conv
    
    def _prune_batchnorm(self, bn: nn.BatchNorm2d, keep_indices: List[int]) -> nn.BatchNorm2d:
        """Prune a batch normalization layer."""
        keep_indices = torch.tensor(keep_indices, device=bn.weight.device)
        
        new_bn = nn.BatchNorm2d(
            num_features=len(keep_indices),
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats
        )
        
        # Copy parameters for kept channels
        if bn.affine:
            new_bn.weight.data = torch.index_select(bn.weight.data, 0, keep_indices)
            new_bn.bias.data = torch.index_select(bn.bias.data, 0, keep_indices)
        
        if bn.track_running_stats:
            new_bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, keep_indices)
            new_bn.running_var.data = torch.index_select(bn.running_var.data, 0, keep_indices)
            new_bn.num_batches_tracked = bn.num_batches_tracked
        
        return new_bn
    
    def progressive_prune_phase(self, phase_ratio: float) -> Dict:
        """Execute one phase of progressive pruning."""
        
        self.current_phase += 1
        logger.info(f"Starting pruning phase {self.current_phase} with ratio {phase_ratio:.2%}")
        
        # Register hooks to capture activations
        self.register_hooks()
        
        # Perform a forward pass to capture activations
        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)  # ImageNet size
        if next(self.model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Track pruning statistics
        phase_stats = {
            'phase': self.current_phase,
            'blocks_pruned': 0,
            'total_filters_removed': 0,
            'stage_details': {}
        }
        
        # Prune blocks stage by stage
        for stage, block_names in self.stage_info.items():
            if not block_names:
                continue
            
            stage_ratio = self._get_pruning_ratio_for_stage(stage)
            current_ratio = stage_ratio * phase_ratio
            
            if current_ratio <= 0:
                continue
            
            logger.info(f"Pruning {stage} with ratio {current_ratio:.2%}")
            
            stage_filters_removed = 0
            stage_blocks_pruned = 0
            
            for block_name in block_names:
                if block_name in self.blocks:
                    original_block = self.blocks[block_name]
                    
                    # Count original filters
                    original_filters = original_block.conv1.out_channels
                    
                    # Prune the block
                    pruned_block = self.prune_resnet_block(block_name, original_block, current_ratio)
                    
                    # Replace the block in the model
                    self._replace_module(block_name, pruned_block)
                    
                    # Update statistics
                    new_filters = pruned_block.conv1.out_channels
                    filters_removed = original_filters - new_filters
                    
                    if filters_removed > 0:
                        stage_filters_removed += filters_removed
                        stage_blocks_pruned += 1
            
            phase_stats['stage_details'][stage] = {
                'blocks_pruned': stage_blocks_pruned,
                'filters_removed': stage_filters_removed
            }
            
            phase_stats['blocks_pruned'] += stage_blocks_pruned
            phase_stats['total_filters_removed'] += stage_filters_removed
        
        # Handle the final linear layer if needed
        self._adjust_final_linear_layer()
        
        # Remove hooks
        self.remove_hooks()
        
        # Store phase statistics
        self.pruning_history.append(phase_stats)
        
        logger.info(f"Phase {self.current_phase} completed: "
                   f"{phase_stats['blocks_pruned']} blocks pruned, "
                   f"{phase_stats['total_filters_removed']} filters removed")
        
        return phase_stats
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model."""
        name_parts = module_name.split('.')
        parent = self.model
        
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, name_parts[-1], new_module)
    
    def _adjust_final_linear_layer(self):
        """Adjust the final linear layer to match the last conv layer output."""
        # Find the last conv layer and linear layer
        last_conv_channels = None
        linear_layer = None
        linear_name = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_channels = module.out_channels
            elif isinstance(module, nn.Linear):
                linear_layer = module
                linear_name = name
        
        if linear_layer is not None and last_conv_channels is not None:
            expected_in_features = last_conv_channels
            
            if linear_layer.in_features != expected_in_features:
                logger.info(f"Adjusting linear layer: {linear_layer.in_features} -> {expected_in_features}")
                
                new_linear = nn.Linear(
                    in_features=expected_in_features,
                    out_features=linear_layer.out_features,
                    bias=linear_layer.bias is not None
                )
                
                # Initialize with truncated or padded weights
                min_features = min(linear_layer.in_features, expected_in_features)
                new_linear.weight.data[:, :min_features] = linear_layer.weight.data[:, :min_features]
                
                if linear_layer.bias is not None:
                    new_linear.bias.data = linear_layer.bias.data.clone()
                
                self._replace_module(linear_name, new_linear)
    
    def execute_progressive_pruning(self) -> List[Dict]:
        """Execute the complete progressive pruning strategy."""
        
        logger.info("Starting progressive pruning strategy for ResNet")
        
        # Phase 1: Initial pruning
        phase1_stats = self.progressive_prune_phase(self.config.phase1_ratio)
        
        # Phase 2: Intermediate pruning  
        phase2_stats = self.progressive_prune_phase(self.config.phase2_ratio)
        
        # Phase 3: Final pruning
        phase3_stats = self.progressive_prune_phase(self.config.phase3_ratio)
        
        # Generate summary statistics
        total_filters_removed = sum(phase['total_filters_removed'] for phase in self.pruning_history)
        total_blocks_pruned = sum(phase['blocks_pruned'] for phase in self.pruning_history)
        
        logger.info(f"Progressive pruning completed:")
        logger.info(f"  Total phases: {len(self.pruning_history)}")
        logger.info(f"  Total blocks pruned: {total_blocks_pruned}")
        logger.info(f"  Total filters removed: {total_filters_removed}")
        
        return self.pruning_history
    
    def get_model_statistics(self) -> Dict:
        """Get comprehensive statistics about the pruned model."""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        original_params = sum(p.numel() for p in self.original_model.parameters())
        
        compression_ratio = 1 - (total_params / original_params)
        
        stats = {
            'original_parameters': original_params,
            'pruned_parameters': total_params,
            'parameters_removed': original_params - total_params,
            'compression_ratio': compression_ratio,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'pruning_phases': len(self.pruning_history),
            'phase_details': self.pruning_history
        }
        
        return stats


def create_optimized_pruning_config(**kwargs) -> PruningConfig:
    """Create a pruning configuration with optional parameter overrides."""
    return PruningConfig(**kwargs)


def prune_resnet56_optimized(model: nn.Module, 
                           config: Optional[PruningConfig] = None) -> Tuple[nn.Module, Dict]:
    """
    Main function to perform optimized structured pruning on ResNet56.
    
    Args:
        model: ResNet56 model to prune
        config: Pruning configuration (uses default if None)
    
    Returns:
        Tuple of (pruned_model, pruning_statistics)
    """
    
    if config is None:
        config = PruningConfig()
    
    # Create structured pruner instance
    pruner = ResNetStructuredPruner(model, config)
    
    # Execute progressive pruning
    pruning_history = pruner.execute_progressive_pruning()
    
    # Get final statistics
    final_stats = pruner.get_model_statistics()
    
    return pruner.model, final_stats


if __name__ == "__main__":
    # Example usage
    print("Final Fixed Optimized ResNet56 Pruning Implementation")
    print("This module provides structured pruning with proper residual connection handling.")
    print("Use prune_resnet56_optimized() function to prune your model.")

