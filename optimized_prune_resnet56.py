"""
Optimized Pruning Implementation for ResNet56
Author: Manus AI
Date: June 30, 2025

This module implements an advanced pruning strategy specifically designed for ResNet56
networks, incorporating multi-criteria importance scoring, progressive pruning schedules,
and layer-wise sensitivity analysis.
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
            return torch.zeros(layer.weight.size(0))
        
        grad = layer.weight.grad.data
        # Compute gradient magnitude for each filter
        importance = torch.sum(torch.abs(grad.view(grad.size(0), -1)), dim=1)
        return importance
    
    def compute_variance_importance(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute activation variance-based importance scores."""
        # activations shape: [batch_size, channels, height, width]
        # Compute variance across spatial and batch dimensions
        variance = torch.var(activations.view(activations.size(0), activations.size(1), -1), 
                           dim=[0, 2])
        return variance
    
    def compute_taylor_importance(self, layer: nn.Conv2d, activations: torch.Tensor) -> torch.Tensor:
        """Compute Taylor expansion-based importance scores."""
        if layer.weight.grad is None:
            logger.warning("No gradient available for Taylor importance")
            return torch.zeros(layer.weight.size(0))
        
        weights = layer.weight.data
        grads = layer.weight.grad.data
        
        # Taylor approximation: importance ≈ |weight * gradient|
        taylor_scores = torch.sum(torch.abs(weights * grads).view(weights.size(0), -1), dim=1)
        return taylor_scores
    
    def compute_combined_importance(self, layer: nn.Conv2d, layer_name: str, 
                                  activations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute combined importance scores using all metrics."""
        
        # Magnitude importance
        mag_importance = self.compute_magnitude_importance(layer)
        mag_importance = mag_importance / (mag_importance.max() + 1e-8)
        
        # Gradient importance
        grad_importance = self.compute_gradient_importance(layer, layer_name)
        grad_importance = grad_importance / (grad_importance.max() + 1e-8)
        
        # Initialize other importance scores
        var_importance = torch.zeros_like(mag_importance)
        taylor_importance = torch.zeros_like(mag_importance)
        
        if activations is not None:
            # Variance importance
            var_importance = self.compute_variance_importance(activations)
            var_importance = var_importance / (var_importance.max() + 1e-8)
            
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


class ResNet56Analyzer:
    """Analyzer for ResNet56 architecture to identify layer stages and dependencies."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_info = self._analyze_architecture()
    
    def _analyze_architecture(self) -> Dict:
        """Analyze ResNet56 architecture and categorize layers."""
        layer_info = {
            'stage1_layers': [],
            'stage2_layers': [], 
            'stage3_layers': [],
            'residual_blocks': [],
            'shortcut_layers': []
        }
        
        # Analyze the model structure
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'layer1' in name:
                    layer_info['stage1_layers'].append((name, module))
                elif 'layer2' in name:
                    layer_info['stage2_layers'].append((name, module))
                elif 'layer3' in name:
                    layer_info['stage3_layers'].append((name, module))
                
                # Identify shortcut connections
                if 'shortcut' in name or 'downsample' in name:
                    layer_info['shortcut_layers'].append((name, module))
        
        return layer_info
    
    def get_stage_for_layer(self, layer_name: str) -> int:
        """Determine which stage a layer belongs to."""
        if any(layer_name in name for name, _ in self.layer_info['stage1_layers']):
            return 1
        elif any(layer_name in name for name, _ in self.layer_info['stage2_layers']):
            return 2
        elif any(layer_name in name for name, _ in self.layer_info['stage3_layers']):
            return 3
        else:
            return 0  # Initial conv or classifier
    
    def get_pruning_ratio_for_layer(self, layer_name: str, config: PruningConfig) -> float:
        """Get the appropriate pruning ratio for a specific layer."""
        stage = self.get_stage_for_layer(layer_name)
        
        if stage == 1:
            return config.stage1_ratio
        elif stage == 2:
            return config.stage2_ratio
        elif stage == 3:
            return config.stage3_ratio
        else:
            # Conservative pruning for initial conv and classifier
            return 0.10


class OptimizedPruner:
    """Main class for optimized ResNet56 pruning."""
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        self.analyzer = ResNet56Analyzer(model)
        self.scorer = ImportanceScorer(config)
        self.original_model = copy.deepcopy(model)
        
        # Track pruning progress
        self.pruning_history = []
        self.current_phase = 0
        
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
    
    def identify_filters_to_prune(self, layer: nn.Conv2d, layer_name: str, 
                                 prune_ratio: float) -> List[int]:
        """Identify which filters to prune based on importance scores."""
        
        # Get activations if available
        activations = self.scorer.activation_cache.get(layer_name, None)
        
        # Compute combined importance scores
        importance_scores = self.scorer.compute_combined_importance(
            layer, layer_name, activations
        )
        
        # Determine number of filters to prune
        num_filters = layer.weight.size(0)
        num_to_prune = int(num_filters * prune_ratio)
        
        if num_to_prune == 0:
            return []
        
        # Select filters with lowest importance scores
        _, indices = torch.sort(importance_scores)
        filters_to_prune = indices[:num_to_prune].tolist()
        
        logger.info(f"Layer {layer_name}: Pruning {num_to_prune}/{num_filters} filters "
                   f"({prune_ratio:.2%})")
        
        return filters_to_prune
    
    def prune_conv_layer(self, layer: nn.Conv2d, filters_to_remove: List[int], 
                        dim: int = 0) -> Tuple[nn.Conv2d, Optional[torch.Tensor]]:
        """Prune a convolutional layer by removing specified filters."""
        
        if not filters_to_remove:
            return layer, None
        
        # Create indices for filters to keep
        all_indices = set(range(layer.weight.size(dim)))
        keep_indices = list(all_indices - set(filters_to_remove))
        keep_indices = torch.tensor(keep_indices, device=layer.weight.device)
        
        if dim == 0:
            # Pruning output channels
            new_out_channels = len(keep_indices)
            new_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=new_out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                bias=layer.bias is not None
            )
            
            # Copy weights for kept filters
            new_layer.weight.data = torch.index_select(layer.weight.data, dim, keep_indices)
            
            if layer.bias is not None:
                new_layer.bias.data = torch.index_select(layer.bias.data, 0, keep_indices)
            
            return new_layer, None
            
        else:
            # Pruning input channels
            new_in_channels = len(keep_indices)
            new_layer = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                bias=layer.bias is not None
            )
            
            # Copy weights for kept input channels
            new_layer.weight.data = torch.index_select(layer.weight.data, dim, keep_indices)
            
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            
            # Return residue for dependency tracking
            remove_indices = torch.tensor(filters_to_remove, device=layer.weight.device)
            residue = torch.index_select(layer.weight.data, dim, remove_indices)
            
            return new_layer, residue
    
    def prune_batch_norm(self, bn_layer: nn.BatchNorm2d, 
                        filters_to_remove: List[int]) -> nn.BatchNorm2d:
        """Prune a batch normalization layer."""
        
        if not filters_to_remove:
            return bn_layer
        
        # Create indices for channels to keep
        all_indices = set(range(bn_layer.num_features))
        keep_indices = list(all_indices - set(filters_to_remove))
        keep_indices = torch.tensor(keep_indices, device=bn_layer.weight.device)
        
        new_num_features = len(keep_indices)
        new_bn = nn.BatchNorm2d(
            num_features=new_num_features,
            eps=bn_layer.eps,
            momentum=bn_layer.momentum,
            affine=bn_layer.affine,
            track_running_stats=bn_layer.track_running_stats
        )
        
        # Copy parameters for kept channels
        if bn_layer.affine:
            new_bn.weight.data = torch.index_select(bn_layer.weight.data, 0, keep_indices)
            new_bn.bias.data = torch.index_select(bn_layer.bias.data, 0, keep_indices)
        
        if bn_layer.track_running_stats:
            new_bn.running_mean.data = torch.index_select(bn_layer.running_mean.data, 0, keep_indices)
            new_bn.running_var.data = torch.index_select(bn_layer.running_var.data, 0, keep_indices)
        
        return new_bn
    
    def progressive_prune_phase(self, phase_ratio: float) -> Dict:
        """Execute one phase of progressive pruning."""
        
        self.current_phase += 1
        logger.info(f"Starting pruning phase {self.current_phase} with ratio {phase_ratio:.2%}")
        
        # Register hooks to capture activations
        self.register_hooks()
        
        # Perform a forward pass to capture activations
        self.model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        if next(self.model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Track pruning statistics
        phase_stats = {
            'phase': self.current_phase,
            'layers_pruned': 0,
            'total_filters_removed': 0,
            'layer_details': {}
        }
        
        # Prune each convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and 'shortcut' not in name:
                # Get layer-specific pruning ratio
                base_ratio = self.analyzer.get_pruning_ratio_for_layer(name, self.config)
                current_ratio = base_ratio * phase_ratio
                
                # Identify filters to prune
                filters_to_prune = self.identify_filters_to_prune(
                    module, name, current_ratio
                )
                
                if filters_to_prune:
                    # Prune the layer
                    new_layer, _ = self.prune_conv_layer(module, filters_to_prune, dim=0)
                    
                    # Replace the layer in the model
                    self._replace_layer(name, new_layer)
                    
                    # Update statistics
                    phase_stats['layers_pruned'] += 1
                    phase_stats['total_filters_removed'] += len(filters_to_prune)
                    phase_stats['layer_details'][name] = {
                        'filters_removed': len(filters_to_prune),
                        'original_filters': module.weight.size(0),
                        'pruning_ratio': current_ratio
                    }
        
        # Remove hooks
        self.remove_hooks()
        
        # Store phase statistics
        self.pruning_history.append(phase_stats)
        
        logger.info(f"Phase {self.current_phase} completed: "
                   f"{phase_stats['layers_pruned']} layers pruned, "
                   f"{phase_stats['total_filters_removed']} filters removed")
        
        return phase_stats
    
    def _replace_layer(self, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model with a new layer."""
        # Navigate to the parent module and replace the layer
        name_parts = layer_name.split('.')
        parent = self.model
        
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, name_parts[-1], new_layer)
    
    def execute_progressive_pruning(self) -> List[Dict]:
        """Execute the complete progressive pruning strategy."""
        
        logger.info("Starting progressive pruning strategy for ResNet56")
        
        # Phase 1: Initial pruning
        phase1_stats = self.progressive_prune_phase(self.config.phase1_ratio)
        
        # Phase 2: Intermediate pruning  
        phase2_stats = self.progressive_prune_phase(self.config.phase2_ratio)
        
        # Phase 3: Final pruning
        phase3_stats = self.progressive_prune_phase(self.config.phase3_ratio)
        
        # Generate summary statistics
        total_filters_removed = sum(phase['total_filters_removed'] for phase in self.pruning_history)
        total_layers_pruned = len(set().union(*[phase['layer_details'].keys() for phase in self.pruning_history]))
        
        logger.info(f"Progressive pruning completed:")
        logger.info(f"  Total phases: {len(self.pruning_history)}")
        logger.info(f"  Total layers pruned: {total_layers_pruned}")
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
    Main function to perform optimized pruning on ResNet56.
    
    Args:
        model: ResNet56 model to prune
        config: Pruning configuration (uses default if None)
    
    Returns:
        Tuple of (pruned_model, pruning_statistics)
    """
    
    if config is None:
        config = PruningConfig()
    
    # Create pruner instance
    pruner = OptimizedPruner(model, config)
    
    # Execute progressive pruning
    pruning_history = pruner.execute_progressive_pruning()
    
    # Get final statistics
    final_stats = pruner.get_model_statistics()
    
    return pruner.model, final_stats


if __name__ == "__main__":
    # Example usage
    print("Optimized ResNet56 Pruning Implementation")
    print("This module provides advanced pruning capabilities for ResNet56 networks.")
    print("Use prune_resnet56_optimized() function to prune your model.")

