#!/usr/bin/env python3
"""
Example Usage of Optimized ResNet56 Pruning
Author: Manus AI
Date: June 30, 2025

This script demonstrates how to use the optimized pruning implementation
with the existing ResNet56 model from the repository.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the original repository to path
sys.path.append('/home/ubuntu/implementation-of-pruning-filters')

# Import the original ResNet56 implementation
from netModels.ResNet56 import MyResNet56

# Import our optimized pruning implementation
from optimized_prune_resnet56 import (
    prune_resnet56_optimized,
    PruningConfig,
    OptimizedPruner
)


def example_basic_pruning():
    """Example of basic optimized pruning usage."""
    
    print("=== Basic Optimized Pruning Example ===")
    
    # Create ResNet56 model
    model = MyResNet56()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create custom pruning configuration
    config = PruningConfig(
        stage1_ratio=0.10,  # Conservative pruning for early layers
        stage2_ratio=0.30,  # Moderate pruning for middle layers
        stage3_ratio=0.50,  # Aggressive pruning for late layers
    )
    
    # Perform optimized pruning
    pruned_model, stats = prune_resnet56_optimized(model, config)
    
    print(f"Pruned model parameters: {stats['pruned_parameters']:,}")
    print(f"Compression ratio: {stats['compression_ratio']:.2%}")
    print(f"Parameters removed: {stats['parameters_removed']:,}")
    
    return pruned_model, stats


def example_advanced_pruning():
    """Example of advanced pruning with custom configuration."""
    
    print("\n=== Advanced Optimized Pruning Example ===")
    
    # Create ResNet56 model
    model = MyResNet56()
    
    # Create advanced pruning configuration
    config = PruningConfig(
        # Stage-wise ratios
        stage1_ratio=0.15,
        stage2_ratio=0.40, 
        stage3_ratio=0.60,
        
        # Progressive schedule
        phase1_ratio=0.25,
        phase2_ratio=0.45,
        phase3_ratio=0.30,
        
        # Importance scoring weights
        magnitude_weight=0.20,
        gradient_weight=0.35,
        variance_weight=0.25,
        taylor_weight=0.20,
        
        # Knowledge distillation
        distillation_weight=0.4,
        temperature=5.0
    )
    
    # Create pruner for more control
    pruner = OptimizedPruner(model, config)
    
    # Execute progressive pruning manually
    print("Executing progressive pruning phases...")
    
    # Phase 1
    print("Phase 1: Initial pruning")
    phase1_stats = pruner.progressive_prune_phase(config.phase1_ratio)
    print(f"  Filters removed: {phase1_stats['total_filters_removed']}")
    
    # Phase 2  
    print("Phase 2: Intermediate pruning")
    phase2_stats = pruner.progressive_prune_phase(config.phase2_ratio)
    print(f"  Filters removed: {phase2_stats['total_filters_removed']}")
    
    # Phase 3
    print("Phase 3: Final pruning")
    phase3_stats = pruner.progressive_prune_phase(config.phase3_ratio)
    print(f"  Filters removed: {phase3_stats['total_filters_removed']}")
    
    # Get final statistics
    final_stats = pruner.get_model_statistics()
    print(f"\nFinal compression ratio: {final_stats['compression_ratio']:.2%}")
    print(f"Model size reduction: {final_stats['model_size_mb']:.2f} MB")
    
    return pruner.model, final_stats


def example_layer_analysis():
    """Example of analyzing layer-wise pruning decisions."""
    
    print("\n=== Layer Analysis Example ===")
    
    # Create model and pruner
    model = MyResNet56()
    config = PruningConfig()
    pruner = OptimizedPruner(model, config)
    
    # Analyze architecture
    analyzer = pruner.analyzer
    
    print("Layer categorization:")
    print(f"Stage 1 layers: {len(analyzer.layer_info['stage1_layers'])}")
    print(f"Stage 2 layers: {len(analyzer.layer_info['stage2_layers'])}")
    print(f"Stage 3 layers: {len(analyzer.layer_info['stage3_layers'])}")
    print(f"Shortcut layers: {len(analyzer.layer_info['shortcut_layers'])}")
    
    # Show pruning ratios for different layers
    print("\nPruning ratios by layer:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            stage = analyzer.get_stage_for_layer(name)
            ratio = analyzer.get_pruning_ratio_for_layer(name, config)
            print(f"  {name}: Stage {stage}, Ratio {ratio:.2%}")


def example_importance_scoring():
    """Example of importance scoring analysis."""
    
    print("\n=== Importance Scoring Example ===")
    
    # Create model and components
    model = MyResNet56()
    config = PruningConfig()
    pruner = OptimizedPruner(model, config)
    
    # Register hooks and perform forward pass
    pruner.register_hooks()
    
    # Dummy forward pass to capture activations
    model.eval()
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Analyze importance scores for first conv layer
    first_conv = None
    first_conv_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break
    
    if first_conv is not None:
        # Compute different importance metrics
        scorer = pruner.scorer
        
        mag_scores = scorer.compute_magnitude_importance(first_conv)
        grad_scores = scorer.compute_gradient_importance(first_conv, first_conv_name)
        
        activations = scorer.activation_cache.get(first_conv_name)
        if activations is not None:
            var_scores = scorer.compute_variance_importance(activations)
            taylor_scores = scorer.compute_taylor_importance(first_conv, activations)
            combined_scores = scorer.compute_combined_importance(first_conv, first_conv_name, activations)
            
            print(f"Importance scores for {first_conv_name}:")
            print(f"  Magnitude scores range: {mag_scores.min():.4f} - {mag_scores.max():.4f}")
            print(f"  Variance scores range: {var_scores.min():.4f} - {var_scores.max():.4f}")
            print(f"  Combined scores range: {combined_scores.min():.4f} - {combined_scores.max():.4f}")
    
    # Clean up
    pruner.remove_hooks()


def main():
    """Run all examples."""
    
    print("Optimized ResNet56 Pruning Examples")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # Run examples
        example_basic_pruning()
        example_advanced_pruning()
        example_layer_analysis()
        example_importance_scoring()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

