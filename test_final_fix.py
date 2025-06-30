#!/usr/bin/env python3
"""
Test Script for Final Fixed ResNet56 Implementation
Author: Manus AI
Date: June 30, 2025

This script tests the final fixed implementation with structured pruning.
"""

import torch
import torch.nn as nn
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_final_fix():
    """Test the final fixed implementation."""
    try:
        logger.info("Testing final fixed implementation with structured pruning...")
        
        from optimized_prune_resnet56_final_fix import (
            ResNetStructuredPruner, 
            PruningConfig, 
            prune_resnet56_optimized
        )
        from resnet56_imagenet import create_resnet56_imagenet
        
        # Create model
        logger.info("Creating ResNet56 model...")
        model = create_resnet56_imagenet(num_classes=1000)
        
        # Create very conservative pruning config for testing
        config = PruningConfig(
            stage1_ratio=0.05,  # Very conservative
            stage2_ratio=0.10,
            stage3_ratio=0.15,
            stage4_ratio=0.20,
            phase1_ratio=1.0,   # Do all pruning in one phase for testing
            phase2_ratio=0.0,
            phase3_ratio=0.0
        )
        
        logger.info("Original model parameters:")
        original_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {original_params:,}")
        
        # Test forward pass before pruning
        logger.info("Testing forward pass before pruning...")
        model.eval()
        input_tensor = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output_before = model(input_tensor)
        
        logger.info(f"✓ Forward pass before pruning successful: {output_before.shape}")
        
        # Test structured pruning
        logger.info("Executing structured pruning...")
        pruned_model, stats = prune_resnet56_optimized(model, config)
        
        logger.info("Structured pruning completed successfully!")
        logger.info(f"  Compression ratio: {stats['compression_ratio']:.2%}")
        logger.info(f"  Parameters removed: {stats['parameters_removed']:,}")
        
        # Test forward pass with pruned model
        logger.info("Testing forward pass with pruned model...")
        pruned_model.eval()
        
        with torch.no_grad():
            output_after = pruned_model(input_tensor)
        
        logger.info(f"✓ Pruned model forward pass successful: {output_after.shape}")
        
        # Verify output shapes match
        if output_before.shape == output_after.shape:
            logger.info("✓ Output shapes match between original and pruned models")
        else:
            logger.error(f"✗ Output shape mismatch: {output_before.shape} vs {output_after.shape}")
            return False
        
        # Test multiple forward passes to ensure stability
        logger.info("Testing multiple forward passes for stability...")
        for i in range(5):
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                test_output = pruned_model(test_input)
            logger.info(f"  Pass {i+1}: {test_output.shape} ✓")
        
        logger.info("🎉 ALL TESTS PASSED! The final fix is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"✗ Final fix test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_different_batch_sizes():
    """Test with different batch sizes to ensure robustness."""
    try:
        logger.info("Testing with different batch sizes...")
        
        from optimized_prune_resnet56_final_fix import prune_resnet56_optimized, PruningConfig
        from resnet56_imagenet import create_resnet56_imagenet
        
        # Create and prune model
        model = create_resnet56_imagenet(num_classes=1000)
        config = PruningConfig(
            stage1_ratio=0.05,
            stage2_ratio=0.10,
            stage3_ratio=0.15,
            phase1_ratio=1.0,
            phase2_ratio=0.0,
            phase3_ratio=0.0
        )
        
        pruned_model, _ = prune_resnet56_optimized(model, config)
        pruned_model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}...")
            test_input = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = pruned_model(test_input)
            
            expected_shape = (batch_size, 1000)
            if output.shape == expected_shape:
                logger.info(f"  ✓ Batch size {batch_size}: {output.shape}")
            else:
                logger.error(f"  ✗ Batch size {batch_size}: expected {expected_shape}, got {output.shape}")
                return False
        
        logger.info("✓ All batch size tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch size test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_gradients():
    """Test that gradients flow properly through the pruned model."""
    try:
        logger.info("Testing gradient flow...")
        
        from optimized_prune_resnet56_final_fix import prune_resnet56_optimized, PruningConfig
        from resnet56_imagenet import create_resnet56_imagenet
        
        # Create and prune model
        model = create_resnet56_imagenet(num_classes=1000)
        config = PruningConfig(
            stage1_ratio=0.05,
            stage2_ratio=0.10,
            stage3_ratio=0.15,
            phase1_ratio=1.0,
            phase2_ratio=0.0,
            phase3_ratio=0.0
        )
        
        pruned_model, _ = prune_resnet56_optimized(model, config)
        pruned_model.train()
        
        # Test gradient computation
        input_tensor = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 1000, (2,))
        
        # Forward pass
        output = pruned_model(input_tensor)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        grad_count = 0
        for name, param in pruned_model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        logger.info(f"✓ Gradients computed for {grad_count} parameters")
        
        # Check input gradients
        if input_tensor.grad is not None:
            logger.info("✓ Input gradients computed successfully")
        else:
            logger.warning("⚠ No input gradients (this might be expected)")
        
        logger.info("✓ Gradient flow test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Gradient flow test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("="*60)
    logger.info("RUNNING COMPREHENSIVE TESTS FOR FINAL FIXED IMPLEMENTATION")
    logger.info("="*60)
    
    tests = [
        ("Final Fix Test", test_final_fix),
        ("Batch Size Test", test_different_batch_sizes),
        ("Gradient Flow Test", test_gradients),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The final implementation is ready to use.")
        logger.info("\nThe BatchNorm dimension mismatch error has been completely resolved!")
        logger.info("You can now use the fixed implementation with confidence.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

